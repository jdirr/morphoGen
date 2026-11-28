# Architecture

This document describes the modules, data model, conventions and test strategy of
morphoGen 1.0, for those who change its code; the configuration files, the commands
and the files of a run are in the [configuration reference](configuration.md). The requirements are in [specification.md](specification.md)
and the formulas in [theory.md](theory.md).

## 1. Overview

```
            ┌──────────── config (YAML/JSON/TOML) ─── cli ─── pipeline ──────────────┐
            │                                                                         │
 Model ─ generate(domain, seed) ─> Microstructure ─ voxelize ──> VoxelImage ─┬─> analysis
 (parameters, theory)              (vector geometry)                         ├─> mesh (hexahedra, STL)
                                        │                                    ├─> homogenization (FFT)
                                        ├ level_set ─> conforming_mesh ─> TetMesh ─> homogenization (FE)
                                        │              (MMG)                 ├─> io (files, solvers)
                                        └ cell_complex ─> polycrystal_mesh ──┘   └─> rve (statistics)
                                          (exact cells)    (MMG, by grain)
 parallel: realizations in processes, tasks, MPI ranks; records, merge; cluster job scripts
```

* A **model** holds the parameters of a random set or of a deterministic architecture,
  together with its closed-form characteristics (`model.theory(domain)`).
* `model.generate(domain, seed)` returns a **microstructure**: the explicit ("vector")
  description of one realization (sphere centres and radii, fibre axes, germs and
  weights, Fourier modes, surface equations). Nothing is discretized at that point.
* A microstructure can be voxelized at any resolution (`voxelize`), turned into a
  level-set field (`level_set`), queried point-wise (`contains`) and serialized
  (`to_dict` / JSON).
* **Images** feed the measurements (`analysis`), the voxel meshes (`mesh`), the FFT
  solver (`homogenization.fft`), the files (`io`) and the RVE statistics (`rve`).
* **Conforming tetrahedral meshes** (`mesh.conforming`, MMG) follow the interfaces of
  the implicit morphologies, and polycrystals are meshed grain by grain from their exact
  cells (`mesh.polycrystal`); the finite element solver (`homogenization.fe`) works on
  them, with isotropic phases or crystals turned by the orientations of the grains.
  **Exact meshes** (`mesh.exact`, `method: exact`, the default through `auto`) need no
  background grid: faceted morphologies are meshed from their exact convex pieces, the
  others from their exact point-wise level set (`level_set_at`) on an adaptive octree
  (`mesh.octree`), their interface nodes moved onto it.
  Voxels are reserved for the FFT solver and for image measurements.
* **pipeline** runs configurations: batches of realizations and RVE studies;
  **parallel** runs their realizations in processes, in tasks of cluster job arrays or
  under MPI, and merges their records.
* **Combinations** (`combination`) build models from models: set operations,
  inclusions in a matrix; tessellations can have a grain-boundary phase.
* **Automata and growth processes** (`automata`) produce lattice microstructures
  (`LatticeField`) through the models `cellular_automaton`, `activator_inhibitor`,
  `reaction_diffusion`, `nucleation_growth` and `aggregation`, which fit the same
  pipeline (voxels, level sets, conforming meshes, homogenization, RVE studies). On
  other lattices (`lattice`: triangles, hexagons, Penrose rhombi, Ammann–Beenker tiles,
  prisms, BCC and FCC Voronoi cells, golden rhombohedra), the automata and
  reaction–diffusion give `TilingField`s, rasterized cell by cell; the quasi-periodic
  tilings fill periodic boxes through their periodic approximants (`approximant`).
* **Combinatorial design** (`design`) assembles tiles (`tile_assembly`), builds strut
  frameworks with defects and analyses their rigidity (`random_lattice`), and searches
  design spaces (annealing, genetic algorithm, inverse design of automaton rules).
* **Topology optimization** (`optimization`) designs periodic cells on voxels with the
  FFT solver; designs become lattice microstructures.
* **Visualization** (`viz`) draws figures and animations; the **interface** (`gui`)
  serves a page in the browser that drives all of the above and draws its morphologies
  and simulations in 3D (WebGL).
* **Nested models** take models as parameters (Cox spheres in aggregates, coatings,
  nested combinations); hierarchical and fractal architectures and **bio-inspired
  models** (nacre, Bouligand plywoods, honeycombs and wood, osteons, open foams and
  trabecular bone, branching networks) are models like the others.
* **Homogenization** solves loadings that mix macroscopic strains and stresses (the
  periodic element), with thermal strains, and follows loading paths with nonlinear
  behaviours by finite elements or by FFT on voxels; **post-processing** gives engineering
  constants, anisotropy indices and normalized values; **results** keeps every run in a
  database and draws normalized Ashby charts.

## 2. Package layout

```
src/morphogen/
  __init__.py        public API (Domain, Grid, VoxelImage, models, analysis, io, mesh, rve)
  domain.py          Domain (box, periodicity, unit), Grid (voxel grid conventions)
  image.py           VoxelImage (array + spacing + origin + periodicity + metadata)
  seeding.py         reproducible seeds (SeedSequence derivation)
  distributions.py   distributions of radii, lengths... (moments, quadrature, configs;
                     scaled: the law of s X)
  orientations.py    fibre-axis distributions (isotropic, aligned, planar, Schladitz's
                     beta law, Gaussian girdle, von Mises-Fisher, mixtures), their
                     quadratures and the law of the angle to a direction, uniform crystal
                     orientations (Bunge)
  texture.py         crystallographic textures: the rotations of the Laue classes,
                     disorientations and Mackenzie's distribution, mean orientations,
                     poles and their projections, pole figures (density in multiples of
                     a random texture), inverse pole figures and their colours, the
                     disorientations of the boundaries of an image of grains, textures
                     drawn from measured orientations
  symmetry.py        finite symmetry groups of periodic cells, real irreps from the
                     regular representation, axial isotropy subgroups (EBL)
  primitives.py      Spheres, Segments (cylinders/spherocylinders), periodic images
  theory.py          shared closed-form results (covariograms of spheres, cylinders and
                     spherocylinders, covariances of oriented grains, Voronoi constants...)
  settings.py        number of threads; the BLAS on one thread while the compiled
                     kernels run on theirs (blas_on_one_thread)
  _public.py         the public interface (docs/stability.md): its modules and names, for
                     the API reference of the documentation site and its test
  stopping.py        cooperative stops of long computations (checkpoints, Stopped)
  gpu.py             optional GPU backend: the choice of the device (CuPy, NVIDIA GPUs)
  kernels/           Numba kernels (array in, array out)
    raster.py          cell lists, union rasterization, signed distances
    union.py           Python front end of raster.py
    rsa.py             random sequential addition (spheres; spherocylinders: closest
                       points of segments)
    implicit.py        TPMS level-set functions
    leaves.py          first covering leaf (sphere or spheroid) of each voxel (dead
                       leaves); the field of transparent leaves
    cells.py           implicit function of a union of convex cells from the planes
                       of their faces (the cells of a tiling that have a phase)
    collective.py      stealthy hyperuniform point patterns (collective coordinates,
                       L-BFGS)
    packing.py         Lubachevsky-Stillinger: event-driven molecular dynamics of
                       growing hard spheres (dense packings)
    force_biased.py    force-biased packing of chains of spherocylinders (Altendorf &
                       Jeulin): repulsions, springs, open angular springs; contact
                       forces (Keilmann et al.); the initial placement of least overlap
                       (Altendorf, Jeulin & Willot)
    long_fibres.py     the sections of long fibres packed in a periodic rectangle (the
                       overlaps of discs pushed apart as they grow), the overlaps of wavy
                       fibres removed in space (closed chains moved in their planes)
    seeds.py           seeds made of spheres placed the largest first within an overlap
                       tolerance (MicroStructPy): grids of linked lists by levels of sizes,
                       the crowded seeds left out or placed where they overlap least
    annealing.py       Yeong-Torquato reconstruction: swaps annealed to S2 and L
    watershed.py       marker-controlled watershed (Meyer's flooding) and the maxima
                       of a given prominence (union-find), of planes and of volumes
                       (26-connected voxels, periodic faces)
    textile.py         the yarns at points and voxel centres: the closest segment of each
                       yarn (runs of a yarn in the cells), the function of its section,
                       the deepest yarn; sub-samples of partial volumes
    meshes.py          the voxels inside closed triangle surfaces (the parity of the
                       crossings of the columns of their centres)
  models/            morphological models and microstructure kinds
    base.py            Model, Microstructure, registries, parameter metadata
    affine.py          Anisotropy, AffineStructure: the affine anisotropy of the models
                       (the isotropic model in the preimage of the domain, mapped)
    union.py           PrimitiveUnion (unions of spheres and segments)
    pieces.py          the smooth pieces of the phases of structures of primitives:
                       overlaps (periodic images included), colourings, pieces of a
                       union, of its coatings and bores (for the meshes that follow
                       sharp edges)
    implicit.py        the implicit functions of the phases: bands, the points of a
                       grid near an object, objects that cover one another
                       (leaves, osteons)
    boolean.py         BooleanSpheres, BooleanCylinders
    fibers.py          PoissonFibers, HardcoreFibers (RSA of spherocylinders),
                       CurvedFibers (worm-like chains, arcs, helices; Boolean or
                       RSA of chains)
    long_fibers.py     LongFibers -> PrimitiveUnion (long fibres along an axis, closed
                       across the period: unidirectional composites, their sections a
                       random packing or an array, straight or wavy, the waviness a
                       random field correlated across the fibres)
    hardcore.py        HardcoreSpheres (RSA or Lubachevsky-Stillinger),
                       PenetrableSpheres (the cherry-pit model)
    tessellation.py    Voronoi, Laguerre (Poisson or packed germs, grain-boundary
                       phase) -> Tessellation
    seeded.py          SeededLaguerre -> SeededTessellation (grains grown from ellipsoidal
                       seeds: the spheres of their medial surfaces, the grain of a seed the
                       union of the Laguerre cells of its spheres; ellipsoids or boxes, in
                       space or in the plane; the seeds of each phase of their own laws,
                       amorphous phases merged; the grains against their seeds)
    gaussian.py        GaussianExcursion -> GaussianField
    tpms.py            TPMS -> TPMSStructure
    lattice.py         StrutLattice (cubic, bcc, fcc, octet, kelvin, diamond,
                       pyrochlore; hollow, rotated or graded struts) -> PrimitiveUnion
    laminate.py        RandomLaminate -> Laminate (layers along an axis, holes)
    deadleaves.py      DeadLeavesModel -> DeadLeaves (leaves in covering order:
                       families, spheroids, background of a finite coverage; their
                       bivariate laws)
    reconstruction.py  Reconstruction -> LatticeField (Yeong-Torquato: Debye media,
                       models, images)
    transparent.py     TransparentDeadLeavesModel -> TransparentDeadLeaves (the random
                       function of transparent leaves, its excursion)
    combination.py     Combination -> Combined (set operations of components,
                       inclusions in a matrix)
    lattice_field.py   LatticeField: phases (labels, field) per cell of a lattice
    ebsd.py            EBSD -> LatticeField (a window of an orientation map: the grains
                       of the map, their phases and mean orientations)
    tiling_field.py    TilingField: a phase per cell of a tiling, rasterized
    automata.py        CellularAutomatonModel, ActivatorInhibitorModel,
                       ReactionDiffusionModel -> LatticeField (TilingField on
                       other lattices)
    growth.py          NucleationGrowthModel, AggregationModel -> LatticeField
    design.py          TileAssembly -> LatticeField, RandomLattice -> PrimitiveUnion
    nested.py          CoxSpheres -> ClusteredSpheres; Coating -> CoatedStructure
                       (coatings, shells, hollow members); FractalSponge ->
                       FractalStructure
    nested_grains.py   NestedGrains -> NestedGrainStructure (grains of a tessellation with
                       subgrains of an inner tessellation, turned by small angles; or filled
                       with the phase 1 of another model, turned with their crystals); the
                       subgrains as exact convex pieces, the functions of filled grains at
                       any point
    hierarchical.py    HierarchicalLattice: frames of generation G of (hollow) tubes
    planar.py          PlanarCells: Voronoi cells of the plane and distances to their
                       sides; hard-core germs of the plane
    nacre.py           Nacre -> NacreStructure (layers of tablets)
    bouligand.py       Bouligand -> BouligandStructure (twisted layers of fibres,
                       periodic fibre families)
    honeycomb.py       Honeycomb -> PrismaticCells (honeycombs; wood: RadialFiles,
                       WoodRays between the files)
    osteons.py         Osteons -> OsteonStructure (dead leaves of cylinders)
    foam.py            OpenFoam -> PrimitiveUnion (edges of periodic Voronoi cells,
                       Lloyd relaxation, anisotropy); DelaunayNetwork -> PrimitiveUnion
                       (fibres on the edges of periodic Delaunay triangulations of
                       Poisson or hard-core points, in space or in a plane; Ban et al.)
    levelset_foam.py   LevelSetFoam -> DistanceFieldFoam (level set of the distance
                       fields of packed ellipsoids: Plateau borders, nodes, walls);
                       calibrate_seeds: seeds fitted to measured section radii
    branching.py       BranchingNetwork -> PrimitiveUnion (space colonization,
                       Murray's law)
    scale_rich.py      ScaleRichNetwork -> ScaleRichStructure (ligaments of power-law
                       thicknesses grown until they meet, Both et al. 2026: convex
                       voids split by strips, exact partition; the dual graph, the
                       orientation tensor; the density and jamming in closed form);
                       with dimension 3 -> ScaleRichPlates or ScaleRichBeams
    scale_rich_3d.py   the growth in space: plates splitting convex polyhedral voids
                       (clipping, sections, volumes; closed cells, exact partition),
                       beams grown until they touch (segment distances); the first
                       plate area in closed form, the beams' mean field; fields of
                       the plates and beams in buckets (voxels, level sets)
    textile.py         WovenTextile -> TextileStructure (yarns swept along the crimped
                       paths of 2D weaves, plies, and 3D weaves with binders: elliptic,
                       lenticular or power-ellipse sections, a partition of their union,
                       the share where they interpenetrate; a label and an orientation
                       per segment)
    microgen.py        Microgen -> MicrogenStructure (a shape of Microgen 2, 3MAH: its
                       field at any point, the inside voxelized, meshed, saved and read
                       back with the description of the shape)
    mesh_particles.py  MeshParticles (particles given by closed triangle surfaces: their
                       voxels by parity, their function the signed distance of the voxels)
    _sizes.py          the sizes of the objects: a number (monodisperse) or a law, a size
                       per object (polydisperse); the scale of a law solved for a volume
                       fraction; the moments of the sum of two sizes
  automata/          cellular automata and lattice growth
    lattice.py         neighbourhoods, neighbour counts (Numba), FFT kernel sums,
                       sublattices
    rules.py           Life-like, majority, annealing, conserved, probabilistic,
                       activator-inhibitor rules
    engine.py          CellularAutomaton (updates, histories), random states
    reaction.py        Gray-Scott reaction-diffusion on the grid and on tilings
                       (finite-volume Laplacian, stable substeps)
    tilings.py         Tiling: triangles, hexagons, Penrose rhombi (pentagrid),
                       Ammann-Beenker tiles (four grids), prisms, BCC/FCC Voronoi cells,
                       golden rhombohedra (six grids of planes), their periodic
                       approximants (rational grids, approximant_period); CSR
                       neighbourhoods, face measures, point location
    growth.py          nucleation and growth (event-driven), Potts grain growth,
                       DLA / Eden / ballistic aggregation
  design/            combinatorial design
    tiles.py           TileSet (labelled faces, rotations, geometry), polarized blocks,
                       pipe tiles
    csp.py             wave function collapse with backtracking, transfer matrices,
                       exact count of polarized blocks
    frames.py          periodic pin-jointed frameworks: lattices, dilution, rigidity
    search.py          simulated annealing, genetic algorithm, fit_rule
  optimization/      topology optimization of periodic cells
    cell.py            graded voxel cells: homogenized tensors and sensitivities (FFT)
    filters.py         density filter, Heaviside projection, symmetry
    mma.py             method of moving asymptotes
    topology.py        CellOptimization (objectives; OC, MMA, BESO, HCA), CellDesign
                       (hinges, re-evaluation on a finer grid), count_hinges
    run.py             optimization configuration files and outputs
  iga/               isogeometric analysis and shape optimization
    basis.py           B-spline functions and derivatives, knot vectors, the operators
                       of their refinement (knots inserted, degree raised)
    geometry.py        Patch (NURBS surfaces and solids) and Multipatch (patches that
                       share sides; the periods of a cell), their refinements and the
                       map design -> analysis, the Jacobians of the patches, their
                       image on a grid (for the voxel solvers)
    shapes.py          rectangles, rings, a plate and periodic cells with holes (exact
                       arcs; boxes and parallelograms), the sheet with alternating
                       holes (rotating squares), tilings, repetitions, extrusions
    analysis.py        Analysis: elasticity (plane strain, plane stress, solids) and
                       conduction; loads on sides, fields, error norms
    homogenization.py  Cell: periodic conditions on matching control points (the
                       sides of a box, or a period apart on any lattice), effective
                       stiffness and conductivity; plane engineering constants
    optimization.py    ShapeDesign (variables on control points, linked by the
                       symmetries of a pattern), analytical shape sensitivities,
                       optimize_cell and optimize_structure (SLSQP, MMA; linear
                       constraints on the variables)
  learning/          behaviours learned from data
    data.py            random strain paths, the responses of a material point or of a
                       microstructure along them (Green-Lagrange measures in finite
                       strains), splits, scales
    networks.py        FeedForward, Potential (input-convex: hyperelastic), Recurrent
                       (gated recurrent unit, Numba backpropagation through time), their
                       gradients and tangents written out; in finite strains each of the
                       logarithmic strain 1/2 ln C (its stress T, S = J^T T; the default)
                       or of the Green-Lagrange strain (the option strain); .npz files
                       with JSON metadata (those without strain: laws of E)
    logarithmic.py     the logarithmic strain 1/2 ln C of Green-Lagrange strains, dH/dE
                       and the pull-back S = J^T T with its geometric tangent (Numba
                       kernels of the divided differences of homogenization.tensors),
                       shared by the models of ln C
    standard.py        Standard: a learned generalized standard material (convex free
                       energy and dissipation of learned internal variables), the
                       incremental variational update by Newton's method, its symmetric
                       tangent, trained through the minimizations by implicit
                       differentiation (Numba kernels, gradients written out); in finite
                       strains its free energy of the logarithmic strain 1/2 ln C
                       (S = J^T T, the kernels of logarithmic.py) or of the
                       Green-Lagrange strain
    training.py        Adam, mini-batches, early stopping, random rotations of the paths
                       (uniform, or the 24 of the cube for a crystal in its axes)
    behaviour.py       learned laws in the solvers (the seam of the MFront points),
                       P = F S in finite strains; turned by the orientation of their
                       grains (the axes of their material), the stress-free thermal
                       strain taken out (in finite strains as the Hencky points do,
                       1/2 ln C - alpha dT: a law of E sees its Green-Lagrange strain)
    run.py             learning configurations (morphogen learn)
    mfront.py          MFront sources of the learned laws (io.mfront.mfront_source,
                       morphogen mfront, the AMITEX and Abaqus exports): the trained
                       parameters as constants, the Default DSL in small strains or with
                       Hencky's (laws of 1/2 ln C) or Green-Lagrange's strain measure,
                       the state of the model as auxiliary state variables, the thermal
                       split of the Hencky points, orthotropic in the axes of the material
    learned.cpp        the C++ port of the four models for those sources (forward passes,
                       consistent tangents, the Newton solve of the standard model), a
                       section per kind, morphoGen's Voigt order inside
  fracture/          fracture
    cohesive.py        cohesive zone elements: laws (bilinear mixed-mode, exponential,
                       the contact of lips), Coulomb's friction of their broken share
                       (radial return, slip and dissipation of each point), interface
                       elements in small and finite strains (three nodes a side, or six
                       between 10-node tetrahedra, on curved faces), insertion by
                       duplicated nodes, quasi-static solver (loadings in stages, a
                       predictor of lips that stick), calibration, DCB
    periodic.py        CohesiveCell (the periodic element of the loading paths with
                       cohesive interfaces between its phases or its grains: debonding,
                       contact and friction of the broken lips, the stress of the
                       interfaces on the faces of the cell), open_cell (the faces opened
                       on the torus of a periodic mesh, the periodic links of the two
                       lips), interface_law (a law from a mapping)
    lefm.py            cracks by level sets, conforming (remeshed by MMG, the front on
                       edges; 4-node elements, or 10-node ones with quarter points at
                       the front) and XFEM solutions, energy release rate, Paris and
                       Griffith propagation
    mixed.py           mixed-mode cracks (I, II, III) in small and finite strains: G-θ,
                       interaction integrals, contact and friction of the lips (the
                       work of their tractions, the bonded ligament ahead of the
                       front), kinked cracks
                       remeshed by MMG, kinking (Erdogan-Sih) and propagation; the
                       driving forces of inelastic bodies (work, free energy, tip)
    surface.py         cracks of any surface and front (SurfaceCrack: rows x stations,
                       closed or open fronts, level sets and axes of the front), grown
                       point by point of the front (propagate_front: Erdogan-Sih kinks
                       and Paris advances of each station)
    bulk.py            the behaviours of morphoGen's points as the bulk of the crack
                       solvers (von Mises, porous and crystal plasticity, MFront), their work and
                       free energies, the configurational forces of internal variables
    ductile.py         remeshed propagation in inelastic bodies: the transfer of the state
                       of the points between meshes (nearest or projected, the lips apart),
                       the equilibrium regained, ductile growth by J_tip or the porosity
                       at the tip, the resistance curve
    phasefield.py      the phase field of brittle fracture on voxel images (AT1, AT2):
                       alternate minimization with the FFT solver, irreversibility by a
                       primal-dual active set, energy splits, a toughness per phase;
                       its result a PathResult (damage, crack surface, energies)
    phasefield_finite.py  the same in finite strains (rubbers, soft solids): the
                       energy of the deformation gradient of hyperelastic or elastic
                       phases, the displacement by the nonlinear FFT solver whose
                       points are degraded, the damage problem of the small strains
                       (polycrystals: grain by grain, crack meshes with grains)
  contact/           contact, self-contact and friction
    surfaces.py        Surface (free faces of a mesh, their points, rims and sizes), Pair
                       (two surfaces, or one against itself), Obstacle (rigid plane,
                       sphere, cylinder, cone that move and rub), assemble (bodies meshed
                       apart in one mesh)
    kernels.py         Numba kernels: the functions of the faces, the nearest point of a
                       flat or curved face (edges, slack, rim), the forces of a slave
                       point on a master face or on an obstacle (penalty, multiplier,
                       Coulomb's radial return, the slip of new contacts), the choice of
                       the master face, tangents by central differences
    problem.py         ContactProblem (the quasi-static solver of the cohesive zones with
                       the forces of the contacts: search, passes, predictor, augmented
                       Lagrangian, records), ContactResult
    periodic.py        ContactCell (the periodic element of the loading paths with the
                       contact of the walls of the pores of a cell: images through its
                       faces, the stress that the contacts carry, the macroscopic
                       tangent), walls (the free faces inside a cell)
    hertz.py           closed forms: Hertz's sphere and cylinder, Sneddon's cone, the
                       flat punch, the stick zone of Cattaneo and Mindlin, the sphere on
                       a layer of finite thickness (Dimitriadis et al.)
    experiments.py     ready-made tests: compression (a sample between two platens,
                       its walls free to close on themselves) and indentation (a
                       sphere or a cone; the modulus read from the force)
  micromorphic/      micromorphic continua
    space.py           quadratic displacements and linear generalized degrees of freedom
                       on tetrahedra, node-wise sparse assembly, Newton solver (with a
                       predictor by the last tangent)
    finite.py          finite strains: logarithmic strains pulled back to the Piola-
                       Kirchhoff stresses, the objective measures χ⁻¹F and χ⁻¹ Grad χ
    microfoam.py       the microfoam model of nickel foams (Dillard, Forest & Ienny 2006):
                       anisotropic compressible plasticity, full microdeformation, small
                       and finite strains
    damage.py          micromorphic damage (implicit gradient), small and finite strains
    ductile.py         micromorphic ductile damage of von Mises elasto-(visco)plasticity,
                       driven by the plastic strain weighted by the triaxiality (Rice &
                       Tracey, Johnson & Cook, Lemaitre), small and finite strains
  viz/               visualization
    png.py             colour maps, sections, PNG encoding (no dependency)
    sections.py        sections of the level sets on their planes (one cell thick,
                       up to 2048 x 2048 pixels without the volume), anti-aliased
                       rendering (of the implicit functions of all the phases of
                       the structures of several phases), measures of the level sets
    figures.py         matplotlib figures, animated GIFs; the beams of a frame in
                       its cell coloured by a value per beam (frame_figure: forces
                       about zero, energies on a log scale; rasterized beyond 2000
                       pieces in vector files), Ban's replicas against s and v;
                       textures: pole figures, inverse pole figures, orientation
                       maps in the colours of their crystals with their grain
                       boundaries and colour key, disorientations against Mackenzie
                       (softening_figure); the cut of a tetrahedral mesh by a plane,
                       its elements in the colours of their phases or of a field
                       (mesh_figure, mesh_section)
    export.py          figures as bitmaps, vector graphics and TikZ (pgfplots)
    video.py           videos (mp4 by ffmpeg, gif, PNG frames; once, loop, bounce):
                       sections, orthoslices, VTK turntables, sweeps, fields, the
                       steps of loading paths; colour bars
    workflow.py        figures and videos of the realizations of a run (the
                       visualization section): image, labels, mesh, fields, curves
    surfaces.py        triangle surfaces for the 3D view of the interface, in groups
                       between two regions (phases, cells): marching cubes of level sets
                       cut by the box, exact facets of polycrystals, faces of voxels,
                       faces of tetrahedral meshes with their fields (nodal, or averaged
                       per region at the nodes) and displacements; volumes for the faces
                       of the box and the sections; a budget of triangles; base64 JSON
  gui/               web interface (index.html, viewer.js, api.py, server.py; desktop.py,
                     the stand-alone application: the page in a window of pywebview,
                     the dialogs of the system, icon.png, icon.ico): menus
                     in four sections (create, characterize, simulate, export &
                     compare), their tabs, and in a tab the menu of its tools under
                     the headings of what they do (of the models by family in
                     Generate); a search over the tabs, the tools, the fields and
                     options of their forms and the models, which opens what it
                     finds with the option set; addresses #tab/tool and
                     #generate/model; the workspace of the session (morphologies with their
                     traits, simulations, designs and the runs of tools under keys,
                     saved and opened), generation by model families with the views of
                     the level sets and the formation, simulations of a morphology (the
                     steps of a configuration run in a thread), exports on demand
                     (figures, videos, files, archive), optimizations from a morphology,
                     tile assemblies, Ashby charts; the 3D view of any item (viewer.js:
                     WebGL 2, no library, drawing the surfaces of GET /api/surface:
                     orbit, sections, phases, colour maps, magnified displacements)
    tools/           the registry of tools (their tab and the heading of its menu
                     that lists them, forms of typed fields in boxes, one or
                     several items of the traits they need, jobs with their progress
                     and their stops, results and behaviours kept in the workspace,
                     archives; the fields of the steps of a run with their manifest,
                     which the 3D view goes through, and the videos on demand: the
                     frames kept, <name>.frames.npz, for the export); one module per
                     tab: statistics (RVE
                     sizes, measures, correlation functions, stereology, percolation
                     thresholds and critical exponents, the grains and the texture
                     of an orientation map), mechanics
                     (loading paths, NTFA, material points, the identification of
                     behaviours, linear tensors, the stiffness of the frame of beams of
                     the networks: the items of the trait framework, its drawing by
                     the forces or energies of the beams, Ban's replicas), learning (a
                     behaviour learned from a material point or a morphology, trained
                     in the background, its
                     curves, its law kept as a behaviour), fracture
                     (stress intensity factors, growth, cohesive zones, polycrystals,
                     the phase-field fracture of a morphology, the growth of a curved
                     crack front in 3D), contact (the compression of a morphology
                     between two platens, its indentation by a sphere or a cone),
                     design (automaton rules of morphologies, rigidity of strut
                     lattices), stability (frames and morphologies, nonlinear paths, Bloch
                     bands), nonlocal_damage (microfoam, ductile with the triaxiality,
                     quasi-brittle), physics (transport properties and their bounds,
                     the flow through the pores with the acoustic parameters and the
                     absorption of a layer, dynamic moduli of viscoelastic composites),
                     couplings (diffusion, oxidation of a plate and of a morphology,
                     hygromorph multilayers), interop (AMITEX_FFTP, Abaqus, mesh and image formats,
                     MFront, Z-set files read and written), compare (the curves of
                     several runs on one figure)
  analysis/          measurements on images
    basic.py           volume and phase fractions
    surface.py         Crofton specific surface (3 or 13 directions)
    covariance.py      covariance along axes, 3D covariance maps
    connectivity.py    connected components, percolation, periodic merging; the wrapping of a phase around the periodic box (wrapping)
    percolation.py     the forest fire (burning) of a phase from a face, compiled, and
                       its chemical distance; percolation thresholds of families of
                       images: bisection of the probability of percolating (labels
                       wrapping around periodic boxes, or the fire); their finite-size
                       scaling (the thresholds of several box sizes fitted to
                       p_c + a L^(-1/nu), weighted, nu given or fitted); the family of
                       the model of a configuration, in its box or, sized, in cubes at
                       its voxel size; planar images (one voxel along z: 4 or 8
                       neighbours, nu = 4/3); the anisotropy: the fire across the
                       core of the box along a direction (burn_along, the cylinder of
                       the inscribed ball), the percolation, chemical distance and
                       geometric tortuosity of one image along directions
                       (directional_percolation), the thresholds of a family along
                       them (percolation_anisotropy: each realization bisected along
                       all, paired deviations, ratio, threshold tensor); the rigidity
                       threshold, where the elastic moduli by the FFT solver vanish
                       (rigidity_threshold: the fraction where half the realizations
                       bear above a cut, the others void, extrapolated over box sizes;
                       images decided by their connectivity through the nodes and the
                       faces of the voxels or by inclusion when they can be, the others
                       by the energies of their solutions, resolved by the stress
                       averages or by the round-off; the bounds tried last); the critical
                       exponents of the conductivity and of the elastic moduli
                       (percolation_exponent: the FFT solutions of the realizations of
                       several box sizes at the threshold, their means fitted to
                       L^(-t/nu), the error of the threshold propagated by the same
                       images shifted, a sweep above it optionally)
    cells.py           cell volumes, adjacency, neighbours (negative labels ignored)
    fractal.py         box-counting dimension
    morphology.py      chords, granulometry and local thickness, Euler characteristic,
                       Minkowski densities, geodesic distances and tortuosity
    correlation.py     Torquato's descriptors: lineal-path, two-point cluster and
                       pore-size functions; radial distribution function and structure
                       factor of point patterns
    random_sets.py     structuring elements: dilations, erosions, openings, closings,
                       Choquet capacities and contact distributions, morphological
                       covariance and Minkowski curves of grey-level images
    topology.py        Betti numbers, connectivity density, Euler characteristic of
                       plane sections and integral of mean curvature
    anisotropy.py      mean intercept lengths along the 13 grid directions, the MIL
                       fabric tensor (principal MILs, axes, degree of anisotropy),
                       the first minima of the covariance (hole effect)
    power_laws.py      power-law tails of samples by maximum likelihood, the bound of
                       the tail by the Kolmogorov-Smirnov distance (Clauset et al.)
    stereology.py      sections by the grid planes (watershed, frame correction),
                       Saltykov's unfolding for spheres and oriented ellipsoids, the
                       consistency of a shape with the section radii
    particles.py       the particles of a phase in the volume: connected parts split by
                       the watershed of their distance map, volumes, radii, centres,
                       number per unit volume (frame correction of Miles-Lantuéjoul)
    measures.py        named measures used by configurations and the CLI (connectivity: the faces joined and, periodic, the wrapping around the box; directional_percolation: the crossings and tortuosities along the lattice directions)
  rve/               statistics of RVE studies
    scaling.py         point variances, ensemble and window statistics, fits, RVE sizes
    study.py           series of properties x boundary conditions (Dirrenberger 2014):
                       intervals, errors, fitted/imposed exponents, bootstrap, isotropy,
                       discrepancy; the mean stiffness and conductivity tensors per volume
    windows.py         the window method for tensors: one periodic realization solved
                       once per unit load by FFT, the mean mutual energies of its windows
    report.py          Markdown, LaTeX, JSON reports and figures
  dynamics/          explicit dynamics with OpenRadioss (a separate program)
    model.py           the consistent units of the inputs, the materials of the phases
                       (elastic, von Mises tabulated, Johnson-Cook, failure), the
                       loadings (compression, tension, shear, impact), the mesh of a
                       morphology and the estimate of its stable time step
    openradioss.py     the installation found (the paths, the executables, their
                       environment) or installed from a release on GitHub, the runs
                       (Starter, Engine, their progress, stops and failures) and the
                       conversion of their outputs (anim_to_vtk, th_to_csv)
    results.py         the results read back: the nominal stress-strain curves of the
                       faces, the energies, the fields of the animations on the mesh,
                       the summary, the frames for the 3D views
  coupling/          transport, chemistry and mechanics
    diffusion.py       transient diffusion in voxel images (finite volumes), Crank's slab
    oxidation.py       thermal oxidation: the closed-loop scheme, thin films,
                       diffusion-limited profiles of a plate, and the voxels of an
                       image (oxidation_field: TR-BDF2, the local species eliminated
                       voxel by voxel, the oxygen by conjugate gradients)
    ageing.py          property laws of a tracer, graded phases, hydrolysis (Ekenstam)
    plates.py          hygromorph multilayers: moisture and curvatures, Timoshenko
    fenicsx.py         diffusion and free swelling of tetrahedral meshes (FEniCSx),
                       curvature of strips
  laser/             architectured sheets by localized laser treatment (docs/laser.md)
    heat.py            moving Gaussian or top-hat beams on a plate, one face or both
                       (quasi-steady integral over the delay, images through the
                       thickness, losses; separable on grids), Rosenthal's limits, the
                       tabulated cycles of a track (TrackKernel), transient finite
                       volumes of a blank (LOD, Numba)
    metallurgy.py      austenite, martensite, bainite and tempering of C-Mn steels
                       (lever rule, Andrews, Koistinen-Marburger, Maynier,
                       Hollomon-Jaffe additivity), peak-temperature hardening
                       (AISI 430), melt-pool carburizing, nitriding and alloying
    materials.py       Ludwik or Voce laws fitted to Rp, Rm and Ag (Considere), the
                       strength of each hardness, the presets of the PIMM steels
    patterns.py        tracks and laser paths: straight, corrugated, Fourier, crossed,
                       honeycomb; length, time, energy, gaps, curvature, G-code
    process.py         the treatment of a path at points of the sheet, track after
                       track (tabulated responses, integrated cycles where tracks meet)
    specimen.py        dog-bone, strip and cell specimens, structured quadrilaterals
    mechanics.py       layered membrane in plane stress with large rotations (Numba
                       elements, corotational log strains, consistent tangent), the
                       tensile test and its measures
    bending.py         three-point bending of a strip: fibre sections, neutral axis,
                       moment-curvature, deflection by virtual work
    design.py          cases, measures, objectives and constraints; grid, Nelder-Mead,
                       differential evolution, genetic and Bayesian optimization
                       (Gaussian process, expected improvement)
    solid.py           the treated sheet in its volume: voxels of graded elastoplastic
                       phases with free faces (flow laws as von Mises plasticity),
                       its tensile test by the nonlinear FFT solver
    run.py             configuration files and outputs (morphogen laser)
    figures.py         hardness maps, curves, strains, histories (matplotlib)
  homogenization/    apparent properties
    core.py            boundary conditions, loadings (strain, stress, mixed,
                       temperature), condensation of the macroscopic unknowns, solutions
    materials.py       isotropic and anisotropic materials (cubic, hexagonal, any
                       stiffness), rotations by grain orientations, material tables
                       (with the rotation of each grain); the plasticity or crystal
                       plasticity of a material
    behaviours.py      von Mises elasto-(visco)plasticity: parameters, implicit
                       integration and consistent tangent at the integration points;
                       Hill's criterion (closest point projection, Newton on the plastic
                       strain increment and dp; its CUDA port in behaviours.cu); the
                       integration of all the points in small strains (Numba: chunks
                       of points in parallel, the work arrays of the laws allocated
                       per chunk before the loop — no reference counted in the kernels)
    porous.py          porous elasto-(visco)plasticity (GTN, Rousselier, elliptic):
                       parameters, matrix stresses, kinematic hardening (backstresses
                       of GTN and elliptic), implicit integration along the invariants
                       of the relative trial stress, its analytical Jacobian, consistent
                       tangent (Numba; its CUDA port in behaviours.cu)
    beremin.py         cleavage by the local approach: the Weibull stress of Beremin on
                       the fields of a loading path (the largest principal stress, or
                       the normal stress of the cleavage planes of the grains), the
                       probability of failure of the weakest link
    crystal.py         crystal plasticity of Méric and Cailletaud: slip families of
                       cubic and hexagonal lattices, twins (one sense, a saturation),
                       cleavage planes (opening and accommodation systems that soften:
                       a crack band), interaction matrices (Franciosi), implicit
                       integration (flow rule as stresses, active set, substeps) and
                       consistent tangents in small and finite (multiplicative) strains
                       (Numba)
    finite.py          finite strains: elastic, von Mises and porous points in the logarithmic
                       strain space, the integration of all the points (dP/dF), total
                       Lagrangian element kernels, Cauchy stress, logarithmic strain,
                       lattice rotations (Numba)
    mfront.py          MFront behaviours through MGIS (optional, conda-forge): the
                       compilation of .mfront sources (cached), the loading and checks
                       of the behaviours, the integration of the points of the kind
                       MFRONT (which the Numba and CUDA kernels skip) in morphoGen's
                       conventions and arrays of internal variables
    tensors.py         3 x 3 algebra of the kernels: Jacobi eigen-decomposition,
                       divided differences of the logarithm, Gaussian elimination
    paths.py           LoadingPath (components in time, strain or stress controlled,
                       temperature) and PathResult (curves, phase averages, fields; in
                       finite strains Cauchy stresses and logarithmic strains)
    nonlinear.py       the periodic element: Newton iterations with line search on the
                       fluctuation and the macroscopic strain components, small or
                       finite strains, time stepping, shared by both solvers
                       (_PathSolver: they give the discretization; _PointArrays: the
                       arrays and the integration of the points on the CPU);
                       tetrahedral finite elements (NonlinearFEProblem); the behaviours
                       of the elements; the material point simulator
    fft_nonlinear.py   loading paths on voxel images (NonlinearFFTProblem): B-bar and
                       F-bar voxels, the tangent applied matrix-free, Newton-Krylov with
                       FFT-preconditioned conjugate gradients (on the CPU or the GPU)
    dilatation.py      selective reduced integration of the 10-node tetrahedra: the
                       volume change of each element as its mean (B-bar, F-bar), the
                       conjugate stresses and the consistent tangent
    fe.py              finite elements on tetrahedral meshes (P1, P2; KUBC, MBC, PBC, SUBC;
                       integration: full or bbar); the reduced tangents T^T K T of a
                       loading path by the places of the stored values of K (Reduction);
                       the colours of the elements, for the assemblies on several
                       threads (element_colours)
    plane.py           finite elements of the plane on triangular meshes (P1, P2; KUBC,
                       PBC, SUBC): the structures invariant across their plane in
                       generalized plane strain (the anti-plane fluctuation free) or in
                       plane strain, conduction; the uniform components that no field of
                       the plane gives as unknowns of their own under SUBC
    fenicsx.py         the same problems solved by FEniCSx (dolfinx, dolfinx_mpc, PETSc)
    ntfa.py            nonuniform transformation field analysis: modes (POD of
                       snapshots), influence fields (FFT), reduced laws along paths
    identification.py  parameters of a behaviour identified on a reference response
                       (least squares on the logarithms, material points, as Zopt); the
                       macroscopic law of a morphology (its loadings run, a law estimated,
                       von Mises or Hill, identified on them)
    bloch.py           Bloch waves: band structures of periodic cells (consistent
                       mass, Bloch-periodic map, shift-invert Lanczos), band gaps; the
                       scalar waves div(a grad u) + w^2 b u = 0 (acoustic, antiplane
                       shear, TM and TE light), speeds at long wavelengths
    photonic.py        light in three-dimensional photonic crystals: the vector waves
                       of Maxwell's equations by plane waves on the voxels (transverse
                       basis, operators by FFT, block Davidson with the inverse medium
                       for preconditioner), cells cut by interfaces averaged (Meade)
    coupled.py         strong couplings on voxel images (CoupledProblem): piezoelectric,
                       piezomagnetic, magneto-electro-elastic, magnetoelectric and
                       thermoelectric laws, their fields solved together (MINRES or
                       conjugate gradients, block Fourier preconditioner; conjugate
                       orthogonal gradients for complex laws), the tensors and constants
                       of the effective law, exact laminates
    flow.py            Stokes flow through the pores of voxel images by the lattice
                       Boltzmann method (two relaxation times, D3Q19): the permeability
                       tensor of Darcy's law
    plane_flow.py      Stokes flow through the pores of the meshes of the plane (the
                       elements of Taylor and Hood) and their flow along z: the
                       permeability of beds of fibres
    acoustics.py       acoustics of porous media: the parameters of the equivalent fluid
                       of a rigid frame (tortuosity, characteristic lengths, viscous and
                       thermal permeabilities), the model of Johnson, Champoux, Allard
                       and Lafarge, the absorption of a layer
    harmonic.py        harmonic regimes: complex stiffness of viscoelastic phases,
                       complex conductivity of leaky dielectrics, effective refractive
                       index (complex symmetric laws, the coupled solver)
    laws.py            behaviours in closed form integrated for all the points of a
                       phase at once: generalized Maxwell viscoelasticity, hyperelastic
                       energies (and finite viscoelasticity), isotropic damage (crack
                       band, viscosity, the unilateral effect), Mazars' damage of
                       concrete, the damage coupled to plasticity (Lemaitre, Kachanov);
                       the seam of the learned and MFront points
    fatigue.py         multiaxial high-cycle fatigue of the points along a cycle: the
                       amplitude of the deviator (the smallest hypersphere that holds
                       its path), the criteria of Crossland, Sines and Dang Van, the
                       critical planes of Findley and Matake, the utilization of the
                       fields of a loading path (a criterion per phase), the factor
                       of safety
    poroelastic.py     weak couplings as eigenstrain problems: swelling (hygroscopic,
                       chemical), the poroelasticity of Biot (drained stiffness, tensor
                       and modulus of Biot, undrained response)
    xfem.py            extended finite elements of level sets on a regular grid
                       (LevelSetSample; ridge enrichment of the cut elements, voids
                       integrated on their solid side): elastic and transport problems,
                       loading paths in small strains
    zmat.py            material files of Z-mat (Z-set): gen_evp behaviours to the native
                       kernels or to MFront's brick StandardElastoViscoPlasticity,
                       porous_plastic ones to the porous kernels, the elastic, von
                       Mises and porous materials written back (write_zmat), and the
                       coverage of Z-mat's library (docs/zmat.md)
    frames.py          periodic frames of Euler-Bernoulli beams, in the plane and in
                       space (cubic, bcc, octet, Kelvin, diamond and pyrochlore
                       lattices of struts; or the
                       beams of circular sections of a framework): stiffness, axial
                       forces of a macroscopic strain, linear buckling, Bloch waves,
                       apparent stiffness, relative density; as an option, exact
                       Timoshenko beams (shear stiffness kGA per beam, Cowper's
                       coefficients) and their consistent geometric stiffness, and
                       beams of frameworks with rigid ends at their joints
                       (rigid_joints, in radii; the effective stiffness only)
    networks.py        effective stiffness of random periodic frames (fibre networks,
                       open foams, random lattices) by the sparse periodic problem:
                       Voigt stiffness, its split into stretching, bending, torsion
                       (and shear for Timoshenko beams), affine estimate,
                       non-affinity, by axis of a uniaxial stress (by_axis); beams with
                       rigid ends (the rigid arms of their nodes); the frame of a
                       realization of a network model (network_frame: its framework
                       and fibre radius, plane for a planar network; for morphogen
                       frame-stiffness and the interface); the axial force and the
                       energy of each part of each beam under a uniaxial stress
                       (beam_forces); networks of non-identical beams (Ban et al.
                       2016): exact second-order softening and scatter (modes axial,
                       bending, modulus, area), replicas with a control variate (their
                       table by CV: Softening.rows; a callback per replica)
    bifurcation.py     post-bifurcated patterns by group theory (Azulay), for frames
                       and continua: isotypic blocks, patterns of the axial
                       subgroups, first bifurcation, initial-stress matrix of
                       tetrahedra, symmetries detected on meshes; cells of several
                       primitive cells split by their translations (Bloch wave
                       vectors) and the little co-groups of their stars; along nonlinear
                       paths in finite strains (path_bifurcation: the tangent of each
                       step in the isotypic blocks, the switch of branch)
    fft.py             FFT-preconditioned solver of voxel images (periodic), on the CPU
                       or the GPU
    voxel_fe.py        the same trilinear voxels assembled on those of the materials,
                       conjugate gradients preconditioned by the sparse factorization of
                       the matrix shifted (VoxelFEElasticProblem, VoxelFEThermalProblem):
                       the FFT solutions in a few iterations near the rigidity threshold;
                       the voxels that bear along an axis (load_bearing: the parts cut at
                       single nodes through which a cycle winds around the box)
    gpu.py             the FFT solvers on NVIDIA GPUs (CuPy): CUDA kernels of the
                       stiffness and tangent operators of the voxels, the Fourier
                       preconditioner (cuFFT), conjugate gradients; the loading paths
                       with their integration points on the GPU (VoxelPaths)
    behaviours.cu      CUDA ports of the integration of the behaviours of the points
                       (von Mises, Hill, porous plasticity, crystal plasticity,
                       logarithmic strains) for gpu.py
    kernels.py         Numba kernels of the trilinear voxel elements (FFT solver: the
                       inverse of the Fourier symbol; the Gauss points of the loading
                       paths)
    elements.py        Voigt notation, isotropic stiffness, voxel element matrices
    solvers.py         direct, AMG-preconditioned and Jacobi-preconditioned CG
                       (MINRES after a direction of negative curvature: the
                       indefinite tangents of softening points); GMRES with
                       deflated restarts for the full tangents that are not
                       symmetric, recycling its space from a solve to the next
                       once one restarts (GCRO-DR, Recycling) and stopped where
                       it stagnates; the two-level preconditioner of 10-node
                       elements, its cycle in the order of the colours of its
                       sweeps; dot products, norms and products of long arrays in a
                       single thread (dot, norm, times), the Krylov solvers with the
                       BLAS on one thread; a solver updated with another matrix (the
                       next tangent of a path) keeps its coarse levels and gives its
                       sweeps the new values, until it is worn (LinearSolver.worn)
    amg.py             smoothed-aggregation algebraic multigrid (Numba); on large
                       matrices, Gauss-Seidel sweeps in the order of the colours of
                       the unknowns, those of a colour in parallel, on a copy of the
                       matrix sorted by colour (single precision, the parts below
                       and above the diagonal apart: Smoother) — by nodes and 3 x 3
                       blocks where the unknowns are the components of displacements
                       —, the products of a sparse matrix and of the dense matrix of
                       the coarsest level on several threads (product, dense_product);
                       the matrix by blocks of nodes in double precision (NodeBlocks:
                       the products of the Krylov iterations, the same to the bit;
                       the values of another matrix of the same pattern at their
                       places, which the sweeps follow)
    apparent.py        apparent moduli and conductivity, tensors, thermal expansion,
                       isotropy, named loadings (Response), homogenize()
    postprocess.py     compliance S = C^-1 and the notation of both (NOTATION: Voigt
                       11 22 33 23 31 12, engineering shears), engineering constants,
                       anisotropy indices, the directional Young's modulus (plane
                       strain or stress), solid fraction, effective density,
                       relative, normalized and specific values
    bounds.py          Voigt, Reuss, Hashin-Shtrikman (Walpole) bounds; bounds of
                       polycrystals of cubic crystals; Levin's thermal expansion
  mesh/              meshes
    tetra.py           TetMesh (4- and 10-node tetrahedra, phases, grains, box faces,
                       outer faces and periodic links of whole periodic grains; the
                       departures of the mid-edge nodes of curved elements, their
                       volumes and Jacobians; the sharp edges that a mesh follows)
    curved.py          10-node meshes and 6-node triangles whose nodes lie on the interfaces
                       (curved_mesh: Newton's method on the implicit functions, exact
                       for unions of spheres and segments, cubic splines otherwise;
                       the nodes of sharp edges on the two surfaces that meet there)
    conforming.py      conforming meshes of level sets with MMG (periodic along all,
                       some or none of the axes); the box cut piece by piece for the
                       structures that give their smooth pieces, whose sharp edges
                       are kept as ridges; mirrored meshes of symmetric cells (exact
                       symmetries; the chambers of the groups 4/mmm and m-3m with
                       diagonal mirrors, point_group), implicit bodies of level-set
                       functions
    exact.py           exact meshes without a background grid: the face-to-face complex
                       of convex pieces (overlaid coplanar faces, T-junctions) meshed by
                       MMG, adaptive meshes of point-wise level sets (octree, MMG -ls,
                       projection of the interface onto the exact level set; matching
                       faces cut exactly), the route of each microstructure (exact_mesh); `labelled`: the label and
                       orientation of each element of the structures that give their labels
                       at any point (woven textiles)
    octree.py          balanced octrees refined by a test of their cells, periodic
                       across chosen axes, split into tetrahedra (Kuhn cells, cones over
                       fanned faces; conforming, matching opposite faces)
    polyhedra.py       exact cells of power diagrams (CellComplex: pieces, facets)
    triangles.py       TriMesh (3- and 6-node triangles of the plane, phases, grains, sides,
                       the layers of tetrahedra extruded from it); the exact polygons of the
                       tessellations of coplanar germs and the exact sections of those in
                       space (the power diagram of the plane, clipped by half planes),
                       their small features removed (min_feature), meshed by mmg2d within
                       their edges graded from what remains; the sections of implicit
                       structures cut by the level sets of mmg2d
    regularize.py      removal of small features of cell complexes (caps, short edges)
    polycrystal.py     conforming meshes of tessellations, one subdomain per grain
                       (MMG): whole periodic cells (translated facets meshed alike), or
                       cells cut by the box (periodic faces triangulated in the plane);
                       exact cells with their bends as ridges (exact=True)
    voxel.py           voxel hexahedral meshes
    surface.py         marching-cubes surfaces
  io/                file formats (vtk.py, metaimage.py, native.py, tiff.py, meshes.py,
                     medit.py, geometry.py) and the dispatch functions save_*/load_*;
                     the Abaqus inputs of tetrahedra and voxels with the materials of
                     the phases (elastic, or MFront laws of the abaqus interface; in
                     finite strains all MFront laws, the axes of the orthotropic ones
                     in their state variables) and the orientations of the grains
    export.py          one entry point: the writer chosen from the object and suffix
    dem.py             the particles of one phase without its matrix, for the discrete
                       element method: closed surfaces whole across periodic faces, the
                       exact spheres and clumps of unions (capsules for segments), their
                       tetrahedra (gmsh, or fans of star-shaped particles), surfaces
                       remeshed by MMG (mmgs); STL, OBJ, VTU, Gmsh 2.2 for LMGC90 (an
                       entity and a physical group per particle), tables, LAMMPS and
                       LIGGGHTS data files, YADE columns
    ebsd.py            orientation maps (EBSDMap): .ang (EDAX/TSL) and .ctf (Oxford
                       Instruments/HKL) files read and written, with the axes of each
                       system and of its hexagonal crystals; the grains of a map
                       (Grains: union-find on the disorientations of neighbouring
                       points, mean orientations, areas) and their polycrystal; the
                       map that an acquisition would give of a layer of a polycrystal
    tables.py          tables as .csv, .txt and .xlsx (no dependency), read back
    xdmf.py            XDMF images (raw binary data), chunked voxelization to files
    solvers.py         periodic constraint equations (Abaqus/CalculiX), DAMASK grids
                       and material files, Neper seeds, FEniCSx XDMF meshes
    amitex.py          AMITEX_FFTP runs of an image (material and zone fields, material,
                       loading and algorithm files, MPI script, manifest; the nonlinear
                       materials as MFront laws of the umat interface; in finite
                       strains every phase) and their results (.std, .mstd, .zstd;
                       apparent stiffness)
    mfront.py          MFront sources of the nonlinear materials (von Mises and Hill:
                       the brick StandardElastoViscoPlasticity, Hill's criterion an
                       orthotropic behaviour in the axes of the sample; porous
                       plasticity: GTN in the brick where it is morphoGen's (its porous
                       criterion, the porosity in its implicit system, f0 until the
                       matrix flows), the others morphoGen's kernel from behaviours.cu
                       (porous_update, small strains or Hencky's); crystals: an
                       implicit scheme with morphoGen's slip systems, in finite strains
                       morphoGen's kernel from behaviours.cu; Hencky's elasticity in
                       finite strains; the learned laws, learning.mfront; thermal
                       expansion from 293.15 K), their
                       internal variables, the functions of the umat interface; the
                       user's MFront laws as the exports need them (UserBehaviour:
                       strain measure, axes for Abaqus), their parameters
  interop/           morphoGen with other programs and libraries
    arrays.py          voxel images as the arrays of JAX, PyTorch and CuPy without copies
                       (DLPack, buffers aligned on 64 bytes); signed distances and
                       diffuse-interface phase fields of the phases, the grids and states
                       of spectral solvers (Exponax), ensembles of realizations
    microgen.py        the phases of morphoGen as shapes of Microgen and its shapes as
                       structures; the tetrahedral meshes of both through PyVista
    blender.py         a microstructure as a scene of Blender: the closed surfaces of its
                       phases (the boundaries of their tetrahedra in the conforming mesh),
                       the fluid of Mantaflow around them (domain, obstacles, inflow,
                       outflow), the grids it bakes read back (.uni); the particles of a
                       phase as its rigid bodies (Bullet), the packing read back
    alloys.py          an alloy from a tool of alloy design (its phases, fractions, radii,
                       moduli), its scales as configurations, their effective elasticity
  config.py          configuration schema and validation
  pipeline.py        batches, RVE studies, reports
  results.py         database of the configurations tested and their results (SQLite),
                     filters, CSV exports, normalized Ashby charts
  parallel.py        realizations in processes, tasks and MPI; records, merge, status;
                     job scripts and submission (SLURM, PBS, Torque, SGE, OAR, LSF)
  cli.py             command line
  templates/         example configurations (morphogen template <name>)
```

## 3. Data model

### 3.1 Domain and grids

`Domain(size, origin=(0, 0, 0), periodic=True, unit="")` is the box
`[origin, origin + size)`.

* **Periodic domains** are tori: grains, fibres and cells wrap around the faces, so that
  samples tile space (periodic boundary conditions, FFT).
* **Non-periodic domains** are windows of the infinite random medium. Germs are drawn in
  the domain dilated by the largest grain extent, so that the window is unbiased (no
  edge effect).

`Grid(domain, shape)` splits the domain into equal voxels. Give either:

* `shape=n`: n voxels along the longest edge, voxels as cubic as possible;
* `shape=(nx, ny, nz)`;
* `voxel_size=h`: the shape is rounded and the spacing adjusted so that the grid covers
  the domain exactly.

### 3.2 Models

Models are frozen dataclasses registered under a name (`boolean_spheres`, ...). Each
parameter is declared with `param(default, doc=..., kind=..., unit=...)`. This metadata
serves configuration parsing (`Model.from_config`), the `morphogen models` command, the
documentation and, later, the GUI forms. Plain values are converted on construction:
numbers or mappings become distributions, strings become orientation distributions.

| Model | Microstructure kind | Phases | Periodic |
|---|---|---|---|
| `boolean_spheres` | `PrimitiveUnion` (spheres) | 0 matrix, 1 spheres | yes / no |
| `boolean_cylinders` | `PrimitiveUnion` (segments) | 0 matrix, 1 fibres | yes / no |
| `poisson_fibers` | `PrimitiveUnion` (segments clipped to the dilated box) | 0 pores, 1 fibres | no |
| `curved_fibers` | `PrimitiveUnion` (chains of spherocylinders) | 0 matrix, 1 fibres | yes / no |
| `hardcore_spheres` | `PrimitiveUnion` (spheres) | 0 matrix, 1 spheres | yes / no |
| `hardcore_fibers` | `PrimitiveUnion` (spherocylinders) | 0 matrix, 1 fibres | yes / no |
| `random_walk_fibers` | `PrimitiveUnion` (chains of spherocylinders, packed, drawn into contact) | 0 matrix, 1 fibres | yes / no |
| `long_fibers` | `PrimitiveUnion` (closed chains of spherocylinders along an axis; the summary: realized fraction, RMS and largest angles, smallest gap) | 0 matrix, 1 fibres | yes / no (a window of a periodic box) |
| `cross_linked_fibers` | `Combined` inclusions (fibres; balls of the cross-links minus the fibres), or a `PrimitiveUnion` if merged | 0 pores, 1 fibres, 2 cross-links | no (infinite fibres) / yes |
| `delaunay_network` | `PrimitiveUnion` (spherocylinders on the edges of a periodic Delaunay triangulation, spheres at the nodes; the framework in the metadata) | 0 pores, 1 fibres | yes / no (the periodic network cut by the box) |
| `voronoi`, `laguerre` | `Tessellation` (germs, weights, phases, orientations, grain-boundary thickness; Poisson, packed or stealthy hyperuniform germs) | per cell (and the boundary phase) | yes / no |
| `seeded_laguerre` | `SeededTessellation` (the spheres of the seeds as germs, their weights the squares of their radii; the grain of each sphere, the phases and orientations of the grains, the centres, semi-axes and axes of their seeds) | per grain | yes / no |
| `nested_grains` | `NestedGrainStructure` (the grains and the inner realization, the frame of each grain: its germ, rotation and shift; the subgrains: their grains, germs, weights, orientations and misorientations) | those of the grains (subgrains), and the inner phase (filled grains) | yes / no |
| `gaussian_excursion` | `GaussianField` (Fourier modes from a seed) | 0, 1 excursion | yes / no |
| `tpms` | `TPMSStructure` (surface, level, cell size) | 0 void, 1 solid | yes / no |
| `microgen` | `MicrogenStructure` (the description of a shape of Microgen: its class, arguments and part; the seed of a spinodoid) | 0 outside, 1 inside the field | yes / no |
| `strut_lattice` | `PrimitiveUnion` (spherocylinders) | 0 void, 1 struts | yes / no (rotated or graded: no) |
| `cellular_automaton` | `LatticeField` (`TilingField` on other lattices) | 0, 1 | yes (quasi-periodic tilings: their approximants) / no (windows) |
| `activator_inhibitor` | `LatticeField` (`TilingField` on other lattices) | 0, 1 | yes (quasi-periodic tilings: their approximants) / no (windows) |
| `reaction_diffusion` | `LatticeField` (with the activator field; `TilingField` on other lattices) | 0, 1 | yes (quasi-periodic tilings: their approximants) / no (windows) |
| `nucleation_growth` | `LatticeField` (grain labels, orientations, arrival-time field for partial transformations) | 0 matrix, 1 grains (or per grain) | yes / no (windows) |
| `aggregation` | `LatticeField` (cluster or tree labels) | 0 pores, 1 aggregate | clusters: yes / no; deposits: no |
| `tile_assembly` | `LatticeField` (tile indices in the attributes) | 0 void, 1 solid | yes / no |
| `random_lattice` | `PrimitiveUnion` (struts; framework and rigidity in the metadata) | 0 void, 1 struts | yes / no |
| `random_laminate` | `Laminate` (intervals of the layers of phase 1, holes per layer) | 0, 1 layers | yes / no |
| `dead_leaves` | `DeadLeaves` (spheres or spheroids in covering order, phases, orientations, the background of a finite coverage) | per leaf (and the background) | yes / no |
| `penetrable_spheres` | `PrimitiveUnion` (spheres whose cores do not overlap) | 0 matrix, 1 spheres | yes / no |
| `transparent_dead_leaves` | `TransparentDeadLeaves` (spheres in covering order, their values, the threshold) | 0 below, 1 above the threshold | yes / no |
| `reconstruction` | `LatticeField` (the energy, trials and accepted swaps in the attributes) | 0, 1 | yes (periodic lattice) |
| `ebsd` | `LatticeField` (one cell through the thickness; the labels and the orientations of the grains; the window, the step and the spread of the grains in the attributes) | the phases of the map, from 1 | no |
| `combination` | `Combined` (the realizations of the components, the operation) | 0, 1 (set operations); the matrix's and one per inclusion model | if every component is |
| `scale_rich_network` | `ScaleRichStructure` (ligaments and voids as convex polygons of the plane, extruded along an axis; ends, neighbours, nucleation points; a frame); in 3D `ScaleRichPlates` (plates and voids as convex polyhedra, the rims of the plates and their neighbours) or `ScaleRichBeams` (the axes of the beams, the beams that stop their ends and those they touch) | 0 voids, 1 ligaments | no |
| `woven_textile` | `TextileStructure` (yarns: polylines of their centrelines with the width and thickness axes at their nodes, half sizes, the translation that closes a periodic yarn; the section; a label and an orientation per segment, the matrix last; conforming meshes of its exact geometry, a label and an orientation per element) | 0 matrix, 1 warp, 2 weft, 3 binder (or 0, 1 merged) | yes (whole repeats in the plane) / no (windows) |

Parameters of kind `models` hold nested models (mappings with a `type` key). The
parameters that are files (`path_parameters` of the class: the `file` of `ebsd`, the
`texture.file` of the tessellations whose texture is that of a map) are taken from the
folder of the configuration file, in the models that a model holds too. Some flags
depend on the parameters: `has_level_set` (tessellations with a grain-boundary phase 0,
set operations of models with level sets), `meshed_by_grains` (tessellations without a
boundary phase), `supports_periodic` (combinations). `Model.active_phases()` gives the
phases the realizations can contain, checked against the materials of a configuration.

**Adding a model** takes four steps:

1. Subclass `Model` as a frozen dataclass with a `name`, `phases` and `param` fields.
2. Implement `_validate`, `generate` (return an existing microstructure kind when
   possible) and `theory`.
3. Decorate the class with `@register` and import it in `models/__init__.py`.
4. Add a template and tests against the theory.

### 3.3 Microstructures

Every microstructure implements:

* `voxelize(shape | voxel_size | grid, supersampling=1)`: phase image (uint8), or
  partial volume fractions of phase 1 (float32) when `supersampling > 1`, with
  `supersampling³` samples per voxel;
* `level_set(...)`: a field negative in phase 1, zero on its boundary:
  * exact signed distance, clipped to a band, for unions of grains;
  * first-order distance (φ − t)/|∇φ| for TPMS;
  * (threshold − Y)·ℓ for Gaussian fields;
  * not available for tessellations (`Model.has_level_set` is false), which are meshed
    grain by grain from their exact cells (`mesh.polycrystal`) or homogenized by FFT on
    voxels; with a grain-boundary phase 0, half its thickness minus the distance to the
    nearest face shared by two cells;
  * min, max or negation of the level sets of the components for set operations;
* `contains(points)`: membership of arbitrary points (unions, tessellations, TPMS);
* `to_dict()` / `Microstructure.from_dict()`: JSON description holding the model, domain,
  seed and explicit geometry; regeneration from JSON is exact;
* `summary()`: counts (spheres, segments, cells, modes).

Tessellations also provide `labels(...)` (int32 image of germ indices), `cell_of(points)`,
`boundary_distance(points)`, `cell_complex()` (exact cells, cached) and `cell_table()`
(germs, weights, phases, Euler angles). Dead leaves label the visible leaves, and
inclusions in a polycrystal keep the labels and orientations of its cells. `has_labels` tells whether a
microstructure has cell labels (tessellations, grain and cluster structures); the
pipeline then uses them for the `cells` measure and the `labels` outputs.

`LatticeField(domain, phases, labels=None, orientations=None, smoothing=0, field=None,
threshold=0, attributes=None)` is the realization of the lattice models: a phase per
cell of a regular lattice of the domain (one cell through the thickness for planar
structures), optional grain labels (int32, -1 outside any grain) and orientations, an
optional continuous field (phase 1 where it exceeds the threshold), and JSON-able
attributes (histories, kinetics, tile indices). It is exact on its lattice; elsewhere
the level set is interpolated trilinearly from the cells (theory §14). Arrays are
stored in JSON as base64 zlib data. Models with a formation history implement
`evolution(domain, seed, frames)`: lattice states along the formation of the
realization of `generate(domain, seed)`, which is the last frame.

### 3.4 Images

`VoxelImage(data, spacing, origin, periodic, name, metadata)`:

* `data` has shape (nx, ny, nz) and is indexed `[i, j, k]` along (x, y, z);
* `origin` is the lower corner of voxel (0, 0, 0);
* `metadata` (model, domain, seed, version, phase names) follows the image into `.npz`
  files and run manifests.

### 3.5 Tetrahedral meshes

`TetMesh(nodes, elements, phase, lower, upper, periodic, metadata, cell, orientations)`
holds 4-node or 10-node tetrahedra (VTK / Abaqus C3D10 order: corners, then mid-edge
nodes of edges 01, 12, 02, 03, 13, 23; isoparametric: a mid-edge node off the middle of
its edge makes the edge a parabola, `deviations()`, `curved`, `straight()`, and
`volumes()` integrates the Jacobian) with one phase per element, in the box
`[lower, upper]`; meshes of polycrystals also give the grain of each element (`cell`)
and the Euler angles of the grains (`orientations`), kept by every mesh format that
can hold them; meshes that follow the sharp edges of their phases give these edges
(`sharp`: pairs of corner nodes) and the two pieces of the structure that meet along
each (`sharp_pieces`), which follow the meshes made of them (`quadratic()`, `linear()`,
`select()`).

* `face_nodes(axis, side)`, `boundary_faces(axis, side)` (outward triangles),
  `periodic_pairs(axis)` and `matching_faces()` describe the faces of the box;
* `quadratic()` adds the mid-edge nodes, `linear()` removes them, `subset(phases)` keeps
  some phases (e.g. the solid of a porous medium);
* `volumes()`, `quality()` (1 for a regular tetrahedron), `phase_fractions()`
  (fractions of the box: a phase left out of the mesh is a pore), `summary()`.

`conforming_mesh(micro, size=..., ...)` builds it (see §4 and theory §11), grain by grain
for tessellations (`polycrystal_mesh`, theory §19).

### 3.6 Homogenization problems

`make_problem(sample, materials, physics, bc)` returns:

* `FEElasticProblem` / `FEThermalProblem` for a `TetMesh` (any boundary condition);
* `FFTElasticProblem` / `FFTThermalProblem` for a `VoxelImage` (periodic).

`problem.solve(Loading.strain(E) | stress | gradient | flux, fields=False)` returns a
`Solution`: macroscopic strain and stress (gradient and flux), energy, the per-phase
means and standard deviations of the fields, the solver data, and optionally the
element fields and their support (mesh or image). The assembled system (and the
multigrid hierarchy) is reused for every loading with the same constraints.

`homogenize(sample, materials, bcs=..., properties=...)` returns the apparent bulk and
shear moduli, conductivity, stiffness and conductivity tensors, as `Apparent` records
(value, energy estimate, isotropy criterion, response, solution). Crystals
(`Material.cubic`, `.hexagonal`, `.anisotropic`) are turned by the orientations of the
grains: those of a polycrystal mesh, or `grains` (label image) and `orientations` for a
voxel image; `material_table` gives one matrix per phase and grain.

The FFT problems, `NonlinearFFTProblem`, `make_problem` and `homogenize` take
`device="cpu" | "gpu" | "auto"` (`morphogen.gpu`): on the GPU, the conjugate gradients run
there, with the same solutions (`info.method == "fft-gpu"`); finite elements run on the
CPU.

## 4. Algorithms

* **Unions of grains.**
  * Primitives are registered in a uniform grid of cells, stored as compressed sparse
    rows. A primitive goes into every cell that may contain a point closer to it than its
    radius plus the level-set band, so each query point only tests the candidates of its
    own cell.
  * Voxels are processed in parallel over x slices.
  * Periodic boxes are handled by adding the periodic images of the primitives that
    cross the faces, before a non-periodic rasterization.
  * Signed distances are exact outside the grains, and bounded by the deepest grain
    inside.
* **Poisson fibres.**
  * Isotropic lines hitting the ball that contains the dilated box are drawn: a Poisson
    number with mean L_V π ρ², uniform directions, uniform points in the projected disk.
  * The lines are clipped to the box dilated by the fibre radius, which gives the exact
    restriction of the infinite model.
* **RSA.** A sequential Numba kernel with linked cell lists; candidate positions come in
  batches from the NumPy generator, so every random draw stays reproducible.
  Non-periodic samples are cut from a larger periodic box (no wall effects).
* **Tessellations.**
  * Each voxel goes to its nearest germ through a SciPy k-d tree, toroidal for periodic
    boxes, processed in slabs.
  * Laguerre (power) cells are Voronoi cells of the germs lifted to 4D with the
    coordinate sqrt(w_max − w_i).
  * Non-periodic germs are drawn in a window dilated by three cell sizes; a warning is
    raised if a voxel lies farther from its germ than that margin.
  * Grain-boundary phases: distances to the planes of the k nearest lifted germs, k
    doubled until no farther germ can bring a closer plane (theory §22).
  * Packed germs: random sequential addition of the scaled spheres (largest first) in
    the periodic box or in the periodic version of the window.
* **Exact cells.** Each cell is the half-space intersection (SciPy/Qhull) of the
  bisector planes of nearby germs (k-d tree, periodic images included), checked
  against the other germs at its vertices and recomputed with the missing ones; the
  pieces inside the box, their facets and vertices are merged into a `CellComplex`
  whose facets are shared exactly by two pieces.
* **Polycrystal meshes.** Regularization of the complex (`mesh.regularize`), cones from
  the centroids of the pieces over their triangulated facets, then `mmg3d` remeshing of
  the subdomains (`-hmax`, `-hausd`, `-hgrad`). Periodic faces: subdivision of the box
  edges alike on parallel edges, planar straight-line graph of the traces of each pair
  of faces, constrained Delaunay triangulation (exact orientation tests with rational
  arithmetic when rounding could change a sign), `mmg2d -nosurf` remeshing with the
  traces required, canonical flips; the face triangles are then required in `mmg3d`.
  About 20 s for 300 grains and 2 × 10⁵ elements.
* **Dead leaves.** Leaves are drawn in batches until the centres of a check grid are
  covered; a Numba kernel gives each voxel the first leaf covering its centre (nearest
  image on the torus) and stops once every voxel is covered.
* **Random laminates.** Layers are intervals along the axis (binary search for each
  slice); the holes of a layer are a k-d tree of discs, lifted to 3D for variable radii
  (inside a disc when the lifted distance is below the largest radius).
* **Chunked images.** `io.voxelize_to_file` voxelizes slabs of the grid along z (sub-
  grids of their own) into a `numpy.memmap` of the raw file; the models evaluated point
  by point give the same voxels as at once.
* **Gaussian fields.**
  * A field is a finite sum of Fourier modes on the generation box: the domain itself if
    periodic, the domain enlarged by 3 correlation lengths otherwise.
  * Amplitudes come from the Gaussian spectrum, cut off at |k|ℓ = 6.
  * The field is evaluated on grids by one inverse FFT (periodic, aligned grids) or by a
    separable contraction (any grid), so one realization gives the same values at any
    resolution.
* **TPMS.** Closed-form level-set functions are evaluated in Numba. The level for a target
  fraction is the quantile of φ (or |φ|) over a fine sampling of the unit cell.
* **Strut lattices.** Unit cells are defined by nodes and neighbour distances, then tiled
  with duplicate struts removed. The radius for a target density comes from a root
  finder on a voxel estimate of one periodic cell.
* **Specific surface.** The Cauchy–Crofton formula S_V = 2 ⟨P_L⟩, with crossings
  counted along 3 or 13 lattice directions weighted by the solid angles of their
  spherical Voronoi cells. The estimate is biased low by roughly (voxel size)/(mean
  chord), because chords shorter than a voxel are missed.
* **Surfaces.** Marching cubes (scikit-image) on level sets padded with outside values;
  cap vertices are moved onto the box faces, which gives watertight, manifold,
  outward-oriented surfaces.
* **Conforming meshes.** The level set is sampled on a Kuhn triangulation of the box
  (reflected in the cubes of odd index: no preferred diagonal) and snapped away from
  zero near the nodes; MMG (`mmg3d -ls`, run as an external
  program from the `pymmg` wheel) discretizes the interface and remeshes. Periodic
  microstructures get matching faces through a first cut without remeshing, then a
  remeshing with the face triangles required. About 0.2–0.5 ms per output element.
* **Finite elements.** Node-level sparsity pattern from the element connectivity, then
  dof-level CSR matrices assembled in Numba; constraints and periodicity as a
  prolongation `T` (reduced matrix `T^T K T`); averages and fields integrated in
  parallel Numba kernels.
* **Multigrid.** Aggregation, candidate fitting (modified Gram-Schmidt per aggregate)
  and Gauss-Seidel sweeps in Numba; sparse products in SciPy. For 10-node elements, a
  two-level preconditioner on top of the 4-node hierarchy.
* **FFT solver.** Matrix-free trilinear operator (loop over nodes, no atomics), real
  FFTs of SciPy (multithreaded), 3×3 inverse symbols precomputed per frequency.
* **Cellular automata.** Neighbour counts by a parallel Numba loop over the cells
  (any offsets, periodic or fixed boundaries); large weighted neighbourhoods
  (activator–inhibitor) by FFT convolution; conserved rules by `argpartition` of the
  scores. Gray–Scott steps are a parallel stencil, on tilings a parallel loop over the
  CSR face weights. On tilings, neighbourhoods are CSR lists (lattice neighbours found
  once per basis point in a block of unit cells, then translated; quasi-periodic
  adjacency from the integer vertices of the multigrids, packed into one integer each
  and numbered by one sort, those of approximants first reduced modulo the periods by an
  integer adjugate), counts parallel sparse sums, and rasterization a
  nearest-centre query (k-d tree) in the undistorted tiling, with a point-in-polygon test
  for rhombi and a test of the coordinates along the edges for rhombohedra (16 nearest
  centres, then the ball of the circumradius for the points left).
* **Nucleation and growth.** An event-driven automaton: a binary heap (Numba) of
  (arrival time, cell, nucleus) events; each cell is finalized once, by its first
  arrival, and pushes candidate arrivals to its 26 neighbours (theory §15). Potts
  sweeps are parallel over colour classes with counter-based random numbers.
* **Aggregation.** A sequential Numba kernel driven by the NumPy generator; contact
  counts maintained incrementally; launching spheres, big jumps and kill radii for
  single clusters, kill heights for deposits; Eden growth over a perimeter list;
  ballistic flights by voxel traversal.
* **Wave function collapse.** Domains are 64-bit sets; arc consistency with a stack;
  least-entropy choice; backtracking with a trail of domain changes, restarts after a
  number of backtracks (Numba).
* **Transfer matrices.** Admissible sections enumerated by depth-first search (Numba),
  the sparse matrix built in blocks, exact counts with Python integers (object
  arrays), uniform sampling from the backward counts.
* **Frameworks.** Lattices from the strut unit cells with nodes identified modulo the
  box; ranks by dense SVD (up to about 1000 nodes).
* **Topology optimization.** The graded operator `sum_e s_e K0` is applied matrix-free
  (as the FFT solver); each load case is solved by preconditioned CG from the previous
  solution (warm start), to a relative residual of 1e-4 during the optimization and
  1e-6 for the final thresholded design; mutual energies of the element fields give the
  tensor and the sensitivities in one parallel pass. The filter and the automaton
  neighbourhoods are FFT convolutions. MMA solves its dual subproblem with NumPy.
* **Parallel runs** (`parallel`). The realizations of a run are items (index, seed
  key, file stem, size). Each one runs in a fresh process (spawn) of a pool, with its
  share of the threads (`NUMBA_NUM_THREADS`, BLAS variables, `settings.threads`), and
  writes a JSON record atomically (`records/<stem>.json`); `--resume` skips the items
  with a record. Tasks take every n-th item (`--task-id`, `--n-tasks`, or `auto` from
  the scheduler's array index), MPI ranks likewise; `merge` gathers the records into
  the results and report once every item has one. `job_script` writes the array and
  merging scripts of SLURM, PBS Pro, Torque, SGE, OAR and LSF; `submit` submits them,
  the merging job depending on the array.
* **Interface.** `ThreadingHTTPServer` on 127.0.0.1; JSON API (`/api/models`,
  `/api/generate`, `/api/evolve`, `/api/count`, `/api/optimize`, `/api/job/<id>`);
  optimizations run in threads and are polled; images are PNG data URLs; downloads come
  from a temporary session folder.

## 5. Tests

`pytest` runs the suite in `tests/`; `ruff check` and `ruff format --check` check the
style.

| File | Content |
|---|---|
| `test_core.py` | domains, grids, seeds, distributions (moments against sampling), orientations (uniformity, Euler round trips), images |
| `test_kernels.py` | rasterization, supersampling, signed distances, periodic images and RSA against brute force |
| `test_models_grains.py` | Boolean spheres (volume fraction, covariance, convergence of S_V, integral range), unbiased non-periodic windows, Boolean cylinders, Crofton anisotropy, Poisson fibres (fraction, covariance, number and isotropy of fibres), hard spheres |
| `test_models_cells_fields.py` | Voronoi/Laguerre cells against brute force, Poisson-Voronoi statistics (CV² of volumes, mean number of faces through Qhull, boundary density), phases and orientations, Gaussian fields (evaluation paths, resolution independence, fraction, Rice formula, covariance), TPMS (symmetry, target fraction, periodicity), lattices (closed-form density, target density) |
| `test_models_base.py` | registry, config and JSON round trips for every model, error messages, point queries |
| `test_analysis.py` | exact answers on synthetic images (planes, balls, stripes, white noise, components, cells) |
| `test_percolation.py` | percolation thresholds: the exact brackets of the step of a slab (rising and falling, open and periodic), wrapping around the periodic box against joining its faces (a staircase, a helix, diagonals through edges and corners), site percolation of the simple cubic lattice on 48³ voxels (0.3116 through faces, open and periodic; 0.0976 through 26 neighbours), Boolean spheres near 0.2895, the seeds, the errors; the family of the model of a configuration (its parameter set to the fraction, the refusals), `morphogen percolation` (its output, the seed of the configuration, its errors); the forest fire against the labels (open) and scipy's components of the graph wrapping transversally (periodic) on 200 small site lattices, its exact times (a channel, a staircase, a diagonal, a periodic corridor) and those of a pure-Python breadth-first search, the threshold and the command by the fire; the wrapping flags of the measure `connectivity` (the helix and the staircase); the finite-size scaling of the sites of the cubic lattice (the three pairs of faces of open cubes 16 to 48 across by the fire, 99 realizations: within 0.0045 of 0.3116, the largest box above it by more than 0.0055); the exact law of the slabs (p_c = 0, a = 1, ν = 1 fitted); the seeds by size, the errors, the sized family of a configuration, `morphogen percolation --sizes`; planar images (along x or y, never trivially z; 4 and 8 neighbours as 6 and 18; the exact duality of 8 and 4 neighbours on 300 rectangles; the square lattice on 128² sites within 0.006 of 0.592746 and 0.407254; the finite-size scaling of 2:1 rectangles with ν = 4/3, within 0.006 of p_c; a planar sized family); the cores along directions (their chemical distances full, a direction and its opposite and the image reversed alike, a channel, the errors), the measure `directional_percolation`; the anisotropy of the square and cubic lattices (paired deviations within 3.5 standard errors, each realization's threshold bracketing its own crossing, independent of the other directions), of sites of 2 × 1 voxels (lower along x, higher along y, the diagonals alike), of layers (inf across), of aligned fibres (along x below half the others), a falling family the rising one reversed, the errors, `morphogen percolation --directions`; the critical exponents (a channel conducting exactly as L^-1 in 2-D and L^-2 in 3-D, its uniaxial-strain modulus as L^-1, realizations alike fitted unweighted; layers conducting as the fraction, the sweep of exponent 1 from a threshold at 0, where no exponent is fitted; sites of the square lattice through 8 neighbours at 1 - 0.59274621 on 16 to 32 sites, t within its error of 1.30, the realizations that percolate conducting more at the fraction shifted up; the errors), `morphogen percolation --exponent` (the threshold extrapolated first, or given, a sweep); the rigidity threshold (chains of pixels joined at their corners: a zigzag of pins, floppy, its moduli at round-off, and a diagonal, a straight chain bearing as 0.2747/L exactly, so that the moduli vanish below 0.6 though the pixels connect from 0.3; one solution of each per realization and size, the other images decided by their connectivity or by inclusion; the diagonal's elastic exponent 1; a contrast and the cut above it; the sites of the square lattice bearing between the fractions where the same realizations join through 8 and through 4 neighbours, the decisions growing with the fraction, the elastic exponent at the threshold and its standard error; the voxels assembled and factorized (`solver="fe"`) deciding as the FFT solver with its moduli in at most 3 iterations, the exponent the same, void around or at a contrast; the decisions of chains of pins, ball joints and hinges, in a line or zigzagging, by both solvers; the decision by the energy, the stress average and the round-off of unresolved energies; the bisection with its bounds tried last; the errors), `morphogen percolation --rigidity` (alone, and before `--exponent elastic`, by either `--solver`; refused without sizes, along any axis, for the conductivity) |
| `test_rve.py` | formulas, fits, published RVE sizes, window method against the exact finite-window variance of the Boolean model; shifted tilings of periodic images (the mean of all their windows the image mean exactly, the variance pooled within the tilings, one tiling unchanged to the bit, squares and their edges), their variance on independent voxels within the error of one tiling, unbiased, of a smaller scatter |
| `test_rve_windows.py` | the window method for tensors: windows tiling the box give its apparent stiffness and conductivity (round-off, Hill's lemma), symmetric and positive tensors, their statistics those of the windows as realizations; the known windows of a laminate of orthotropic phases (shear slots 23, 31, 12); planar images (squares, the plane stiffness), pores (NaN compliances); the pipeline (results.csv, run.json, report), refused configurations, the RVE tool of the interface; shifted tilings: every window of a laminate, a random image (no new solve, the first tiling the unshifted one to the bit, each tiling of the box its tensor, the variances within the error of one tiling), `rve.shifts` in the pipeline and its refusals; the edges of planar windows (roots of their areas) in the report, rve_report.json, rve_stiffness.csv and the RVE tool; the shifts of the RVE tool (its volume fraction and mean stiffness those of the API with the shifts, 1 by default the study of before byte for byte, the refusals of the API), the axes of the figures of planar windows (areas, not volumes) |
| `test_rve_study.py` | Dirrenberger et al. (2014): RVE sizes and fibre counts of Table 10, discrepancy (eq. 69), relative errors, recovery of K and γ, isotropy, tables, reports |
| `test_homogenization.py` | finite elements: element matrices, exact homogeneous media under the four conditions (P1, P2), Backus laminates, energy ordering and bounds, averages and fields, porous media and floating clusters, scikit-fem agreement, iterative solvers against direct, non-matching periodic faces, Table 4 bounds |
| `test_solvers.py` | the kernels of the iterative solvers: the product of a matrix and a vector on several threads against SciPy's, to the bit; the colours of the unknowns (no stored value couples two of a colour, a value stored on one side of the diagonal included); the sweeps in the order of the colours against Gauss–Seidel on the matrix sorted by colour, the forward sweep from zero and its residual on half of the matrix each; the two-level preconditioner (symmetric positive definite, the iterations of the sweeps in the order of the unknowns); the same results to the bit on one thread and on several; the reduced tangent by the places of its stored values against the products of sparse matrices; the sweeps by nodes (3 × 3 blocks, components that a condition holds) against Gauss–Seidel in the order of the colours of the nodes; the products without BLAS against NumPy's; the BLAS on one thread within a Krylov solve, its threads back afterwards; the matrix by blocks of nodes (its product against that of the stored values to the bit, the values of another matrix at their places, a matrix with a value outside the blocks); the sweeps that follow the blocks against those built for the new matrix, to the bit; a solver updated with another matrix (products, sweeps, solution; a matrix that is not symmetric); the wear of a preconditioner; a plastic path with its preconditioner kept from a step to the next against the direct solver |
| `test_assembly.py` | the assembly of the paths on several threads: the colours of the elements (no node in common within a colour); the tangent, the nodal forces, their sums and the couplings assembled colour by colour against the elements taken one by one (small and finite strains, full integration and B-bar, one and several elements per chunk), the same to the bit on one thread and on several |
| `test_texture.py` | crystallographic textures and orientation maps: the rotations of the eleven Laue classes (orders, closure); disorientations of known rotations and of random pairs against Mackenzie's distribution; the mean orientation of a grain; the axes of the families of planes, poles and projections; pole figures (a mean of one, a random texture, the kernel at the poles of one crystal, weights); inverse pole figures and their colours; textures drawn from measured orientations; maps written and read (`.ang`, `.ctf`), files as the systems write them (a hexagonal grid, decimal commas), the axes of the orientations of each system and of the hexagonal crystals; the grains of a noisy map of known grains, their boundaries; maps turned, cropped and coloured; the model `ebsd` (windows, axes, errors), a map of one crystal against the rotation of its stiffness; the map of a layer of a polycrystal and back; the demonstration map; windows of a map homogenized from a configuration (the exact bulk modulus); the density of inverse pole figures (random textures, one crystal: the kernel over the number of equivalent axes); the table, the boundaries (lengths, disorientations) and the summary of the grains of a map; the figures of the textures; the command `morphogen texture` (summary, tables, images, conversion); a Voronoi polycrystal with the texture of a map (`texture: {type: measured}`); the lattice of a crystal in single slip against Schmid and Boas (tension, compression), the fibres of a Taylor polycrystal, the texture of a cell along a path (from its fields, from their files, by the command) |
| `test_fft.py` | FFT solver against a direct solution of the same operator, homogeneous media and laminates, translation invariance, Hill's lemma, porous media |
| `test_voxel_fe.py` | the voxels assembled and factorized: the matrix against the operator of the FFT solvers on the nodes of the materials (elastic and thermal, planar, one or two voxels along an axis), the stiffness and conductivity tensors of the FFT solvers (soft or void phase, three spacings), the moduli near the rigidity threshold (sites of the square and cubic lattices, floppy at round-off and bearing to the tolerance, 2 iterations against 300 to 900), uniform and void images, the errors; the voxels that bear (chains of pins, a hanging pixel and block, a column, a hinged voxel) and their moduli and conductivities, those of the whole images |
| `test_conforming.py` | MMG meshes: tiling, mirrored meshes of symmetric cells (exact symmetries, matching faces, the group of a 2 x 2 array of holes), volume fractions, exact faces, matching periodic faces and exact PBC, porous lattices, exports (VTU, INP, MSH, Medit) and readers |
| `test_iga.py` | B-spline functions against the recurrence of Cox and de Boor, their derivatives and the embeddings of their refinements; exact arcs, refined patches and assemblies; the patch test (plane strain, plane stress, a solid); Lamé's thick-walled cylinder (orders 3 and 2, 4 and 3) and Kirsch's plate with a hole; the effective tensors of a homogeneous cell, of a laminate (elasticity, conduction) and of a cell with a hole, against the curved finite elements of the solid (slow); cells of any lattice (other cuts of the same cell, strips with gaps, a cell and its neighbour), the square and hexagonal arrays of holes against the series of Perrins et al. and the isotropy of the hexagonal one, the sheet with alternating holes (exact fraction, symmetries of the square, the square array with circles, Poisson's ratio against the ligaments; against the FFT solver, slow) |
| `test_iga_optimization.py` | the design and its geometries (linear maps, symmetric radial variables); the analytical shape derivatives of the tensor and of the fraction of a cell (elasticity, conduction) and of the compliance and the area of a structure against central differences; the hole of largest bulk modulus against the bound of Hashin and Shtrikman (Vigdergauz; SLSQP and MMA); the hole of a plate under biaxial tension against the ellipse (slow); folded designs and the history of the kept iterates; variables linked by the symmetries of a pattern (images and directions, periods, derivatives, errors); the holes of an auxetic sheet found again from their effective constants; its lowest Poisson's ratio at a given stiffness, with linear constraints |
| `test_implicit.py` | the implicit geometry of every model: for the 46 models, the shapes of Microgen and 19 other configurations of several phases, an implicit function of each phase whose sign is that of the voxel image on two grids and which partition the points; the Menger sponge on the grid of its cubes, two cells and their boundary layer against their planes, leaves and a transparent leaf against their spheres; the interpolation on the background mesh; the conforming mesh of a coated sphere (three phases cut one after the other: volumes, matching faces, the phases kept, finite elements against the FFT solver), cells of three phases meshed by their grains, coated hard spheres (slow) |
| `test_radioss.py` | the inputs of OpenRadioss read back card by card against the columns of the Radioss Reference Guide (units, nodes, bricks and tetrahedra oriented for Radioss, their mid-side nodes on their edges, parts, the solid property, LAW1, LAW36 with its yield curves at the rates of Norton's viscosity, LAW2, node groups, held and driven faces, the anchor of free faces, a shear, an impact on a rigid wall, the self contact, the time histories and their sampling, the Engine input, the length scale); the model (units, materials, loadings, time steps); the listings, time histories and legacy VTK grids (ASCII and binary, classic and VTK 5.1) |
| `test_dynamics.py` | the installation of OpenRadioss found (the paths, the environment, the permissions restored), runs with a stand-in of OpenRadioss (`fake_openradioss.py`): the curves of a bar against its modulus, the forces from the time history or the animations, the frames for the 3D view, a shear, an impact, the failures of the Starter, the stops, the conversions; a release installed from a GitHub API served locally (the archive of each system, no member leaving its folder); the command line |
| `test_gui_tools_dynamics.py` | the tab Explicit dynamics: the inputs written without OpenRadioss (engineering units, the materials of the fields and of the JSON), the errors, a dynamic test with the stand-in (its summary in MPa and MJ/m³, the curves in morphoGen's components compared with another run, the 3D view of its last frame), the results read back, the installation found, an impact |
| `test_desktop.py` | the stand-alone application: the dialogs, files and folders of the page (a stand-in of the window), the confirmation of closing an unsaved workspace, the jobs stopped with the window, the check without a window, the fallback on the browser, the hooks of the page; the real window with `MORPHOGEN_DESKTOP_TEST=1` |
| `test_io.py` | round trips of every format and data type, independent readers (VTK, Gmsh) when installed, mesh structure and files, watertight surfaces, geometry files |
| `test_config_cli.py` | templates, configuration errors, batch outputs, split runs equal to single runs, window studies, CLI commands, `python -m morphogen` |
| `test_homogenization_pipeline.py` | configuration of meshes and methods, FFT and FE batch runs with fields and meshes, a run on a planar section of its volume (`voxels.section`: the layer of the image, of the level set and of the measures, the stiffness of the layer given to the solver; what refuses it), FE RVE study split across tasks, `homogenize` and `fit` commands |
| `test_macroscopic.py` | mixed strain and stress loadings, thermal expansion (Levin, Rosen-Hashin), porous media, the post-processing of apparent properties; the stiffness and its compliance in every output (`S C = I`, `S11 = 1/E1`, `S12 = -nu12/E1 = -nu21/E2`, `S44 = 1/G23`, `S55 = 1/G31`, `S66 = 1/G12` in Voigt with engineering shears, on an orthotropic material of three distinct shear moduli: `homogenize`, the columns of a run and the notation of its `run.json`, the labels of the Ashby charts, the text and JSON of `morphogen homogenize`); the directional Young's modulus (isotropic in plane stress and strain, an orthotropic stiffness against the fourth-order tensor turned by each angle, the plane block, other planes, the angles from the second axis, the errors), and along the directions of space (isotropic, the plane stress modulus in a plane of the axes, a cubic crystal along <100> and <111>; the half-sphere of Fibonacci directions) |
| `test_automata.py` | neighbourhood sizes and sums against brute force, FFT kernel sums, Life notation, glider and blinker, majority and annealing, coarsening, conservation, probabilistic and activator–inhibitor rules (wavelength), updates, histories, Gray–Scott patterns |
| `test_tilings.py` | neighbour counts and symmetry of every lattice (BCC shells 8 + 6, FCC 12 + 6), Voronoi cell volumes, rasterized cell areas and volumes, Penrose ratio of thick to thin rhombi, vertex stars and their frequencies; Ammann–Beenker tiles (ratio √2, equal areas, the six vertex stars and their window frequencies, clipped areas summing to the box's, a disc); golden rhombohedra (volumes, ratio τ, faces shared by two cells, V = C, E = F = 3C, a ball, every point in one cell); the 24 vertex stars of the icosahedral tiling (their frequencies in τ against the identities of the tiling and the shares of the acceptance window, the stars of a patch and of the approximants 5/3 and 8/5, the kinds against the stars up to the 120 symmetries of the icosahedron); the square tiling against the grid automaton, growth from a seed (hexagons, cuboctahedra, rhombic dodecahedra), rules on irregular neighbourhoods, colourings, activator–inhibitor weights, models (JSON, windows, supersampling, frames), inverse design, square realizations unchanged (digests); finite-volume Laplacians (face pyramids, plane waves, homogenized diffusivity of rhombi), Gray–Scott on the square tiling against the grid (round-off), Turing wavelengths on hexagons and triangles, reaction–diffusion models on tilings; periodic approximants of the three quasi-periodic tilings (tiles per period, edges and areas of the tiles, filling the box, every edge or face shared by two cells across the faces of the box, invariance by a period, every point in one cell, no seam in the images, Euler's relation on the torus, tori too small for their cells refused (otherwise the neighbourhoods of the shared sides, edges and faces), the periods, boxes fitted to them and their face measures, refusals, tile ratios and vertex stars converging with the order, the models in periodic boxes and windows, Gray–Scott through the FFT solver) |
| `test_models_automata.py` | lattice fields of the automaton models: exactness on the lattice, other resolutions, level sets, exact fractions, JSON, windows, planar structures, smoothing, conforming meshes |
| `test_growth.py` | Voronoi distances, faceted metrics, phantom nuclei, Avrami kinetics, Johnson–Mehl densities, planar kinetics, partial transformations, Potts energy decrease, DLA dimensions, Eden, deposits, models |
| `test_design.py` | tile sets and rotations, exact counts (brute force, transfer matrix, gauge), uniform sampling, wave function collapse, framework rigidity and Maxwell's transition, annealing and genetic search, inverse design of rules, models |
| `test_optimization.py` | laminates, finite-difference sensitivities, filters and symmetries, MMA against SLSQP, Hashin–Shtrikman bounds, auxetic and target designs, `morphogen optimize` |
| `test_laser.py` | moving beams against Rosenthal and Rykalin and the energy of a cross-section, separable grids, beams of both faces, transient finite volumes against the quasi-steady cycles; Andrews, Maynier, Sieverts, the lever rule, Koistinen–Marburger, Hollomon–Jaffe additivity, peak hardening and its tempering, melt-pool mass balances; flow laws against their tensile properties; paths (arc length, curvature, gaps, G-code, honeycomb sides); tracks that harden and temper; the membrane against exact tension, finite differences of its tangent, rigid rotations, bands at equal strain; the Gaussian process; a case and its optimization; `morphogen laser` |
| `test_laser_solid.py` | a treated sheet in its volume: the flow laws of the sheets as von Mises plasticity (Voce exactly, the power laws within 1.5 %), the volume of a hardness map (classes and their mean hardness, layers, the gap of its free faces, refusals), a sheet of one hardness against its flow law in uniaxial stress, layers and stripes along the tension against the mean of their laws, stripes across it between its bounds, the cell of a laser treatment against its membrane along and across the tracks |
| `test_contact.py` | contact of surfaces: the nearest point of a face (the regions of a triangle against a search, curved faces of six nodes, edges, the slack and the rim); the distances of the rigid obstacles (their gradients are their normals), their motions; the surfaces of a mesh (free faces, phases, points and weights, rims, bodies assembled); two blocks of meshes that do not match pressed together (the pressure of the uniaxial stress, the works; 4- and 10-node elements, one pass and two); the contact points at the nodes of the faces (their weights, the same blocks, a block whose sides carry nothing, the corner of a cube against a plane and against a block); the augmented Lagrangian; a block slid far along a track (T = μ N with the points on either surface or both, the work of friction); Hertz's line contact (pressure, half-width, profile, the work of the cylinder); a platen that rubs; a slot that closes on itself (the stiffness of the solid block, the pressure at the nodes, the file); periodic cells (layers that close on their images through the faces of the cell: the stress of the solid in uniaxial strain and stress; a cell cut through its slot or beside it; the walls cut by the faces of a cell have no rim there); two slots parted by a ligament thinner than its faces; the option `contact` of the paths of `homogenize`, of the command line and of configurations; the errors; the closed forms; Hertz's sphere and a clip whose arms come to rest on each other in finite strains (slow) |
| `test_cohesive_cell.py` | cells whose interfaces debond: the interfaces opened on the torus (inside a cell, on its faces, cells one element thick, 10-node elements, grains, a function of centroids; the periodic links of the two lips; the errors); a bar with one interface against its closed forms (the bilinear law in tension and in shear, the work that breaks it, the lips pushed back, Coulomb's bound over the elastic slip and the energy dissipated, lips without adhesion, the exponential law, uniaxial stress, finite strains, the recurrence of the viscous damage and the interface that breaks at once); grains; kinematic conditions against the solver of bodies with interfaces; a law for each pair of phases (three interfaces in series, the fields of the damage and of the opening of the elements, grains of one phase); the grain boundaries of a polycrystal that break (the work of the stress against the fracture energy times the area of the crack, whole grains against grains cut by the box; slow); beads that debond from their matrix (slow); `homogenize`, the configurations, the command line; an elastic cell brought back to zero strain (finite elements and FFT) |
| `test_contact_experiments.py` | the ready-made tests of contact: a block between lubricated platens (the uniaxial stress with the penalties in series, 4- and 10-node elements, the contact points inside the faces and at their nodes), between platens that rub (stiffer, below the block held on its sides), a block with a slot (softer until the slot closes on itself); the indentation by a sphere and by a cone (the powers 3/2 and 2 of the depth, the modulus with the base of the sample, the unloading along the loading); the correction of a layer against its published coefficients; the errors; the commands `compress` and `indent`; a sphere on a block of 12 x 12 x 6, glued or sliding (slow) |
| `test_gui_tools_contact.py` | the tab Contact: its catalog and forms; a porous morphology between two platens (softer than its solid, the test run directly, with friction); a morphology indented by a sphere (the test run directly, the modulus with the base, the unloading) and by a cone; the errors of the forms |
| `test_friction.py` | Coulomb's friction of interfaces whose nodes match: the law of a point (stick, radial return, the energy of a cycle, the strength left to a broken cohesive point); two blocks pressed then sheared (stick, then T = μ N, in small and finite strains, 4- and 10-node elements; the balance of the works); a cohesive interface that keeps Coulomb's strength once broken, or broken from the start; a closed crack that slides by the shear less Coulomb's and sticks below it, the tractions of its lips; the stress intensity factor and the release rate of an edge crack whose lips rub (slow) |
| `test_sharp.py` | the meshes that follow the sharp edges of the phases: the overlaps of primitives (depths, thicker radii, the faces of a periodic box against the search of every pair, segments), colourings, the pieces of unions, coatings and hollow members; the ridges of mesh files; the sharp edges through the meshes derived from one; the configuration key; two overlapping spheres (the circle of their groove, the exact volume of their union with curved elements, the 4-node mesh with and without it), options, one phase alone, errors and the fallback; three spheres and their two triple points, a groove through the faces of a periodic box, coated spheres against the exact volume of their interphase, crossing fibres (slow) |
| `test_sections.py` | sections computed on their planes: the slices of the voxel images at the same resolution for every model, without the volumes; the coverage of the pixels and the blended images; the structures of several phases (the fraction of each phase in each pixel from its implicit function: they sum to one, the phase of the centre has the largest, three bands against their exact shares); the measures of the level sets against closed forms |
| `test_viz_gui.py` | PNG round trips and colours, formation histories, figures and `morphogen view`, interface actions and server; the whole chain from the form (FFT and finite elements: views, video, files, the configuration rerun, views redrawn, errors), files shown inline; the search of the page (every tool by its title and every model by its name first, the options of the forms with the field and value to set, words by their start), the documentation of the models as running text, a morphology of several phases (the sections of the implicit functions of its phases, the function and the surface of each phase exported) |
| `test_gui_tools.py` | the registry of the tools (catalog and forms, the items their traits offer, the runs kept in the workspace, the tabs in their sections, the menus of the tabs: their headings and the tools under each, the long forms in boxes) and the tab Statistics: RVE sizes by the ensemble and window methods, measures against the theory, correlation functions, stereology, the RVE fit of pasted tables, reconstructions, the orientation map tool (numbers, tables, figures, the polycrystal that joins the workspace), the percolation threshold of the model of a morphology (its bracket, the probabilities across 1/2, a missing parameter refused), its anisotropy (the 13 lattice directions of spheres as bars, 6 sampled in the plane of a planar morphology as a rose; the table, the ratio, the eigenvalues), its critical exponent (the means of the boxes at a threshold given, drawn on logarithmic axes and tabulated, the conductivity and the elastic modulus; a threshold extrapolated from boxes too small refused), its rigidity threshold (discs of a planar morphology over box sizes: the shares that bear and the moduli against the fraction, the finite-size scaling; the same by the voxels assembled and factorized, in fewer iterations; the elastic exponent at it when no threshold is given; refused without sizes) |
| `test_gui_compare.py` | the curves of the runs of a loading path and of material points on one figure (the strain that varies most, the plastic strain along the time), their values and gaps to the points of a test, the runs a form of several items refuses (no curves, not a run), the tab Comparisons in its section |
| `test_gui_tools_learning.py` | the tab Learning: its form generated from the sections of `morphogen learn` (their defaults), isotropic elasticity learned at a material point (the paths and the losses of each epoch reported, the learning curves, the validation path within 1 %, the model, data and configuration written and the configuration learning it again to the bit, the behaviour of the workspace naming its model in its folder), given to the matrix of a morphology by a loading path (the elastic response within 1 %), saved and opened with the workspace; a cubic crystal learned with the rotations of the cube and checked turned, the material point of Mechanics turning its law; a standard law of von Mises plasticity (its options in the configuration that learns it again to the bit; in finite strains, the strain of its free energy, and that of a recurrent network, logarithmic by default, in the model and the configuration); a recurrent law of the homogenized responses of a morphology (its voxel image and configuration); the errors of the form |
| `test_gui_behaviours.py` | the behaviours of the workspace: kept by a material point (its run shown again, run again without a second behaviour) and an import of Z-set (saved and opened), their MFront sources in their folders, taken by a loading path (the same response as the behaviour of its form), refused by the NTFA when not von Mises, given to one phase of the inputs of AMITEX_FFTP (its MFront law), to the MFront sources, to the bulk of a cracked plate and a phase of a cracked morphology; a crack grown through hard-core spheres in a plastic matrix (J at the tip at the toughness, a rising load, the plastic work); a behaviour identified on the curves of a material point and on the points of a test, on two material points followed together (a tension and a cycle in shear: the isotropic and kinematic hardenings), weights one per run, runs all in small or all in finite strains; Gurson's q1, q2 on two material points of a porous behaviour (two triaxialities), an identification started from a behaviour of the workspace |
| `test_stopping.py` | the checkpoints (raised in the thread asked to stop only, the innermost stop, not an `Exception`), the conjugate gradients stopped, a tool and a simulation stopped keeping nothing, an optimization stopped keeping its design, `POST /api/job/stop` and its errors |
| `test_surfaces.py` | the 3D view: closed, outward and measured surfaces of a level set (the area of its measures), budgets of triangles, voxel faces, exact facets of polycrystals and voxel faces of grains, mesh faces with their fields on each side, displacements and sections, voxel images of fields, `GET /api/surface` and its errors, the functions of `viewer.js` run by Node (matrices, section planes, colour maps against matplotlib, decoding) |
| `test_polycrystal.py` | exact cells (volumes, lattices, periodic pieces), regularization (closed valid pieces, volumes), constrained triangulations of faces, polycrystal meshes (grains, matching faces, exports), hard cases of the face meshing |
| `test_exact_mesh.py` | point-wise level sets against the grid (compositions, partitions of several phases), the split test of the octree, conforming octree tetrahedra (periodic faces matching), adaptive meshes of spheres (interface nodes on the sphere, periodic faces matching), curved 10-node elements, thin sheets, exact convex pieces (volumes, nodes on the planes of their pieces), exact cells, greedy boxes, the routes, `mesh.method` |
| `test_crystals.py` | rotations of stiffness and conductivity, cubic and hexagonal crystals, orientation averages and bounds, FE and FFT polycrystals |
| `test_parallel.py` | runs in processes, split and merged, resumed and under MPI give the same results; records and status; job scripts of the six schedulers; submission chaining the merging job |
| `test_amitex.py` | AMITEX inputs: ten-line VTK headers, big-endian fields and zone files, an independent reader (VTK), the laws and constants, the stiffness of every crystal voxel rebuilt from the exported axes against morphoGen's, loadings, paths and finite strains, the errors (plasticity, temperature, non-orthotropic crystals); the runs of finite strains (Small_Perturbations, nine components, every phase an MFront law, its function and coefficients, the elasticity of the nonlinear phases, the errors, the results read in finite strains); results of the cell, materials and zones in morphoGen's orders, numbers written by Fortran, finite strains, the apparent stiffness; `outputs.amitex` and `morphogen amitex` |
| `test_exports.py` | periodic equations against affine fields, FEniCSx XDMF tags, DAMASK quaternions and files, Neper files, XDMF images and chunked voxelization, outputs of runs; the materials of Abaqus inputs of tetrahedra and voxels (sections per grain with their orientations, per phase without; in finite strains MFront laws, no orientation, the axes of the grains in *INITIAL CONDITIONS, the brick's expanding laws refused, a porous kernel's taken), of the outputs of runs (the 21 anisotropic constants in Abaqus's order, expansion and the initial temperature of the nodes, density, voids, MFront laws) and, with CalculiX (ccx), the stresses of an isotropic and a turned anisotropic grain under a homogeneous strain, also expanding and heated from the initial 293.15 K of the input, against morphoGen's |
| `test_morphology.py` | chords of Boolean models, Euler characteristics of known shapes, granulometry and local thickness, geometric tortuosity, registered measures |
| `test_combination.py` | set operations against their components, theory of unions, model errors, grain-boundary distances against brute force, inclusions in a polycrystal homogenized with its crystals |
| `test_nonlinear.py` | plasticity parameters and loading paths, the material point against closed forms, the consistent tangent against finite differences, homogeneous elements and an exact periodic laminate along cyclic mixed thermal paths, the condensation of linear behaviours, divided steps, the prediction of a step (exact at the first order for a straight line, at the second for a parabola with unequal steps; the choice of the order), the secant of the condensed block (the secant equation, BFGS that stays positive definite with Powell's damping, Broyden), `homogenize`, configurations and the command line |
| `test_nonsymmetric.py` | nonsymmetric consistent tangents: which laws are symmetric and the asymmetry of the others, GMRES on dense nonsymmetric systems (CPU and GPU), its deflated restarts against the plain ones on an indefinite system (fewer products, the same solution; without restart the plain GMRES to the bit), GCRO-DR's recycling against them along a sequence of systems (the same solutions, fewer products; empty and to the bit until a solve restarts; CPU and GPU), the stop of a GMRES that stagnates (an indefinite system with more eigenvalues near zero than it keeps), the full tangent operator of the voxels, its coupling and `K_Ew` (voxels and elements) against finite differences, a learned recurrent network on voxels and by finite elements (its symmetrized tangent fails or takes three times the iterations), crystals with latent hardening (fewer Newton iterations at a material point and by finite elements), symmetric laws unchanged to the bit, the decisions of `auto` by the measured asymmetry (an Armstrong-Frederick composite symmetrized to the curve of `symmetric`, the recurrent network full to that of `full`, LU unmeasured, multigrid measured; per phase, a learned voxel at 0.8 % of the volume full), the softening points of a Rousselier composite with full tangents (deflated restarts and recycling; on the GPU, slow), multigrid and GMRES against LU, the full tangents and paths on the GPU against the CPU, its measure of the asymmetry and decisions those of the CPU |
| `test_crystal.py` | slip systems and Franciosi's interactions, configurations, Schmid's law along [001] and [111], consistent tangents (small and finite strains) and warm starts, large steps, points integrated together and alone (the same to the bit: slip, cleavage, twins that turn the lattice), homogeneous single crystal elements, a polycrystal (Sachs and Taylor estimates, lattice rotations, uniaxial Cauchy stress), the template through the pipeline and the command line |
| `test_finite.py` | eigen-decomposition and divided differences, tangents and objectivity of the logarithmic strain space, Hencky's tension and simple shear, the small strain curve of von Mises plasticity in the logarithmic space, finite element paths, macroscopic measures |
| `test_fft_nonlinear.py` | loading paths on voxels: the tangent of B-bar, F-bar and full integration against finite differences and its symmetry, homogeneous images against the material point (von Mises, single crystals in small and finite strains), the linear FFT solution with full integration (pores included), an exact laminate, no locking with B-bar and F-bar, a polycrystal (Sachs and Taylor, uniaxial Cauchy stress, lattice rotations), divided steps, `homogenize`, configurations, the template through the pipeline and the command line; fewer Newton and linear iterations for the same path with the second-order prediction and the secant of the condensed block |
| `test_mfront.py` | MFront behaviours: materials and configurations (sources relative to the file), `morphogen info`; the conventions of MGIS (TFEL's order 12, 13, 23 of the shear components against morphoGen's 23, 31, 12, also on an orthotropic law whose shear moduli differ; the factors √2, the orders of finite strains, tangents and rotations against morphoGen's matrices); MFront twins (`tests/mfront`) of von Mises plasticity (small strains, logarithmic space), Méric–Cailletaud crystals and elasticity with material properties against morphoGen's kernels at a material point, by finite elements and on voxels, small and finite strains, on the CPU and the GPU; the errors; the command line. Without MGIS and MFront (conda-forge) the twins are skipped |
| `test_mfront_export.py` | MFront sources of morphoGen's behaviours: the brick of von Mises plasticity (and of Hill's criterion, orthotropic: compiled for MGIS, called as AMITEX and Abaqus call it), their thermal expansion in small and finite strains (von Mises, Hill's orthotropic one, a turned crystal's, Hencky's elasticity) against the kernels while the temperature changes and its refusals, porous plasticity (GTN of the brick: its criterion, coalescence, constant nucleation, f0 until the matrix flows, the internal variables; the others with morphoGen's kernel in the source: Rousselier's and the elliptic criteria, the Gaussian and exponential nucleations, thresholds, other effective porosities, q3 > q1², kinematic hardening, their row of parameters and internal variables; the refusal of an anisotropic expansion), the slip systems of crystals in TFEL's components, the orthotropic constants in MFront's order, the internal variables, the errors, `morphogen mfront`; compiled, the behaviours against morphoGen's kernels through MGIS (von Mises in small and finite strains, GTN along triaxial paths that grow the porosity: rate-independent with Voce terms, Norton with the coalescence, the nucleation with q3 ≠ q1², expanding, finite strains; the porous kernels along such paths: Rousselier's criterion with the Gaussian and the exponential nucleations from thresholds, the elliptic one with backstresses, GTN with a table of f* and with q3 > q1², expanding, finite strains, and in a tension with the same Newton iterations; crystals with Franciosi's interactions, two families, bcc systems, turned) and an AMITEX export through the umat interface called as AMITEX calls it (coefficients and internal variables against MFront's example of Cast3M, each zone of a polycrystal, porous phases of the brick and of the kernel), an Abaqus input through the abaqus interface called as Abaqus calls it (in the axes of the orientations of its sections; porous phases of the brick and of the kernel; expanding laws too, their temperatures from 293.15 K, in small and finite strains); crystals in finite strains (morphoGen's kernel in the source) against the kernel, with the same Newton iterations; the user's MFront laws in AMITEX and Abaqus inputs (their coefficients against MFront's examples, glossary names merged, their parameters from the directory of the runs, called as the codes call them against MGIS); Hencky's elasticity through MGIS; AMITEX and Abaqus inputs of finite strains, their laws called as the codes call them in finite strains (the deformation gradients; the axes of the zones, or of the state variables in Abaqus) against morphoGen's finite strains, and the user's laws of finite strains (`tests/mfront`: SaintVenantKirchhoff, OrthotropicHencky) against MGIS. Without MGIS and MFront those tests are skipped |
| `test_learning_mfront.py` | MFront sources of the learned laws: the sources of the four models (their scales and trained parameters as constants, to the bit; Hencky's strain measure for the laws of `ln C`, Green-Lagrange's for the others; orthotropic in the axes of the material, the axes of Abaqus in finite strains; the tangents declared symmetric for the potential and the standard model; the state of the recurrent and standard models; the thermal expansion in morphoGen's components, a tensor with shears too), a law in the axes of the sample, `morphogen mfront`, the errors; compiled, the four models against morphoGen's integration of the same laws through MGIS along paths of the six components (small strains, finite strains of `ln C` and of `E`, heated, turned; to the round-off), tensions with free components (the same Newton iterations for the potential and the standard model), their tangents against central differences in MGIS's components from states off zero and heated, the shear order on a linear law of three shear moduli (TFEL's slots 12, 13, 23 against morphoGen's 23, 31, 12); the learned phases of AMITEX exports (umat) and Abaqus inputs (abaqus), small and finite strains, called as the codes call them. Without MGIS and MFront those tests are skipped |
| `test_gpu.py` | the choice of the device and its errors (API, configurations, command line); on a GPU, the linear FFT solutions and iterations of the CPU, the behaviours of the points against the Numba kernels (every behaviour, Hill's criterion included, small and finite strains, substeps), the porous points apart (GTN with its nucleation and coalescence, viscous GTN, Rousselier, the elliptic criterion), the CUDA measures, conjugate stresses and tangent operators of the voxels (B-bar, F-bar, full integration) and their coupling columns against the CPU kernels, paths of polycrystals and of a viscoplastic composite with their fields |
| `test_models_extra.py` | random laminates (layers, holes, level set, renewal variance), dead leaves (same-leaf probability; families, spheroids isotropic and aligned, finite coverage over a background against the bivariate laws, their limits, round trips; transparent dead leaves against the moments and covariance of their field, opaque leaves as the dead leaves), Kelvin cells, Laguerre tessellations of packings |
| `test_torquato.py` | Lubachevsky-Stillinger packings (no overlap over all the pairs, the fraction, contacts of g(r), minimal gap, jamming, polydisperse and non-periodic), the cherry-pit model (its limits and bounds, cores apart, dense cores), the reconstruction (Debye media, a Boolean model with and without the lineal path, planar, from an image file, errors), stealthy hyperuniform germs (S(k) zero on the constrained wave vectors, the regular Voronoi cells) |
| `test_levelset_foam.py` | foams as level sets: struts on the edges of Kelvin cells, the position along them, their thickening towards the nodes, the curvature of their sections, a solved fraction (percolating, the JSON geometry), closed cells (a pore per seed), the copper foam of Degdeg & Dirrenberger (fraction, ligament thickness and its spread, elongated pores), the calibration of round seeds 20 % too small on the sections of a foam of known seeds (slow) |
| `test_fenicsx_swelling.py` | (FEniCSx installed: WSL) diffusion and swelling of meshes: Crank's slab, uniform free swelling (no stress), a bilayer strip bending in time to Timoshenko's curvature, curling into an arc in finite strains (curvature, shortening of the chord) |
| `test_fenicsx_oxidation.py` | (FEniCSx installed: WSL) the stresses of an oxidized epoxy plate against the free laminated plate (skin in tension, core in compression) |
| `test_learning.py` | random paths (isochoric, seeded, free components, rates), rotations of Voigt components; the gradients of the models (three variants of the standard one) and their tangents against central differences (and `dP/dF` of a law of `E`), the steps of a point as the recurrence of the paths; the potential convex, its tangent symmetric and positive, its stress its gradient; isotropic elasticity (and its constants), Hencky's elasticity in finite strains (against the linear fit, at a finite strain material point), von Mises plasticity by the recurrent model on unseen paths (a cycle, a shear in another direction, a turn) against a feed-forward baseline (slow); the standard model: its dissipation nonnegative at every step, its tangent symmetric positive definite and the derivative of its stress, its stress, forces and dissipation those of its free energy, whatever its parameters; a standard linear solid represented to round-off (one Newton iteration a step, its dissipation) and learned at random rates (its relaxation within 1 %); a standard law at a material point, on voxels and by finite elements, from its file and its configuration; von Mises plasticity by the standard model against the recurrent one, its dissipation along a cycle between the reference's dissipation and plastic work (slow); the standard model in finite strains, of the logarithmic and of the Green-Lagrange strain: its training gradients against central differences, its dissipation nonnegative, `S = ∂ψ/∂E`, its tangent symmetric and the derivative of its stress, `dP/dF` at its points, objective (a superposed rotation turns `P` and its tangent along two steps), the small strain model recovered to the first order of the gradient; its law in finite strains at a material point, on voxels and by finite elements, turned with its material, heated free (`F = exp(α ΔT) I`), its option in the configurations; von Mises plasticity at large strains against the recurrent networks of `E` and of `ln C` (slow); the perceptron, the potential and the recurrent network of the logarithmic strain: their training gradients, their tangents `dS/dE` the derivatives of their stresses (the potential's symmetric, its stress the derivative of its energy), `dP/dF` at their points, objective, the small strain models recovered to the first order of the gradient, at a material point, on voxels and by finite elements; the files written before the option `strain` loaded as the laws of `E` they were, to the bit; Hencky's elasticity the quadratic term of a potential of `ln C` (its start to round-off, learned to 0.5 %); reproducibility from the seeds, files of models and data, a material of the file; a learned phase on voxels and the homogenized responses as data; the errors; laws turned with their material: an orthotropic stiffness of three shear moduli learned (its linear fit) and turned at a material point and in its tangent, to round-off, not oriented in the axes of the sample; their equivariance to round-off (a recurrent law along steps, a potential of `E` and of `ln C` in finite strains with `dP/dF`, the tangents the derivatives of the turned stresses); a polycrystal of copper on voxels with the learned crystal against its crystals (10⁻¹⁰), crystal plasticity learned with the rotations of the cube against the crystal law on a polycrystal (slow); thermal strains: free and constrained heating, a heated tension, `F = exp(α ΔT)` in finite strains as the Hencky points (the linear law of `ln C`, constrained and along a heated tension, their stresses), a turned crystal of three expansions; heated in finite strains as the Hencky points: a law that is Hencky's elasticity, of `ln C` or of `E`, isotropic or a turned crystal of three expansions, their stresses to round-off under constraint, along a heated gradient and a heated tension, at a material point, on voxels and by finite elements, free of stress heated free, `dP/dF` of every model in both strains heated and turned, unheated or without an expansion the law of before to the bit; a heated point of a singular `C` failing as a Hencky point does, whatever its model; `morphogen learn` |
| `test_identification.py` | the plastic parameters of a cycle recovered from values 50 % off, the Young modulus from a curve of one component; the isotropic and kinematic hardenings of a tension and a shear cycle followed together, the weights of the references (whatever their numbers of points); Gurson's q1, q2 and the initial porosity of a tension and a uniaxial strain, the parameters of porous plasticity named (matrix, criteria, nucleation, tables refused); a stop during an identification, the errors of the parameters and references (names, terms, Norton flow, positive starting values, components, weights); Hill's coefficients on three tensions and a shear; the macroscopic law of a Hill material taken as a morphology (its nine parameters from the estimated law), a von Mises law of two loadings, the errors |
| `test_hill.py` | Hill's criterion: all its coefficients 1 against von Mises's radial return (cycle of tension and shear, isotropic and kinematic hardenings, rate-independent and viscous), the yield stresses along the axes and in the three shears of distinct coefficients and the ratio of the lateral plastic strains (closed forms), the consistent tangent against finite differences, homogeneous voxels and elements following their material point in small and finite strains, the NTFA refusing it, its MFront source an orthotropic behaviour, its configurations |
| `test_ntfa.py` | the NTFA: exact for uniform fields (plastic and viscoplastic, load-unload), its reduced operators (the apparent stiffness, Hill's reciprocity, symmetry), a stiff sphere in a plastic matrix against full fields (tension, cycle, shear; free lateral stresses), the local fields, storage, the mean fields of the phases, configurations and `homogenize(..., ntfa=...)`, kinematic hardening (uniform fields, a composite along cycles) |
| `test_bifurcation.py` | the symmetry group of the honeycomb in a 2 x 2 cell and its irreps (the axial classes of Table 3 of Azulay & Combescure), Euler-Bernoulli frames (a cantilever, equibiaxial forces, symmetric stiffness, the primitive cell at pi^2, four lattices), Timoshenko beams (a cantilever and a fixed-fixed beam deflecting with shear, Engesser's load, the Euler-Bernoulli limit, a strut in space, Cowper's coefficients), the first bifurcations of the honeycomb (triple in T2 under equibiaxial compression, double under uniaxial), the invariance of the patterns, the figure; the Bloch-wave stability of the primitive cell (the lowest load at the M points, that of the 2 x 2 cell); continua: the symmetries of meshes (O_h, a 2 x 2 x 2 cell of 384 elements, a random image), a lattice of bars (the axial classes, a pattern per class, invariance on the whole mesh, the projections of the first modes, VTK output), a column against Timoshenko (slow); frames of struts in space (an oblique cantilever: Euler's load, deflection, twist; the groups of four lattices, equivariance, hydrostatic forces; the octet at rho/9 and nu = 1/3, the cubic lattice at rho/3, Kelvin at 0.6 rho^2, the bcc lattice bending-dominated), the bifurcations of the cubic lattice (Euler's load of its columns clamped with sway, projections, invariant patterns, Bloch waves, 3D figure and video), Schoenflies symbols of 21 point groups; along nonlinear paths in finite strains: a slender column within 1 % of its linear buckling (same irreps, four modes), a square array of holes in uniaxial compression (a simple point of period doubling, alternating ellipses, the bifurcated branch: growing pattern, plateau, negative Poisson's ratio), an elastoplastic column bifurcating earlier; the pattern transformation of an array of holes on a mirrored conforming mesh |
| `test_porous.py` | porous plasticity: MINRES on a symmetric indefinite system and after the conjugate gradients, the paths of a porous composite with held lateral strains (softening points) without a stalled solve nor a divided step, on the CPU and the GPU; GTN's hydrostatic yield and the von Mises limit, the porosity and the plastic volume change, Chu and Needleman's nucleation, the yield of compaction along the densification (small and finite strains), tangents by central differences, objectivity, Norton's flow, the Jacobian of the update against central differences (each criterion, the nucleations, Norton, Rousselier's vertex), GTN's matrix stress bracketed from a guess (nearby or a thousand times off) against the root from its estimate, finite element and FFT volume elements, configurations; kinematic hardening: its configuration and consumers (the form, the identification, the NTFA refused), von Mises's kernel at f = 0 along a turning cycle, the Bauschinger effect, Prager's hydrostatic backstress and the yield on the shifted surface, tangents (small and finite strains) and the Jacobian against central differences |
| `test_zmat.py` | Z-mat material files (the examples of its manual): native translations, MFront sources (orthotropic Hill, several potentials, user-defined flows, softening), porous_plastic files (tables, nucleation, initial porosity), errors; Hill's coefficients in TFEL's convention on random stresses; configurations and `morphogen zmat`; the coverage table; with MGIS and MFront, the native kernels against MFront (gaps below 1e-8) and Hill's yield stresses in tension and in three shears; the materials written as Z-mat files and read back as the same (each isotropic hardening brick, kinematic terms named, Norton, thermal strain, cubic and anisotropic elasticities with distinct shear moduli; porous_plastic: Gurson with its coalescence and a nucleation, modified Rousselier, elliptic tables, the initial porosity), the materials refused |
| `test_surface_crack.py` | cracks of any surface and front: the level sets and axes of a surface crack against the penny, straight and kinked cracks, growth by rows, the mesh refined around the front; an inclined penny crack against Kassir & Sih (the three modes along the front, the angles of kinking), an elliptical crack against Irwin; the growth of each point of the front (an ellipse towards a circle, an inclined crack towards local symmetry); a straight front as an open surface crack |
| `test_lefm.py` | crack level sets and cut tetrahedra (volumes, sides, refinement), the energy release rate of an edge crack by XFEM (4 % of the handbook) and on conforming meshes (converging), 10-node elements with quarter points at the front (within 2 % with elements of 1, where the 4-node ones are 24 % low) and the consistent loads of faces of six nodes, the remeshing by MMG and the opening of its crack faces, fatigue growth by Paris' law (factors within 4 %) |
| `test_curved.py` | the isoparametric 10-node tetrahedra (volumes of curved elements, their matrices against the formulas of the reference element, patch tests under the four boundary conditions, plastic paths against the material point) and the meshes whose nodes lie on the interfaces: the functions of the phases at any point, the volume of a sphere, the conductivity of a cubic array of spheres against Sangani & Acrivos (1983), a coated sphere on its two interfaces, random coated spheres with their grooves followed or rounded and the faces of periodic meshes of three phases (slow), the options |
| `test_dilatation.py` | the mean dilatation of the 10-node tetrahedra (integration: bbar): its names and options, a homogeneous nearly incompressible solid and a rubber along a path against its material point (small and finite strains), the tangents of B-bar and F-bar against central differences of the forces and of the macroscopic stress, the linear matrix as the tangent of the paths, Lamé's thick-walled cylinder at a Poisson's ratio of 0.4999, stiff spheres in a nearly incompressible matrix (slow) |
| `test_cohesive.py` | the laws (energies of the modes and of their mixture, secant unloading, compression, shear directions), the insertion (an inclusion detached, a crack front kept closed, the orientation of the normals), a bar debonding at N A with the work G A in small and finite strains, frame indifference in finite strains, the calibration formulas and the corrected beam theory, the debonding of particles in a composite, the double cantilever beam against the beam theory (slow); between 10-node tetrahedra: the insertion (each lip its mid-edge nodes, the front its own), the bar with the loads of the 4-node elements at every step, the area and the translation force of the curved interface of a sphere, the beam with two elements across each arm (slow) |
| `test_workflow.py` | the visualization section (defaults, names, 30 errors), the dependencies checked before the run, the whole chain on voxels (FFT: figures, videos, the fields of a path at every step, the files of the fields at the ends of the segments only, tables as .txt and .xlsx equal to the .csv) and the template `workflow` scaled down (finite elements: 3D views, videos), views of meshes (phase legend, deformation, histories), colour bars, `morphogen export` |
| `test_viz_exports.py` | tables (csv, txt, xlsx: records, columns, arrays, sheets, NaN, booleans), figures (png, tif, svg, pdf) and TikZ (axes, legends, explicit tick labels, images, fills; compiled by pdflatex when present, slow), video sequences (once, loop, bounce) and writers (gif, PNG frames, mp4), sections and orthoslices, VTK turntables, sweeps and fields, the entry point `export` |
| `test_bloch.py` | consistent mass matrices (4 and 10 nodes) and the path of the Brillouin zone, a homogeneous cell against the folded acoustic branches, a bilayer against Rytov's relation (ten bands, the first gap along Γ-X), the figure; the scalar waves: layered fluids against their exact dispersion and Wood's law, the TM gap of alumina rods (Joannopoulos et al.), the sound slowed by rigid cylinders |
| `test_transport.py` | the transport phenomena of a scalar potential (electrical, dielectric, magnetic, diffusion): the thermal solvers to the bit, layers, inert phases, bounds, loadings with a physics, RVE studies of their scalars and tensors, configurations and the command line |
| `test_coupled.py` | strong couplings: the laws and their rotations, layers against the exact laminate of every coupling, a product property, the uncoupled limits, the dense system of a small grid, uniform thermal strains and Levin's relation, thermoelectric composites, crystals turned by their grains, configurations and RVE statistics of the constants |
| `test_closure_reorientation.py` | cleavage cracks that close and twins that turn the lattice: a crystal pulled, pushed and pulled again (the plane closes by its flow rule, carries the pressure as the sound crystal, opens again at the strength left; the work of the cycle); the friction of the lips of a closed plane and of a sound one under pressure; twins that shear back to none (the fraction against the strain, there and back); the disorientations of the twins against their closed forms, the stiffness of a turned lattice; a lattice that turns at a threshold and turns back (magnesium off its axis c, small and finite strains; a cubic crystal sheared along one twin: no strain left); the configuration and its errors; the consistent tangents against finite differences (a plane that closes, held closed, whose lips rub; twins that shear back, undone; a turned lattice and the step that turns it) and the warm starts; homogeneous cells against the point (voxels, mesh); a textured magnesium that untwins; the same without detwinning and with turned lattices, a cracked polycrystal of iron that closes (slow); twins that shear back and planes that close at a share of their resistance (the flow rules, the tangents); the rule that turns the lattices of a grain on points of given fractions, a material point (the constant threshold), a grain of 64 voxels and a polycrystal whose turned volumes follow their twinned volumes; the basal plane of a turned lattice under the criterion of Beremin |
| `test_slip_cleavage.py` | cleavage that the slip nucleates: a crystal that slips at its yield stress until the strength of its plane has fallen to it (the onset, the flow rules of the slips and of the opening); the damage of the plane that eases the slip in compression; the configuration and the bound of the coupling; the Jacobian of the local residual and the consistent tangents against finite differences (small and finite strains); homogeneous cells against the point; a polycrystal of iron that breaks sooner (slow) |
| `test_microdamage.py` | cleavage planes regularized by their microdamage: the closed form of the band of a plane that breaks and its parameters; the configuration; the law of the planes at a prescribed microdamage (one plane, two planes, a plane broken through, a point alone) and its consistent tangent (small and finite strains); the equation of the microdamage on voxels and on meshes against the closed form of a slab; a bar that breaks across its band (softening branch, fields, work against the fracture energy, planes broken through), on grids of three sizes (slow) and on meshes; homogeneous cells against the point; the solvers that refuse; a columnar polycrystal (slow); the acceleration of the alternate scheme (the updates for maps with a known fixed point; the bar with the plain scheme and with the accelerated one) and its options (solvers, configuration) |
| `test_beremin.py` | the criterion of cleavage of Beremin: the largest principal stress and the normal stress of the cleavage planes of turned crystals against closed forms, the Weibull stress (a uniform stress, two regions, compressed points, large moduli), the weakest link; a homogeneous cell pulled, unloaded and pulled again (the plastic zone, the probability of the largest Weibull stress); configurations and their errors; the pipeline (curves, field, planes against the principal stress) |
| `test_twinning_cleavage.py` | twins and cleavage of crystals: the slip systems of the hexagonal lattice (numbers, closure under its rotations), the twins and their shears against the formulas of their modes (the second plane undistorted, the sense of {10-12} on either side of c/a = √3), the systems of the cleavage planes, configurations and their errors; at a material point: twins along the axis c of magnesium and of zinc (one sense, Schmid's law with the barrier of the saturation at each step, the strain of the twins, the saturation), an fcc crystal that twins at half the stress in compression, basal slip; a plane that opens at its strength and softens (σ = Y₀ exp(−w δ), the work Y₀/w, unloading, compression), an inclined plane and its accommodation, slip that shields cleavage; the consistent tangent of a crystal that slips, twins and cleaves against finite differences; homogeneous cells against the point (voxels, mesh, small and finite strains); the fracture energy of a crack band; a fibre texture; a textured magnesium (asymmetry); transgranular cleavage of a polycrystal of iron; cracks between the grains or through them (cohesive boundaries and cleavage planes) |
| `test_fatigue.py` | multiaxial high-cycle fatigue (the pores of a voxel image are not judged): the amplitude of a cycle against closed forms (proportional cycles, a tension and a torsion a quarter of a cycle apart, the circles of three states of shear, a cycle of pressure, arrays of points); the criteria of Crossland, Sines, Dang Van, Findley and Matake at the two limits that calibrate them, under a mean stress, under a pressure held, in any frame; criteria from mappings and their errors; points judged by phase, the factor of safety and the share beyond the limit; the fields of a cycle of a homogeneous cell at the closed form (finite elements and FFT), around a stiff particle; the configurations and the pipeline (the map, the results, the quantities of the statistics); Goodier's stresses at a spherical pore (slow) |
| `test_damage_laws.py` | the damage laws coupled to plasticity, of concrete and the unilateral effect, at a material point against closed forms: Lemaitre (a perfectly plastic and a hardening bar, a bar of Hill's criterion, the triaxiality, the closure of the microcracks, cycles, a point that breaks, the delayed damage), Kachanov (the time to rupture, the damage and the tertiary creep, Hayhurst's stress), Mazars (tension, compression, the shares), the spectral split (the stiffness recovered in compression); their tangents against differences; configurations and errors; a composite whose matrix is damaged, by FFT in small and finite strains |
| `test_laws.py` | viscoelasticity, hyperelasticity and damage: material points against closed forms (Prony series, incompressible rubbers, softening curves), tangents against finite differences, the laws in the solvers (exact laminates by FFT and finite elements), the crack band independent of the voxels, a crack in a hyperelastic plate, the identification of a Prony series |
| `test_harmonic.py` | harmonic regimes: complex laminates, Dykhne's checkerboard, static and instantaneous limits, the harmonic regime followed in time, Maxwell-Wagner polarization, effective indices, laws in a half-plane, conjugate orthogonal gradients against a dense system and MINRES, RVE statistics |
| `test_xfem.py` | extended finite elements of level sets: sub-tetrahedra, oblique laminates to round-off, spheres against voxels and estimates, voids, transport, loading paths with damage, configurations (`method: xfem`) |
| `test_phasefield.py` | phase field of brittle fracture: homogeneous closed forms of AT1 and AT2 with each split, optimal profiles and areas of a crack, the fracture energy of a bar on two grids, compression, a row of slits against Griffith, unbreakable inclusions and voids, `homogenize(fracture=...)`, configurations and RVE statistics of the strength and of the work |
| `test_phasefield_finite.py` | phase field of fracture in finite strains: a cell of rubber and of an elastic solid stretched to 1.5 against the closed forms (damage, nominal stress, energies; uniaxial strain and uniaxial stress), a compression that breaks only without a split, a cracked plate at small strains against the solver of the small strains, a cracked strip of rubber that breaks at the stretch of Rivlin and Thomas, the phases and options refused, `homogenize` and the configurations |
| `test_photonic.py` | light in three-dimensional photonic crystals by plane waves: the light cones of a homogeneous medium and of a uniaxial crystal to round-off, the Bragg mirror against its exact dispersion at normal and oblique incidence for both polarizations (second order), magnetic layers and the duality of the fields, interfaces inside the cells of the plane waves (the averaged tensor), rods against the two scalar polarizations and out of their plane, long waves against the homogenized permittivity, the band gap of the diamond lattice of spheres (Ho, Chan & Soukoulis; MPB), the Davidson solver against the matrices, the command line |
| `test_poroelastic.py` | weak couplings: Levin's relation for the swelling of two phases, layers by finite elements, the relations of Biot and Gassmann for pores in an isotropic solid and in a crystal, the tensor of Biot of a heterogeneous solid by reciprocity, finite elements and level sets, undrained limits, configurations, RVE statistics and the command line |
| `test_flow.py` | Stokes flow through the pores by the lattice Boltzmann method: the flow between two planes exactly, whatever the viscosity of the lattice, a square duct, square arrays of cylinders (Sangani & Acrivos; Drummond & Tahir), simple cubic arrays of spheres (Zick & Homsy), closed pores, the permeability in `homogenize`, configurations, RVE statistics and the command line |
| `test_acoustics.py` | acoustics of porous media: the thermal permeability of slits and tubes, the tortuosity and the viscous length of arrays of cylinders and spheres (Rayleigh, Maxwell), closed and narrow pores, the equivalent fluid against the exact responses of slits and tubes and its limits, the absorption of a layer, the property `acoustic` in `homogenize`, configurations, RVE statistics and the command line |
| `test_gui_tools_physics.py` | the tab Physics of the interface on small voxel images: its forms, the transport properties between their bounds and with an inert phase, the flow through the pores with the acoustic parameters and the absorption curve of a layer, the dynamic moduli of a viscoelastic composite (the loss peak, the limits), the errors of the forms |
| `test_physics_guide.py` | the guide to the physics (docs/physics.md): its Python snippets run in one session and give the physics they announce (bounds, couplings, loss peaks, band gaps, softening and broken cells, a filled rubber and a strip torn in finite strains), its RVE configuration runs, its commands parse |
| `test_tutorial.py` | the tutorials (docs/tutorial_fibres.md: the fibres of a ply meshed by their exact discs, the axial conductivity the arithmetic mean, the connection of Hill to round-off, the bounds of Hashin and Shtrikman, the conditions ordered, the permeability of the preform, the plane elements of Abaqus; docs/tutorial_plane.md: the exact polygons of seeds of the plane and their small features removed, the bulk modulus of cubic crystals under the three conditions, the layer in space, the section of a polycrystal of space, the bounds of an orientation map, a plastic path, the plane elements of Abaqus; docs/tutorial_texture.md: the grains and the texture of the map, its stiffness, the lattices that a tension turns, a Taylor polycrystal, a polycrystal of space with the texture of the map; docs/tutorial.md, tutorial_rve.md, tutorial_rubber.md, tutorial_geometry.md, tutorial_iga.md, tutorial_contact.md, tutorial_fracture.md: layers that debond at their strength and with their fracture energy, a pore under a cyclic load, magnesium that twins one way, iron that cleaves, the probability of Beremin): the snippets of each run one after another at reduced sizes and give what the page says (measures against theory, the size of the RVE; the variance of the volume fraction of Boolean spheres against their integral range, the exponent 2/3 of Poisson fibres, the windows of one image, the report of a study; the laws fitted to the tests of a rubber and the stretch at which its strip tears; the fractions of a conforming mesh of three phases and the moduli of three solvers; the orders of an isogeometric analysis, the arrays of holes, the hole of Vigdergauz and the auxetic sheet); the sizes replaced are in the pages, the figures shown are in `docs/_static/tutorial/`, the commands parse |
| `test_howto.py` | the recipes (docs/howto.md): their snippets run in one session and give what they announce; the `morphogen` commands of the recipes and of the tour of the interface parse |
| `test_gallery.py` | the gallery of examples (docs/gallery.md): a card and an image for every template, in the families of `docs/gallery/make_gallery.py`, with the result figures of the templates that have one and no image left without a card; the page is the one that the script writes |
| `test_fenicsx.py` | (FEniCSx installed: WSL) the FEniCSx solver against morphoGen's: stiffness and expansion of random P1 and P2 meshes under PBC, KUBC, SUBC, MBC, phase statistics, conductivities, `homogenize`, oriented cubic grains |
| `test_fenicsx_options.py` | the solver names, configurations, the errors without FEniCSx or on voxel images |
| `test_oxidation_field.py` | thermal oxidation in the voxels of an image: one voxel against the thin film, a plate against its one-dimensional profile (second order), a block and impermeable fibres along the diffusion against the plate, barriers and inert skins, units, exposed faces, the tracer of the ageing laws, the refusals |
| `test_coupling.py` | transient diffusion against Crank's slab (uptake, profile, sqrt(t) start), a periodic plate against the line, solubilities; thin-film oxidation against Ernault et al.'s amides and oxidized layer; property laws, graded phases and the mean eigenstrain; Ekenstam's hydrolysis; a bilayer against Timoshenko, the bending sequence of a hygromorph, Crank's uptake of a layer; the `diffusion` section of configurations (outputs, errors); orthotropic layers (turned fibres, twist) and layers from microstructures (homogeneous, laminates normal to z and to x) |
| `test_random_sets.py` | Choquet capacities of Boolean spheres (balls, segments along x and (1, 1, 1), pairs) and Poisson fibres (minus sampling) against their closed forms, dilations and the capacity, openings and closings, the morphological covariance of a Gaussian field, Minkowski functionals of Boolean spheres, of a ball and of Gaussian excursions (Adler), Betti numbers of a torus, a shell, two balls and joined tori |
| `test_anisotropy.py` | the MIL fabric tensor of isotropic Boolean spheres (degree of anisotropy, the MIL along the axes against the mean chords), of the same voxels on a stretched grid (ratios 2 : 1.5 : 1 and their axes), of an elongated level-set foam (both phases); the first minima of the covariance of an egg crate (half the periods, exactly), of lamellae and of a dense level-set foam; the measures `anisotropy` and `covariance_minima` |
| `test_affine_anisotropy.py` | the affine anisotropy of the models: its configuration (mapping, 3 or 6 numbers, an elongation, errors, the periodicity rule) and the same GUI field everywhere; for each mapped model, periodic images equal to those of the isotropic model in the preimage box (labels, JSON round trips, the invariant theory); correlation lengths (1 : 1 : 2 within 4 %) and mean chords (5 %) along the axes of Boolean and hard-core spheres, Voronoi cells, dead leaves and the Gaussian excursion, the volume fraction kept; the Gaussian covariances along the axes against theory and a metric turned in a periodic box; turned axes in a non-periodic box (fabric tensor, exact point queries, labels); the identity and the open foam's two forms |
| `test_particles.py` | the particles of a phase in a volume: two overlapping balls cut by the plane of their neck against the volumes of their caps, across the faces of a periodic image; a packing of hard spheres recovered sphere by sphere (count, centres, radii); the frame of windows that are not periodic (Miles-Lantuéjoul); discs of a planar image against the sections of the stereology; the kernels of the watershed of a volume; the named measure |
| `test_stereology.py` | Saltykov's matrix and the scales of ellipsoids, the unfolding of exact sections of spheres and of turned ellipsoids in three directions, the shape consistency against the values of Degdeg & Dirrenberger, sections of voxel hard spheres, the watershed of two discs, the frame weights of non-periodic windows (exact on average) |
| `test_correlation.py` | the lineal-path function (exact counts, the Boolean model), the pore-size function of the Boolean model, the two-point cluster function, the radial distribution function and the structure factor (Poisson points, hard spheres, a lattice) |
| `test_curved_fibres.py` | arcs (radii of curvature of consecutive points, chords, end-to-end distances, distributed radii, aligned middle tangents), helices (radius, pitch, end-to-end distance), curved fibres without overlap (no two segments closer than the gap over the nearby pairs and their periodic images, the fraction; in a box), hollow fibres (lumens on the axes, walls, solid fractions, storage) |
| `test_fibres.py` | closest points of segments; hard-core fibres (no two closer than the gap over the periodic images, fraction, aligned axes, non-periodic windows, jamming); curved fibres (tangent correlations, mean square end-to-end distance, fractions of straight and tortuous fibres) |
| `test_fibre_covariances.py` | the quadratures of the orientation laws (moments against draws, round trips), the von Mises-Fisher and mixed laws (Kolmogorov-Smirnov, proportions, errors), the law of the angle to a direction; the covariograms of cylinders and spherocylinders (K(0) = V, across and along the axis in closed form, ∫K = V², an oblique shift against points drawn in the capsule), the lens of two disks; the dilute integral range E[V²]/E[V] (isotropic and β laws, random sizes); the covariances of Boolean fibres (β law about (1, 1, 1), round and flat caps) and of Poisson lines (girdle, von Mises-Fisher) measured along x, y, z and the mean fibre axis against the closed forms; Poisson lines of any law (isotropic unchanged, aligned lines exactly, the count of axes, the Choquet capacity of segments); the curves of the interface along each axis |
| `test_networks.py` | the sparse periodic problem of frames against the dense one of the four lattices of struts (and from their frameworks) to round-off, the honeycomb (Masters & Evans) and triangular frames against their closed forms, of Euler-Bernoulli and of Timoshenko walls (the shear compliance of the walls, the energy split, the struts in space against the dense problem, shear coefficients of frameworks); beams with rigid ends at their joints (the triangular lattice in closed form, random frames against their beams split with stiff ends, in space and in the plane, the cubic lattice at l/l', the lengths, the shortest beams kept flexible, the dense paths refused); frames of frameworks (sections, dictionaries, plane networks, Voronoi frameworks, errors, mechanisms); slender Delaunay networks against an independent truss, stretching-dominated at 0.82 of the affine estimate with its Poisson's ratio 1/4 (0.76 and 0.38 in the plane); open foams bending-dominated (E ~ phi^2, coefficient about 1); the softening of non-identical beams (Ban et al.): exact derivatives against finite differences, replicas against s CV^2 and v CV^2, the laws of the factors; for Timoshenko beams whose shear stiffness follows the area, the derivatives of the four modes (the concave chord stiffness) and replicas of the area; the frameworks and frame stiffness in the summaries of open_foam, delaunay_network and random_lattice; the frames of their realizations (`network_frame`: the radius of the summary or of the model, plane frames of planar networks, a microstructure without a framework refused) and `by_axis` against the means of the summary; `morphogen frame-stiffness` (the realizations of a configuration seeded as its batch run, a geometry, Euler-Bernoulli and Timoshenko beams of Cowper's or a given coefficient, rigid joints, the modulus and Poisson's ratio of the fibres, the softening coefficients against `softening_coefficients`, a plane frame, the text and the errors) and its replicas (`--replicas`, `--cv`, `--distribution`, `--replica-seed` against `softening` of the same frame and seed, the JSON, the text, `--table` and `--figure`, the errors); `beam_forces` (the forces of the dense path of a kagome and an octet lattice, the energies of the beams summing to 1/2E split as `split`, the stretching energy N^2 l/2EA with rigid ends and Timoshenko beams, the virial of a truss), the frame drawn in its cell (`frame_figure`: the pieces of each beam as long as it, inside the cell; the scales, the errors), `softening_figure`, `Softening.rows` and the callback of the replicas |
| `test_gui_tools_mechanics.py` | the tab Mechanics: the loading path of a morphology (FFT on its voxels, finite elements on its conforming mesh); crystals of the forms that twin (magnesium along its axis c) and cleave (iron along [001]; a matrix that cleaves along a path); the criterion of Beremin along a path (the Weibull stress of the field written, the figure, the configuration); its interfaces that debond (strong ones: the bonded phases; weak ones soften the response; their fields and the configuration that runs them again) and the fatigue of its cycle (the fields of the criterion on the stresses of the steps, the figure, the limits of the form), the reduced model of the NTFA against the full field, the material point, the linear tensors (the stiffness and its compliance, S C = I; by extended finite elements on a grid cut by the level set: between the bounds, within a few percent of the FFT solver, Levin's relation, what they refuse); the stiffness of the frame of beams of the networks of the workspace (the items of the trait framework, the spheres refused plainly; its values, tables of C and S, figure and file against `network_stiffness`, Timoshenko beams with rigid joints, the softening along an axis, a plane frame per unit area; Ban's replicas of two modes against `softening`, their table, figure, file and progress replica by replica; the drawing of the frame coloured by the forces or energies of its beams along an axis, in its plane for a planar network, the share of the beams storing half the energy against `beam_forces`; the errors) and the frame of beams in the summaries of Generate and the card of the morphology |
| `test_scale_rich.py` | scale-rich networks: the convex polygons of the growth (clipping with the faces of the edges, chords, strips that fit); the first length in closed form against the integral of l² over the lines and 4,000 first ligaments; the exact partition of the box by ligaments and voids, the T-tessellation of the thin network (N + 1 voids of four vertices on average, Euler), each end on a face of its neighbour, the degrees of the dual graph; the thicknesses exactly, the mean-field density within 3 % of realizations; no jamming at 0.75 of the jamming thickness, jamming at 1.25; the tails of 30,000 ligaments (lengths and degrees near 3); aligned, spread and polarized angles (lines through the pole or across it), linear and radial nucleation densities against their exact means; voxels, partial volumes, the level set (the distance to the ligaments near them, or less), point queries, JSON round trips, prisms along x, the frame; the solved volume fraction (within 1.5·10⁻³, with a frame, aligned), its refusal beyond jamming; the parameters, the template and the interface |
| `test_scale_rich_3d.py` | scale-rich networks in 3D: convex polyhedra clipped by planes (volumes, the hexagonal section of the cube, faces oriented and convex after 300 cuts, volume kept to 10⁻¹²); the mean inverse distance in the cube against its closed form (Bailey, Borwein & Crandall) and the first plate area against 20,000 sections; plates: the exact partition into N + 1 closed voids, thin plates their rims times their thickness, rims in their mid-planes ending on earlier plates or the box, the links and degrees; the density within 6 % of the mean field `1 - (1 - rho_1/3)^3`, jamming; the tails of 10,000 thin plates (areas and degrees near 5/2); normals in a plane (vertical plates through the box: the planar network extruded), aligned, about a pole (shells, radial plates), spread, a gradient along z; beams: no axis enters an earlier beam, the contacts recorded are those of the beams that touch, each end on the box or on the beam that stopped it, nucleation points clear of the beams, the mean field within 15 %; the model in 3D (the solved density of plates and of beams in a frame, voxels and partial volumes, the theory, the errors of the parameters); JSON round trips, point queries and level sets of both; the template `scale_rich_plates` and beams from the interface |
| `test_power_laws.py` | power-law tails by maximum likelihood: Pareto laws above a bound and the zeta law of integers within three standard errors, the error of the discrete estimate against the variance of ln x under the law, the bound of a tail above a uniform body by the Kolmogorov-Smirnov distance, the bias of the continuous estimate on integers, the errors |
| `test_delaunay_network.py` | the periodic Delaunay triangulation against that of all the images (2-D, 3-D), Euler's E = 3V on the torus; the coordination 2 + 48π²/35 and the mean fibre length of Poisson-Delaunay graphs (in space and in the plane, 32/(9π) λ^(-1/2)), hard-core points (lower coordination, more even fibres, least distance); the fibres as struts (a solved fraction, the framework, JSON round trips, planar networks, windows, errors); the template and the form of the interface |
| `test_lattices.py` | rotated lattices (struts along the turned axes, fraction), graded lattices (interpolated radii, mean and local fractions), their errors |
| `test_public_api.py` | the public interface (`morphogen._public`) against the list kept in `public_api.txt`: names of the `__all__` of the packages, functions with their parameters (names, kinds, defaults, without annotations), classes with their public methods and properties, data; the writing of the parameters |
| `test_long_fibres.py` | long fibres along an axis: straight fibres along the axis, closed across the period, as many as the fraction; wavy fibres without overlap (every pair of segments within reach, nearest periodic images, Ericson's closest points), with or without a gap, waving together or each on its own; the realized fraction against the exact volume of the chains and the voxels; the RMS angle exact before the corrections and near it after; the covariances of the waviness along and across the fibres; `secant_mean` against quadratures and its series; lognormal radii; the arrays (spacing, nearest neighbours, exact fraction, largest fractions with a gap); windows, the axes x and y, the theory, round trips, errors |
| `test_textile.py` | woven textiles: the interlacing matrices of the 2D weaves; the half cosines and their arc lengths against quadratures, the rounded rises of binders; the areas of the sections by Monte Carlo; the crossings, crimp and fractions of a plain weave; the partition of interpenetrating yarns; the orthogonal weave (exact fractions, the crimp of its binders) and the angle interlocks; periodicity, windows, stacking axes, shifted plies; the functions of the phases at points and on grids; polydisperse yarns; round trips; the FFT stiffness of tows turned along their paths against the same tows in the axes of the sample; the conforming mesh of the exact geometry (the route, matching periodic faces, the volume of each phase within the Hausdorff distance times the area of its interfaces, a label of its phase and the orientation of its segment per element), the finite elements of oriented tows against the same tows in the axes of the sample (10⁻⁸) and against the FFT solver (6 %) |
| `test_nested_grains.py` | grains with subgrains, or filled with another structure: the subgrains refine the grains and hold their germs, the nearest germ of the grain by brute force, the number of germs (exact without turns); laths and anisotropic grains of a window; the misorientations of their laws, the angle between two subgrains of one grain against products of random rotations and the boundaries of an image; lamellae turned with the crystals of their grains; spheres in the grains of one phase at their fraction; one realization cut by the grains; periodicity, round trips, partial volumes, errors; copper polycrystals with subgrains by FFT (the exact bulk modulus, subgrains turned by nothing and by 0.01°); the subgrains as exact convex pieces (the box partitioned, points in the pieces of their subgrains and phases, Monte Carlo volumes; a grain-boundary phase, laths and anisotropic grains of windows), their exact mesh with the crystals of the elements and its bulk modulus by finite elements; the functions of filled grains at any point (signs, continuity across the faces of the grains, grids) and their exact mesh |
| `test_seeded_grains.py` | grains grown from seeds: the spheres of the breakdown of ellipsoids touching them at the nearest points in closed form and inside them, their coverage; the overlap tolerance of Hart and Rimoli; seeds placed within the tolerance, crowded seeds left out or placed; the grains of spheres filling the box with the volumes of their seeds, periodic or not; elongated and flat seeds giving grains of their shapes along their axes; the labels, the table, the realization saved and read back, the export to Neper refused; the exact surfaces, the elements of crack meshes and the conforming mesh by grain |
| `test_seeded_phases.py` | seeds of several phases and the verification of their grains: the exact moments of the cells summing to those of a box, the shapes of the grains (the ellipsoids of their second moments; the box of a single germ), the grains against seeds of lognormal sizes and uniform ratios (errors, Kolmogorov–Smirnov, `exact_grains`), two phases of their own sizes in their fractions, an amorphous matrix (one region, the distance to the crystals inside it, saved without NaN, meshed or left out), seeds of very different sizes placed within the tolerance by brute force; the seeds of the plane (ellipses in the middle plane, columnar grains, the tolerance of the plane, their sections against their seeds); the figure of the verification; the spheres of boxes and the circles of rectangles inside them and filling them, seeds of cubes and of rectangles against their grains |
| `test_plane_flow.py` | the flow through the pores of the plane: the channel of Poiseuille exactly, the square arrays of cylinders against the drags of Sangani & Acrivos and the series of Drummond & Tahir, closed pores at rest across the plane and the pipes of Hagen and Poiseuille along it, `homogenize`, the configurations and a run; the lattice Boltzmann method on the voxels of the same fibres (slow) |
| `test_plane_meshes.py` | meshes of the plane and their finite elements: triangles counterclockwise, their sides, six-node elements and layers of tetrahedra; the power diagram of coplanar germs against the exact cells; the grains of seeds of the plane meshed exactly (their areas to round-off, angles above 3°, matching sides), sections of spheres against their level set; the files .vtu, .mesh, .msh and .inp read back, the crystals of the grains in Abaqus; uniform fields under every condition, laminates of turned crystals exactly, the checkerboard of Dykhne converging from above, the layer of tetrahedra in space giving the same tensors, the ordering of the conditions; `homogenize` of a mesh of the plane; the small features of the polygons removed (fewer elements, none flat, the areas within a few percent), the columnar grains in space (their polygons extruded), the exact sections of Laguerre tessellations against sampled points; a run of seeds of the plane (its files, the bulk modulus of the crystals), the keys of the configurations, the command line, the figures; the curved six-node triangles of sections: the area of fibres falling as the fourth power of the size, the conductivity of square arrays of cylinders against Perrins et al. (conducting and insulating), the reciprocal theorem of Keller, the neutral coated cylinder of Hashin under the three conditions, their layer in space and their files; the exact discs of the sections of fibres and spheres (their nodes on the circles, their gaps graded, a box not periodic) and what they leave to the level sets (oblique fibres, discs that meet); the rings of coated and hollow fibres and spheres (the neutral coated cylinder on its rings, shells, coated spheres, coatings that meet) |
| `test_polydispersity.py` | monodisperse or polydisperse sizes: laws scaled (moments, largest values, draws, configurations); the moments of the sum of two radii against Monte Carlo; polydisperse Poisson lines (the Choquet capacity of balls against the exact distances of realizations); curved fibres without overlap by every pair of segments; the contacts of cross-linked fibres against an independent computation; plywoods of fibres of their own radii on their voxels; each strut of a periodic lattice once (3, 8 and 24 per cell) and the length per unit volume; lattices, foams and networks of struts of radii of a law, the scales solved for the fractions, the spheres of the nodes, the frames of beams |
| `test_dem.py` | the particles of a phase for the discrete element method: spheres whole across the periodic faces (volumes, areas, centroids beyond the face, equivalent radii), a cap cut by a window, a fibre that winds around the box and the matrix of spheres as clusters, a voxel image; the exact spheres and the clumps of segments; the files (STL solids and files, OBJ, VTU, tables, LAMMPS data files, YADE columns) read back; the Gmsh files read as LMGC90's pre-processor reads them (`readGmshv2`, `separateMeshes`: an entity and a physical group per particle, closed outward triangles of the volume of each particle, positive tetrahedra that fill it, their skin its triangles, the cut particles named), and by gmsh and meshio; capsules of the exact volumes of segments, the fans of star-shaped surfaces without gmsh and the error for the others; the surfaces remeshed by MMG (sizes, qualities, volumes within 2 %); the command line and the tab Export |
| `test_arrays.py` | voxel images as arrays of other libraries: aligned buffers, DLPack without copies (NumPy, JAX: the same buffer); the phase field of a sphere against the tanh of its exact distance (10⁻⁶ near the interface, 2.4·10⁻⁵ beyond 8ε), the distance of a gyroid of unit slope; the grids and states of spectral solvers, ensembles of realizations; with JAX and Exponax, a disc shrinking by motion by curvature in Allen–Cahn's equation (R² = R0² − 2νt within 1 %), the mass of Cahn–Hilliard, ensembles stepped by `jax.vmap` |
| `test_microgen.py` | Microgen: a TPMS sheet of the model against the shape built by Microgen, voxels its sign, saved and read back, spinodoids seeded; a sphere placed and a box turned; the errors; the phases of morphoGen as shapes of Microgen (the implicit function of a phase, the grid's interpolated), meshed by Microgen and back; tetrahedral meshes of 4 and 10 nodes through PyVista both ways |
| `test_blender.py` | Blender: the scene (spheres inside the walls, the box, the script); the transforms read back (translations, a capsule turned, surfaces moved; the window up to the top); particles given by closed surfaces (their voxels those of the spheres they approximate, across the periodic faces; their function; round trips); with `bpy`, hard spheres settled on the floor inside the walls overlapping by less than 1 % of their radius, fibres kept and lower; the scene of a microstructure (closed surfaces of the phases and their volumes, the fluid along x and z, the marching cubes without MMG), the grids of Mantaflow read back, the archive of the tab Export; with Blender, the scene built and a flow baked (nil in the obstacles) |
| `test_alloys.py` | alloys: the description (the composition normalized, the matrix the rest, JSON and YAML round trips, its errors); the scales (60 precipitates by the third moment of the truncated law, 64 grains, the particles as inclusions); the moduli of each scale within the Hashin–Shtrikman bounds of its phases, a dual-phase steel of cubic ferrite crystals; `morphogen alloy` |

Principles:

* **Statistical tests** use fixed seeds; tolerances are a few standard deviations
  computed from the theory (e.g. σ = sqrt(p(1−p)A₃/V)).
* **Exact checks against brute force** are used wherever possible.
* **No cross-platform bitwise comparisons**: floating-point details (e.g. fused
  multiply-add on arm64) may flip voxels lying exactly on a boundary.
* **Optional checks** run only when the tools are installed (VTK, Gmsh,
  scikit-image, tifffile, scikit-fem; MMG for conforming meshes; CuPy and an NVIDIA GPU
  for the GPU backend).

Continuous integration (`.github/workflows/ci.yml`) runs ruff, then the tests on Ubuntu
(Python 3.12 and 3.14), Windows and macOS on Apple Silicon (Python 3.14), in parallel
(pytest-xdist, one file per process: `tests/conftest.py` starts the longest files first
and gives each process its share of the cores), and builds the documentation site with
its warnings as errors (kept as the artifact `site` of the run), for every pull request
and every push to `main`;
`tools/ci_local.sh` runs the same jobs on the development machine (D18). The slowest
tests (marker `slow`: the numerical checks of more than about 15 s each, two thirds of
the time) run in one job, Ubuntu with Python 3.14 (locally the last Python version),
the others take the fast tier (`-m "not slow"`). The tests of the interface check what
its tools pass to the library and give back, on small problems; the physics they show is
checked by the tests of the library.

The documentation site (`docs/conf.py`, Sphinx with MyST and Furo) takes the Markdown
pages of `docs/` and the change log, and writes at each build the pages of the API reference in `docs/api/`
(ignored by git): a package lists the names of its `__all__`, linked to the module that
defines them, and its modules; a module documents what it defines, its undocumented
public functions and methods with their signature. The pages stand in six groups:
getting started (installation, tutorials, gallery, recipes, questions), the user guide (a
page for each step of a study, from the models to the design, and the guide to the
physics), the applications (laser-architectured sheets, explicit dynamics), running
studies (configurations and commands, the interface, other codes), the reference
(API, configuration keys, theory, Z-mat, glossary, performance) and the project.
`docs/index.md` is the landing page (the steps of a study as cards, a first study that
`tests/test_howto.py` runs, where to read what). The README is a short presentation of
its own, whose first study is that of the landing page. `docs/_static/` holds the logo, the styles (`gallery.css`, `site.css`) and
`site.js`, which adds to the pages, without any library, a button that copies each code
block, a filter of the cards of the gallery and the enlarged screenshots.

The images of the pages are written by scripts, never by hand:
`docs/gallery/make_gallery.py` (the cards, images and result figures of the templates,
and `docs/gallery.md` itself), `docs/tutorials/make_figures.py` (it runs the snippets
of the tutorials as written and keeps the figures they save, or that a study they run
writes) and
`docs/tutorials/make_screenshots.py` (it starts `morphogen gui` and drives the study of
the tour in the page with Playwright: `docs/tutorials/screenshots.js`).

## 6. Performance notes

* Numba compiles the kernels on first use (a few seconds) and caches them on disk
  (`cache=True`).
* `morphogen.set_num_threads(n)` or `--threads n` limits the threads used by Numba and
  SciPy.
* Memory goes mainly into the output arrays: 1 byte per voxel for phases, 4 bytes for
  labels and fields. Tessellations are labelled in slabs of about 2 M voxels, and
  covariances use 1D FFTs along each axis.
* Automata: 30 updates of a 64³ Moore automaton take about half a second; 4000
  Gray–Scott steps on 48³ cells about 0.7 s; nucleation and growth of 64³ cells about
  0.2 s; a DLA cluster of 30 000 particles a few seconds (first calls include the
  compilation).
* Topology optimization: about 0.06 s per iteration for 48² planar cells, 0.25 s for
  64², 0.5 s for 24³ cells; designs converge in 50–300 iterations. The conjugate
  gradients compute their dot products in a single thread: BLAS would spread them over
  threads that compete with those of the compiled kernels, which made mid-size solves
  several times slower.
* Conforming meshes take about 0.2–0.5 ms per element (MMG is sequential); finite element
  systems of 10⁵–10⁶ unknowns are solved in seconds to a minute per loading, the
  hierarchy being reused for all loadings of a boundary condition. Realizations are
  independent: run them in processes (`-j`) or split studies into tasks (`--task-id`,
  `--n-tasks`, `morphogen submit`) to use several cores or nodes.
* Polycrystals: the exact cells of 300 grains take about 2 s, their regularization 1–2
  s, the meshing 15–20 s for 2 × 10⁵ elements (mean quality 0.73–0.74).
* Grain-boundary phases cost a k-nearest query (k ≥ 16) per voxel: about 6 s for 128³
  voxels of 200 cells. Dead leaves cover 96³ voxels in about 0.1 s; laminates
  voxelize slice by slice, a k-d tree query per layer.
* Chunked voxelization holds one slab of about `max_voxels` (2²⁴ by default) in
  memory.
* GPU (`device: gpu`): the linear FFT solves are 5 to 7 times faster than on 20 CPU
  threads from 64³ to 192³ voxels (NVIDIA RTX 3000 Ada laptop GPU), the loading paths
  of a polycrystal 1.6 to 2.8 times and of a porous composite 2.4 to 3.5 times, the integration of the behaviours being bound by the double precision
  of this GPU (theory notes, §35; `examples/gpu_benchmark.py`, `--paths` for the paths
  alone). The first solve compiles the CUDA kernels (NVRTC, the behaviours in a few
  seconds; CuPy caches them on disk). The integration points of a path take on the GPU
  the memory they take on the CPU (about 1 kB per point for a crystal with 12 slip
  systems in small strains).
* Behaviours of the points on 20 CPU threads, an increment in the middle of a loading
  (`examples/behaviour_benchmark.py`): 10 ns per point for elasticity, 88 for von
  Mises, 345 for Hill's criterion, 522 for GTN and 1.4 µs for an fcc crystal in small
  strains; the logarithmic strain of finite strains adds about 0.25 µs (its tangent
  from the six directions of dC). The kernels of
  Hill and of porous plasticity take their work arrays once per block of points
  (`_hill_work`, `_porous_work`); the porous Newton iteration and its tangent take the
  analytical Jacobian of its equations (`_jacobian`, `_matrix_derivatives`), GTN's
  matrix stress at the points of the line search bracketed from the current point's.
