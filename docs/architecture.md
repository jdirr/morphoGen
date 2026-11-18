# morphoGen — Architecture

This document describes the modules, data model, conventions, file formats and test
strategy of morphoGen 1.0. The requirements are in [specification.md](specification.md)
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
  distributions.py   distributions of radii, lengths... (moments, quadrature, configs)
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
    annealing.py       Yeong-Torquato reconstruction: swaps annealed to S2 and L
    watershed.py       marker-controlled watershed (Meyer's flooding) and the maxima
                       of a given prominence (union-find), of planes and of volumes
                       (26-connected voxels, periodic faces)
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
    hardcore.py        HardcoreSpheres (RSA or Lubachevsky-Stillinger),
                       PenetrableSpheres (the cherry-pit model)
    tessellation.py    Voronoi, Laguerre (Poisson or packed germs, grain-boundary
                       phase) -> Tessellation
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
                     the dialogs of the system, icon.png): menus
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
                     archives); one module per tab: statistics (RVE
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
    curved.py          10-node meshes whose nodes lie on the interfaces of the phases
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
                       faces cut exactly), the route of each microstructure (exact_mesh)
    octree.py          balanced octrees refined by a test of their cells, periodic
                       across chosen axes, split into tetrahedra (Kuhn cells, cones over
                       fanned faces; conforming, matching opposite faces)
    polyhedra.py       exact cells of power diagrams (CellComplex: pieces, facets)
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
| `cross_linked_fibers` | `Combined` inclusions (fibres; balls of the cross-links minus the fibres), or a `PrimitiveUnion` if merged | 0 pores, 1 fibres, 2 cross-links | no (infinite fibres) / yes |
| `delaunay_network` | `PrimitiveUnion` (spherocylinders on the edges of a periodic Delaunay triangulation, spheres at the nodes; the framework in the metadata) | 0 pores, 1 fibres | yes / no (the periodic network cut by the box) |
| `voronoi`, `laguerre` | `Tessellation` (germs, weights, phases, orientations, grain-boundary thickness; Poisson, packed or stealthy hyperuniform germs) | per cell (and the boundary phase) | yes / no |
| `gaussian_excursion` | `GaussianField` (Fourier modes from a seed) | 0, 1 excursion | yes / no |
| `tpms` | `TPMSStructure` (surface, level, cell size) | 0 void, 1 solid | yes / no |
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

## 5. Configuration files

```yaml
model:                 # required: type + parameters (morphogen models <type>)
  type: boolean_spheres
  radius: 1.0          # number, or {type: uniform|lognormal|normal|discrete, ...}
  volume_fraction: 0.3
domain:                # size required except for RVE studies
  size: 20.0           # or [Lx, Ly, Lz]
  periodic: true
  unit: um
seed: 1                # base seed (realization r uses SeedSequence(seed, spawn_key=(r,)))
realizations: 2
voxels:                # images, voxel meshes, surfaces, measurements, FFT solver
  voxel_size: 0.2      # or shape: n | [nx, ny, nz]
  supersampling: 1     # >1: exported images hold partial volume fractions
  section: null        # x | y | z: the run studies one layer of voxels through the middle
                       # of the volume, normal to that axis (a planar study: a plane strain
                       # for the FFT solver, far fewer voxels; not with xfem nor windows)
mesh:                  # conforming tetrahedral meshes (MMG), FE solver
  size: 0.5            # largest edge (required)
  order: 2             # 10-node tetrahedra, the default of a mesh analysed by finite
                       # elements; 1: 4-node tetrahedra, the default of a mesh only written
  curved: true         # 10-node elements: their nodes on the interfaces of the phases
                       # (curved edges); false: the flat faces of the mesher
  phases: auto         # all | solid (pores not meshed); auto: solid if a phase is a void
  hausdorff: null      # interface accuracy (default size / 20)
  min_size: null       # default size / 10
  gradation: 2.0       # size ratio of neighbouring elements
  background: null     # level-set sampling (default size / 2): must resolve thin features
  grains: null         # tessellations: one subdomain per grain (default), false: phases
  min_feature: null    # grains: features below this length removed (default size / 4)
  cells: whole         # periodic tessellations: whole grains (default) or box (cut cells)
  method: auto         # exact: no background grid, the exact cells, convex pieces or
                       # point-wise level set (theory §60; background unused); grid: the
                       # level set on a grid; auto: exact when possible, grid otherwise
outputs:
  directory: out/run   # relative to the configuration file
  name: sample         # file stem
  geometry: [json, csv, geo]
  image: [vti, vtk, mhd, npz, npy, tif]
  labels: [vti]        # tessellations: cell labels
  level_set: [vti]
  mesh: {formats: [inp, msh, vtu], phases: [1]}   # voxel hexahedra
  conforming_mesh: [vtu, inp, msh, mesh, xdmf]    # tetrahedra (needs the mesh section)
  periodic_pairs: true
  periodic_equations: true   # *EQUATION files for the meshes written (periodic domains)
  damask: true         # tessellations with orientations: DAMASK grid and material file
  neper: true          # tessellations: Neper seeds, weights, orientations, command
  amitex: true         # AMITEX_FFTP runs of the image (needs homogenization), or options:
                       # {fields, tolerance, elastic_only, processes, library}; the
                       # nonlinear materials as MFront laws, along the paths
  surface: [stl]
  database: null       # SQLite file recording the run and its results (results.py)
  tables: [txt, xlsx]  # the tables of the run in these formats too (besides .csv)
analysis: [volume_fraction, specific_surface, connectivity, phase_fractions, cells]
homogenization:        # apparent properties of every realization
  method: auto         # fe (mesh section) | fft (voxels, periodic) | xfem (the level set
                       # of the model cut by the grid of the voxels section, periodic) |
                       # auto: fe if mesh given
  materials:           # one per phase: void, {young, poisson, conductivity}, {bulk, shear}
    0: void            # or crystals: {cubic: [C11, C12, C44]}, {hexagonal: [C11, C12,
    1: {young: 210000.0, poisson: 0.3, conductivity: 100.0}   # C13, C33, C44]},
                       # {stiffness: 6 x 6}, conductivity: k | [k_a, k_c] | 3 x 3
                       # paths: plasticity: {...}, crystal_plasticity: {families,
                       # tau0, Q, b, C, D, K, n, interaction; c_over_a (hexagonal
                       # families); twins: {twin1012: {tau0: ..}}, twin_saturation,
                       # twin_reorientation (the fraction at which the lattice of a
                       # point becomes that of its twin, or {threshold, growth}:
                       # grain by grain), twin_detwinning (true; a share of the
                       # resistance); cleavage: {planes: cube, strength, energy and
                       # length (or softening), shear, K, n, closure (true; a share
                       # of the strength), friction, coupling (the rate at which
                       # the slip lowers the strength of the planes),
                       # regularization: {width, ratio} of the damage band of the
                       # planes, with their energy (or {modulus, length} of their
                       # microdamage, with their linear softening)}} or
                       # mfront: law.mfront | {source | library, behaviour,
                       # material_properties, parameters, external_state_variables,
                       # temperature, plastic_strain} (sources relative to this file)
                       # other physics: electrical_conductivity, permittivity,
                       # permeability, diffusivity (as conductivity; 0: an inert phase),
                       # refractive_index: n | [n, k], swelling: beta | [..] | 3 x 3,
                       # coupling: {piezoelectric: {e31, e33, e15}, piezomagnetic: {q31,
                       # q33, q15}, magnetoelectric: 3 x 3, seebeck: s, pyroelectric: [..]}
                       # behaviours: viscoelasticity: {shear: [[g, tau], ...], bulk:
                       # [[k, tau], ...], wlf: [C1, C2]}, hyperelasticity: {model:
                       # neo_hookean | mooney_rivlin | yeoh | arruda_boyce | gent | ogden |
                       # saint_venant_kirchhoff, mu | c10, c01 | ..., bulk}, damage:
                       # {strength | threshold, fracture_energy + length | ultimate,
                       # softening, equivalent, split: none | volumetric | spectral,
                       # viscosity}; concrete: damage: {law: mazars, threshold,
                       # tension: [A_t, B_t], compression: [A_c, B_c], beta}; with a
                       # plasticity: damage: {law: lemaitre, S, s, threshold, closure}
                       # or {law: kachanov, A, r, k, alpha, beta} (and critical,
                       # viscosity: the damage of the effective stress)
  boundary_conditions: [kubc, mixed, periodic, subc]   # aliases: utg, mbc/mtbc, pbc, uhf
  properties: [bulk_modulus, shear_modulus, conductivity, stiffness, conductivity_tensor,
               thermal_expansion]   # materials with expansion: alpha | [a_a, a_c] | 3 x 3
               # the other transports: electrical_conductivity, permittivity, permeability,
               # diffusivity and their tensors (<name>_tensor); weak couplings: swelling,
               # poroelastic; strong couplings (fft): piezoelectric, piezomagnetic,
               # magnetoelectroelastic, magnetoelectric, thermoelectric; harmonic regimes
               # (fft): dynamic_stiffness, complex_conductivity, refractive_index; flow
               # through the pores (fft, the voids, cubic voxels): hydraulic_permeability,
               # acoustic
  frequencies: [0.1, 10.0]   # angular frequencies of dynamic_stiffness, complex_conductivity
  wavelengths: [0.6]         # of refractive_index
  mean_temperature: 300.0    # thermoelectric: the absolute temperature of the law (K)
  fluid_bulk_modulus: null   # poroelastic: the fluid of the pores (undrained response)
  fracture:                  # the paths break the voxels (fft): phase field
    {toughness: {0: 100.0}, length: 0.5, model: AT1, split: spectral}   # or AT2;
                       # irreversibility: bound | history; staggered_tol, max_staggered
  fatigue:                   # the points along a cycle of a path (fe, fft): its fields are
    {criterion: dang_van, bending: 110.0, torsion: 80.0, path: cycle, cycle: [1, 3]}
                       # kept at each step. criterion: crossland | sines | dang_van |
                       # findley | matake, with the fatigue limits of the phase (bending
                       # and torsion, fully reversed; torsion and repeated for sines) or
                       # its coefficients (alpha, beta; k, limit); phases: those judged
                       # (default: the solid ones), or criteria: {phase: {...}}; cycle:
                       # its times (default: the whole path). Results: fatigue_utilization,
                       # fatigue_safety, fatigue_beyond; the map <path>_<bc>_fatigue
  beremin:                   # cleavage by the local approach, on the fields of a path (fe,
    {modulus: 22.0, scale: 1550.0, volume: 1.25e-10, planes: cube, plastic: 1.0e-4}
                       # fft): the Weibull stress of the points of the plastic zone (those
                       # beyond the plastic strain) and the probability that the weakest
                       # link has failed. planes: the cleavage planes of the grains, whose
                       # normal stress drives them (left out: the principal stress); path.
                       # Curves: beremin_weibull, beremin_probability; the field
                       # <path>_<bc>_beremin
  loadings:            # named macroscopic loadings (the periodic element): components not
    tension: {stress: {"11": 100.0}}                       # given are stress-free
    mixed: {strain: {"11": 0.001, "22": 0.0}}             # prescribed strains, free rest
    heating: {strain: {"11": 0, "22": 0, "33": 0, "23": 0, "31": 0, "12": 0},
              temperature: 100.0}                       # blocked: thermal stresses
  paths:               # loading paths with the nonlinear behaviours (fe or fft)
    tension: {times: [0, 20], increments: 10, strain: {"11": [0, 0.02]}}
  nonlinear:           # options of the Newton iterations of the paths
    tolerance: 1.0e-6
    max_iterations: 25
    max_cuts: 10
    kinematics: small  # or finite
    integration: bbar  # of the paths: bbar (B-bar, F-bar) | full; default: bbar on
                       # voxels (fft), the integration of the section on a mesh (fe)
    tangent: auto      # symmetric (conjugate gradients) | full (GMRES) | auto: by the laws
                       # and, for Krylov solvers, by the asymmetry of their tangents
    microdamage: {tolerance: 1.0e-3, passes: 200, acceleration: true}
                       # cleavage planes regularized by their microdamage: the change
                       # of the microdamage at which a step has settled (over the
                       # damage that breaks a plane), its most equilibria, and whether
                       # the alternate scheme is accelerated
    contact: {friction: 0.3}   # fe, porous samples: the walls of the pores that close
                       # press on each other, and on their images across the faces of a
                       # periodic cell (morphogen.contact); true, or friction, stiffness,
                       # tangential, depth, augmentations, tolerance, points (gauss |
                       # nodes: corners and edges that touch); finite strains
                       # unless kinematics: small; the curves get contact_area,
                       # contact_pressure, contact_penetration, contact_dissipation
    interfaces: {normal_strength: 30.0, mode_I: 0.05, friction: 0.3, viscosity: 0.02}
                       # fe: the interfaces of the phases debond (morphogen.fracture.
                       # periodic). law: bilinear (normal_strength, mode_I; shear_strength,
                       # mode_II, eta, stiffness) | exponential (strength, energy; beta,
                       # penalty) | contact (lips without adhesion; stiffness); friction
                       # of the broken lips; faces: phases | grains | pairs [[0, 1]];
                       # viscosity: the time of the regularization of the damage; slip:
                       # the elastic slip of the friction; the curves get debonded_area,
                       # interface_damage, interface_opening, interface_dissipation.
                       # A law per pair of phases: laws: {0-1: {...}, 0-2: {...}} (with
                       # faces: grains, 1-1: the boundaries of the grains of phase 1)
  integration: full    # fe, 10-node tetrahedra: full | bbar (also: reduced), the volume
                       # change of each element as its mean, for nearly incompressible
                       # phases (elastomers) and plastic flow; the paths of voxels too
  device: cpu          # FFT solvers: cpu | gpu | auto (NVIDIA GPUs, CuPy)
  solver: auto         # auto | direct | amg | cg (FE linear solver)
  tolerance: 1.0e-8
  max_iterations: null
  fields: false        # write the local fields (.vtu for FE, .vti for FFT)
visualization:         # figures and videos of the first realizations (viz/workflow.py)
  realizations: 1
  cmap: viridis        # matplotlib colormap of every view
  fps: 24
  size: [640, 480]     # pixels of the 3D views (VTK) and of the videos
  elevation: 25        # the camera of the 3D views (degrees)
  azimuth: 30
  dpi: 200             # figures
  figures:             # of: image | labels | mesh | <property|loading|path>.<bc>
    - {of: image, kind: sections, formats: [png, svg, tex]}   # or surface (phase)
    - {of: mesh, color: phase}                                # 3D view (bitmaps)
    - {of: shear_modulus.periodic, color: von_mises, magnification: null,
       part: total, section: {axis: 2, fraction: 0.5}, clim: null}
    - {of: tension.periodic, kind: curves, formats: [pdf]}   # or kind: field
  videos:              # formats: mp4 | gif | frames; mode: once | loop | bounce
    - {of: image, kind: orthoslices, mode: bounce}   # or sections (axis), surface
    - {of: mesh, kind: turntable, frames: 72}        # or sweep (axis)
    - {of: shear_modulus.periodic, wave: sine, mode: loop, repeat: 3}   # or ramp
    - {of: tension.periodic, color: p, frames: 60}   # the steps, interpolated
                       # the fields of a path: von_mises, p, displacement, stress_11...,
                       # strain_11..., plastic_strain_11...; interface_damage and
                       # interface_opening (nonlinear.interfaces); fatigue_utilization,
                       # fatigue_equivalent, fatigue_amplitude, fatigue_hydrostatic_max
                       # (homogenization.fatigue, at the end of the path)
rve:                   # turns the run into an RVE study
  method: ensemble     # or windows
  sizes: [10, 20, 40]  # cube edges (ensemble)
  realizations: [64, 16, 4]
  size: 100.0          # windows: edge of the single realization
  window_sizes: [5, 10, 20]
  shifts: 1            # windows: tilings per axis, shifted around the periodic realization
  properties: [volume_fraction, bulk_modulus.kubc, conductivity]   # measures, <prop>.<bc>,
               # quantities of the paths: <path>.<quantity>[@<time>][.<bc>], tension.S11@5,
               # tension.strength, tension.work (and .fracture_energy with a fracture);
               # named results: <property>.<result>[.<bc>], piezoelectric.e33,
               # dynamic_stiffness.shear_loss_1, poroelastic.biot_coefficient,
               # hydraulic_permeability.mean, acoustic.tortuosity;
               # stiffness (or stiffness.<bc>): the mean C and S = C^-1 per size, and
               # conductivity_tensor; windows: volume_fraction and these tensors (by FFT)
  relative_errors: [0.01, 0.05]
  n_realizations: [1, 10]
  gamma: null          # impose the exponent (e.g. 0.6667 for Poisson fibres)
  min_volume: null     # range of volumes used in the fit
  max_volume: null
  bootstrap: 1000      # replicates for the confidence intervals (0: none)
  weights: none        # none | n: weight the sizes by (n - 1) / 2 in the fit
threads: null
```

Without a `voxels` section, the volume fraction is measured on the conforming mesh.
Crystals are turned by the orientations of the grains of tessellations
(`crystal_orientations: true`), on conforming meshes (grain by grain) and on voxels.
`morphogen template` lists the example configurations shipped with the package.

Topology optimizations have their own configuration files (`morphogen optimize`):

```yaml
optimization:
  objective: poisson_ratio   # bulk_modulus | shear_modulus | poisson_ratio | conductivity | target
  volume_fraction: 0.35      # required
  grid: [64, 64, 1]          # required: voxels (nz = 1: planar, prismatic cell)
  cell_size: 1.0             # edge of the cell along x
  material: {young: 1.0, poisson: 0.3, conductivity: 1.0}
  method: mma                # oc | mma | beso | hca (oc, beso, hca: monotone objectives)
  penalty: 3.0               # SIMP exponent
  filter_radius: 3.0         # voxels
  projection: 8.0            # final Heaviside sharpness (0: none)
  symmetry: cubic            # none | orthotropic | cubic (square in 2D)
  stiffness_floor: 0.001     # SIMP stiffness of void voxels
  min_stiffness: 0.05        # poisson_ratio: mean normal stiffness / E
  iterations: 300
  tolerance: 0.01            # largest change of the design variables
  move: 0.2
  target: null               # 6 x 6 stiffness (NaN entries ignored) for objective target
  evolution_rate: 0.02       # beso
  initial: centered_void     # centered_void | random | uniform
seed: 1
outputs:
  directory: out/cell
  name: cell
  image: [vti]               # thresholded design
  density: [vti]             # densities
  geometry: [json]           # LatticeField
  check_refinement: 2        # constants of the design on a grid twice as fine (0: none)
```

Laser-architectured sheets have their own configuration files (`morphogen laser`,
[laser.md](laser.md)): a `laser_sheet` section (the material, a preset or a mapping
`preset` + changes; the `treatment` and its `alloying`; the `configuration`, single or
symmetric; the `layers`; the `laser` beams `beam` and `bottom` and the `pause` between
tracks; the `specimen`; the `element` size; the `pattern`, its `kind` and parameters;
`compare`, named reference patterns; `loading`, options of the tensile test), an
optional `optimization` (`method`, `evaluations`, `objective`, `parameters` as dotted
names with bounds, `constraints` on measures, `target`, `check_element`), `seed` and
`outputs` (`directory`, `figures`, `fields`, `gcode`, `resolution`); `loading` may be
`{kind: bending, span, deflection}` on a strip, and `directions` a list of angles. The
template `laser_corrugated` is the reference case of the thesis with its optimization,
`laser_bending` a strip hardened on one face in three-point bending.

## 6. Command line

| Command | Purpose |
|---|---|
| `morphogen run CONFIG [-o DIR] [--seed S] [--task-id I\|auto --n-tasks N] [-j J] [--threads T] [--resume] [--mpi]` | batch or RVE study, in J processes, one task of a split run, or one task per MPI rank |
| `morphogen merge DIR [--no-report]` | gather the records of the tasks of a split run into its results and report |
| `morphogen status DIR [--json]` | realizations done and missing in a run directory |
| `morphogen submit CONFIG --scheduler slurm\|pbs\|torque\|sge\|oar\|lsf --tasks N [-j J] [--threads T] [--time T] [--memory M] [--queue Q] [--account A] [--setup LINE...] [--submit]` | job array and merging job scripts for a cluster, submitted with `--submit` (the merging job waits for the array) |
| `morphogen fit RESULTS... [--property P] [--group-by C] [--volume-column C] [--eps E...] [--n N...] [--gamma G] [--bootstrap B] [--weights n] [-o DIR]` | statistical RVE analysis of CSV tables (morphoGen results directories, or external FE/FFT results) |
| `morphogen homogenize SAMPLE --material P=... [--bc ...] [--property ...] [--size H --order O --min-feature F] [--voxels N] [--device cpu\|gpu\|auto] [--fields DIR] [--save-mesh FILE] [--json]` | apparent properties of a geometry (`.json`, meshed, or voxelized with `--voxels`), a tetrahedral mesh (`.vtu`, `.mesh`: FE) or an image (FFT, on the CPU or an NVIDIA GPU); crystals as `P=cubic:C11/C12/C44` or `hexagonal:...`; a stiffness printed with its compliance S = C^-1, their notation and its engineering constants (in the JSON: `tensor`, `compliance`, `notation`) |
| `morphogen homogenize ... [--path [NAME:]PATH ...] [--paths FILE] [--finite] [--contact [MU]] [--interfaces KEY:VALUE,...] [--curves DIR]` | loading paths followed with the nonlinear behaviours of the materials (their curves as `.csv`); in finite strains; with the contact of the walls of the pores of a mesh (friction `MU`); with the interfaces of the phases of a mesh that debond (`normal_strength:30,mode_I:0.05,friction:0.3,viscosity:0.02`; `law:exponential,strength:..,energy:..`; `law:contact`; `faces:phases\|grains\|0-1+0-2`) |
| `morphogen analyze IMAGE [--array A] [--measure M...] [--json]` | measurements on an image file |
| `morphogen texture MAP [--threshold T] [--min-size N] [--axes map\|file] [--setting 1\|2\|3\|4] [--poles HKL...] [--direction x\|y\|z\|a,b,c] [--halfwidth H] [-o DIR] [--format png\|svg\|pdf] [--convert FILE [--clean]] [--json]` | the grains and the texture of an orientation map (`.ang`, `.ctf`): their numbers phase by phase; the tables of the grains and of their boundaries, the map in the colours of its orientations, the pole figures, the inverse pole figure and the disorientations (`-o`); the map written in the other format, as it is or cleaned |
| `morphogen texture FIELDS [--lattice cubic\|hexagonal] [--poles HKL...] [--direction D] [--halfwidth H] [-o DIR] [--json]` | the texture of the lattices of the fields of a loading path (`.vti`, `.vtu`: crystals in finite strains, or turned by their twins): the largest densities of its pole figures and of its inverse pole figure, and their figures |
| `morphogen compress SAMPLE --material P=... --strain E [--axis 0\|1\|2] [--friction MU] [--no-self-contact] [--points gauss\|nodes] [--steps N] [--small] [--size H --order O] [--curve F] [--fields F] [-o F] [--json]` | compression of a sample (a geometry, meshed, or a tetrahedral mesh; its solid phases) between two rigid platens, with contact and friction on the platens and of its walls on themselves: nominal stress against strain, areas in contact |
| `morphogen indent SAMPLE --material P=... --depth D [--indenter sphere\|cone] [--radius R] [--half-angle A] [--point U V] [--unload] [--base fixed\|smooth] [--poisson NU] [--steps N] [--small] [--curve F] [--fields F] [-o F] [--json]` | indentation of the upper face of a sample by a rigid sphere or cone: force against depth, area of the contact, the reduced modulus read (Hertz, Sneddon, the layer on its base, the slope of the unloading) |
| `morphogen bands SAMPLE --material P=... [--waves elastic\|acoustic\|shear\|tm\|te\|electromagnetic] [--path G X M G] [--bands N] [--points N] [--order O] [--coarsen N] [-o bands.csv] [--figure F] [--json]` | band structure of the Bloch waves of a periodic cell (an image or a tetrahedral mesh): elastic waves, the scalar waves of sonic and planar photonic crystals, or light in a three-dimensional crystal (plane waves on the voxels of an image, `--coarsen` times coarser with averaged interfaces); the band gaps along the path |
| `morphogen percolation CONFIG [--parameter P] [--lower L] [--upper U] [--phase N] [--axis x\|y\|z\|any\|all] [--connectivity 4\|6\|8\|18\|26] [--periodic\|--no-periodic] [--method labels\|fire] [--directions axes\|lattice\|N] [--sizes L1,L2,...] [--nu NU] [--fit-nu] [--rigidity] [--cut C] [--exponent thermal\|elastic] [--threshold P] [--threshold-stderr S] [--fractions F1,F2,...] [--contrast C] [--device cpu\|gpu\|auto] [--solver fft\|fe] [--realizations N] [--tol T] [--seed S] [--json]` | percolation threshold of the model of a configuration (its domain and voxel grid) against a parameter, by default the volume fraction: the threshold, its bracket and the probabilities tried (a 2-D configuration, one voxel along z: along x or y, through 4 or 8 neighbours); with `--directions`, its anisotropy: the thresholds across the cores of the box along the axes, the axes and diagonals of the lattice or N directions sampled, their paired deviations and ratio, the threshold tensor; with `--sizes`, the thresholds of cubes of these sizes at its voxel size extrapolated to p_c + a L^(-1/nu); with `--rigidity`, those of its elastic moduli by the solver of the voxels (where half the realizations bear a uniaxial strain above `--cut`, 1e-8 of the phase's modulus, the others void), extrapolated alike, with the solutions counted; with `--exponent`, the critical exponent of the conductivity (t) or of the elastic moduli (f) by the solver of the voxels: the means of the realizations of the `--sizes` at the threshold (given, or extrapolated first: through 26 neighbours wrapping around the box for t, by the moduli as with `--rigidity` for f) fitted to L^(-t/nu), the error of the threshold propagated, a sweep over `--fractions`, the other phases at `--contrast` (1e-8); `--solver fe`: the voxels assembled and factorized instead of the FFT solver (on the CPU; a few iterations near the rigidity threshold, void around) |
| `morphogen homogenize ... [--cells whole\|box] [--loading [NAME:]COMPONENTS ...]` | whole periodic grains; macroscopic loadings such as `tension: S11=100`, `E11=0.001 S22=0 dT=10` (strain `E`, stress `S`, gradient `G`, flux `Q`; components not given are free) |
| `morphogen frame-stiffness FILE [--seed S] [--realization R] [--beams euler-bernoulli\|timoshenko] [--shear-coefficient K] [--rigid-joints C] [--young E] [--poisson NU] [--softening axial\|bending\|modulus\|area ...] [--axis x\|y\|z] [--replicas N] [--cv CV ...] [--distribution normal\|lognormal\|bimodal] [--replica-seed S] [--table CSV] [--figure FILE] [--json]` | the stiffness of the frame of beams of a realization of `delaunay_network`, `open_foam` or `random_lattice` (a configuration, realized as the realization R of its batch run, or a geometry `.json`; its framework kept in its metadata): Euler–Bernoulli or Timoshenko beams (Cowper's coefficient, or K), rigid over C radii at the joints, fibres of modulus E and Poisson's ratio NU; the Voigt stiffness (per unit area for a planar network) and its compliance S = C^-1 with their notation (JSON: `stiffness`, `compliance`, `notation`), its engineering constants, along each axis the Young's modulus, Poisson's ratio, ratio to the affine estimate, energy split (stretching, bending, torsion, shear) and non-affinity of a uniaxial stress, their means (the summary of the models), and Ban et al.'s softening coefficients s and v of the modes given along an axis; with `--replicas`, N replicas per CV of the stiffness of the beams of each mode (the law of `--distribution`, seed S) against them: the plain and corrected means of E/E0 with their standard errors, 1 - s CV^2, the scatter against v CV^2 (`--table`: CSV; `--figure`: the means against CV and the scatter against v CV^2) |
| `morphogen amitex export SAMPLE --material P=...\|--materials FILE [--voxels N] [--loading ...] [--path ...] [--stiffness] [--fields] [--tol T] [--elastic-only] [--name NAME] [-o DIR]` | AMITEX_FFTP runs of an image (or of a geometry with `--voxels`): the six unit strains of the stiffness by default, named loadings and paths; the files and `<name>_run.sh` in DIR |
| `morphogen mfront --material P=...\|--materials FILE [--kinematics small\|finite] [--name NAME] [-o DIR]` | the MFront sources of the nonlinear materials (`<name>Phase<p>.mfront`; no underscore, for Abaqus; the learned laws too), with the command that compiles them for an interface |
| `morphogen amitex read DIR [--name NAME] [--json] [-o CSV]` | the results of the runs of an export: averages, apparent stiffness and its compliance (JSON: `stiffness`, `compliance`, `notation`) and moduli (a row appended to a CSV table, for `morphogen fit`: `C11` ... `C66`, `S11` ... `S66`) |
| `morphogen db add\|list\|keys\|export\|remove [ITEMS] [--db FILE] [--where COND...] [-o CSV]` | database of the runs: add run directories, list them, the keys of their records, export the records, remove runs |
| `morphogen ashby [--db FILE] [--x KEY] [--y KEY] [--by KEY] [--where COND...] [--guides N...] [--theme light\|dark] [-o CHART] [--table CSV]` | normalized Ashby chart of the records |
| `morphogen run CONFIG --db FILE` | also record the run in a database (`outputs.database`) |
| `morphogen models [NAME] [--json]` | list the models / describe their parameters |
| `morphogen template [NAME] [-o FILE]` | example configurations |
| `morphogen info` | versions, threads, models, the GPU (or why none is usable) |
| `morphogen optimize CONFIG [-o DIR] [--seed S]` | topology optimization of a periodic cell |
| `morphogen laser CONFIG [-o DIR] [--seed S] [--evaluations N] [--no-optimization] [--json]` | laser-architectured sheet: the treatment of its pattern, its tensile test, the reference patterns and the optimization of the laser path |
| `morphogen learn CONFIG [-o MODEL] [--data FILE] [--model feedforward\|potential\|recurrent\|standard] [--epochs N] [--paths N] [--seed S] [--json]` | a behaviour learned from data: random paths of a material point (`material`) or of a voxel image (`sample`, `materials`), a network trained and saved (`.npz`), its errors on the validation paths and on named `paths` that the learned law runs at a material point |
| `morphogen fit-rule IMAGE [--neighbourhood N] [--steps S] [--lattice L] [--tiling T] [--population P] [--generations G] [-o CONFIG] [--json]` | automaton rule whose structures match an image, on the grid or the cells of a tiling `T`; writes the configuration that generates them |
| `morphogen view FILE [--shape N] [--labels] [--tile T] [--surface] [--history] [--kinetics] [--animate GIF --frames F] [-o PNG]` | figures of a geometry, configuration or image; animation of the formation of a structure |
| `morphogen export INPUT OUTPUT [--kind K] [--field F] [--displacement U] [--magnification M] [--wave ramp\|sine] [--section AXIS FRACTION] [--frames N] [--fps F] [--mode once\|loop\|bounce] [--repeat R] [--cmap C] [--clim LO HI] [--size W [H]] [--shape N] [--format frames]` | an image, a mesh with its arrays (`.vtu`), a geometry or a table to another format (by the suffix), a figure or a video |
| `morphogen zmat [FILE] [--behaviour K] [-o DIR] [--coverage] [--json]` | how a material file of Z-mat is translated (native kernels, or the MFront source written in DIR), or the coverage of Z-mat's library |
| `morphogen gui [--port P] [--no-browser]` | web interface on http://127.0.0.1:P/ |
| `morphogen desktop [--session DIR] [--browser] [--debug]` | the interface as a stand-alone application, in a window of its own (pywebview), with the dialogs of the system; `morphogen-desktop [--check]` |
| `morphogen radioss info\|install\|export\|run\|read [SOURCE] [-o DIR] [--materials FILE] [--loading compression\|tension\|shear\|impact] [--strain E] [--rate R] [--velocity V] [--voxels N \| --mesh conforming --element-size H] [--units "Mg mm s"] [--threads T] [--mpi N] [--json]` | explicit dynamics with OpenRadioss: the installation found or installed from GitHub, the inputs of a geometry or an image, their run, their results (summary, `curves.csv`, `energies.csv`, frames) |

**Parallel and cluster runs.** A study is a list of items `(size index, realization)`.
Task `i` of `N` processes items `i, i + N, i + 2N, ...`, and each item's seed depends
only on its indices, so results do not depend on `N`, nor on the number of processes.
On one machine, `morphogen run study.yaml -j 8` runs eight realizations at a time
(spawned processes sharing the threads). On a cluster:

```bash
morphogen submit study.yaml --scheduler slurm --tasks 32 -j 4 --time 02:00:00 --submit
# or by hand, in a job array script:
morphogen run study.yaml --task-id auto --n-tasks 32 --resume --seed 7
# then, once all tasks are done (submit chains it after the array):
morphogen merge out/study
```

The seed must be fixed for split runs (`submit` draws one if the configuration has
none). `--resume` reruns only the realizations without a record, e.g. after a time
limit; `morphogen status` lists them. Under MPI, `mpirun -n 64 morphogen run study.yaml
--mpi` gives each rank its share and merges at the end (mpi4py, optional). Scripts using
`-j` above 1 must guard their entry point with `if __name__ == "__main__":` (processes
are spawned).

## 7. Output files

A run directory contains:

* `<name>_<index>.*` for each realization (`<name>_L<size>_<index>.*` in RVE studies);
* `<name>_<index>_tet.*`: conforming meshes; `<name>_<index>_<property>_<bc>.vtu`
  (FE) or `.vti` (FFT): local fields (energy density, von Mises stress, stress and
  strain components, displacement or temperature at the nodes);
* `results.csv`, or `results_task<i>of<n>.csv` for split runs: one row per
  realization (measures, apparent properties `<property>.<bc>`, tensor components:
  of a stiffness the upper triangles of `C` and of its compliance `S = C^-1`,
  `stiffness.<bc>.C11` ... `C66` and `stiffness.<bc>.S11` ... `S66`, beside its
  engineering constants `E1`, `nu12`, `G23`..., phase fractions, mesh size and quality);
* `homogenization.csv` (`homogenization_task<i>of<n>.csv`): one row per realization,
  property and boundary condition (value, energy estimate, isotropy, response, solver
  data);
* with `outputs.tables`, `results.txt`, `results.xlsx`, `homogenization.*` and the curves
  of the paths in these formats too;
* with a `visualization` section, `<stem>_<view>.<format>`: the figures and videos of
  the first realizations (`<stem>_image_orthoslices.mp4`, `<stem>_mesh.png`,
  `<stem>_<result>_<bc>_<field>.mp4`, `<stem>_<path>_<bc>_curves.pdf`...; the images of
  TikZ figures beside them);
* `records/<stem>.json`: the record of each realization (its rows and timing), from
  which `merge` rebuilds the tables of split or interrupted runs;
* `run.json` (`run_task<i>of<n>.json` for the tasks of split runs): configuration, model,
  theory, seed, versions and timing, and `stiffness_notation` when the stiffness is
  computed (Voigt order 11 22 33 23 31 12, engineering shears; TFEL and Abaqus order
  their shears 12 13 23);
* `<scheduler>_array.sh`, `<scheduler>_merge.sh`: job scripts written by `submit`;
* a laser run (`morphogen laser`) writes `laser.json` (the configuration, versions and the
  measures of each design: the pattern, the references, the optimum and its check on
  finer elements), `curves.csv` (the engineering curves of every design), `history.csv`
  (the evaluations of the optimization: parameters, measures, objective, violation,
  cost), `path.gcode` and `path.csv` (the tracks of the best design), `fields.vti`
  (hardness, treated share, peak temperature and carbon of the parallel part, a pixel
  per `resolution` mm and a slice per layer) and the figures `hardness.png`,
  `curves.png`, `strain.png` and `history.png`;
* exports for other codes: `<stem>_tet_equations.inp` and `<stem>_equations.inp`
  (periodic constraints of the conforming and voxel meshes), `<stem>_damask.vti` and
  `<stem>_damask_material.yaml`, `<stem>_seeds.txt`, `<stem>_weights.txt`,
  `<stem>_ori.txt` and `<stem>_neper.sh`;
* for RVE studies, `rve_report.json`, `rve_report.md`, `rve_report.tex` and figures
  (`rve_<property>.png`, when matplotlib is installed); with the stiffness, the mean
  stiffness C̄ and its compliance S = C̄⁻¹ per condition and size with the standard
  deviation of each component (`tensors` in `rve_report.json`, its tables, the RVE size
  of each component, `rve_stiffness.csv` and `rve_stiffness_<bc>.png`; Voigt order 11 22
  33 23 31 12, engineering shears).

An optimization directory contains `<name>.json` (the design as a `LatticeField`),
`<name>.vti` (thresholded design), `<name>_density.vti`, `history.csv` (objective,
volume, change, projection sharpness, solver iterations per iteration) and
`optimization.json` (settings, homogenized tensors of the density and thresholded
designs, `tensor` and `binary_tensor`, their compliances `compliance` and
`binary_compliance` with their `notation` for the elastic designs, engineering
constants, convergence, versions, the number of hinges, and the
constants of the design voxelized `outputs.check_refinement` times finer from its level
set).

| Format | Content |
|---|---|
| `.vti` | VTK XML ImageData, cell data (one array per image, e.g. `phase`, `cell`), appended raw binary, zlib-compressed by default; read by ParaView/VTK |
| `.vtk` | legacy STRUCTURED_POINTS, CELL_DATA, big-endian binary (FFT codes) |
| `.mhd` + `.raw` | MetaImage (ITK, Fiji, 3D Slicer); `Offset` is the centre of the first voxel |
| `.xdmf` + `.raw` | XDMF 3 (ParaView, VisIt): an XML header and one little-endian raw file per array (images, also written slab by slab); tetrahedral meshes: ASCII XDMF with the grids `mesh`, `phase`, `grain`, `faces` (FEniCSx tags) |
| `.npz` / `.npy` | NumPy; `.npz` keeps spacing, origin, periodicity and metadata |
| `.tif` | multi-page TIFF (one page per z slice), ImageJ metadata when possible |
| `.json` | full microstructure description (regenerates the realization exactly) |
| `.csv` | `<stem>_spheres.csv`, `<stem>_segments.csv`, `<stem>_cells.csv` |
| `.geo` | Gmsh OpenCASCADE script of phase 1 (union clipped by the box) or phase 0 |
| `.inp` | Abaqus/CalculiX: `*NODE`, `*ELEMENT, TYPE=C3D8` (voxels) or `C3D4`/`C3D10` (tetrahedra)`, ELSET=PHASE_<p>`, `*ELSET CELL_<id>` (grains), `*ORIENTATION` per grain and `*SOLID SECTION` with the materials, `*NSET X0 ... Z1` |
| `_equations.inp` | `*EQUATION` of periodic displacements (or temperatures) between facing nodes, with reference nodes `REF_X`, `REF_Y`, `REF_Z` carrying the macroscopic strain, and the `CORNER` node set |
| `_damask.vti`, `_damask_material.yaml` | DAMASK grid (cell data `material`, origin and size) and material file: one material per grain (quaternion with P = −1 convention), elastic constants of the phases |
| `.msh` | Gmsh 2.2 ASCII: hexahedra or tetrahedra (types 4, 11) with physical volumes `phase_<p>` (tag p + 1), element faces on the box faces with physical surfaces `x0 ... z1` (tags 101–106) |
| `.vtu` | VTK XML UnstructuredGrid with cell data `phase` (and `cell`); tetrahedral meshes store their box and periodicity as field data, and fields as cell and point data |
| `.mesh` | Medit (MMG): tetrahedra (reference phase + 1), boundary triangles (references 1–6), box in a comment |
| `_periodic.csv` | facing nodes on opposite faces (1-based, as in `.inp`/`.msh`) |
| `.stl` | binary STL, watertight, outward normals |

## 8. Tests

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
| `test_implicit.py` | the implicit geometry of every model: for the 39 models and 19 other configurations of several phases, an implicit function of each phase whose sign is that of the voxel image on two grids and which partition the points; the Menger sponge on the grid of its cubes, two cells and their boundary layer against their planes, leaves and a transparent leaf against their spheres; the interpolation on the background mesh; the conforming mesh of a coated sphere (three phases cut one after the other: volumes, matching faces, the phases kept, finite elements against the FFT solver), cells of three phases meshed by their grains, coated hard spheres (slow) |
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
| `test_tutorial.py` | the tutorials (docs/tutorial_texture.md: the grains and the texture of the map, its stiffness, the lattices that a tension turns, a Taylor polycrystal, a polycrystal of space with the texture of the map; docs/tutorial.md, tutorial_rve.md, tutorial_rubber.md, tutorial_geometry.md, tutorial_iga.md, tutorial_contact.md, tutorial_fracture.md: layers that debond at their strength and with their fracture energy, a pore under a cyclic load, magnesium that twins one way, iron that cleaves, the probability of Beremin): the snippets of each run one after another at reduced sizes and give what the page says (measures against theory, the size of the RVE; the variance of the volume fraction of Boolean spheres against their integral range, the exponent 2/3 of Poisson fibres, the windows of one image, the report of a study; the laws fitted to the tests of a rubber and the stretch at which its strip tears; the fractions of a conforming mesh of three phases and the moduli of three solvers; the orders of an isogeometric analysis, the arrays of holes, the hole of Vigdergauz and the auxetic sheet); the sizes replaced are in the pages, the figures shown are in `docs/_static/tutorial/`, the commands parse |
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
pages of `docs/`, the README in parts (between its `<!-- section: ... -->` marks) and the
change log, and writes at each build the pages of the API reference in `docs/api/`
(ignored by git): a package lists the names of its `__all__`, linked to the module that
defines them, and its modules; a module documents what it defines, its undocumented
public functions and methods with their signature. The pages stand in four groups: to
start (installation, tutorials, recipes, questions), the guides (features, physics,
options, gallery, laser, performance), the reference (theory, Z-mat, API, glossary,
stability) and the project. `docs/index.md` is the landing page (what morphoGen does as cards, a first
study that `tests/test_howto.py` runs, where to read what, then the introduction of the
README). `docs/_static/` holds the logo, the styles (`gallery.css`, `site.css`) and
`site.js`, which adds to the pages, without any library, a button that copies each code
block, a filter of the cards of the gallery and the enlarged screenshots.

The images of the pages are written by scripts, never by hand:
`docs/gallery/make_gallery.py` (the cards, images and result figures of the templates,
and `docs/gallery.md` itself), `docs/tutorials/make_figures.py` (it runs the snippets
of the tutorials as written and keeps the figures they save, or that a study they run
writes) and
`docs/tutorials/make_screenshots.py` (it starts `morphogen gui` and drives the study of
the tour in the page with Playwright: `docs/tutorials/screenshots.js`).

## 9. Performance notes

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
