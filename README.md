# morphoGen
<!-- section: introduction -->

Morphological modelling and generation of virtual microstructures — stochastic or
deterministic — for the study of representative volume elements (RVE).

morphoGen does ten things:

* It generates random media from models of mathematical morphology and stochastic
  geometry, and architected materials (TPMS, strut lattices, scale-rich networks),
  periodic or not.
* It grows microstructures with cellular automata and lattice processes: annealing and
  Turing patterns, reaction–diffusion, nucleation and growth of grains (Avrami
  kinetics), grain coarsening, fractal aggregates.
* It designs architected materials: compatible assemblies of building blocks
  (combinatorial metamaterials), strut lattices with defects and their rigidity, and
  periodic cells optimized for their homogenized properties (topology optimization).
* It nests models — particles clustered in aggregates (Cox models of nanocomposites),
  coatings and hollow members, hierarchical lattices of frames of tubes, fractal
  sponges — and draws on biology: nacre, Bouligand plywoods, honeycombs and wood,
  Haversian and trabecular bone, open-cell foams, vascular trees grown by space
  colonization.
* It gives their morphological characteristics, both theoretical and measured.
* It exports them as vector, implicit or voxel geometry, as conforming tetrahedral
  meshes or voxel meshes for finite elements, and as volumes for FFT solvers. The
  conforming meshes of two phases need no voxels nor background grid (by default,
  `mesh: {method: auto}`): tessellations from their exact cells, faceted morphologies
  from their exact convex pieces, the others from their exact level set, refined where
  the interface needs it, its nodes on the exact surface.
* It computes their apparent properties — elastic, thermal, electrical, dielectric,
  magnetic and diffusive, with their weak and strong couplings (thermal expansion,
  swelling, poroelasticity, piezoelectricity...), harmonic regimes, Bloch waves, the
  flow through the pores and the sound that a porous layer absorbs: finite elements on
  conforming meshes that follow the interfaces or the grains of polycrystals (kinematic,
  mixed, periodic and static boundary conditions), an FFT solver on voxel images, on
  the CPU or an NVIDIA GPU, or extended finite elements on a grid that the interfaces
  cut; phases are isotropic or anisotropic crystals turned by the orientations of the
  grains.
  Loadings mix macroscopic strains and stresses (the periodic element of Z-set), with
  thermal expansion; the results are post-processed into engineering constants and
  normalized values, kept in a database of every configuration tested, and shown on
  normalized Ashby charts ([docs/physics.md](docs/physics.md) is the guide to these
  problems and their solvers).
* It follows nonlinear behaviours in time, up to fracture: phases that yield (von Mises
  or Hill elasto-viscoplasticity with isotropic and kinematic hardening), polycrystals
  that slip on the systems of their lattices (crystal plasticity of Méric and
  Cailletaud), viscoelastic and hyperelastic phases (rubbers, soft tissues, foams),
  ductile and quasi-brittle damage, along loading paths (tension at a strain rate,
  creep, relaxation, cycles, thermal histories), in small or finite strains, with the
  macroscopic strain components as degrees of freedom (the periodic element): by finite
  elements on conforming meshes, by the FFT solver on voxel images, or by extended
  finite elements. Cracks grow in meshes (stress intensity factors, cohesive zones) or
  nucleate and merge in voxel images (the phase field, in small or finite strains).
  Surfaces touch and rub (contact, self-contact and Coulomb's friction): the walls of the
  pores of a cell that densifies, a sample between platens or under an indenter. The
  interfaces of the phases of a cell debond along a path (cohesive zones on its torus).
  The stresses of a cycle are judged by criteria of multiaxial fatigue (Crossland, Sines,
  Dang Van, Findley, Matake): where a cell would start to crack, and its factor of safety.
  Behaviours written in MFront can take the place of its own (through MGIS).
* It runs RVE-size studies with the statistical method of Kanit et al. (2003) and
  Dirrenberger et al. (2014): generalized scaling law D²(V) = K V^-γ, RVE sizes with
  confidence intervals, discrepancy between boundary conditions, isotropy. Realizations
  run in parallel on a workstation and as job arrays on clusters.
* It architectures metal sheets by localized laser treatment (`morphogen.laser`): the
  thermal cycles of laser tracks on one face or both, the martensite, tempering,
  carburizing, nitriding or alloying they leave, the tensile test of the specimen by
  finite elements with large rotations, and the optimization of the laser path for a
  strength–ductility objective ([docs/laser.md](docs/laser.md)).

It comes with figures, animations and a graphical interface in the web browser. The
[tutorials](docs/tutorials.md) walk through whole studies — the first from the
installation to a behaviour learned from the responses of a microstructure —, the
[recipes](docs/howto.md) answer "how do I..." in a snippet each, and the
[gallery of examples](docs/gallery.md) shows every template with its image, its
configuration and the commands that reproduce it.

> **Status: 0.6, version 1.0 in preparation** (phase 4: the documentation site, the
> stable public interface, the citation and the release steps), after the work of
> November 2026 (multiphysics, fracture and contact, explicit dynamics with OpenRadioss
> and the stand-alone application, crystallographic textures and orientation maps,
> scale-rich networks and exact meshes, laser-architectured sheets), phase 3b (the FFT
> solvers on NVIDIA GPUs; hard-core and curved fibres, rotated and graded lattices;
> AMITEX_FFTP workflows; MFront behaviours through MGIS; 3D views), phases 2a
> (homogenization and statistical analysis), 3a (cellular automata, combinatorial
> design, topology optimization, visualization), 2b (polycrystals in finite elements,
> anisotropic crystals, parallel and cluster runs, exports) and 2c (whole periodic
> grains, the periodic element with mixed macroscopic loadings, thermo-elasticity,
> nonlinear behaviours along loading paths, post-processing, database of results and
> Ashby charts, nested and bio-inspired models). The specification
> ([docs/specification.md](docs/specification.md)) was validated on 2026-10-21 (1.1);
> its later versions describe the later phases. morphoGen is free software under the GNU
> GPL v3 or later; its repository is public since 18 November 2026.

<!-- section: features -->
## Features

| Models | Type |
|---|---|
| `boolean_spheres` | Boolean model of spheres, constant or random radii (porous and granular media) |
| `boolean_cylinders` | Boolean model of finite fibres (cylinders or spherocylinders) of any law of the axes, with its exact covariance along any direction (Peyrega et al. 2009) and integral range |
| `poisson_fibers` | infinite fibres on Poisson lines (Dirrenberger et al. 2014), isotropic or of any law of directions, with their exact covariance along any direction |
| `curved_fibers` | Curved fibres: worm-like chains whose tangents decorrelate along the contour (nonwovens, entangled networks), circular arcs of distributed radii of curvature (as in PuMA) or helices (crimp); a Boolean model or fibres without overlap; hollow (lumens) |
| `hardcore_spheres` | non-overlapping spheres by random sequential addition, or dense random packings grown by the Lubachevsky–Stillinger algorithm (up to about 0.64) |
| `penetrable_spheres` | penetrable spheres with impenetrable cores: Torquato's cherry-pit model, between the Boolean model and hard spheres |
| `hardcore_fibers` | non-overlapping fibres (spherocylinders) by random sequential addition, isotropic, aligned or planar, with a minimal gap |
| `random_walk_fibers` | bending fibres without overlap at high volume fractions (Altendorf & Jeulin 2011): random walks whose steps follow a von Mises–Fisher law between the main direction of their fibre (isotropic, aligned, planar, Schladitz's beta law, or mixtures of laws, as the orthotropic glass fibres of the 2014 paper) and the previous step, packed by repulsions against springs on the segments and the joints after an initial placement of least overlap (Altendorf, Jeulin & Willot 2014), then drawn into contact by the contact forces of Keilmann, Redenbach & Willot (2026) |
| `cross_linked_fibers` | entangled cross-linked fibres (Mezeix et al. 2009; Chatti et al. 2018, 2020): straight fibres, infinite on Poisson lines or finite, isotropic or in the Gaussian girdle of moulded fibres, bonded by cross-links (phase 2) at a fraction of their contacts, with the exact mean spacing of the contacts and cross-links |
| `delaunay_network` | fibre networks on the edges of the periodic Delaunay triangulation of Poisson or hard-core points (Ban et al. 2016), in space or in a plane, cross-linked at every node (15.5 fibres per node in space, 6 in a plane), with the stiffness of their frame of beams |
| `voronoi`, `laguerre` | Poisson–Voronoi and Laguerre tessellations (Poisson germs, germs of sphere packings, or stealthy hyperuniform germs: disordered but regular cells): polycrystals with random orientations, multiphase mosaics, grain-boundary phases |
| `ebsd` | measured polycrystals: windows of an orientation map (EBSD: `.ang` and `.ctf` files), its grains found from the disorientations of its points, each with its mean orientation |
| `gaussian_excursion` | excursions of Gaussian random fields (bicontinuous porous media) |
| `tpms` | gyroid, Schwarz P, Schwarz D and Neovius, sheet or network |
| `strut_lattice` | cubic, BCC, FCC, octet, Kelvin, diamond and pyrochlore lattices, with solid or hollow struts; rotated and functionally graded lattices (non-periodic samples) |
| `cellular_automaton` | two-phase structures grown by a cellular automaton from noise: annealing (bicontinuous), majority, conserved volume fraction, any `B/S` rule; on squares or cubes, triangles, hexagons, Penrose rhombi, Ammann–Beenker tiles, prisms, BCC or FCC cells, golden rhombohedra (quasi-periodic tilings in periodic boxes: their periodic approximants) |
| `activator_inhibitor` | Turing patterns of Young's activator–inhibitor automaton (spots, stripes, labyrinths) |
| `reaction_diffusion` | Gray–Scott reaction–diffusion patterns (spots, worms, labyrinths, holes), on the grid or the cells of any tiling |
| `nucleation_growth` | nucleation and growth of grains (JMAK kinetics): Voronoi or Johnson–Mehl polycrystals, partially crystallized structures, faceted grains, Potts coarsening |
| `aggregation` | diffusion-limited, Eden and ballistic aggregates: fractal clusters and deposits |
| `tile_assembly` | compatible assemblies of building blocks: polarized blocks (combinatorial metamaterials), pipe networks |
| `random_lattice` | strut lattices with missing struts and displaced nodes, with their rigidity (Maxwell–Calladine) |
| `random_laminate` | layers of random thicknesses with holes through the thin layers (nanolayered films, Bironeau et al. 2016) |
| `dead_leaves` | dead leaves of spheres or spheroids: mosaics of non-convex grains, families of leaves of their own sizes, finite coverage over a background, with their exact covariances (Jeulin 1997) |
| `transparent_dead_leaves` | excursions of the random function of transparent leaves (Jeulin 1997), with its exact covariance |
| `reconstruction` | stochastic reconstruction (Yeong & Torquato 1998): a lattice annealed to the two-point probability and lineal-path function of a Debye medium, a model or an image |
| `combination` | union, intersection, difference and complement of models; inclusions in a matrix, e.g. a polycrystal; nested models (each fills the phase 1 of the previous one) |
| `cox_spheres` | spheres clustered in the phase 1 of another model: Cox models of nanocomposites (particles in aggregates), hard-core or not |
| `coating` | a coating (interphase) around the phase 1 of any model, or a shell inside it: hollow spheres, fibres and struts, core–shell particles |
| `hierarchical_lattice` | hierarchical lattices of (hollow) tubes: frames of generation G of regular octahedra (Rayneau-Kirkhope, Mao & Farr 2012) |
| `fractal_sponge` | Menger sponge, Vicsek fractal, Cantor dust and any iterated function system on a cubic grid |
| `nacre` | brick-and-mortar stacks of mineral tablets, polygonal or staggered bricks |
| `bouligand` | twisted plywoods of fibres (arthropod cuticles, fish scales), periodic fibre families |
| `honeycomb` | prismatic cells: hexagonal, square or random honeycombs; softwood with radial files, growth rings and rays |
| `osteons` | Haversian bone: osteons with canals and cement lines, remodelled one after another (dead leaves) |
| `open_foam` | open-cell foams and trabecular bone: struts on the edges of relaxed, elongated Voronoi cells, at a given relative density, with their framework and the stiffness of its frame of beams |
| `level_set_foam` | open- or closed-cell foams as level sets of the distance fields of packed aligned ellipsoids: Plateau-border struts of controlled curvature thickening towards the nodes, at a given thickness or solid fraction (Degdeg & Dirrenberger); `calibrate_seeds` fits the seeds to measured section radii |
| `branching` | vascular trees, roots and corals grown by space colonization, radii by Murray's law |
| `scale_rich_network` | scale-rich networks (Both et al. 2026): ligaments of thicknesses `lambda_0 t^-alpha` grown one after another from points of the voids until they meet, extruded prisms; their thicknesses, lengths and degrees span orders of magnitude (power-law tails `1 + 1/alpha`, 3 and 3); the relative density in closed form (mean field) or solved on each realization, the jamming threshold; uniform, aligned or polarized ligaments, gradients of nucleation, a frame for periodic cells; in 3D (`dimension: 3`), plates that split convex voids into closed cells (areas and degrees of tails 5/2, the density and the jamming threshold in closed form) or beams grown until they touch (open cells, a mean-field density) |

Point processes, tessellations, Gaussian fields, dead leaves, growths and lattices take the
same field `anisotropy: {stretch: [1, 1, 2], euler: [0, 0, 0]}` (or `[1, 1, 2]`): the
realization of the isotropic model mapped by stretches along turned principal axes,
preserving volumes (aligned ellipsoids, elongated cells and grains, anisotropic
covariances); periodic boxes take principal axes along their axes, the Gaussian field any.

* **Representations.**
  * Each realization is first an explicit description: centres and radii, fibre axes,
    germs and weights, Fourier modes, surface equations. It is saved to JSON and
    regenerated exactly.
  * It is voxelized at any resolution, with optional partial-volume voxels; volumes
    larger than memory are written slab by slab (`io.voxelize_to_file`).
  * Every phase of every model has an implicit function (`phase_level_set`: negative
    in the phase, zero on its boundary): signed distances for unions of grains, minima
    and maxima for combinations, coatings and inclusions, distances to the spheres of
    the leaves, to the faces of the cells of tessellations and tilings, to the cubes of
    a sponge. The exports of the geometry — conforming meshes, surfaces, sections — are
    cut from them, not from voxels ([docs/theory.md](docs/theory.md) §75).
* **Theory.** Each model knows its closed-form characteristics when they exist: volume
  fraction, specific surface, covariance, integral range, variance exponent, cell
  statistics. See [docs/theory.md](docs/theory.md).
* **Measurements.** Volume fraction, specific surface (Cauchy–Crofton, 13 directions),
  covariance, connected components and percolation (labels, or a forest fire and its chemical
  distance), percolation thresholds of families
  of microstructures and their finite-size scaling to the infinite medium, in 2D and 3D, the
  anisotropy of percolation (along any direction across a core of the box, the thresholds
  of a family along directions, their ratio and threshold tensor), the rigidity threshold
  where the elastic moduli of the voxels vanish, and the critical exponents
  of the conductivity and of the elastic moduli at the threshold (t and f, by the
  finite-size scaling of the FFT solutions, or of the voxels assembled and factorized near
  the rigidity threshold, with their errors), cell volumes and neighbours,
  fractal dimension (box counting), chord lengths, granulometry by openings, local
  thickness and pore sizes, Euler characteristic and Minkowski densities, geometric
  tortuosity; Torquato's descriptors: lineal-path, two-point cluster and pore-size
  functions, radial distribution function and structure factor (hyperuniformity) of
  point patterns; mathematical morphology: Choquet capacities of balls, segments, pairs
  and cubes, dilations, erosions, openings, closings and contact distributions, the
  morphological covariance of grey-level images, the four Minkowski functionals (with
  the integral of mean curvature) and their curves for excursion sets, Betti numbers;
  the mean intercept length and its fabric tensor (degree of anisotropy), the first
  minima of the covariance (hole effect); stereology: section sizes and Saltykov's
  unfolding of the size distribution of spheres or oriented ellipsoids; the particles of
  a phase measured one by one in the volume (`analysis.particle_sizes`: touching
  particles split by a watershed of their distance map, their sizes and number per unit
  volume, with the frame correction of images that are not periodic).
* **Cellular automata and growth** (`morphogen.automata`).
  * Von Neumann, Moore, edge and Euclidean neighbourhoods in 2D and 3D; Life-like
    `B/S` rules, majority, Vichniac's annealing rule, majority at constant volume
    fraction, probabilistic rules, activator–inhibitor rules; synchronous,
    asynchronous and sequential updates.
  * Gray–Scott reaction–diffusion, on the grid or on any tiling (finite-volume
    Laplacians).
  * Automata on other lattices than the square grid: triangles, hexagons, Penrose
    rhombi (de Bruijn's pentagrid; quasi-periodic), Ammann–Beenker squares and rhombi
    (octagonal, quasi-periodic), the prisms stacking them in 3D, truncated octahedra
    (BCC), rhombic dodecahedra (FCC) and the golden rhombohedra of the icosahedral
    quasi-periodic tiling, rasterized to images cell by cell (`lattice: hexagonal`);
    the periodic approximants of the quasi-periodic tilings for periodic boxes (the same
    tiles in a rectangle or a cube: `approximant: 4`; the cubic approximants 1/1, 2/1,
    3/2... of the icosahedral tiling).
  * Nucleation and growth by an event-driven automaton with exact (or faceted,
    oriented) growth fronts, checked against the Avrami law; normal grain growth by
    the Monte Carlo Potts model; diffusion-limited, Eden and ballistic aggregation.
  * Lattice structures are exact on their lattice and smooth elsewhere (a level set
    for conforming meshes); non-periodic domains are cut from larger periodic
    simulations, so that their structures do not wrap around. Their formation can be
    watched step by step.
* **Combinatorial design** (`morphogen.design`).
  * Tile sets with labelled faces; compatible assemblies by wave function collapse
    with backtracking; exact counts and exact uniform sampling by transfer matrices;
    polarized blocks after Coulais et al. (2016), with their exact counts.
  * Periodic pin-jointed frameworks: random dilution and perturbation, rigidity
    (mechanisms, states of self-stress, floppy macroscopic strains).
  * Simulated annealing, genetic algorithm, and the inverse design of automaton rules
    that reproduce the morphology of an image (`morphogen fit-rule`).
* **Topology optimization** (`morphogen.optimization`). Periodic cells of maximal bulk
  or shear modulus or conductivity, of negative Poisson's ratio, or of a target
  stiffness tensor, by inverse homogenization with the FFT solver and exact
  sensitivities: SIMP, density filter, Heaviside projection, cubic or orthotropic
  symmetry; optimality criteria, method of moving asymptotes, BESO and the hybrid
  cellular automaton. Designs are microstructures like any other (meshing, finite
  element validation, exports).
* **Isogeometric analysis and shape optimization** (`morphogen.iga`). NURBS patches
  and their assemblies — exact circles and ellipses —, refined without changing the
  geometry; linear elasticity (plane strain, plane stress, solids) and conduction with
  the functions of the geometry; the effective tensors of periodic cells, boxes or
  cells of any lattice (hexagonal arrays, patterns cut along no straight line); shape
  optimization whose variables are control points, linked by the symmetries of the
  pattern, with analytical sensitivities (the energy-momentum tensor) and SLSQP or the
  moving asymptotes: the hole of largest bulk modulus of a cell reaches the bound of
  Hashin and Shtrikman, and a sheet with alternating holes is brought to a Poisson's
  ratio of -0.8 at a given stiffness ([tutorial](docs/tutorial_iga.md)).
* **Laser-architectured sheets** (`morphogen.laser`, [docs/laser.md](docs/laser.md)).
  After Wang's thesis (PIMM 2022) and the project SCOLASTIC: moving Gaussian or top-hat
  beams on a plate, on one face or on both at once (Rosenthal's limits, images through
  the thickness, losses; transient finite volumes of a blank); austenite, martensite and
  tempering of C–Mn steels (lever rule, Andrews, Koistinen–Marburger, Maynier,
  Hollomon–Jaffe), the peak-temperature hardening of AISI 430, melt-pool carburizing,
  nitriding and alloying; flow laws fitted to `R_p`, `R_m` and `A_g` by Considère; laser
  paths (straight, corrugated, Fourier, crossed, honeycomb; G-code); a layered membrane
  in plane stress with large rotations for the tensile test of dog-bone specimens; the
  optimization of the path (grid, Nelder–Mead, differential evolution, genetic, Bayesian)
  under constraints of strength and process (`morphogen laser`). Corrugated martensite
  in an annealed 0.23 C steel trades yield strength for uniform elongation as measured;
  a nonlocal fracture of the martensite gives the failure strains of the six samples of
  Wang et al. (2023) within 0.006; the strength contrast of the reinforcement, from the
  matrix to the martensite, decides whether a corrugation pays (the maps of 1450 designs
  within the limits of the process); morphoGen's 2D morphologies treated at the scale of
  the spot (1 mm²) on whole sheets, the sheets they make, the steering of a ductile crack
  by soft corridors, the materials and the stiffness; the geometric work hardening of
  corrugated tracks over a whole sheet (the strength contrast, not the stiffness, unbends
  them) and where it works (along the axis of the tracks in uniaxial tension, the tracks
  lying on lines that the deformation first shortens; every architecture lowers
  the forming limit on the stretching side), and the architecture this rule selects: two
  families of straight tracks crossed about the load, a scissor network that locks at
  the necking of the matrix (an angle in closed form), raising the uniform elongation and
  the forming limit near uniaxial tension. Cellular design (the loop of SCOLASTIC): phases on a
  grid of cells, sensitivities by replays of the deformation history, BESO and
  neighbour-exchange automata tested by finite elements, HCA and MMA, the tracks that
  draw a design. Structures beyond the tensile test: a compact tension specimen (hard
  bands that shield a crack, softened bands around it in hard sheets) and an L-shaped
  plate, the growth of its crack (bridged by hard tracks); the plastic anisotropy of
  treated sheets on the periodic cell of the sheet (Lankford, Hill 1948, Yld2000-2d),
  the anisotropy set by an architecture, a rolling texture compensated by tracks along
  the rolling direction; 22MnB5 and AA6082-T6 (softened by the laser). The periodic
  cell of a treated sheet in its volume (`case_volume`: voxels of graded elastoplastic
  phases with free faces) for the solvers of three-dimensional fields, within 1 to 3 %
  of the membrane in tension.
* **NTFA.** Reduced models of nonlinear voxel microstructures (nonuniform transformation
  field analysis, Michel & Suquet): modes from full-field snapshots, then loading paths
  in milliseconds, within a few percent of the full fields; also from configurations
  (`homogenization: ntfa`).
* **Bloch waves.** Band structures of the elastic waves of periodic cells (meshes or
  voxel images) along the edges of the Brillouin zone, their band gaps and figures
  (`homogenization.band_structure`); the scalar waves of sonic and photonic crystals
  (`waves="acoustic"`, `"shear"`, `"tm"`, `"te"`: the pressure of fluids, antiplane
  shear, the two polarizations of light in planar crystals), rigid scatterers, and the
  speeds of the homogenized medium at long wavelengths (Wood's law); light in
  three-dimensional photonic crystals (`waves="electromagnetic"`: the vector waves of
  Maxwell's equations by plane waves on the voxels, dielectric, anisotropic or magnetic
  phases, interfaces averaged within the cells; the band gap of the diamond lattice).
* **Post-bifurcated patterns.** The patterns that a periodic architectured material (a
  frame of beams, a tetrahedral mesh or a voxel image) may adopt after buckling, from
  the symmetry of its cell (group theory and the equivariant
  branching lemma, after Azulay & Combescure), with the first bifurcation of a loading by
  linear buckling, and the cell they need from a Bloch-wave stability analysis (cells of
  several primitive cells split by their translations and the little groups of their wave
  vectors, the patterns that double the period); periodic
  frames of Euler-Bernoulli beams in the plane (honeycomb, triangular, square, kagome)
  and in space (`SpaceFrame`: cubic, body-centred cubic, octet, Kelvin, diamond and
  pyrochlore lattices of struts, their apparent stiffness and relative density),
  patterns drawn and animated
  in 3D (`homogenization.post_bifurcation`, `morphogen.symmetry`); along nonlinear
  loading paths in finite strains with nonlinear behaviours (Hencky elasticity, von Mises
  elasto-(visco)plasticity, crystals, MFront laws), the tangent of each step block by
  block, the first bifurcation, its patterns and the bifurcated branch
  (`homogenization.path_bifurcation`).
* **Fibre networks as frames of beams.** The effective stiffness of random periodic
  frameworks (Delaunay fibre networks, Voronoi open foams, random lattices) as frames of
  Euler-Bernoulli beams of circular sections with rigid joints, or of exact Timoshenko
  beams (`shear_coefficient`: a coefficient per beam, or Cowper's), by a sparse periodic
  problem: the Voigt stiffness, its split into stretching, bending, torsion and shear,
  the affine estimate and the non-affinity of the deformation (slender Delaunay
  networks at 0.82 of the affine estimate, open foams at E = 1.27 E_s phi^2); the
  softening of networks of non-identical beams of Ban et al. (2016), exact to second
  order in the coefficient of variation and against replicas; with the axial stiffness
  lowered, Timoshenko beams whose shear stiffness follows the area of their general
  sections reproduce their Voronoi case (s = 0.46) and come within a factor 1.4 of
  their Delaunay case. Against the FFT solver on the voxels of the same realizations,
  the frames are 2 to 30 % softer than the continuum, the struts merging in its joints;
  Timoshenko beams rigid over a third of their radius at the joints (`rigid_joints`)
  follow it within 4 % for open foams from 6 to 15 % (7 % at 2 %) and a Delaunay network
  at 12 % (`homogenization.networks`; in the summaries of the network models with
  `frame_stiffness: true`; for any realization of them, with every option, by
  `morphogen frame-stiffness` and the tool Stiffness of the frame of beams of the
  interface). Both run Ban's replicas of the stiffness of the beams (normal, lognormal or
  bimodal, several coefficients of variation) against the exact `1 - s CV^2` and
  `v CV^2`, with their table and figure; the axial force and the energy of each beam
  under a uniaxial stress (`beam_forces`) draw the force chains and the beams that bend
  (`viz.figures.frame_figure`, in the tool).
* **FEniCSx.** `solver: fenicsx` assembles and solves the finite element problems of
  tetrahedral meshes with FEniCSx (dolfinx, dolfinx_mpc, PETSc; Linux or WSL), with the
  same results as morphoGen's own solver.
* **Couplings.** Transient diffusion in microstructures (moisture, oxygen, solvents),
  thermal oxidation of polymers (closed-loop scheme, diffusion-limited profiles of a
  plate, and in the volume of a microstructure around its fibres or fillers),
  properties that evolve with an ageing tracer mapped to graded phases, hydrolysis of
  cellulose, hygromorph multilayers bending as moisture diffuses (orthotropic layers,
  their properties homogenized from microstructures); diffusion from
  configurations (`diffusion` section); with FEniCSx, the diffusion and free swelling of
  meshed samples (hygromorph strips bending in time) and the stresses of oxidized ones.
* **Physics.** One homogenization module for the mechanical, thermal, electrical,
  electromagnetic, chemical, diffusive, optical and acoustic problems of heterogeneous
  materials, static and dynamic, with their couplings; [docs/physics.md](docs/physics.md)
  is the guide to them for researchers (which problem, which solver, which name).
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
  * Behaviours in closed form for every solver: viscoelasticity (Prony series, WLF),
    hyperelasticity (neo-Hookean, the polynomial of Rivlin with Mooney–Rivlin and Yeoh,
    Arruda–Boyce and the eight chains, Gent, van der Waals, Fung, Ogden, the fibres of
    Holzapfel–Gasser–Ogden, the foams of Blatz–Ko and Ogden–Hill; the softening of
    Mullins, finite viscoelasticity; `fit_hyperelasticity` fits them to the uniaxial,
    equibiaxial, planar and shear tests of a rubber) and isotropic damage
    (crack band, viscous regularization, the unilateral effect of cracks that close),
    Mazars' damage of concrete, and the damage coupled to plasticity (Lemaitre's ductile
    damage, the creep damage of Kachanov and Rabotnov: `damage: {law: lemaitre, S: ...}`
    beside `plasticity`), beside plasticity, crystal plasticity and porous plasticity.
  * Fracture: the phase field of brittle fracture on voxel images
    (`homogenize(..., fracture=...)`: AT1 and AT2, a toughness per phase; in finite
    strains with `nonlinear: {kinematics: finite}`, for hyperelastic phases), beside the
    cracks, cohesive zones and ductile damage of `morphogen.fracture`.
  * Extended finite elements (`method: xfem`, `LevelSetSample`): a regular grid cut by
    the interfaces of the morphology, without a mesher and without the stairs of the
    voxels; elasticity, transport and loading paths.
  * Flow and sound in porous media: the permeability of Darcy's law by a Stokes flow
    through the pores (`hydraulic_permeability`, a lattice Boltzmann method on the
    voxels), the parameters of the equivalent fluid of a rigid frame (`acoustic`:
    tortuosity, viscous and thermal characteristic lengths, thermal permeability) and the
    sound absorption of a layer (`homogenization.absorption`, the model of Johnson,
    Champoux, Allard and Lafarge).
* **Explicit dynamics with OpenRadioss** (`morphogen.dynamics`). The crushing, tension,
  shear and impact of a morphology at the rates of crashes, shocks and drops, simulated
  by [OpenRadioss](https://github.com/OpenRadioss/OpenRadioss), the open-source explicit
  finite element solver (AGPL-3.0; run as a separate program, installed from its
  releases by `morphogen radioss install`): its voxels as stabilized 8-node bricks or a
  conforming mesh of its level set as tetrahedra, its phases elastic, von Mises
  elastoplastic (the linear, Voce and Norton hardenings of morphoGen tabulated in LAW36)
  or Johnson–Cook (LAW2), with failure by element deletion, the self contact of its
  folding struts and closing cells (TYPE24), its faces held and driven after a smooth
  start, or the whole thrown against a rigid wall; its nominal stress–strain curves at
  both faces (their gap telling a quasi-static test from a dynamic one), energies, peak
  and plateau stresses, densification strain, absorbed energy per volume and mass, and
  its fields over time (`morphogen radioss export|run|read`, the tab Explicit
  dynamics). The inputs are the Radioss Block Format of the reference guide, written
  with their manifest so that a run made elsewhere (a cluster) is read back.
* **Homogenization.**
  * Conforming tetrahedral meshes of every model, made with MMG from the implicit
    functions of its phases: the interfaces are followed within a Hausdorff distance,
    and opposite faces match for periodic media. Porous media are meshed without their
    pores; structures of several phases (coated inclusions, inclusions in a polycrystal,
    osteons, leaves of several phases) are cut phase by phase, conforming to all their
    interfaces.
  * Polycrystals (Voronoi and Laguerre tessellations) are meshed grain by grain from
    their exact cells, after removing features smaller than a chosen length: conforming
    grain boundaries, one element set and orientation per grain, matching periodic
    faces.
  * Crystals: cubic, hexagonal or any stiffness and conductivity, turned by the
    orientation of each grain, in both solvers; the bulk modulus of cubic polycrystals
    comes out exact, and the shear modulus within the Hashin–Shtrikman bounds.
  * Textures and orientation maps (`morphogen.texture`, `morphogen.io.ebsd`): the
    symmetries of the eleven Laue classes, disorientations (against Mackenzie's
    distribution for cubic crystals), mean orientations, pole figures in multiples of a
    random texture, inverse pole figures and their colours; EBSD maps read and written
    (`.ang`, `.ctf`, with the axes of each acquisition system), their grains, and the
    polycrystal of a map for every solver (the model `ebsd`); the texture of a map given
    to the cells of a tessellation of space (`texture: {type: measured, file: map.ctf}`);
    `morphogen texture` for the numbers, the tables and the figures of a map. Along a
    loading path in finite strains, the lattices that slip turns (`PathResult.texture`,
    the `orientation` of a material point), checked against Schmid and Boas in single
    slip and against the fibres of Taylor's polycrystal.
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
  * Periodic polycrystals meshed from their whole grains (periodized at the design):
    node-to-node periodic links, no cell cut by the faces of the box.
  * The periodic element: loadings that prescribe any mix of macroscopic strain and
    stress components (uniaxial stress, plane strain...), solved exactly by
    condensation of the macroscopic unknowns, for every boundary condition and both
    solvers; thermal expansion of the phases, thermal stresses and the apparent thermal
    expansion tensor (Levin's relation and bounds).
  * Nonlinear behaviours along loading paths: von Mises elasto-viscoplasticity (Norton
    flow, linear and Voce isotropic hardening, Prager and Armstrong–Frederick
    kinematic hardening) or rate-independent plasticity, integrated implicitly with
    the consistent tangent, or with Hill's anisotropic criterion (Z-set's coefficients
    `h11` … `h31` by component; the closest point projection by Newton's method); paths that prescribe each component as a strain or a
    stress in time, with a change of temperature; Newton iterations whose unknowns
    include the macroscopic strain components (periodic and kinematic conditions on
    meshes, periodic on voxels), with line search and step division; macroscopic curves,
    phase averages and fields of plastic strain; a material point simulator.
  * Crystal plasticity of Méric and Cailletaud for polycrystals: octahedral, cube and
    body-centred cubic slip families, Norton flow, isotropic hardening coupled by an
    interaction matrix (Franciosi's six interactions), kinematic hardening per system,
    slip systems turned by the orientation of each grain; robust implicit integration
    with its consistent tangent.
  * Finite strains for every nonlinear behaviour: total Lagrangian finite elements,
    Hencky elasticity and von Mises plasticity in the logarithmic strain space,
    multiplicative crystal plasticity with lattice rotations; nominal and true (Cauchy)
    stresses, logarithmic strains.
  * Behaviours written in MFront (TFEL), integrated by MGIS (optional, from conda-forge):
    standard behaviours in small or finite strains in place of morphoGen's own, by
    finite elements, on voxels (CPU or GPU) and at a material point, crystals turned by
    the orientation of each grain; `.mfront` sources compiled on first use and cached;
    validated against MFront twins of morphoGen's behaviours.
  * Behaviours learned from data (`morphogen.learning`, `morphogen learn`; NumPy and
    Numba only): random strain paths at a material point or on a microstructure, a
    perceptron, a learned convex potential (hyperelastic: a symmetric positive tangent),
    a recurrent network over the increments (plasticity) or a learned generalized
    standard material (a convex free energy and dissipation potential of learned
    internal variables, the state of a step the minimizer of the incremental potential:
    a nonnegative dissipation and a symmetric positive tangent by construction, trained
    through the minimizations by implicit differentiation), every model in finite
    strains of the logarithmic strain (by default, as the Hencky points: its stress
    conjugate to `½ ln C`, objective) or of the Green-Lagrange strain, trained with their
    gradients written out, then integrated in place of a behaviour (`learned` in a
    material) at a material point, by finite elements and on voxels; a law learned in
    the axes of a crystal (its paths turned by the rotations of the cube) turned grain
    by grain in polycrystals, the stress-free thermal strain of its material taken out;
    a tool of the interface that learns a behaviour in the background and keeps its law;
    every learned law exported as an MFront behaviour (its trained parameters and a C++
    port of its model, in small or finite strains, its state as state variables), for
    Abaqus, Cast3M, Code_Aster and AMITEX_FFTP, checked against morphoGen's own
    integration to round-off.
  * Post-processing: the compliance S = C⁻¹ beside every stiffness C, in tables, JSON
    and the text of the commands (morphoGen's Voigt order 11, 22, 33, 23, 31, 12 with
    engineering shears, stated with each, TFEL and Abaqus ordering their shears 12, 13,
    23), engineering constants, anisotropy indices (universal index, Zener
    ratio, distance to isotropy), macroscopic values of porous media
    (Σ = φ⟨σ⟩_solid), relative, normalized and specific properties.
  * Voigt, Reuss and Hashin–Shtrikman bounds, also for polycrystals of cubic crystals.
* **Results database and Ashby charts** (`morphogen.results`). Every run can be
  recorded in a database (one SQLite file): its configuration and the results of each
  realization, filtered by conditions on any key (`model.type == gyroid`,
  `solid_fraction < 0.3`). Normalized Ashby charts (relative property against relative
  density, scaling laws of cellular solids) from Python, `morphogen ashby` and the
  graphical interface.
* **RVE.**
  * ensemble method (realizations at several sizes) and window method (sub-volumes of
    one large sample, or of any scalar field; the stiffness and conductivity tensors of
    the windows of one periodic realization, their mean mutual energies by FFT; tilings
    shifted around a periodic sample, their variances pooled within the tilings);
  * per property and boundary condition: means and intervals, relative errors, fit of
    K and γ (free or imposed), RVE sizes and the number of objects (fibres, grains)
    they contain, bootstrap confidence intervals, discrepancy between conditions;
  * for every physical quantity: the scalar properties and tensors of each physics, the
    named results of the couplings and of the harmonic properties (`piezoelectric.e33`,
    `dynamic_stiffness.shear_loss_1`, `poroelastic.biot_coefficient`), and the quantities
    of nonlinear loading paths (a stress at a time, the strength, the work of fracture);
  * reports in Markdown, LaTeX and JSON, with figures;
  * fit of external tables, e.g. apparent moduli computed with another FE/FFT code.
* **Exports.**
  * Volumes: VTK (`.vti`, legacy `.vtk` for FFT codes), XDMF, MetaImage, NumPy, TIFF.
  * Conforming tetrahedral meshes and hexahedral voxel meshes, with phase, grain and
    face sets: Abaqus/CalculiX `.inp` (with orientations, sections and the materials of
    the phases: elastic, the nonlinear ones as MFront laws), Gmsh `.msh`,
    `.vtu`, Medit `.mesh`, FEniCSx XDMF with tags, and more through meshio; periodic
    node pairs and periodic constraint equations (`*EQUATION`).
  * Polycrystals for other codes: DAMASK grids and material files, Neper seeds,
    weights and orientations.
  * Runs of AMITEX_FFTP (the FFT solver of the CEA, MPI-parallel): material and zone
    fields, material file (isotropic phases, voids and crystals with the orientation of
    each grain; the nonlinear materials as MFront laws), loading files (the six unit
    strains of the stiffness, mixed loadings, loading paths, in small or finite
    strains), algorithm file and the MPI script; their results read back (averages of the cell, materials and zones,
    apparent stiffness and moduli) for RVE studies (`morphogen amitex export|read`,
    `outputs.amitex`).
  * MFront sources of the nonlinear materials (von Mises or Hill plasticity, porous
    plasticity, crystal plasticity and the learned laws, in small and finite strains,
    with their thermal expansion from 293.15 K,
    where the Abaqus inputs then start their nodes), for AMITEX_FFTP, Cast3M, Abaqus or
    Code_Aster (`morphogen mfront`), checked against morphoGen's own kernels; the user's
    own MFront laws go to AMITEX and Abaqus inputs too, in small or finite strains
    (every phase an MFront law then, Hencky's elasticity for the elastic ones).
  * Watertight STL surfaces for additive manufacturing.
  * Gmsh OpenCASCADE scripts.
  * Raw data and figures: tables as `.csv`, `.txt` and `.xlsx` (no dependency),
    figures as `.png`, `.tif`, `.svg`, `.pdf` and TikZ (pgfplots code for LaTeX);
    `morphogen.export(obj, path)` picks the writer from the object and the suffix.
* **Visualization.** Sections, periodic tilings, 3D surfaces, histories, Avrami
  kinetics and animated GIFs of the formation of structures (`morphogen view`).
  Videos (`.mp4` through ffmpeg, `.gif`), played once, in a loop or back and forth:
  sections sweeping through a volume and the three orthogonal sections (as in
  Avizo), turntables of phases and meshes, cross-sections sweeping through a mesh,
  fields and modes animated with a magnified displacement, matplotlib colormaps
  (3D views rendered off screen by VTK).
* **Interfaces.** Python API, command line, YAML/JSON/TOML configurations and a
  graphical interface, in the web browser (`morphogen gui`) or as a stand-alone
  application in a window of its own (`morphogen desktop`, `morphogen-desktop`; a
  bundle that needs no Python from `packaging/desktop`), with the dialogs of the system
  for its folders and files.
* **Parallel and cluster runs.** Realizations run in parallel processes (`-j`), in
  job arrays written and submitted for SLURM, PBS Pro, Torque, SGE, OAR and LSF
  (`morphogen submit`), or under MPI; each one leaves a record, so that runs can be
  resumed, checked (`morphogen status`) and merged, with results independent of the
  splitting.

<!-- section: installation -->
## Installation

morphoGen needs Python 3.12 or newer on Windows, macOS (Apple Silicon) or Linux. All
dependencies are installed from binary wheels; no compiler is needed.

**With uv** (recommended, it can also install Python itself):

```bash
git clone https://github.com/jdirr/morphoGen.git
cd morphoGen
uv venv --python 3.13
uv pip install -e ".[all]"            # [all] = scikit-image, tifffile, meshio, matplotlib, pymmg
```

Activate the environment with `.venv\Scripts\activate` (Windows) or
`source .venv/bin/activate` (macOS, Linux).

**With pip and venv:**

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS, Linux
python -m pip install -e ".[all]"
```

**With conda** (Miniforge or Anaconda):

```bash
conda create -n morphogen python=3.13
conda activate morphogen
python -m pip install -e ".[all]"
```

The core needs only NumPy, SciPy, Numba and PyYAML; the graphical interface needs
nothing else. The optional extras are `surface` (STL surfaces), `io` (TIFF and other
mesh formats), `plot` (figures and animations, with matplotlib) and `mesh` (the MMG
mesher, from the `pymmg` wheels: conforming meshes for the finite element solver). Where pymmg has no wheel (e.g. Linux on ARM), build `mmg3d` from
<https://github.com/MmgTools/mmg> and set `MORPHOGEN_MMG3D` to its path;
`morphogen info` shows the mesher found.

**GPU (optional).** On a machine with an NVIDIA GPU, `python -m pip install -e ".[gpu]"`
adds CuPy and the CUDA libraries it loads (the runtime, NVRTC and cuFFT, about 270 MB;
neither the CUDA toolkit nor a compiler is needed, only a driver of CUDA 13). With a
driver of CUDA 12, install
`python -m pip install cupy-cuda12x "cuda-toolkit[cudart,nvrtc,cufft]==12.*"` instead.
The FFT solvers then run on the GPU with `device: gpu` (or `auto`: the GPU when there is
one); `morphogen info` shows the GPU found, or why none is usable. These libraries are
proprietary (NVIDIA's licence, free of charge), and morphoGen never requires them.

**Stand-alone application (optional).** `python -m pip install -e ".[desktop]"` adds
pywebview, which shows the interface in a window of the web engine of the system
(WebView2 on Windows, WebKit on macOS; on Linux, Qt WebEngine from its wheels, with the
system's EGL and X libraries, e.g. `apt install libegl1 libxkbcommon-x11-0
libxcb-cursor0`); `morphogen desktop` (or the program `morphogen-desktop`) opens it.
A folder that runs without Python (`morphoGen.exe` on Windows, `morphoGen.app` on macOS)
is built with PyInstaller by the recipe of `packaging/desktop`.

**OpenRadioss (optional).** The explicit dynamic simulations run OpenRadioss, a
separate program: `morphogen radioss install` downloads its latest release from GitHub
(Linux x86-64 and ARM, Windows) into `~/morphogen/openradioss`, where morphoGen finds it
(or set `OPENRADIOSS_PATH` to the folder of an installation, e.g. one built from its
sources on macOS); `morphogen radioss info` shows the installation found. Without it,
morphoGen writes the inputs and the scripts that run them elsewhere.

**MFront behaviours (optional).** MGIS and MFront are not on PyPI: install them from
conda-forge in morphoGen's environment, with a C++ compiler for the behaviours:

```bash
conda install -c conda-forge mgis mfront cxx-compiler
```

On Windows, MFront needs the C++ compiler of Visual Studio (its Build Tools); WSL with
Ubuntu and Miniforge is simpler, and is where morphoGen's MFront tests run.
`morphogen info` shows MGIS and the version of MFront found.

Check the installation:

```bash
morphogen info
python -m pip install -e ".[dev]" && python -m pytest
```

The first call of each kernel compiles it (a few seconds); compiled code is cached.

<!-- section: quick start -->
## Quick start

The [tutorial](docs/tutorial.md) follows a whole study step by step; the
[gallery](docs/gallery.md) starts from any of the templates.

### Python

```python
import morphogen as mg

model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3)
micro = model.generate(mg.Domain.cube(20.0, periodic=True), seed=42)  # explicit geometry
image = micro.voxelize(shape=128)                                       # phase image

print(mg.analysis.volume_fraction(image), model.theory()["volume_fraction"])
print(mg.analysis.specific_surface(image), model.theory()["specific_surface"])

mg.io.save_image(image, "spheres.vti")                      # ParaView, FFT codes
mg.io.save_mesh(mg.mesh.voxel_mesh(image, phases=[1]), "spheres.inp")
mg.io.save_microstructure(micro, "spheres.json")            # regenerates it exactly
```

### Apparent properties

```python
from morphogen.homogenization import VOID, Material, homogenize

mesh = mg.mesh.conforming_mesh(micro, size=0.5)       # tetrahedra following the spheres
materials = {0: Material(young=1.0, poisson=0.3, conductivity=1.0),
             1: Material(young=20.0, poisson=0.3, conductivity=10.0)}
results = homogenize(mesh, materials, bcs=["kubc", "mixed", "periodic"],
                     properties=["bulk_modulus", "shear_modulus", "conductivity"])
print(results["bulk_modulus.kubc"].value)

# porous media: mesh only the solid (phases="solid"), and give the pores a VOID
# voxel images (e.g. tessellations) are solved by FFT, under periodic conditions:
results = homogenize(image, materials, properties=["stiffness"])
stiffness = results["stiffness.periodic"]
print(stiffness["tensor"])      # C: Voigt 11 22 33 23 31 12, engineering shears
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

A cell symmetric about its mid-planes can be meshed in its irreducible part and
mirrored: the mesh then has its symmetries exactly (and matching opposite faces), as the
analyses by group theory need; any level-set function makes a microstructure:

```python
from morphogen.mesh import ImplicitBody, mirrored_mesh

def holes(points):                   # negative in the solid: a square array of holes
    d = points[:, :2] % 1.0 - 0.5
    return 0.4348 - np.hypot(d[:, 0], d[:, 1])

body = ImplicitBody(mg.Domain(size=(1.0, 1.0, 0.13), periodic=True), holes)
mesh = mirrored_mesh(body, size=0.065, cells=(2, 2, 1), order=2, phases="solid")
# with the diagonal mirrors of the square (4/mmm), or of the cube: ((0, 1), (1, 2))
square = mirrored_mesh(body, size=0.065, diagonals=((0, 1),), cells=(2, 2, 1), phases="solid")
```

### Polycrystals

```python
from morphogen.homogenization import Material, homogenize

# a copper polycrystal: 50 grains with random orientations, meshed grain by grain
grains = mg.models.Voronoi(n_cells=50, crystal_orientations=True).generate(
    mg.Domain.cube(1.0, periodic=True), seed=1)
mesh = mg.mesh.conforming_mesh(grains, size=0.08, min_feature=0.02)
copper = Material.cubic(168.4, 121.4, 75.4)
results = homogenize(mesh, {1: copper}, bcs=["kubc", "periodic", "subc"],
                     properties=["bulk_modulus", "shear_modulus"])
print(results["bulk_modulus.subc"].value)        # (C11 + 2 C12) / 3 = 137.07, exactly

mg.io.save_mesh(mesh, "grains.inp")              # element sets and orientations per grain
mg.io.write_periodic_equations("grains_equations.inp", mesh)
mg.io.save_mesh(mesh, "grains.xdmf")             # FEniCSx, with phase, grain and face tags
mg.io.write_damask("grains", grains.labels(shape=64), grains.orientations,
                   materials={1: copper})        # DAMASK grid and material.yaml

# combinations: alumina inclusions and an amorphous boundary phase in the polycrystal
model = mg.models.Combination(operation="inclusions", models=(
    mg.models.Voronoi(n_cells=50, crystal_orientations=True, boundary_thickness=0.01),
    mg.models.HardcoreSpheres(radius=0.05, volume_fraction=0.05)))
print(model.active_phases())     # {1: 'cells', 0: 'grain boundaries', 2: 'inclusions ...'}
```

Features smaller than `min_feature` (short cell edges, thin caps of cells on the faces
of the box) are removed before meshing; `min_feature=0` meshes the exact cells. Without
MMG, or for voxel workflows, the same polycrystals are homogenized by FFT on their
images with the grain labels (`homogenize(image, materials, grains=labels,
orientations=grains.orientations)`).

More examples are in [`examples/`](https://github.com/jdirr/morphoGen/tree/main/examples): `minimal.py`, `rve_study.py` (Voronoi
mosaic, integral range recovered), `homogenization.py` (finite elements and FFT),
`multiphysics.py` (one microstructure homogenized for its elastic, thermal, dielectric
and diffusive properties, its expansions, its poroelastic and piezoelectric laws, its
dynamic moduli, and cut by extended finite elements), `acoustic_foam.py` (from an
open-cell foam to the parameters of its equivalent fluid and the sound absorption of
layers of it),
`polycrystal.py` (meshes of grains, crystals, exports, a parallel batch),
`ebsd_map.py` (an orientation map from end to end: its grains, its texture, the
stiffness of its polycrystal, windows of it as the realizations of a study),
`automata_and_design.py` (automata, growth, aggregation, tile assemblies, rigidity,
inverse design of rules), `topology_optimization.py` (bulk modulus and auxetic cells),
`architectures_ashby.py` (lattices, foams, trabecular bone, honeycombs and fractal
sponges homogenized, recorded in a database and placed on a normalized Ashby chart),
`fracture.py` (cohesive zones of a double cantilever beam against the corrected beam
theory, an edge crack by XFEM against the handbook), `microfoam.py` (the hole size effect
of nickel foams with the microfoam model, a crack as a band of micromorphic damage on two
meshes, the densification of the foam compressed to half its height in finite strains, a
ductile crack by micromorphic ductile damage on two meshes), `hill_of_a_laminate.py`
(Hill's law of a laminate identified on its responses, written for Z-set),
`gurson_of_pores.py` (Gurson's q1 and q2 of spherical pores in a von Mises matrix, from a
tension and a confined one), `benchmark.py`, `gpu_benchmark.py` and
`behaviour_benchmark.py` (the cost of each behaviour of the integration points).

### Cellular automata and growth

```python
import numpy as np
from morphogen import automata
from morphogen.models import CellularAutomatonModel, NucleationGrowthModel

# an automaton on its own: Vichniac's annealing rule coarsens noise into a
# bicontinuous structure
rng = np.random.default_rng(1)
run = automata.CellularAutomaton("anneal", neighbourhood="moore").run(
    automata.random_state((64, 64, 64), 0.5, rng), steps=30, rng=rng)
print(run.history["interface"][-1])                      # interface density

# as a model: any domain, periodic or not, smooth interfaces off the lattice
micro = CellularAutomatonModel(rule="anneal", steps=30, cell_size=0.5).generate(
    mg.Domain.cube(20.0), seed=1)

# a partially crystallized structure, and its kinetics against the Avrami law
model = NucleationGrowthModel(nucleation="continuous", nucleation_rate=1e-3, time=6.0)
grains = model.generate(mg.Domain.cube(40.0), seed=2)
print(grains.voxelize().data.mean(), model.theory()["transformed_fraction"])
frames = model.evolution(mg.Domain.cube(40.0), seed=2, frames=20)  # formation
```

### Combinatorial design

```python
from morphogen import design
from morphogen.models import RandomLattice, TileAssembly

design.count_polarized_blocks((4, 4, 4))           # 145 456 074 compatible assemblies
blocks = design.polarized_blocks()
assembly = design.wave_function_collapse(blocks, (8, 8, 8), rng)
metamaterial = TileAssembly(tiles="polarized").generate(mg.Domain.cube(8.0), seed=3)

# an octet lattice diluted to a mean coordination of 5: below Maxwell's 6, it is floppy
lattice = RandomLattice(cell="octet", coordination=5.0).generate(mg.Domain.cube(4.0), seed=4)
print(lattice.summary()["floppy_strains"])
```

### Topology optimization

```python
from morphogen.optimization import CellOptimization

# planar cell (one voxel through the thickness) with a negative Poisson's ratio
opt = CellOptimization((64, 64, 1), 0.35, objective="poisson_ratio", method="mma",
                       filter_radius=3.0, projection=8.0, iterations=300)
cell = opt.run()
print(cell.properties())                   # moduli of the black-and-white design
print(cell.refined_properties(2), cell.hinges)  # checks: finer grid, hinges
micro = cell.to_microstructure()           # smooth level set: mesh it, homogenize it
```

The same from a configuration file: `morphogen template optimize_auxetic -o
auxetic.yaml`, then `morphogen optimize auxetic.yaml`; in three dimensions, the template
`optimize_auxetic_3d` (a cell auxetic along its three axes, on 24³ voxels).

### Laser-architectured sheets

```python
from morphogen.laser import Beam, SheetCase, Specimen, optimize

# corrugated martensite in an annealed 0.23 C steel, a fibre laser above and a diode below
case = SheetCase(
    material="annealed_ms1500",
    specimen=Specimen(kind="dogbone", length=30.0, width=5.0, thickness=1.0),
    beam=Beam(150.0, speed=10.0, radius=0.5, absorptivity=0.55),
    bottom=Beam(100.0, speed=10.0, radius=0.6, absorptivity=0.55, profile="tophat"),
    pattern={"kind": "corrugated", "count": 1, "amplitude": 1.0, "period": 15.0},
)
print(case.evaluate().measures)          # Rp, Rm, Ag, energy, treated fraction, path...
best = optimize(case, {"pattern.amplitude": [0.0, 2.5], "pattern.period": [6.0, 30.0]},
                constraints={"yield_strength": [400.0, None]}, evaluations=30)
print(best.values, best.result.path.to_gcode()[:200])
```

The same from a configuration file: `morphogen template laser_corrugated -o
laser.yaml`, then `morphogen laser laser.yaml` (curves, maps, history, G-code).

### Nested and bio-inspired models

```python
# carbon black in rubber: particles in aggregates (a Cox model), with a bound-rubber phase
nano = mg.models.CoxSpheres(
    aggregates={"type": "boolean_spheres", "radius": 0.1, "volume_fraction": 0.25},
    radius=0.012, local_fraction=0.35, hardcore=True, aggregate_phase=2)
# hollow spheres, and a hierarchical lattice of hollow tubes (Rayneau-Kirkhope et al.)
shells = mg.models.Coating(model={"type": "hardcore_spheres", "radius": 0.1,
                                  "volume_fraction": 0.2}, thickness=0.02, side="inside")
frames = mg.models.HierarchicalLattice(generations=1, cells=3, tube_radius=0.012,
                                       wall_thickness=0.004)
menger = mg.models.FractalSponge(pattern="menger", level=3)
# from biology
nacre = mg.models.Nacre(arrangement="brick", tablet_size=1.0, tablet_thickness=0.2,
                        mortar_thickness=0.05)
plywood = mg.models.Bouligand(fiber_radius=0.03, fiber_spacing=0.08, pitch_angle=15.0)
wood = mg.models.Honeycomb(arrangement="wood", cell_size=0.4, wall_thickness=0.04,
                           ring_width=4.0, ray_fraction=0.08)   # rays: 3D
bone = mg.models.OpenFoam(n_cells=64, volume_fraction=0.15, relaxation=10, anisotropy=1.6)
cortical = mg.models.Osteons(radius=100.0, coverage=0.65, cement_thickness=4.0)
tree = mg.models.BranchingNetwork(n_attractors=3000, step=0.02, murray_exponent=3.0)
print(cortical.theory())        # exact fractions of the four phases (dead leaves)
```

### Scale-rich networks

```python
from morphogen.homogenization.postprocess import directional_modulus

# Both et al. (2026): 500 ligaments of thicknesses lambda_0 t^-0.6 at 40 % relative density
model = mg.models.ScaleRichNetwork(n_ligaments=500, decay=0.6, volume_fraction=0.4)
plane = mg.Domain(size=(1.0, 1.0, 1 / 512), periodic=False)   # one voxel thick
net = model.generate(plane, seed=1)
print(model.theory(plane))      # lambda_0 of the density, the jamming threshold...
print(net.metadata["summary"])  # ligaments, density, degrees, geometric anisotropy
print({k: v.exponent for k, v in net.statistics().items()})  # power-law tails
# aligned, polarized (tangential for the first 20 %), graded nucleation; a frame for a
# periodic cell (FFT, Bloch waves)
aligned = mg.models.ScaleRichNetwork(volume_fraction=0.4, angles=[0, 90])
swirl = mg.models.ScaleRichNetwork(volume_fraction=0.4, pole=[0.5, 0.5], pole_angle=90,
                                   polarized_fraction=0.2)
cell = mg.models.ScaleRichNetwork(volume_fraction=0.4, frame=0.006).generate(plane, seed=1)
# the elastic anisotropy E_min / E_max of a stiffness tensor C, in plane strain
# eta = directional_modulus(C, np.linspace(0, np.pi, 361)).min() / ....max()
```

In 3D, the plates or the beams of Fig. 6a of the paper fill the box:

```python
cube = mg.Domain(size=(1.0, 1.0, 1.0), periodic=False)
plates = mg.models.ScaleRichNetwork(dimension=3, n_ligaments=2000, decay=0.5,
                                    volume_fraction=0.3)   # closed cells
cells = plates.generate(cube, seed=1)
print(plates.theory(cube))       # first plate area, lambda_0, jamming, tails 5/2...
print(cells.volume_fraction, cells.areas[:5], cells.degrees().mean())
beams = mg.models.ScaleRichNetwork(dimension=3, element="beams", n_ligaments=500,
                                   decay=0.3, volume_fraction=0.15)  # open cells, struts
struts = beams.generate(cube, seed=1)
segments = struts.segments()     # the beams as capsules (meshing, STL)
# layers, vertical walls, shells about a pole, a gradient along z
layers = mg.models.ScaleRichNetwork(dimension=3, volume_fraction=0.3,
                                    orientation={"type": "aligned", "direction": [0, 0, 1]})
shells = mg.models.ScaleRichNetwork(dimension=3, volume_fraction=0.3, pole=[0.5, 0.5, 0.5],
                                    pole_angle=90, nucleation_gradient=[1, 4], gradient_axis=2)
image = cells.voxelize(shape=128)
# the elastic anisotropy E_min / E_max of a 3D stiffness C (homogenization.postprocess):
# e = spatial_modulus(C, sphere_directions()); eta = e.min() / e.max()
```

`examples/scale_rich_networks.py` reproduces the laws of the paper and the anisotropy of
its networks, `examples/scale_rich_networks_3d.py` those of plates and beams.

### Macroscopic loadings and thermal expansion

```python
from morphogen.homogenization import Loading, homogenize

materials = {0: Material(young=2800.0, poisson=0.39, expansion=95e-6),     # epoxy
             1: Material(young=89000.0, poisson=0.3, expansion=5.5e-6)}    # glass
results = homogenize(image, materials, properties=["stiffness", "thermal_expansion"],
                     loadings={"tension": {"stress": {"33": 100.0}},       # the rest free
                               "plane": Loading.parse("E11=0.001 E33=0"),
                               "heating": {"temperature": 100.0}})
print(results["thermal_expansion.periodic"]["tensor"])
print(results["tension.periodic"].components())   # E11 ... E12, S11 ... S12, energy
```

### Multiphysics: transport, couplings, harmonic regimes

Every transport phenomenon of a scalar potential has its coefficient and its properties;
couplings and harmonic regimes are properties too ([docs/physics.md](docs/physics.md)
goes through all of them, with fracture and the choice of a solver):

```python
import dataclasses

phases = {0: Material(conductivity=0.2, electrical_conductivity=1e-12, permittivity=3.0),
          1: Material(conductivity=30.0, electrical_conductivity=1e4, permittivity=10.0)}
results = homogenize(image, phases, properties=["conductivity", "electrical_conductivity",
                                               "permittivity_tensor"])

# a piezoelectric composite: the fields solved together, the constants of the law
ceramic = dataclasses.replace(Material.hexagonal(166e9, 77e9, 78e9, 162e9, 43e9),
                              permittivity=(11.2e-9, 11.2e-9, 12.6e-9),
                              coupling={"piezoelectric": {"e31": -4.4, "e33": 18.6, "e15": 11.6}})
polymer = Material(young=4e9, poisson=0.35, permittivity=3.7e-11)
law = homogenize(image, {0: polymer, 1: ceramic}, properties=["piezoelectric"])
print(law["piezoelectric.periodic"]["d33"], law["piezoelectric.periodic"]["k_t"])

# storage and loss moduli of a viscoelastic matrix at three angular frequencies
resin = Material(young=3e9, poisson=0.35,
                 viscoelasticity={"shear": [[0.5, 1e-2], [0.3, 1.0]], "bulk": [[0.2, 0.1]]})
dma = homogenize(image, {0: resin, 1: Material(young=70e9, poisson=0.22)},
                 properties=["dynamic_stiffness"], frequencies=[0.1, 10.0, 1000.0])
print(dma["dynamic_stiffness.periodic"]["shear_tan_delta_2"])

# a porous rock: Biot's tensor and modulus, Skempton's coefficient with water in the pores
rock = homogenize(image, {0: Material(young=20e9, poisson=0.2), 1: VOID},
                  properties=["poroelastic"], fluid_bulk_modulus=2.2e9)
print(rock["poroelastic.periodic"]["biot_coefficient"])
```

### Nonlinear behaviours along loading paths

```python
from morphogen.homogenization import LoadingPath, Material, homogenize, simulate_point

matrix = Material(young=70e3, poisson=0.33, plasticity={      # aluminium alloy (MPa, s)
    "yield": 100, "isotropic": [{"Q": 80, "b": 20}], "kinematic": [{"C": 1e4, "D": 100}],
    "norton": {"K": 50, "n": 5}})
particles = Material(young=400e3, poisson=0.2)                   # SiC
# tension at 1e-3 /s to 2 %, relaxation for 100 s, unloading; lateral stresses free
path = LoadingPath.parse("t=0,20,120,130 n=10,10,5 E11=0,0.02,0.02,0.017")
out = homogenize(mesh, {0: matrix, 1: particles}, properties=[], paths={"tension": path})
curve = out["tension.periodic"]            # times, strain, stress, plastic, phases, fields
print(curve.stress[:, 0], curve.phases[0]["plastic"])
print(simulate_point(matrix, path).stress[-1])                  # the matrix alone
```

Crystal plasticity (Méric–Cailletaud) in the grains of a polycrystal mesh, in finite
strains:

```python
from morphogen.homogenization import CrystalPlasticity, Material, NonlinearFEProblem

nickel = Material.from_config({
    "cubic": [250e3, 160e3, 120e3],                               # MPa
    "crystal_plasticity": {
        "families": ["octahedral"],                               # also cube, bcc110, bcc112
        "tau0": 80, "Q": 20, "b": 10,                             # r = tau0 + Q sum h (1 - e^-bv)
        "C": 2e4, "D": 300,                                       # x = C alpha
        "K": 25, "n": 10,                                         # <(|tau - x| - r) / K>^n
        "interaction": [1, 1, 0.6, 12.3, 1.6, 1.3]}})             # Franciosi
problem = NonlinearFEProblem(poly_mesh, {1: nickel}, "periodic", kinematics="finite")
curve = problem.run({"times": [0, 100], "increments": 40, "strain": {"11": 0.1}},
                    fields=True)
print(curve.stress[:, 0], curve.cauchy[:, 0], curve.log_strain[:, 0])  # S, TS, LE
print(curve.fields[100.0]["orientation"][:5])     # current lattice orientations (Bunge)
```

Crystals also twin and cleave ([docs/theory.md](docs/theory.md) §86–87). Hexagonal
crystals have their slip families (`basal`, `prismatic`, `pyramidal_a`, `pyramidal_ca`,
`pyramidal2_ca`, with the ratio `c_over_a`); twins shear in one sense until the crystal
has twinned; cleavage planes open at a normal stress and soften, with the fracture
energy of a crack band:

```python
magnesium = Material.from_config({
    "hexagonal": [59400, 25600, 21400, 61600, 16400],             # c11, c12, c13, c33, c44
    "crystal_plasticity": {
        "c_over_a": 1.624, "K": 5, "n": 10,
        "families": {"basal": {"tau0": 10}, "prismatic": {"tau0": 60},
                     "pyramidal2_ca": {"tau0": 100}},
        "twins": {"twin1012": {"tau0": 25}}}})                    # {10-12}: lengthens c
iron = Material.from_config({
    "cubic": [230e3, 135e3, 117e3],
    "crystal_plasticity": {
        "K": 10, "n": 10, "families": {"bcc110": {"tau0": 300}},
        "cleavage": {"planes": "cube", "strength": 500,           # {100}: opens at 500 MPa
                     "energy": 0.05, "length": 0.008,             # G_c, in elements of 8 µm
                     "K": 50, "n": 1}}})
curve = simulate_point(magnesium, {"times": [0, 1], "increments": 20, "strain": {"33": 0.03}})
curve.series["twin_fraction"]         # of a crystal pulled along c; cleavage_damage for iron
```

A textured magnesium flows at half the stress in the sense that twins it (the template
`homogenization_twinning`); a polycrystal of iron breaks through its grains from
cleavage plane to cleavage plane (`homogenization_cleavage`), or between them when its
boundaries are cohesive and weaker (`interfaces: {faces: grains}`, the template
`homogenization_grain_fracture`). The fields map the `twin_fraction`, the
`cleavage_opening` and the `cleavage_damage`. Under a stress that is reversed the twins
shear back and the cleavage planes close ([docs/theory.md](docs/theory.md) §89): a
closed crack carries the pressure as the sound crystal, its lips rubbing with
`friction` (the template `homogenization_crack_closure`); and with `twin_reorientation`
the lattice of a point becomes that of its predominant twin at a threshold of its
twinned fraction, slips and twins as the twin does, and twins back (the field
`twin_misorientation`; the template `homogenization_detwinning`) — point by point, or
grain by grain (`{threshold, growth}`: the turned volume of a grain, `twin_reoriented`,
then follows its twinned volume). `twin_detwinning` and `closure` also take the share of
their resistance at which the twins shear back and the planes close. And the slip can
nucleate the cleavage ([docs/theory.md](docs/theory.md) §90): with `cleavage: {coupling:
Θ}` the strength of the planes falls as `exp(−Θ γ)` of the cumulated slip of the
crystal, and their damage eases its slip.

How likely is cleavage to start? The local approach of Beremin
([docs/theory.md](docs/theory.md) §88) reads it from the fields of a path: the Weibull
stress of the points that have yielded, and the probability that the weakest link has
failed.

```python
from morphogen.homogenization import Beremin, path_beremin

law = Beremin(modulus=22, scale=2600.0, volume=1.25e-4,        # m, sigma_u (MPa), V0 (mm³)
              planes="cube", plastic=1e-4)                     # the {100} planes of the grains
found = path_beremin(curve, law, orientations=euler_of_each_voxel)   # fields="steps"
found.weibull, found.probability       # at each step; found.driving: the stress of each point
```

In a configuration, `homogenization: {beremin: {modulus: 22, scale: 1550, volume:
1.25e-10, planes: cube, plastic: 1e-4}}` adds the curves `beremin_weibull` and
`beremin_probability` to those of the path (the template `homogenization_beremin`); in
the graphical interface, the box *Cleavage* of the tool *Loading path of a morphology*.

In finite strains the macroscopic gradient is upper triangular: `E11 ... E12` are its
components (nominal strains, and simple shears for `E23, E31, E12`), `S11 ... S12` the
same components of the nominal (first Piola–Kirchhoff) stress; free components leave
the faces free of tractions, so that a tension is a uniaxial Cauchy stress.

The same paths are followed on voxel images by the FFT solver, with the grains and
orientations of a tessellation:

```python
from morphogen.homogenization import NonlinearFFTProblem

image = poly.voxelize(shape=32)                   # a Voronoi realization, 32³ voxels
problem = NonlinearFFTProblem(image, {1: nickel}, grains=poly.labels(grid=image.grid()),
                              orientations=poly.orientations, kinematics="finite")
curve = problem.run({"times": [0, 50], "increments": 25, "strain": {"11": 0.05}})
```

Each voxel is a trilinear element with 8 Gauss points; the volumetric part of the strain
is averaged over the voxel (`integration="bbar"`: B-bar, and F-bar in finite strains),
without which the voxels lock under plastic flow (the flow stress of elastic cubes in a
plastic matrix is twice too high on 10³ voxels with `integration="full"`). The memory
is that of the integration points: about 8 kB per voxel for a crystal with 12 slip
systems in small strains (2 GB for 64³ voxels).

Behaviours written in MFront take the place of morphoGen's, with MGIS and MFront
installed (the source is compiled on first use and cached):

```python
from morphogen.homogenization.mfront import MFrontBehaviour

alloy = Material(young=70e3, poisson=0.33,     # the linear problems, the FFT reference
                 mfront=MFrontBehaviour(source="tests/mfront/VonMisesTwin.mfront"))
print(simulate_point(alloy, path).stress[-1])  # its own elasticity and viscoplasticity
out = homogenize(mesh, {0: alloy, 1: particles}, properties=[], paths={"tension": path})
```

In configurations and materials files, `mfront` is the path of the source (relative to
the file) or a mapping: `source`, or `library` and `behaviour`; `material_properties`,
`parameters` and `external_state_variables` (constants); `temperature` (the reference, in
K, to which the paths add their change of temperature); `plastic_strain` (the internal
variable of the cumulated plastic strain of the outputs).

### Identification of a macroscopic law

The parameters of a behaviour are identified on reference responses, as Zopt does for
Zsim: the curves of `homogenize` (`PathResult`: `curve.path`, `curve.times`,
`curve.stress`) of a morphology under several paths give it a macroscopic law, material
points or the points of a test calibrate a law; each reference counts by its weight
whatever its number of points. Gurson's q1 and q2 under two triaxialities, written back
for Z-set:

```python
from morphogen.homogenization import Material, simulate_point
from morphogen.homogenization.identification import Reference, identify_references
from morphogen.homogenization.zmat import write_zmat

tension = {"times": [0, 1], "increments": 12, "strain": {"11": [0, 0.05]}}  # triaxiality 1/3
confined = {**tension, "strain": {"11": [0, 0.05], "22": [0, 0], "33": [0, 0]}}  # held laterally
voids = {"criterion": "gurson", "yield": 100, "hardening": 1000, "porosity": 0.07}
law = Material.from_config({"young": 70e3, "poisson": 0.33,
                            "porous_plasticity": {**voids, "q1": 1.5, "q2": 1.0}})
refs = [Reference(r.path, r.times, r.stress, components) for r, components in (
    (simulate_point(law, tension), [0]), (simulate_point(law, confined), [0, 1, 2]))]
start = {"young": 70e3, "poisson": 0.33, "porous_plasticity": {**voids, "q1": 1.0, "q2": 1.3}}
fit = identify_references(start, refs, ["q1", "q2"])
print(fit.values, fit.gaps)              # q1 1.5, q2 1.0; the gap to each response
write_zmat(fit.material, "gurson.mat", digits=6)  # porous_plastic, for Z-set
```

In the interface, "Identification of a behaviour" (tab Mechanics) takes the runs of the
workspace, each under its own path, and "Z-set file of a behaviour" (Solvers & formats)
writes the behaviour found.

An anisotropic morphology calls for an anisotropic law: Hill's criterion
(`hill: {11: …, 22: …, 33: …, 12: …, 23: …, 31: …}`, Z-set's form, 1 where not given),
whose coefficients `hill11` … `hill31` the identification fits from von Mises's. Two
plastic phases in layers normal to z (`examples/hill_of_a_laminate.py`, a few seconds)
yield alike in tension along x and z, but their shear across the layers (31, the layers in
series) saturates at 0.63 where the shear in their plane (12, in parallel) reaches 0.89:
von Mises misses the first by 38 %, Hill's law found (`h31 ≈ 2`, `h12 ≈ h33 ≈ 1`) follows
all four responses within 4 %, and `write_zmat` writes it as a `gen_evp` with
`*criterion hill` for Z-set. `macroscopic_law(respond)` does it in one go: it runs the
tensions and shears chosen through `respond(path)` (the response of the morphology),
estimates a starting law on them (elasticity of the first increment, yield stress,
hardening; with `cycle=True` the loadings reverse and the law takes a kinematic
hardening, the Bauschinger effect of the morphology) and identifies the von Mises or Hill
law on them together; in the interface,
"Macroscopic law of a morphology" (tab Mechanics) does the same on its voxels and writes
the law for Z-set. `law="gurson"` gives a porous morphology Gurson's law: the matrix
of its phase, its voids as the porosity, q1 and q2 identified on loadings of several
triaxialities (`loadings=["11", "confined11"]`: a tension, then one whose lateral
strains are held), as on unit cells.

### Database of results and Ashby charts

```bash
morphogen run gyroid.yaml --db study.db          # or outputs.database in the file
morphogen db list --db study.db
morphogen db export --db study.db --where "model.type == tpms" -o gyroids.csv
morphogen ashby --db study.db -o ashby.svg       # relative modulus vs relative density
```

### Command line

```bash
morphogen models                          # list the models
morphogen models poisson_fibers           # parameters of a model
morphogen template                        # list the example configurations
morphogen template tpms -o gyroid.yaml    # start from an example
morphogen run gyroid.yaml                 # generate, measure, export
morphogen analyze out/tpms/gyroid_0000.vti --measure volume_fraction specific_surface
morphogen texture map.ctf -o texture/      # grains, pole figures, disorientations of an EBSD map
morphogen texture out/polycrystal_0000_tension_periodic_4.vti -o texture/  # lattices of a path
morphogen percolation spheres.yaml --lower 0.15 --upper 0.45  # percolation threshold
morphogen percolation fibres.yaml --lower 0.02 --upper 0.6 --directions lattice  # anisotropy
morphogen percolation spheres.yaml --lower 0.1 --upper 0.4 --sizes 24,32,48,64 \
    --exponent thermal --realizations 49   # the exponent t of the conductivity at p_c
morphogen percolation discs.yaml --lower 0.4 --upper 0.8 --sizes 32,48,64 \
    --exponent elastic --contrast 0        # f at the threshold of the moduli (--rigidity)
morphogen frame-stiffness foam.yaml --beams timoshenko --rigid-joints 0.33 \
    --softening modulus           # a network as a frame of beams: stiffness, energy split
morphogen frame-stiffness foam.yaml --softening bending --replicas 40 --cv 0.1 0.2 0.3 \
    --table replicas.csv --figure replicas.png   # Ban's replicas against s CV^2 and v CV^2
morphogen view out/tpms/gyroid_0000.vti --surface       # 3D figure (matplotlib)
morphogen homogenize cell.json --voxels 64 --material 0=void 1=young:1,poisson:0.3 \
    --property stiffness --loading "tension: S11=1" "E11=0.01 S22=0"
morphogen homogenize foam.vti --material 0=void 1=young:1,poisson:0.3 \
    --property hydraulic_permeability acoustic   # Darcy's permeability, acoustic parameters
morphogen bands rods.vti --material 0=permittivity:1 1=permittivity:8.9 --waves tm \
    --path G X M G --figure bands.png      # band structure and gaps of a photonic crystal
morphogen bands diamond.vti --material 0=permittivity:1 1=permittivity:11.56 \
    --waves electromagnetic --coarsen 4    # light in a three-dimensional crystal
morphogen gui                             # graphical interface in the browser
morphogen template workflow -o chain.yaml # the whole chain: model, mesh, fields, videos
morphogen laser laser.yaml                # a laser-architectured sheet and its laser path
morphogen export fields.vtu stress.mp4 --field von_mises --displacement displacement \
    --wave sine --mode loop               # files of a run to other formats and videos
```

A configuration file describes the model, the domain, the voxels, the outputs and the
measurements:

```yaml
model:
  type: poisson_fibers
  radius: 1.0
  volume_fraction: 0.16
domain:
  size: 30.0
  periodic: false          # infinite fibres cannot be periodic
seed: 2014
voxels:
  voxel_size: 0.25
outputs:
  directory: out/poisson_fibers
  geometry: [json, geo]    # geo: Gmsh script
  image: [vtk]             # legacy VTK for FFT solvers
  mesh: {formats: [msh], phases: [1]}
analysis: [volume_fraction, specific_surface, connectivity]
```

### The whole chain in a configuration

A `visualization` section draws figures and videos of the first realizations: their
voxel image, conforming mesh and the fields of the homogenization, on the deformed mesh
(finite elements) or the voxels (FFT); `outputs.tables` adds `.txt` and `.xlsx` copies
of the tables. The template `workflow` follows the whole chain:

```yaml
outputs:
  tables: [txt, xlsx]
visualization:
  cmap: viridis
  figures:
    - {of: image, formats: [png, svg, tex]}      # three sections, TikZ for LaTeX
    - {of: mesh}                                 # 3D view coloured by phase
    - {of: shear_modulus.periodic, color: von_mises, section: {axis: 2, fraction: 0.5}}
    - {of: tension.periodic, formats: [pdf]}     # curves of a loading path
  videos:
    - {of: image, kind: orthoslices, mode: bounce}   # as in Avizo
    - {of: mesh, kind: sweep, axis: 0}               # a cross-section sweeping
    - {of: shear_modulus.periodic, wave: sine, mode: loop, part: fluctuation}
    - {of: tension.periodic, color: p, frames: 60}   # the steps of the path
```

Views: `image` (sections, orthoslices, the surface of a phase), `labels`, `mesh`
(view, turntable, sweep) and the results `<property>.<bc>`, `<loading>.<bc>`,
`<path>.<bc>` coloured by a field (`von_mises`, `stress_12`, `p`, `displacement`,
`flux`...; of a path, also `interface_damage` and `interface_opening` where interfaces
debond, `fatigue_utilization` and `fatigue_amplitude` where a cycle is judged),
magnified (`magnification`, automatic by default), cut by a plane, with
the colormap and range of your choice; formats `png`, `tif`, `jpg`, `svg`, `pdf`,
`tex` (TikZ) for figures and `mp4`, `gif`, `frames` for videos, played `once`, in a
`loop` or back and forth (`bounce`). The 3D views need VTK and the mp4 videos ffmpeg
(`pip install 'morphogen[video]'`); a run checks them before it starts.

### Cohesive zones: interfaces and fracture

`morphogen.fracture` opens the interfaces of a tetrahedral mesh (between phases, or
along chosen faces) and gives them cohesive zone elements, in small or finite strains:

```python
from morphogen.fracture import Bilinear, CohesiveProblem, insert_interfaces

opened, interfaces = insert_interfaces(mesh, "phases")   # debonding of the particles
law = Bilinear(stiffness=1e5, normal_strength=30.0, shear_strength=40.0,
               mode_I=0.3, mode_II=0.6, eta=2.0)        # Benzeggagh-Kenane mixity
problem = CohesiveProblem(opened, interfaces, materials, law, kinematics="finite")
result = problem.run([(bottom, [0, 1, 2], 0.0), (top, 2, 1.0)], load=0.05,
                     steps=50, viscosity=1e-4, fields=True)
problem.write("debonding", result)    # the bulk and the interfaces (damage), VTU
```

Laws: bilinear mixed-mode and the exponential law of Ortiz & Pandolfi; helpers: the
length of the cohesive zone, the strength of coarse meshes (Turon et al.), the double
cantilever beam (`dcb_specimen`) and its corrected beam theory (`dcb_beam_theory`).

Cracks (`morphogen.fracture.lefm`): planar cracks by level sets, their energy release
rate on conforming meshes remeshed at each step (MMG, as Z-cracks) or by XFEM (jump and
branch enrichments of linear tetrahedra), and their growth by Paris' law or Griffith:

```python
from morphogen.fracture import Crack, propagate, xfem_work

crack = Crack("penny", centre, normal=(0, 0, 1), size=2.0)
growth = propagate(xfem_work(mesh, material, forces, supports), crack, young=E,
                   poisson=nu, increment=0.5, steps=10, paris=(C, m))
```

Mixed modes, in small or finite strains (`morphogen.fracture.mixed`): the energy release
rate along the front by the G-θ method (exact for the neo-Hookean body of finite strains),
the factors of modes I, II and III by interaction integrals, the frictionless contact of
the lips, and kinked cracks remeshed by MMG at each step, growing in the direction of
Erdogan & Sih:

```python
from morphogen.fracture import KinkedCrack, crack_mesh, propagate_kinked, solve_crack

solution = solve_crack(mesh, crack, material, supports, forces=loads,
                       kinematics="finite", contact=True)
modes = solution.modes(stations=5)   # K_I, K_II, K_III, G, G_I, G_II, G_III along the front
crack = KinkedCrack(((0.0, 16.0, 0.0), (4.5, 16.0, 0.0)))  # mouth to front, front along z
growth = propagate_kinked(lambda c: crack_mesh(lower, upper, c, size=0.5), crack,
                          material, supports, forces=loads, increment=1.5, steps=10,
                          paris=(C, m))
```

Cracks of any surface and front (`morphogen.fracture.surface`): the surface swept by the
successive positions of a front, closed (a crack inside the body) or open (through, edge
and surface cracks), each point of which advances by its own amount and kinks by its own
angle; the mesh is refined around the front alone:

```python
from morphogen.fracture import SurfaceCrack, crack_mesh, propagate_front

crack = SurfaceCrack.ellipse(centre, normal=(0.5, 0, 0.87), a=1.0, b=0.6)  # inclined
mesh_of = lambda c: crack_mesh(lower, upper, c, size=0.6, front_size=0.08)
growth = propagate_front(mesh_of, crack, material, supports, forces=loads,
                         increment=0.3, steps=5, stations=12, paris=(C, m))
growth.rows()        # per step: area, K_I, K_II, K_III along the front, kink angles, cycles
growth.cracks[-1].points   # rows x stations x 3: the surface, its last row the front
```

Inelastic solids (`morphogen.fracture.bulk`): the crack solvers take the behaviours of
morphoGen's points as their bulk (von Mises elasto-viscoplasticity, with Hencky's
logarithmic strains in finite strains; crystal plasticity with the orientations of the
grains; MFront laws) and give the driving forces of dissipative solids:

```python
steel = Material(young=210e3, poisson=0.3, plasticity={
    "yield": 300.0, "hardening": 1e3, "norton": {"K": 50.0, "n": 5.0}})
solution = solve_crack(mesh, crack, steel, grips, kinematics="finite", steps=20)
solution.release_rate(energy="work")   # the J of the deformation theory (Begley-Landes)
solution.release_rate(energy="tip")    # the driving force at the tip (configurational forces)
opened, boundaries = insert_interfaces(polycrystal, faces=lambda c: c[:, 0] > -1e9,
                                       sides=polycrystal.cell)   # the grain boundaries
problem = CohesiveProblem(opened, boundaries, {0: crystal}, law, kinematics="finite")
result = problem.run(grips, 0.02, steps=40, duration=1.0)  # result.work: loads, bulk, interfaces
```

Ductile growth on meshes rebuilt at each advance (`morphogen.fracture.ductile`): the state
of the points (internal variables, dissipated energy, stresses) and the displacement move
from mesh to mesh, the lips apart, the body is brought back to equilibrium, and the crack
advances when its driving force at the tip reaches a toughness, or the porosity of its GTN
points a critical value; the far J at each advance draws the resistance curve:

```python
from morphogen.fracture import KinkedCrack, crack_mesh, propagate_ductile, solve_crack

growth = propagate_ductile(lambda c: crack_mesh(lower, upper, c, size=1.0), crack, steel,
                           grips, load_step=0.25, increment=1.5, steps=5, toughness=30.0,
                           kinematics="finite", transfer="nearest")
advance, resistance = growth.resistance()           # the R-curve (far J against the advance)
moved = solve_crack(new_mesh, longer_crack, steel, grips, 1.2, initial=solution)  # one transfer
```

A polycrystal grows its crack through its grains: the crack meshes carry the grains of a
tessellation and their orientations, and the state of each grain moves within the grain
from one mesh to the next:

```python
mesh_of = lambda c: crack_mesh(lower, upper, c, size=0.4, grains=tessellation)
growth = propagate_ductile(mesh_of, KinkedCrack(((0, 4, 0), (2.5, 4, 0))), crystal, grips,
                           load_step=1e-3, increment=1.0, steps=3, toughness=0.6)
```

Any morphology can be cracked: the crack meshes carry its phases (the materials, in
`mesh.phase`) and the sides of the crack apart (`mesh.side`), as staircases or conforming
to the level set of a two-phase morphology (MMG cuts the grid along it once, then each
crack cuts that mesh and remeshes it), its void phases left out. Stiff inclusions deflect
the crack, soft ones and pores attract it, and a front reaching a pore runs through it;
the state of inelastic points moves material by material:

```python
from morphogen.fracture import phase_mesh

phases = phase_mesh(micro, lower, upper, 0.5)            # cut along its level set, once
mesh_of = lambda c: crack_mesh(lower, upper, c, size=0.5, morphology=phases, voids=[1])
growth = propagate_kinked(mesh_of, crack, {0: matrix, 1: Material(void=True)}, held,
                          forces=tension, increment=0.75, steps=4)
```

The phases and their interfaces have toughnesses of their own: each advance is the
candidate of the largest `G(θ)/Γ(θ)`, the kink of the bulk or a path along an interface
met within an increment (the first-order kinked factors of Cotterell & Rice), so that a
crack turns along interfaces weaker than about a quarter of the phases (He & Hutchinson),
climbs a staircase through the weak boundaries of a brick-and-mortar structure, and
needs a higher load to enter a tough phase:

```python
from morphogen.fracture import Toughness

growth = propagate_kinked(mesh_of, crack, {0: mortar, 1: tablets}, held, forces=tension,
                          increment=0.5, steps=6,
                          toughness=Toughness({0: 1.0, 1: 4.0}, interfaces=0.2))
growth.rows()        # each advance: path (bulk or interface), G / toughness, critical load
```

### Contact, self-contact and friction

`morphogen.contact` keeps the surfaces of a mesh out of each other, out of themselves and
out of rigid obstacles, with Coulomb's friction, in small or finite strains
([docs/theory.md](docs/theory.md) §82): the integration points of the faces against the
nearest point of the surface that they face, a penalty or the augmented Lagrangian;
with `points="nodes"`, the nodes of the faces instead, for bodies that meet by their
corners and edges.

```python
from morphogen.contact import ContactProblem, Obstacle, Pair, Surface

skin = Surface.of(mesh)                                   # the free faces of the mesh
walls = Pair(skin, friction=0.3)                          # a surface against itself
punch = Obstacle.sphere(skin, centre, radius, motion=(0, 0, -0.1))
problem = ContactProblem(mesh, materials, [walls], [punch])
result = problem.run(prescribed, steps=20)                # forces, areas, pressures, works
```

The walls of the pores of a periodic cell close on themselves, and on their images
through the faces of the cell, along a loading path (`ContactCell`; in a configuration,
`homogenization: {nonlinear: {contact: {friction: 0.3}}}`; the template
`homogenization_contact`). Two tests come ready-made — a sample between two platens, and
under an indenter, whose force reads its modulus:

```python
from morphogen.contact import compression, indentation

crush = compression(mesh, materials, -0.2, friction=0.1)      # stress against strain
crush.modulus, crush.tangent(), crush.self_area               # stiffer as the walls close
dent = indentation(mesh, materials, 0.05, indenter="sphere", radius=10.0, unload=True)
dent.modulus, dent.layer_modulus(poisson=0.3), dent.hardness  # Hertz; with the base
```

```bash
morphogen compress foam.json --size 0.2 --material 0=void 1=young:1000,poisson:0.3 \
    --strain -0.3 --friction 0.1 --fields crushed
morphogen indent block.vtu --material 0=young:1000,poisson:0.3 --depth 0.05 --radius 10 \
    --unload --poisson 0.3
```

### Interfaces that debond

The interfaces of the phases of a periodic cell — or the boundaries of its grains — open,
break, close again and rub along a loading path ([docs/theory.md](docs/theory.md) §84):
cohesive interface elements inserted on the torus of the cell, so that an interface that
crosses a face goes on in the next cell.

```python
from morphogen.fracture import Bilinear, CohesiveCell

law = Bilinear(stiffness=1e7, normal_strength=30.0, shear_strength=30.0,
               mode_I=0.05, mode_II=0.05, friction=0.3)
cell = CohesiveCell(mesh, materials, law, viscosity=0.02)   # faces="phases", "grains", pairs
res = cell.run({"times": [0, 1, 2], "increments": [24, 8], "strain": {"11": [0, 0.03, 0]}})
res.stress, res.series["debonded_area"], res.series["interface_opening"]
```

In a configuration, `homogenization: {nonlinear: {interfaces: {normal_strength: 30,
mode_I: 0.05, friction: 0.3, viscosity: 0.02}}}` (the template
`homogenization_debonding`: glass beads that debond from their matrix, soft in tension
and nearly whole again in compression); on the command line, `morphogen homogenize ...
--interfaces normal_strength:30,mode_I:0.05,friction:0.3,viscosity:0.02`. A bead lets go
at once: the `viscosity` (a time of the path) slows the damage and lets the steps
through. Each pair of phases may have its law (`{(0, 1): law, (0, 2): other}`; `laws: {0-1:
{...}}`), and `faces: grains` makes the grain boundaries of a polycrystal cohesive (the
template `homogenization_intergranular`: a crack crosses the cell from boundary to
boundary, and the work that broke it is the fracture energy times the area of the
crack). The fields of the path map the damage and the opening of the interfaces. In the
graphical interface, the box *Interfaces* of the tool *Loading path of a morphology*.

### High-cycle fatigue

In the high-cycle regime a cell stays elastic, and whether a point lasts is read from
the cycle of its stress ([docs/theory.md](docs/theory.md) §85): the amplitude of its
deviator — the radius of the smallest hypersphere that holds its path — and a normal or
hydrostatic stress, in a criterion calibrated on two fatigue limits of the phase.

```python
from morphogen.homogenization import DangVan, homogenize, path_fatigue

cycle = {"times": [0, 1, 2, 3], "increments": [1, 2, 2], "stress": {"11": [0, 40, -40, 40]}}
out = homogenize(mesh, materials, properties=[], paths={"cycle": cycle}, fields="steps")
alloy = DangVan.from_limits(bending=110.0, torsion=80.0)     # or Crossland, Sines, Findley, Matake
res = path_fatigue(out["cycle.periodic"], {0: alloy}, cycle=(1, 3))   # a criterion per phase
res.utilization      # of each point: its equivalent stress over its limit
res.largest, res.safety, res.fraction()   # of the cell; the share of its volume beyond the limit
```

In a configuration, `homogenization: {fatigue: {criterion: dang_van, bending: 110,
torsion: 80, cycle: [1, 3]}}` writes the map of the utilization beside the curve of the
path and adds `fatigue_utilization`, `fatigue_safety` and `fatigue_beyond` to the
results (the template `homogenization_fatigue`: pores in a cast alloy; a spherical pore
doubles the stress at its equator, where the utilization is the largest). In the
graphical interface, the box *Fatigue* of the tool *Loading path of a morphology*.

### Micromorphic continua: microfoam and nonlocal damage

`morphogen.micromorphic` solves micromorphic media on tetrahedral meshes (quadratic
displacements, linear generalized degrees of freedom), in small or finite strains: the
microfoam model of nickel foams (Dillard, Forest & Ienny 2006: anisotropic compressible
plasticity in a micromorphic continuum, whose length removes the strain concentration of
holes as small as the cells; in finite strains the plasticity of the logarithmic strains
and the objective measures `χ⁻¹ F` and `χ⁻¹ Grad χ` of Forest & Sievert), a micromorphic
damage model (the implicit gradient model of Peerlings et al.) and micromorphic ductile
damage (von Mises elasto-(visco)plasticity whose damage follows the micromorphic
cumulated plastic strain), whose cracks are bands of damage independent of the mesh:

```python
from morphogen.micromorphic import (CHI, DamageProblem, DuctileProblem, Microfoam,
                                    MicrofoamProblem, MicromorphicDamage, MicromorphicDuctile)

foam = Microfoam.nickel_foam(length=0.1)          # the article's Tables 2 and 3 (MPa, mm)
problem = MicrofoamProblem(mesh, foam)            # u (P2) and the microdeformation (P1)
result = problem.run([(bottom, 1, 0.0), (held, CHI["chi12"], 0.0), (top, 1, 0.1)], steps=10)
fields = problem.fields(result.values, result.state)   # strain, stress, p, chi
crushed = MicrofoamProblem(mesh, foam, kinematics="finite")   # fields: log strain, Cauchy, J

damage = MicromorphicDamage(young=1000.0, poisson=0.2, threshold=1e-3, length=0.08)
cracked = DamageProblem(mesh, damage, kinematics="finite").run(supports, steps=40)

steel = MicromorphicDuctile(young=200e3, poisson=0.3, yield_stress=400.0, hardening=1e3,
                            voce=((200.0, 20.0),), length=0.1, onset=0.02, critical=0.1,
                            triaxiality="rice_tracey")   # or johnson_cook, lemaitre, none
ductile = DuctileProblem(mesh, steel, kinematics="finite")
result = ductile.run(supports, steps=40)          # a predictor by the last tangent
ductile.damage(result.state), ductile.plastic_strain(result.state)
```

### Z-mat material files

The material files of Z-mat (Z-set) run as they are: a `gen_evp` behaviour of von Mises
or Hill elasto-(visco)plasticity on morphoGen's own kernels, the others (hyperbolic or
double Norton flows, power-law hardening, several potentials...) as MFront behaviours
(MFront's brick StandardElastoViscoPlasticity):

```yaml
homogenization:
  materials:
    0: {zmat: alloy.mat}            # ***behavior gen_evp ... ***return
    1: {young: 400000, poisson: 0.2}
```

`morphogen zmat alloy.mat` shows the translation, `morphogen zmat --coverage` what of
Z-mat's library morphoGen covers (docs/zmat.md). The other way, `write_zmat` writes an
elastic, von Mises, Hill or porous material of morphoGen (a behaviour identified on the response
of a morphology: its Gurson model) as a material file of Z-mat, read back as the same:

```python
from morphogen.homogenization.zmat import write_zmat

write_zmat({"young": 200000, "poisson": 0.3, "plasticity": {
    "yield": 130, "kinematic": [{"C": 15000, "D": 300}], "norton": {"K": 400, "n": 7}}},
    "alloy.mat")
```

Porous plasticity (`porous_plastic` files of Z-mat, or a material of its own): the ductile
damage of Gurson-Tvergaard-Needleman (coalescence, Chu and Needleman's nucleation) and of
Rousselier, and the compaction of powders and foams by the elliptic criterion of Abouaf,
with the Armstrong-Frederick backstresses of the matrix for GTN and the elliptic criterion
(`kinematic: [{C: 30000, D: 200}]`: the potential of σ - X), in small and finite strains,
on morphoGen's kernels (CPU); the curves give the porosity `f`:

```yaml
homogenization:
  materials:
    0:
      young: 200000
      poisson: 0.3
      porous_plasticity: {criterion: gurson, yield: 300, porosity: 0.01, q1: 1.5, q2: 1.0,
                          fc: 0.05, fF: 0.2, hardening: 1000}
    1: {zmat: {file: powder.mat, porosity: 0.4}}   # ***behavior porous_plastic ...
```

### Graphical interface and figures

`morphogen gui` starts a local web server (standard library only, address
`127.0.0.1`) and opens the interface in the browser; `morphogen desktop` (or the program
`morphogen-desktop`, `pip install ".[desktop]"`) opens the same interface as a
stand-alone application, in a window of its own whose server is private to it (a free
port of the local machine): its menu holds the workspace (Open…, Save…), the folder of
the session and the help; the folders of the workspaces and the paths that the forms
take (**Browse…**) are chosen in the dialogs of the system, the files of the results are
saved where one chooses or opened by their applications, the folder of a result is
shown in the file manager (**Show its folder**), and closing the window asks first when
the workspace holds items not saved. The forms of every model open
filled with its example configuration (its template: parameters and domain), so that
any model runs as it opens. Its menus follow the work in four sections, each with its
tabs: **Create** (Generate, Topology optimization, Inverse design, Tile assemblies),
**Characterize** (Statistics), **Simulate** (Linear homogenization, Mechanics, Learning,
Fracture, Contact, Nonlocal damage, Stability, Physics, Couplings, Explicit
dynamics) and **Export & compare**
(Export, Solvers & formats, Comparisons, Ashby charts). A tab lists its tools in a menu
on its left, under the headings of what they do — Mechanics: *Loading paths* (of a
morphology, its macroscopic law, the NTFA), *Behaviours of materials* (material point,
identification), *Linear elasticity* (tensors, frames of beams); Statistics:
*Morphology*, *Representative volume*, *Reconstruction*; Fracture: *Cracks*, *Cohesive
zones*, *Phase field*; Solvers & formats: *AMITEX_FFTP*, *Abaqus and meshes*,
*Behaviours of materials* — and Generate lists its models by family, with a filter. The
**search** of the header (`/`) finds a tool, a model or an option of their forms (a
behaviour, a strain energy, a file format) and opens it with the option set: "arruda"
opens the loading path of a morphology with the energy of Arruda and Boyce chosen,
"finite strains" its kinematics, "stl" the surfaces of Export. The address of the page
names what is shown (`#mechanics/path`, `#generate/voronoi`): a link to keep or to
give. **?** opens the map of the interface (every tab and every tool, as links),
`Ctrl + Enter` runs the form shown (its button stays in sight under the long forms), and
the colours follow the system or are set light or dark. Each morphology, simulation and optimized design is
kept in the **workspace** of the session (a strip of its items under the menus), from
which any tab takes its input, before or after another as needed: an item is sent on by
the buttons of its usual next steps or the menu of the other tools (by section), which
set the forms to take it (e.g. a morphology as the specimen of a crack), and the item
stays when another tool of the tab is chosen; a click on an item shows it, sends it on,
renames or removes it. Each run of a tool joins the workspace too (R), so that a run with other
parameters keeps the first: shown again, its form filled with its parameters, to change
them and run again. The long forms stand in boxes (specimen, material, loading, solver,
output; the details seldom changed folded until opened). A long computation is stopped
by **Stop** beside its progress, at its next step (an iteration of its solver, a
realization, a step it reports): a tool or a simulation then keeps nothing, an
optimization keeps the design of its last iteration. The behaviour of a material
defined at a material point (calibrated on the points of a test), identified, learned
from data or imported from a
Z-set file joins the workspace too (B), with its own elasticity: the loading paths of
morphologies, the NTFA, the bulk of a cracked plate or one phase of a cracked
morphology, the inputs of AMITEX_FFTP and Abaqus and the MFront sources take it, so
that it is given once. A tool runs on several items one after another (**Run on several
items** below its form: the same parameters for each morphology or behaviour checked),
and the curves of these runs are then compared in one click.

* **Generate**: any model, chosen in a menu by family (random sets and particles, fibres
   and layers, tessellations and foams, lattices and TPMS, bio-inspired, growth and
   patterns, assemblies), with a form built from its parameters; sections of its
   level set (anti-aliased boundaries; of the implicit functions of its phases when it
   has several: coatings, inclusions, dead leaves, osteons, cells with phases), computed
   at up to 2048 pixels per edge without
   a voxel image of the volume, measures of its level set against theory; for the
   models grown by a process (automata, reaction–diffusion, nucleation and growth,
   aggregation), the animation of their formation with the share of the cells changed
   at each step; for the network models with `frame_stiffness`, the stiffness of their
   frame of beams in a table of its own and in the card of the morphology;
* **Statistics**: the size of the representative volume element of the model of a
  morphology, by the ensemble or the window method (its windows also shifted around a
  periodic realization), for its volume fraction or apparent moduli (the variance fitted
  as `K V^-γ`, the integral range, the volumes for chosen errors and numbers of
  realizations, the tables and figures of the report); its morphological measures
  against the theory of its model; its correlation functions and
  distributions (covariance, lineal path, chords, granulometry, pore sizes, Choquet
  capacities) with the closed forms of its model; its stereology (Saltykov's unfolding);
  the RVE fit of a pasted table of results of any origin; a morphology reconstructed
  from its two-point function (Yeong & Torquato); the percolation threshold of its model
  against a parameter (the volume fraction), with the curve of the probabilities against
  the fraction, the threshold of its elastic moduli (its rigidity), and the critical
  exponent of the conductivity or of the elastic moduli at the threshold;
* **Simulate**: a morphology of the workspace, its conforming mesh of its level set
  (finite elements) or its voxels (FFT, on an NVIDIA GPU when there is one), the
  materials of its phases, the apparent shear or bulk modulus with its fields;
* **Mechanics**: loading paths of a morphology (elastic, von Mises elasto-viscoplastic,
  Méric–Cailletaud crystal plasticity, GTN, Rousselier and elliptic porous plasticity,
  viscoelasticity by Prony series, hyperelastic energies, isotropic damage with a
  fracture energy, Z-set files, MFront laws; small or finite strains; FFT, on the CPU or an NVIDIA GPU,
  or finite elements; strains, stresses or both over time) with their curves, fields
  and videos; the NTFA reduced
  model trained on paths and compared with the full field; a material point under a
  path, with the points of a test, its behaviour kept in the workspace for the other
  tools; the identification of a von Mises, Hill or porous behaviour (yield stress,
  hardenings, viscosity, elastic constants, Hill's coefficients; Gurson's q1, q2, the
  initial porosity...) on the curves of
  runs followed together — the macroscopic law of a morphology, a tension and a cycle in
  shear to tell its isotropic hardening from its kinematic one, the Gurson model of a
  porous morphology under several triaxialities — or on the points of a test, as Zopt
  does, from the values of the form or a behaviour of the workspace, with the stiffness
  and compliance of the elasticity identified; the linear tensors (stiffness with its
  compliance, thermal expansion,
  conductivity) with their bounds, by FFT, by finite elements on a conforming mesh or
  by extended finite elements (XFEM) on a grid cut by the level set; the stiffness of a fibre network, an open foam or a
  random lattice of the workspace as the frame of beams of its framework (Euler–Bernoulli
  or Timoshenko beams, rigid joints, the modulus and Poisson's ratio of the fibres): its
  stiffness and compliance, engineering constants and directional Young's modulus, the energy of
  uniaxial stresses split into stretching, bending, torsion and shear, its ratio to the
  affine estimate, the non-affinity, a drawing of its beams coloured by their axial
  forces (the force chains) or their stretching or bending energy under a uniaxial
  stress, and Ban et al.'s softening of non-identical beams, with replicas of the
  stiffness of the beams against it (their table and figure, in the background with
  their progress);
* **Learning**: a behaviour learned from data — the responses of a behaviour at a
  material point (those of Mechanics, a material of the configurations, a behaviour of
  the workspace) or the homogenized responses of a morphology (FFT on its voxels) along
  random strain paths, a perceptron, a learned convex potential, a recurrent network or
  a generalized standard material (its internal variables, a viscous dissipation; the
  strain that every model sees in finite strains) trained on them in the background (the losses of each epoch reported, **Stop**
  between two of them), with the fields of the paths, the model and the training of
  `morphogen learn`; its learning curves, its errors on the validation paths, its tangent at the
  natural state (stiffness and compliance) against the elasticity of the reference, and its law
  along a validation path against the reference (turned by an orientation for a crystal
  learned in its axes); the model saved (`learned.npz`, with its data and the
  configuration that learns it again) and the learned law kept in the workspace (B),
  which the loading paths of morphologies give to a phase;
* **Fracture**: the stress intensity factors of a crack (edge, through or penny; K_I,
  K_II, K_III by interaction integrals, G by G-θ, J) in a plate or a morphology, in small
  or finite strains; its growth in a plate or through a morphology of the workspace (E
  and ν or void for each phase, conforming interfaces; kinking, toughness, ductile by
  J_tip or the porosity of GTN points, the R-curve, the phase at each tip, the path over
  the phases, a video; the toughness of each phase and of their interfaces, the
  deflections along weak interfaces marked); cohesive zones (a ligament, the interfaces of a
  double cantilever beam); the growth of a crack through a polycrystal; the phase-field
  fracture of a morphology, in small or finite strains, whose cracks start and merge in its phases along a loading
  path (the curve, the strength, the work of fracture, the damage field); the growth of a
  crack front in 3D (an elliptical crack inclined to the tension inside a block: K_I,
  K_II, K_III along its curved front against Irwin and Kassir & Sih, each point of the
  front advancing and kinking on its own);
* **Stability**: the post-bifurcated patterns of a lattice frame (honeycomb,
  triangular, square, kagome; cubic, bcc, octet, Kelvin, diamond and pyrochlore space
  frames) or of a
  morphology (its mirrored conforming mesh when its cell is symmetric, else its voxels)
  under a macroscopic strain, with their irreps, critical loads, wave vectors and the
  Bloch-wave stability that tells the cell needed; the bifurcation along a nonlinear
  path in finite strains (the pattern transformation of a square array of holes) and the
  post-bifurcated branch; the Bloch-wave band structure of a morphology and its gaps;
* **Nonlocal damage**: the microfoam model of metallic foams (Dillard, Forest & Ienny),
  micromorphic ductile damage with the stress triaxiality (Rice & Tracey, Johnson &
  Cook, Lemaitre) and micromorphic damage of quasi-brittle solids, on notched plates,
  bars or morphologies of the workspace: force–displacement curves, the bands and their
  width set by the internal length, videos of the damage;
* **Physics**: the other physics of a morphology of the workspace, on its voxels — its
  transport properties (thermal and electrical conductivities, permittivity, permeability,
  diffusivity) and their tensor between the bounds of Wiener and of Hashin and Shtrikman;
  the flow through its pores (Darcy's permeability), the acoustic parameters of the rigid
  frame and the absorption curve of a layer of it on a rigid wall; the storage and loss
  moduli and the loss factor of a composite with a viscoelastic phase over a range of
  frequencies;
* **Couplings**: transient diffusion into a morphology (uptake curve, fields, the
  effective diffusivity), the thermal oxidation of an epoxy plate (Ernault et al.) and
  of a morphology in its volume (the oxidized layer around impermeable phases), and
  hygromorph multilayers bending and twisting as moisture enters them, their layers
  homogenized from morphologies of the workspace;
* **Explicit dynamics**: the crushing, tension, shear or impact of a morphology run by
  OpenRadioss (its voxels or a conforming mesh; its phases elastic, von Mises
  elastoplastic or Johnson–Cook, with failure; the self contact of its outer surface;
  engineering units: MPa, kg/m³, m/s, 1/s, ms), its nominal stress–strain curves at both
  faces, energies, peak and plateau stresses, densification strain and absorbed energy,
  its fields over time in 3D and a video; without OpenRadioss here, its inputs and the
  scripts to run them elsewhere; the results of a run made elsewhere read back; the
  installation found, or a release installed from GitHub. Its curves go to Comparisons
  with the loading paths (E33 and S33 for a compression along z);
* **Export**: the files of an item of the workspace with an archive of them all — of a
   morphology, its geometry, sections, voxel images of its level set and phases (up to
   2048 voxels per edge, memory permitting), its surface (STL) and conforming mesh (for
   a morphology of several phases: the implicit function and the surface of each phase,
   `level_set_phase1.vti`, `surface_phase1.stl`..., and the mesh conforming to all their
   interfaces) and a
   video (sections of its level set sweeping it, orthoslices, a turntable); of a
   simulation, a view of its fields (field, colormap, magnification, cut, range) and a
   video (the field on the deformed mesh under a growing or oscillating load, the
   colours following the load; a turntable, a sweep), its tables, fields, mesh and
   configuration (to rerun it with `morphogen run`), drawn without solving again;
* **Design**: the rule of a cellular automaton whose structures match a morphology of
  the workspace (a genetic search over its birth and survival sets), and the rigidity of
  strut lattices as their struts are removed (the mechanisms, states of self-stress and
  floppy strains of Maxwell and Calladine: stretching- or bending-dominated);
* **Optimize**: topology optimization, drawn live, from a void ball, noise, or a
   morphology or simulation of the workspace (its densities, its solid material); the
   design joins the workspace as a morphology (its level set: the densities about 1/2),
   to simulate on a conforming mesh or to export;
* **Solvers & formats**: the inputs of AMITEX_FFTP (and the reading of its results) and
  of Abaqus (materials, orientations, MFront laws, periodic equations, loading steps) of
  a morphology; its meshes and images in the formats of other codes (Gmsh, Medit,
  FEniCSx, Neper, DAMASK, TIFF, MetaImage; the level set as an image of floats, one per
  phase when it has several); the MFront sources of morphoGen's laws; the
  import of a Z-set material file, and the behaviours of the workspace written as such
  files (a behaviour identified on a morphology, for Z-set);
* **Comparisons**: the curves of several runs of the workspace on one figure (loading
  paths of morphologies, material points; a strain component or the time against a
  stress, the plastic strain or the porosity, true measures in finite strains), with the
  points of a test and the gap of each run to them, their final and extreme values;
* **Tile assemblies**: exact number of the compatible assemblies of a tile set in a box
  (combinatorial metamaterials), and one assembly drawn by wave function collapse;
* **Ashby chart**: the records of a database, filtered, as a normalized chart with a
  table view (light and dark themes).

The workspace is saved to a folder and opened again in another session (**Save…** and
**Open…** beside it): the morphologies, simulations and results with their files.

Every item of the workspace is also drawn in 3D in the page (**View in 3D** in the menu
of its chip, and below the results of Generate and Linear homogenization), with WebGL
and no library, so that the page works offline: turned by dragging, moved by
shift-dragging, zoomed by the wheel, cut by a section plane along x, y or z, the box and
each phase shown or hidden. A morphology is drawn by the interfaces of its phases (the
marching cubes of its level set, the exact facets of polycrystals, the faces of the
voxels of cells and grains), a simulation by the faces of its mesh or voxels coloured by
a field (colour map and scale), on the mesh deformed by its magnified displacement; the
faces of the box and the section show the phases or the field inside, as the sections
do. The surfaces (`morphogen.viz.surfaces`, `GET /api/surface`) hold at most 300 000
triangles: coarser grids, or simplified meshes, keep them within.

The tabs of tools are built from a registry (`morphogen.gui.tools`): each tool declares
its form (typed fields, fields shown on conditions, items of the workspace, the
materials of their phases) and runs as a job whose progress the page follows; its
values, tables, figures, videos and files come back with an archive, and the
morphologies it makes join the workspace. A module of tools added to the package
appears in the interface by itself.

`morphogen view` draws the same figures from the command line: sections, `--tile 3`
(periodic tiling), `--surface` (3D view), `--labels` (grains), `--history`,
`--kinetics` (Avrami), and `--animate formation.gif` for the formation of a structure
described by a configuration.

`morphogen fit-rule image.vti -o rule.yaml` finds the automaton rule, and the density
of its initial noise, whose structures match the volume fraction, interface density and
correlation function of a two-phase image, and writes the configuration that generates
them; `--tiling hexagonal` (or any other lattice: `triangular`, `penrose`,
`ammann_beenker`, `bcc`, `fcc`, `icosahedral`) searches the rules of the automaton on
the cells of that tiling.

### Explicit dynamics with OpenRadioss

```python
import morphogen as mg
from morphogen import dynamics

lattice = mg.model_from_config({"type": "tpms", "surface": "gyroid", "volume_fraction": 0.2,
                                "cell_size": 5.0})  # an aluminium gyroid, its lengths in mm
micro = lattice.generate(mg.Domain.cube(10.0), seed=1)
materials = {0: "void", 1: {"young": 70e3, "poisson": 0.33, "density": 2.7e-9,  # MPa, Mg/mm³
                            "plasticity": {"yield": 150.0, "hardening": 600.0},
                            "failure_strain": 0.4}}
test = dynamics.explicit_test(micro, materials,
                              {"kind": "compression", "strain": 0.6, "strain_rate": 500.0},
                              "gyroid_crush", resolution=32)   # units: Mg, mm, s by default
values = dynamics.summary(test.result)  # peak and plateau stresses (MPa), densification...
```

The loadings are `compression`, `tension` and `shear` (the lower face held, the upper
one driven at the nominal strain rate after a smooth start, laterally `fixed` or
`free`) and `impact` (the morphology thrown at a `velocity` against a rigid wall), with
the self contact of the outer surface by default. The values are in the consistent unit
system of `units` (`dynamics.UnitSystem`: Mg, mm, s by default, i.e. MPa, Mg/mm³, mm/s, N,
mJ; `length_scale` converts the lengths of the morphology). The time step of an explicit
run is that of the smallest element (`dynamics.critical_time_step`; the manifest of the
inputs gives the number of cycles before the run): a coarser mesh, a higher rate or a
minimum time step (`time_step`, mass scaling) shortens it. The command line does the
same from a geometry or an image, and reads back a run made elsewhere:

```bash
morphogen radioss install                       # the latest release, from GitHub
morphogen radioss export gyroid.json --materials aluminium.yaml --voxels 32 \
    --loading compression --strain 0.6 --rate 500 -o crush
morphogen radioss run crush --threads 8         # Starter, Engine, conversions, summary
morphogen radioss read crush --frames-vtu       # curves.csv, energies.csv, frames/*.vtu
```

The inputs follow the fields of the Radioss Reference Guide for the input version 2022
(`/BEGIN`, `/NODE`, `/BRICK`, `/TETRA4`, `/TETRA10`, `/PART`, `/PROP/SOLID`, `/MAT/LAW1`,
`/MAT/LAW2`, `/MAT/LAW36`, `/FUNCT`, `/GRNOD`, `/BCS`, `/IMPVEL`, `/INIVEL`, `/RWALL`,
`/SURF/PART/EXT`, `/INTER/TYPE24`, `/TH/NODE`, `/TH/PART`, `/TH/RWALL`; `/RUN`, `/TFILE`,
`/ANIM`, `/DT/NODA/CST`); the forces of the faces come from the reactions of their nodes
in the time history (`th_to_csv`) or in the animations (`anim_to_vtk`), the forces on a
rigid wall from the change of the momentum.

### Homogenization in a configuration

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

`method: fft` (with a `voxels` section and `boundary_conditions: [periodic]`) uses the
FFT solver instead, on an NVIDIA GPU with `device: gpu`. Crystals are given as `{cubic: [C11, C12, C44]}`,
`{hexagonal: [C11, C12, C13, C33, C44]}` or `{stiffness: <6 x 6>}`; tessellations with
`crystal_orientations: true` are meshed grain by grain (`mesh: {size: 0.08,
min_feature: 0.02}`) and their crystals turned by the orientations of the grains.
Templates: `homogenization_spheres` (the four conditions on a composite),
`homogenization_gyroid` (stiffness tensor of a lattice, local fields),
`homogenization_voronoi` (FFT), `homogenization_polycrystal` (copper, finite elements
on grains), `homogenization_polycrystal_inclusions` (crystals, grain boundaries and
inclusions, FFT), `homogenization_mixed_loading` (thermo-elastic loadings mixing strains
and stresses), `homogenization_viscoplastic` (tension, relaxation and unloading of a
particle-reinforced viscoplastic alloy), `homogenization_crystal_plasticity` (tension of
a polycrystal to 10 % in finite strains, Méric–Cailletaud crystal plasticity),
`homogenization_crystal_plasticity_fft` (the same on 24³ voxels of 20 grains, to 5 %,
FFT), `homogenization_twinning` (the twins of a textured magnesium, pulled and
compressed), `homogenization_detwinning` (the same pulled and brought back: twins that
shear back, lattices that turn), `homogenization_cleavage` (a polycrystal of iron that
cleaves through its grains), `homogenization_crack_closure` (the same pushed back: its
crack closes), `homogenization_slip_cleavage` (a polycrystal that yields first: the slip
of its grains nucleates their cleavage), `homogenization_cleavage_band` (planes regularized
by their microdamage: a crack across a band of a given width, whatever the grid), `homogenization_grain_fracture` (cohesive grain boundaries and cleavage planes:
cracks between the grains or through them), `homogenization_beremin` (the probability of
cleavage of a ferritic steel, by the Weibull stress of Beremin), `rve_poisson_fibers_fe` (RVE study of the 2014 paper), `rve_crystal_plasticity_fft`
(RVE sizes of the elastic and viscoplastic responses of polycrystals). The other physics:
`homogenization_transport` (thermal and electrical conductivities, permittivity and
diffusivity on the same voxels), `homogenization_piezoelectric` (a strong coupling),
`homogenization_viscoelastic` (storage and loss moduli at `frequencies`, relaxation in
time), `homogenization_poroelastic` (Biot's law, `fluid_bulk_modulus`),
`homogenization_xfem` (`method: xfem`: the level set of the model cuts the grid of the
`voxels` section), `homogenization_acoustic_foam` (Darcy's permeability and the acoustic
parameters of an open-cell foam) and `rve_multiphysics` (the RVE sizes of several physics
at once);
`homogenization.fracture` (`{toughness: ..., length: ...}`) makes the loading paths break
the voxels by the phase-field method. Nested and
bio-inspired templates: `cox_nanocomposite`, `coated_spheres`, `hierarchical_lattice`,
`menger_sponge`, `nacre`, `bouligand`, `wood`, `osteons`, `trabecular_bone`,
`vascular_tree`. Fibres and lattices: `hardcore_fibers`, `curved_fibers`,
`random_walk_fibers` (bending fibres packed by forces to 0.35), `contact_fibers`
(the same drawn into contact, Keilmann et al. 2026), `cross_linked_fibers`
(entangled carbon fibres bonded by epoxy, Chatti et al.), `delaunay_network` (a
fibre network on the edges of a Delaunay triangulation, Ban et al.), `graded_lattice`
(an octet lattice graded and turned, with its STL surface for additive manufacturing).
Dead leaves and Torquato: `sequential_dead_leaves`, `transparent_dead_leaves`,
`dense_packing`, `penetrable_spheres`, `reconstruction`, `hyperuniform_cells`.

From the command line, for one sample (a geometry `.json`, a mesh `.vtu`/`.mesh`, or an
image for the FFT solver):

```bash
morphogen homogenize spheres.json --size 0.5 --material 0=void 1=young:210e3,poisson:0.3 \
    --bc kubc mixed --property bulk_modulus shear_modulus --fields fields/
# a loading path with the plasticity of the materials (YAML), curves in curves/
morphogen homogenize composite.vtu --materials alloy.yaml --property stiffness \
    --path "tension: t=0,20,120 n=10,10 E11=0,0.02,0.02" --curves curves/
# the same in finite strains: the curves add the true strains LE and stresses TS
morphogen homogenize polycrystal.vtu --materials nickel.yaml --periodic --finite \
    --path "tension: t=0,100 n=40 E11=0.1" --curves curves/ --fields fields/
# on a voxel image: the FFT solver (fields as .vti)
morphogen homogenize composite.vti --materials alloy.yaml \
    --path "tension: t=0,20 n=10 E11=0,0.02" --curves curves/ --fields fields/
# the same on an NVIDIA GPU (CuPy, the extra [gpu])
morphogen homogenize composite.vti --materials alloy.yaml --device gpu \
    --path "tension: t=0,20 n=10 E11=0,0.02" --curves curves/
# the same image for AMITEX_FFTP: its inputs and the MPI script, then its results
morphogen amitex export composite.vti --materials alloy.yaml -o amitex/ \
    --path "tension: t=0,20 n=10 E11=0,0.02"   # the plastic phases as MFront laws
sh amitex/composite_run.sh            # where AMITEX_FFTP and MFront are installed (Linux)
morphogen amitex read amitex/ --json  # the curves of the runs
# the elasticity of the phases only: the apparent stiffness and moduli (a CSV row)
morphogen amitex export composite.vti --materials alloy.yaml --elastic-only -o elastic/
sh elastic/composite_run.sh && morphogen amitex read elastic/ -o stiffness.csv
# the MFront sources of the plastic phases, for another code (umat: Cast3M, abaqus...)
morphogen mfront --materials alloy.yaml -o laws/
cd laws && mfront --obuild --interface=abaqus alloyPhase0.mfront
```

### RVE study

```bash
morphogen template rve_poisson_fibers_fe -o study.yaml   # or rve_poisson_fibers (morphology)
morphogen run study.yaml
```

The run writes:

* `results.csv`: one row per realization (measures and apparent properties);
* `homogenization.csv`: one row per realization, property and boundary condition;
* `rve_report.md`, `rve_report.tex` and `rve_report.json`: for each property and
  condition, the statistics per size, K, γ (free and imposed), R², the RVE sizes for
  each (ε_rel, n) with confidence intervals, the discrepancy between conditions, the
  isotropy criteria and the bounds;
* when the homogenization computes the `stiffness`, for each condition and size the mean
  stiffness C̄_ij and its compliance S_ij = C̄⁻¹ (the inverse of the mean C, not the mean
  of the inverses), each component ± 2 standard deviations, and the RVE size of each
  component (Voigt order 11 22 33 23 31 12, engineering shears): in the report, in
  `rve_stiffness.csv` (one row per condition and size: `C11` ... `C66`, `S11` ... `S66`,
  `C11_std` ...) and in `rve_stiffness_<bc>.png`;
* figures per property.

To analyse apparent properties computed elsewhere (CSV with a `volume` column, one
row per realization; `--group-by` splits the rows by boundary condition):

```bash
morphogen fit fe_results.csv --property k_app mu_app --group-by bc --gamma 0.6667 \
    --eps 0.01 0.05 --n 1 10 -o report/
```

Nonlinear responses are analysed in the same way: `rve.properties` may name quantities
of the curves of the loading paths, `<path>.<quantity>` at the end of the path or
`<path>.<quantity>@<time>` at the end of one of its steps (`tension.S11@3`: the stress
of the path `tension` at the time 3; also `TS11` in finite strains, `p`, or the mean of
a phase, `S11_1`), followed by a boundary condition to keep only that one
(`tension.S11@3.periodic`). The template `rve_crystal_plasticity_fft`, after Yang et al.
(2019), compares the RVE sizes of the elastic moduli of a polycrystal with those of its
stress at 0.1, 0.3 and 1 % of a tension (crystal plasticity, nonlinear FFT solver):

```bash
morphogen template rve_crystal_plasticity_fft -o study.yaml && morphogen run study.yaml
morphogen fit out/rve_crystal_plasticity_fft --property tension.S11@3   # again, from the files
```

### Parallel and cluster runs

Realizations are independent, and the seed of each one depends only on its place in
the study: results do not depend on how a run is split.

```bash
morphogen run study.yaml -j 8                  # 8 realizations at a time on this machine
morphogen submit study.yaml --scheduler slurm --tasks 32 -j 4 --time 02:00:00 \
    --memory 16G --setup "module load python" --submit
morphogen status out/study                     # realizations done and missing
morphogen merge out/study                      # done by the merging job of 'submit'
mpirun -n 64 morphogen run study.yaml --mpi    # or one task per MPI rank (mpi4py)
```

`submit` writes a job array and a merging job for SLURM, PBS Pro, Torque, SGE, OAR or
LSF, and with `--submit` submits both, the merging job waiting for the array. Each task
runs `morphogen run --task-id auto` (the index of the array, read from the scheduler)
with `--resume`: a task resubmitted after a time limit only runs the realizations
without a record. A script that runs realizations in processes (`jobs > 1` in Python)
must guard its entry point with `if __name__ == "__main__":`.

<!-- section: performance -->
## Performance

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
| Voronoi mosaic, contrast 20 | 64³ voxels (FFT) | — | 3–3.5 s, 37–38 iterations |

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
precision of this laptop GPU ([docs/theory.md](docs/theory.md), §35).

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

<!-- section: documentation -->
## Documentation

* [docs/tutorials.md](docs/tutorials.md): nine tutorials, the first eight in snippets
  that run in seconds to a minute and that the tests run —
  [a whole study](docs/tutorial.md) (generation, characterization, apparent properties
  and RVE size, a loading path, exports, a learned behaviour),
  [how large is a representative volume?](docs/tutorial_rve.md) (the statistics of the
  RVE on spheres and on Poisson fibres, windows, a study from a configuration),
  [a rubber, from its tests to its tearing](docs/tutorial_rubber.md) (finite strains),
  [exact geometry, from implicit functions to meshes](docs/tutorial_geometry.md),
  [isogeometric analysis, from a patch to an auxetic sheet](docs/tutorial_iga.md),
  [contact and damage](docs/tutorial_contact.md) (a block crushed and indented, the
  damage laws, a composite whose matrix fails),
  [how a microstructure breaks](docs/tutorial_fracture.md) (interfaces that debond, the
  fatigue of a pore, twins, cleavage and its probability),
  [from an orientation map to a textured polycrystal](docs/tutorial_texture.md) (an
  EBSD map, its grains and its texture, the lattices that a tension turns) and
  [a study in the graphical interface](docs/tutorial_interface.md); their figures are
  written by `docs/tutorials/make_figures.py` (which runs the snippets) and
  `docs/tutorials/make_screenshots.py` (which drives the interface).
* [docs/howto.md](docs/howto.md): recipes, short answers to "how do I..." in snippets
  run by the tests; [docs/faq.md](docs/faq.md): questions and troubles (installing,
  units and phases, the messages of the solvers, the files of a run);
  [docs/glossary.md](docs/glossary.md): the words that have a precise meaning here.
* [docs/gallery.md](docs/gallery.md): the gallery of examples, a card per template
  (written with its images by `docs/gallery/make_gallery.py`).
* [docs/physics.md](docs/physics.md): a guide to the physics for researchers in the
  mechanics and physics of materials — every problem of the homogenization module
  (elasticity, transport, weak and strong couplings, harmonic regimes and waves,
  nonlinear behaviours, fracture), the solver for each, the dependence on the mesh and
  the size of the representative volume, in snippets run by the tests.
* [docs/specification.md](docs/specification.md): specification (cahier des charges) —
  objectives, requirements, technical choices, roadmap, decisions.
* [docs/architecture.md](docs/architecture.md): modules, data model, conventions,
  configuration schema, file formats, tests.
* [docs/theory.md](docs/theory.md): formulas implemented and validated.
* [docs/laser.md](docs/laser.md): architectured sheets by localized laser treatment —
  the choices of a study, the chain of models, its checks and first results.
* [docs/stability.md](docs/stability.md): the public interface and its versioning;
  [docs/release.md](docs/release.md): the steps of a release; [CITATION.cff](CITATION.cff)
  and [docs/citing.md](docs/citing.md): how to cite morphoGen.
* [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).
* The documentation site gathers these pages, this README and the API reference written
  from the docstrings, behind a landing page: `python -m pip install ".[docs]"`, then
  `python -m sphinx -b html docs site` (open `site/index.html`). Its pages copy their
  code blocks at a click, and its gallery filters its cards by words. The continuous
  integration builds it too and keeps it as the artifact `site` of each run.

<!-- section: development -->
## Development

```bash
python -m pip install -e ".[dev]"
python -m pip install vtk gmsh   # optional: independent readers of the exported files
python -m pytest -n auto         # 1900+ tests in parallel (pytest-xdist): about 25 minutes
                                 # on 4 cores
python -m pytest -n auto -m "not slow"   # the fast tier: about 10 minutes on 4 cores
python tests/test_public_api.py  # the list of the public interface written again,
                                 # after a change of it made on purpose
ruff check src tests examples docs && ruff format --check src tests examples docs
python -m sphinx -b html -W docs site   # the documentation site, without warnings
tools/ci_local.sh                # the CI jobs of this machine's system, locally
CI_LOCAL_PACKAGES="vtk gmsh" tools/ci_local.sh   # the same with the VTK and Gmsh checks
```

The statistical tests compare every model with its theory, using fixed seeds and
tolerances derived from that theory. The homogenization tests check exact solutions
(homogeneous media, laminates, the bulk modulus of cubic polycrystals), the ordering of
the boundary conditions and the agreement with scikit-fem. The phase-3 tests check automata against known patterns
(gliders, blinkers), growth against the Avrami law and exact distances, exact counts
against brute force, rigidity against Maxwell–Calladine, sensitivities against finite
differences and optimized cells against the Hashin–Shtrikman bounds. Optional checks
with VTK and Gmsh (their Python packages: VTK reads the `.vti` files, Gmsh the `.msh`
meshes and meshes the `.geo` scripts) run when those packages are installed; the GitHub
workflow does not install them. The GPU tests compare the solutions of both devices
where CuPy finds a GPU, and the MFront tests morphoGen's behaviours with their MFront
twins where MGIS and MFront are installed (in WSL on the development machine).
Continuous integration runs the tests on Linux, Windows and macOS
(`.github/workflows/ci.yml`); `tools/ci_local.sh` runs the jobs of the development
machine's system (an environment per Python version, the tests against the built wheel,
the command-line smoke test, the documentation site), for when GitHub Actions is not
available. The Python versions run at once, each on its share of the cores
(`CI_LOCAL_WORKERS` test processes per version); the slowest tests (marker `slow`,
two thirds of the time) run under the last version only (`CI_LOCAL_SLOW`), which gets
three times the cores of each other version, and the GitHub workflow runs them in its
job Ubuntu with Python 3.14. An environment is kept from one run to the next while
`pyproject.toml` does not change (`CI_LOCAL_FRESH=1` makes new ones). It runs in Git
Bash on Windows. With uv it finds or installs
Python 3.12 and 3.14; without uv it uses venv and pip, and takes the versions
(`pythonX.Y` on the PATH or `py -X.Y`) or the paths of the interpreters:
`tools/ci_local.sh /path/to/python3.12 /path/to/python3.14`. Changes go through pull
requests to `main`.

<!-- section: references -->
## References

* Kanit, T., Forest, S., Galliet, I., Mounoury, V., Jeulin, D. (2003). *Int. J. Solids
  Struct.* 40, 3647–3679.
* Dirrenberger, J., Forest, S., Jeulin, D. (2014). Towards gigantic RVE sizes for 3D
  stochastic fibrous networks. *Int. J. Solids Struct.* 51, 359–376.
* Yang, S., Dirrenberger, J., Monteiro, E., Ranc, N. (2019). *Int. J. Solids Struct.*
  158, 210–219.
* Bironeau, A., Dirrenberger, J., Sollogoub, C., Miquelard-Garnier, G., Roland, S.
  (2016). *J. Microscopy* 264, 48–58.
* Coulais, C., Teomy, E., de Reus, K., Shokef, Y., van Hecke, M. (2016). Combinatorial
  design of textured mechanical metamaterials. *Nature* 535, 529–532.
* Hashin, Z., Shtrikman, S. (1962). A variational approach to the theory of the elastic
  behaviour of polycrystals. *J. Mech. Phys. Solids* 10, 343–352.
* Jeulin, D. (1997). Dead leaves models: from space tessellation to random functions.
  In *Advances in Theory and Applications of Random Sets*, World Scientific, 137–156.
* Sigmund, O. (1994). Materials with prescribed constitutive parameters: an inverse
  homogenization problem. *Int. J. Solids Struct.* 31, 2313–2329.
* Tovar, A., Patel, N. M., Niebur, G. L., Sen, M., Renaud, J. E. (2006). Topology
  optimization using a hybrid cellular automaton method with local control rules.
  *J. Mech. Des.* 128, 1205–1216.

The formulas and the other references are in [docs/theory.md](docs/theory.md).

<!-- section: licence -->
## Licence

Copyright © 2026 Justin Dirrenberger.

morphoGen is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version. It is distributed in
the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See [LICENSE](https://github.com/jdirr/morphoGen/blob/main/LICENSE).
