# Change log

## Unreleased

* The models as editors show them: the decorators `register` and `register_kind` typed as
  returning the class they are given, where the type of `Model` made editors that follow
  it (Spyder 6, through Jedi 0.20) show `Model()` and its documentation for every model,
  and guess a signature from the first parenthesis of its docstring; the messages of two
  alternative parameters (`n_points` and `intensity`, `radius` and `volume_fraction`...)
  say whether both or neither were given, with the values given; the docstring of
  `DelaunayNetwork` names its two pairs.
* Microgen credited as the work of its authors: the link to its repository, its team and the
  DOI of its record in the pages on the models, the other codes, the installation, the
  citation and the theory notes, and Microgen among the references to cite; the model
  `microgen` presented as the shapes of Microgen, which morphoGen calls without copying any
  of it, beside its own 43 models (the README, the user guide, the API reference, the JOSS
  draft).
* Mathematical morphology on the implicit geometry of the realizations (`morphogen.morphology`,
  [docs/theory.md](docs/theory.md) §104, §105): the measures ask the realization itself, not a
  voxel image of it. The four Minkowski functionals of a union of spheres — volume,
  surface, integral of mean curvature, Euler characteristic — exactly, from the power
  diagram of the spheres (the pyramids of each cell over its faces cut by its ball, the
  solid angles of Van Oosterom and Strackee, the circles where two spheres meet in the
  common faces of their cells) and the dual complex of their regular triangulation (Qhull),
  on the torus of a periodic box (the balls repeated within a margin checked by the powers
  of the orthocentres) or in space (`union_of_balls`, `minkowski_functionals`); the three of
  the plane sections of a union of spheres, unions of discs (`union_of_discs`,
  `section_functionals`). The probes of any structure through the implicit functions of its
  phases (`morphology.measures`): points (the fraction), pairs (the covariance along a
  direction or over all), segments marched by sphere tracing (the Choquet capacity and the
  linear contact distribution, every length at once), balls through the distance (the
  capacity and the spherical contact distribution), lines with their crossings located by
  bisection (the chords of each phase, weighted for those cut by the ends, and the specific
  surface by Crofton's formula), with the standard deviations of their estimators; the
  exact dilation of unions of spheres and segments; `compare_with_theory`, the measures of a
  realization beside the closed forms of its model. Tested (`tests/test_morphology_exact.py`,
  `tests/test_morphology_probes.py`): one and two balls and discs in closed form (10⁻¹⁰),
  the topology of rings (χ = 0), shells (χ = 2) and annuli, `dV/dt = S` and `dA/dt = L`
  along the radii (10⁻⁶), Monte Carlo volumes and areas; eight realizations of Boolean
  spheres within four standard errors of Miles' formulae for the four densities, the Euler
  density included (−0.0536 ± 0.0021 for −0.0525), and their sections within the
  stereological identities `L_A = (π/4) S_V` and `χ_A = M_V / 2π`; rays, crossings and chords
  of a ball (10⁻⁸), the covariance and capacities of Boolean spheres and of a Gaussian
  excursion against their closed forms. The named measures of the implicit geometry
  (`EXACT_MEASURES`, `measure_exact`: `exact_fraction`, `exact_surface`, `exact_chords`,
  `exact_minkowski`) in the `analysis` of a configuration, which need no voxels (their
  columns in `results.csv`), by `morphogen analyze` on a geometry beside the theory of its
  model, and in the tool *Morphological measures* of the interface, the exact densities
  beside the closed forms; the tool *Reconstruction from the statistics* with the gradient
  descent of §114 (`method`, `iterations`).
* Models of images (`morphology.segmentation`, `morphology.identification`, the model
  `image`; §106, §107). Grey-level images and stacks turned into phases by Otsu's
  thresholds (one or several, by dynamic programming over the histogram), after a Gaussian
  smoothing; the particles of a phase with the ellipsoids of their moments and the laws of
  their sizes; the scores of a segmentation (accuracy, precision, recall, intersection over
  union, Dice). The model of an image identified among candidates — Boolean spheres of one
  radius or of a lognormal law of radii, hard spheres, a Gaussian excursion, Poisson fibres,
  a Voronoi mosaic — each fitted to the fraction and the specific surface of the image by its
  closed forms, simulated on voxels of the size of the image, calibrated to its specific
  surface and ranked by the shapes of the covariance and of the contact distributions of
  balls and segments; from a volume, or from one section (stereology: `S_V = 4 L_A / π`, the
  discs of the plane). The model `image` (45 models): a volume or a section of a file
  segmented into its phases and described by implicit functions at any point (the
  iso-surfaces of its interpolated grey levels, or the indicators of its phases), in windows
  at its origin or drawn at random for each seed, for the probes, the exact meshes, the
  sections and the solvers; a synthetic tomography of a particulate composite comes with
  morphoGen (`example_image.npz`). `morphogen identify` ranks the candidates of an image
  and writes the best as a configuration (`-o model.yaml`); the templates `image` (windows
  of the tomography) and `spinodal`, with their cards in the gallery. Tested
  (`tests/test_model_identification.py`): Otsu's
  criterion over every bin, the segmentation of the tomography (intersection over union
  0.887, Dice 0.94), the ellipsoids of particles and of an ellipse (1 %), the phases of the
  image structure at the centres of its voxels exactly those of its thresholded levels;
  Boolean spheres, hard spheres, a Gaussian excursion and a Voronoi mosaic each identified
  first from an image of 64³ voxels (the radii within 2 %, the correlation length 0.818 for
  0.8), Boolean spheres from one section of 256² pixels.
* The tutorial of mathematical morphology (`docs/tutorial_morphology.md`), from random closed
  sets to the model of an image: the Choquet capacity probed on the exact geometry and
  Matheron's tests of a Boolean model in straight lines, dilation, erosion, opening, closing
  and granulometry, the four Minkowski functionals exactly against Miles' formulae and the
  estimates of the voxels, the stereology of sections, the identification of the tomography;
  its figures, its snippets run by the tests.
* Spinodal decomposition (the model `spinodal`, §113): the two phases that a mixture quenched
  inside its spinodal forms by the Cahn–Hilliard equation, integrated on a periodic grid by
  the stabilized semi-implicit Fourier spectral method, at an age of the coarsening of their
  pattern; the fraction of the lever rule, the wavelength of the fastest fluctuations of the
  early stage as the unit of length; the composition interpolated between the points of its
  grid as the implicit function. Tested (`tests/test_spinodal.py`): the peak of the linear
  stage at the fastest wave number, the free energy decreasing at each step, the mean
  composition kept to 10⁻¹², the length of the pattern doubling over eight times the age
  (Lifshitz–Slyozov, 2.005), the fraction.
* Reconstruction from descriptors by gradient descent (`morphogen.morphology.reconstruction`,
  §114; the model `reconstruction` with `method: gradient`, as MCRpy): the cells of a
  periodic grid relaxed to `[0, 1]`, their two-point probability at every lag within the
  largest (two Fourier transforms, its gradient exact) and their lineal paths along the axes
  (the products of the cells of the segments, their gradient the products of the others)
  brought to their targets by L-BFGS-B within the bounds, in stages that draw the cells to
  0 and 1, then thresholded at the fraction. The targets in closed form — the covariance of
  the isotropic models at every lag of the grid, the lineal paths of the pores of the models
  with a Choquet capacity — or measured on an image, a planar one giving a volume through the
  three coordinate planes of the grid. Tested (`tests/test_gradient_reconstruction.py`): the
  descriptors of binary fields those of `morphogen.analysis` exactly, the gradient against
  finite differences, a Debye medium within 0.5 % of the variance, the pores of sections of
  a Boolean model of spheres — their exact covariance alone missing their lineal path by 7 %,
  both within 0.5 % with it —, a volume from a section within 3.5 % of the exact 3D
  covariance off the planes, the method of the model. Public interface: the module
  `morphology.reconstruction`, `Reconstruction(method=..., iterations=...)`,
  `Reconstruction.descriptors` (`tests/public_api.txt`).
* Grains grown from seeds, as MicroStructPy (the model `seeded_laguerre`,
  `morphogen.models.seeded`, its kernel `morphogen.kernels.seeds`, §115): ellipsoidal seeds of
  laws of equivalent diameters and of ratios a/b and b/c, their axes at random or aligned with
  the box, drawn until their volumes fill it, broken down into the spheres inscribed about the
  points of their medial surfaces (their radii in closed form; 94 to 97 % of the ellipsoid at
  the default spacing, 98 to 99 % at half of it) and placed the largest first within the
  overlap tolerance of Hart and Rimoli; the grain of a seed is the union of the Laguerre cells
  of its spheres (`SeededTessellation`: the grains as labels, their exact volumes, their
  phases and crystal orientations, the table of their seeds). The seeds that clear at none of
  their positions — elongated seeds jam the sequential addition — are placed where they
  overlap least by default, every seed then its grain, or left out as MicroStructPy does
  (`crowded: leave`). The conforming meshes of polycrystals and the surfaces of their cells by
  grain: the cells of a grain merged in the cell complex (`CellComplex.merged`), one subdomain
  per grain, MMG free inside it (the exact facets of whole periodic cells meshed cell by cell,
  MMG not ending with their merged grains); the elements of the crack meshes of polycrystals take the
  grains of their centroids (`Tessellation.grain_of`); the export to Neper refuses these
  grains, which its tessellations do not describe. Against MicroStructPy 1.6.1 on the same
  seeds: spheres alike (cv of the grain volumes 0.147 and 0.150); seeds of aspect ratio 3, all
  413 placed in 2 s and grains of their volume, where MicroStructPy keeps 276 of them in
  375 s, its grains 1.5 times larger; the grains elongated as their seeds (a/b 3.08) along
  their axes. Tested (`tests/test_seeded_grains.py`), the template `seeded_laguerre` and its
  card in the gallery. Public interface: `SeededLaguerre`, `SeededTessellation`,
  `ellipsoid_spheres`, `overlap_tolerance`, `Tessellation.n_grains`,
  `Tessellation.cell_grains`, `Tessellation.grain_of`, `CellComplex.merged`
  (`tests/public_api.txt`).
* Seeds of several phases and the verification of their grains, as MicroStructPy
  (`seeded_laguerre`, §116): the seeds of each phase of their own laws (`phase_diameter`,
  `phase_aspect_ratio`, `phase_flatness`, `phase_orientation`), of the phase i with the
  probability p_i ∝ q_i / E[V_i] so that each phase fills its fraction of the volume;
  amorphous phases (`amorphous_phases`: a matrix, a glass), their seeds merged into one region
  without grain boundaries, no seed of its own and the identity orientation (`null` in the
  files), a void being an amorphous phase that the meshes leave out (`polycrystal_mesh` keeps
  the phases it is given). The exact shapes of the grains of any tessellation
  (`Tessellation.grain_shapes`): the volumes, centroids and second moments of the cells
  exactly (`CellComplex.piece_moments`, the tetrahedra of the fans of their facets), the
  ellipsoids of the same second moments; the grains against their seeds
  (`SeededTessellation.seed_comparison`, the verification of MicroStructPy on the exact
  cells): the diameters, the ratios a/b and b/c and the axes of the seeds and of the
  ellipsoids of their grains, their errors (bias, mean absolute, RMS, largest, R²), the
  Kolmogorov–Smirnov distance of the grains of each phase to the laws of its seeds
  (`Distribution.cdf` in closed form for every law, `ks_statistic`), the fractions of the
  phases; the named measure `exact_grains` gives them as columns of a run, of the command
  `analyze` and of the tool *Morphological measures*. The placement of seeds of very different
  sizes by grids by levels of sizes, the likeliest overlaps searched first: the same
  placements, 6 times faster for a glass of small seeds about crystals (31.6 s and 5.3 s),
  twice for elongated seeds. The template `seeded_matrix` (laths and equiaxed crystals in a
  glass) and its card in the gallery. Tested (`tests/test_seeded_phases.py`, the distribution
  functions in `tests/test_polydispersity.py`). Public interface: `SeededLaguerre`
  (`phase_diameter`, `phase_aspect_ratio`, `phase_flatness`, `phase_orientation`,
  `amorphous_phases`, `families`), `SeedFamily`, `SeededTessellation.amorphous`,
  `SeededTessellation.seed_comparison`, `Tessellation.grain_shapes`,
  `CellComplex.piece_moments`, `Distribution.cdf`, `ks_statistic` (`tests/public_api.txt`).
* Seeds of the plane and the figures of the verification, as MicroStructPy (`seeded_laguerre`,
  §117): with `dimension: 2`, elliptical seeds in the middle plane z of the box (the diameter
  of the disc of their area, their ratio a/b, a uniform angle in the plane or along x), broken
  down into the circles of their medial segments and placed in that plane with the tolerance
  of MicroStructPy for the plane: columnar grains, the prisms of the power diagram of their
  circles, periodic or not (a box that is not periodic widened in the plane only); their
  grains measured by their sections (`grain_shapes`: their areas, the discs of those areas and
  the ellipses of their second moments) and compared with their seeds (`seed_comparison`,
  `exact_grains`). `morphogen.viz.figures.seed_figure`: the distribution functions of the
  grains of each phase, of their seeds and of the laws of the seeds, the angles of their axes
  and the fractions of the phases; the tool *Grains and their seeds* of the interface
  (Statistics, the trait `seeds` of the items grown from seeds) with the tables of the errors
  and of the fractions and this figure. The template `seeded_plane` (the section of a rolled
  sheet) and its card in the gallery. Tested (`tests/test_seeded_phases.py`,
  `tests/test_gui_tools.py`). Public interface: `SeededLaguerre(dimension=...)`,
  `overlap_tolerance(volumes, dimension)`, `SeedFamily.dimension`, `seed_figure`
  (`tests/public_api.txt`).
* Seeds of boxes and of rectangles (`seeded_laguerre`, §118): `shape: box` (per phase
  `phase_shape`) makes the seeds boxes of the half sides a >= b >= c in the ratios a/b and b/c
  (cubes for ratios of 1; rectangles and squares in the plane), of the volume of the sphere of
  their equivalent diameter. A box is broken down into the spheres inscribed about its medial
  rectangle and those that touch the two faces of its edges, centred on the edges of the boxes
  eroded by their radii c/2 and c/4 (`box_spheres`: 88 to 94 % of it, 53 spheres for a cube),
  a rectangle into the circles of its medial segment and of its corners at three levels (98 to
  99 %, as the corner chains of MicroStructPy, which breaks down no box); the grains of boxes
  compared with their seeds by the ratios of their ellipsoids and by the volumes of the seeds,
  now kept with them (`seed_volume` in the table). The seeds of each phase are drawn until
  they fill its fraction of the volume, the fractions of the seeds exact to a seed, where the
  phase drawn at random for each seed made them fluctuate (0.559 and 0.610 of precipitates for
  0.6 in two realizations). The template `seeded_precipitates` (the cubes of gamma prime of a
  nickel superalloy in their matrix) and its card in the gallery. Tested
  (`tests/test_seeded_phases.py`). Public interface: `SeededLaguerre(shape=...,
  phase_shape=...)`, `SEED_SHAPES`, `BOX_LEVELS`, `RECTANGLE_LEVELS`, `box_spheres`,
  `SeedFamily.shape` (`tests/public_api.txt`).
* Meshes of the plane and their finite elements (`morphogen.mesh.triangles`,
  `morphogen.homogenization.plane`, §119). `TriMesh`, three- or six-node triangles of the
  rectangle of a box in its plane z (phases, grains and their orientations, sides, thickness;
  `extruded`: the layer of tetrahedra of the prisms over the triangles). The columnar grains of
  the tessellations whose germs lie in one plane (the seeds of the plane), which the mesher in
  space cannot take, meshed exactly (`tessellation_mesh`): the power diagram of the plane by
  half planes (`planar_pieces`), its edges subdivided at a size graded from their small
  features (short edges, thin grains, close corners), the opposite sides of a periodic box
  matching node for node, `mmg2d` remeshing the grains within them — the areas of the grains
  within 10⁻¹³ of the exact ones, angles above 5°; the sections of any other structure by the
  level sets of its phases (`section_mesh`; `plane_mesh` chooses). Their files: the plane
  elements of Abaqus and CalculiX (CPE3, CPE6, CPS3, CPS6; the sets of the sides; the crystals
  of each grain turned into the axes of the box), Gmsh (with the edges of the sides), VTK and
  Medit of dimension 2 (`save_mesh`, `read_tri_mesh`, `write_periodic_pairs`). The problems of
  the plane (`PlaneElasticProblem`, `PlaneThermalProblem`, and `homogenize` of a `TriMesh`,
  `plane`): the structure invariant across its plane in generalized plane strain (the
  anti-plane fluctuation free: the 6 × 6 stiffness of the structure in space) or in plane
  strain, conduction (the 3 × 3 tensor), periodic, kinematic and static conditions on the sides
  (the uniform out-of-plane components unknowns of their own), the loadings, thermal
  expansion, statistics and fields of the solvers in space. Tested
  (`tests/test_plane_meshes.py`): exact areas, matching sides, files read back; uniform fields,
  laminates of turned crystals to round-off, the checkerboard of Dykhne converging to the
  geometric mean, the extruded layer giving the same tensors in space to 10⁻¹⁵, kinematic
  conditions above periodic above static. Public interface: `morphogen.mesh.triangles`,
  `morphogen.homogenization.plane`, `homogenize(..., plane=...)`, `make_problem(...,
  plane=...)`, `read_tri_mesh`, `write_inp_tri`, `write_msh_tri`, `write_vtu_tri`,
  `write_medit_tri` (`tests/public_api.txt`).
* The exact discs of the sections of straight fibres along z and of spheres
  (`disc_sections`, `disc_mesh`; `plane_mesh` by default for them): their circles
  discretized exactly, at sizes graded from the gaps between them, three elements across the
  narrowest, the sides of a periodic box matching, `mmg2d` meshing the regions within them;
  the discs that meet and the oblique fibres by the level sets of their union. The areas of
  fibres as the fourth power of the size on six-node triangles (8·10⁻⁷ at 0.05), the
  conductivity of square arrays within 1.4·10⁻⁵, every gap of a random bed open in 0.35 s.
  The template `fibre_bed_plane` without its sampling grid.
* The flow through the pores of the plane (`homogenization.plane_flow`,
  `plane_permeability`): the Stokes flow across a structure that does not vary along z by
  the elements of Taylor and Hood on its six-node triangles (the velocity quadratic, the
  pressure linear), and its flow along z on the same elements, give its permeability tensor;
  `homogenize` of a mesh of the plane takes `hydraulic_permeability` (periodic conditions),
  and so do the configurations of the meshes of the plane (`method: fe` or `auto`), whose
  pores are then meshed with the solid. The channel of Poiseuille exactly, the drags of
  Sangani and Acrivos within their digits, the pipes of Hagen and Poiseuille within 2·10⁻⁶,
  the lattice Boltzmann method within a few percent. The template `fibre_bed_plane` (the
  permeability of a dry bed of fibres) and its card in the gallery, the speeds of its flows.
* Curved six-node triangles: the nodes of the interfaces of the sections of the plane on the
  curves of the functions of their phases (`curved_triangles`; `section_mesh`, `plane_mesh`
  and `mesh.curved` at order 2, by default), isoparametric elements in the finite elements of
  the plane (`TriMesh.curved`, `deviations`, `min_jacobian`, `straight`; their areas exact)
  and in their layer in space, whose edges over the curved edges are curved as they are. The
  area of fibres within 10⁻⁶ at the size 0.1 (4·10⁻³ with straight edges); the conductivity
  of square arrays of cylinders against Perrins, McKenzie & McPhedran (1979) within 3·10⁻⁵,
  the reciprocal theorem of Keller within 7·10⁻⁷, the neutral coated cylinder of Hashin in
  conduction and plane strain within 2·10⁻⁵ under every condition.
* The tutorial *Polycrystals of the plane, meshed and solved in their plane*
  (`docs/tutorial_plane.md`, run by the tests): the grains of a rolled sheet grown from
  seeds of the plane, their exact polygons and their small features removed, the stiffness
  of the sheet under the three uniform conditions (the bulk modulus of the crystals of
  copper), plane strain and the layer in space, the exact section of a polycrystal of space,
  an orientation map meshed by its pixels, a plastic path and the files of Abaqus; its card
  among the tutorials. The small features of the polygons of the plane removed at half the
  size where the polygons would fold, three times at most, before the exact polygons; the
  vertices near a side join the vertex of the side nearer than the features, instead of a
  short edge along the side (the smallest angle of 152 624 triangles of 1415 grains 5.4°
  instead of 0.9°). The power diagram of the plane skips the half planes that do not cut a polygon (the pieces of
  9905 cells in 3.5 s instead of 8.4 s). The interface exports the triangles of its
  simulations of structures of the plane.
* The loading paths of the meshes of the plane factorize the tangents of their layer by
  default (`solver: auto`): the algebraic multigrid converged poorly on a layer tied across
  its thickness (a plastic tension of 658 six-node triangles in 22 s instead of 207 s).
* The orientation maps (EBSD) meshed in their plane by their pixels (§119; `lattice_mesh`,
  `lattice_map`; `plane_mesh` and the configurations of the model `ebsd` by default): two
  triangles per pixel, each the grain and the orientation of its pixel, for the finite
  elements of the plane — the tetrahedra of the lattice field, meshed by its phases, turned
  no crystal; the mesher in space extrudes them through the box (`conforming_mesh`), each
  tetrahedron the grain of its pixel. The bulk modulus of the crystals of nickel of a window
  of the demonstration map
  exactly, kinematic conditions above static ones. The template `ebsd_plane` (windows of the
  map by the finite elements of the plane, under kinematic and static conditions) and its
  card in the gallery. Tested (`tests/test_plane_meshes.py`).
  Public interface: `lattice_map`, `lattice_mesh` (`tests/public_api.txt`).
* The loading paths of the meshes of the plane (§119): `homogenize` of a `TriMesh` follows
  them with the nonlinear behaviours on its layer of tetrahedra in space (`TriMesh.layer`:
  one layer of the thickness of its mean edge, ten-node tetrahedra for six-node triangles),
  periodic across it — generalized plane strain —, under periodic conditions (the others
  skipped); the configurations of the meshes of the plane take them. The stiffness of the
  problems of the plane assembled by chunks of elements (`CHUNK`: the memory of the element
  matrices bounded). The periodic equations of Abaqus and CalculiX of the meshes of the
  plane (`write_periodic_equations`, `outputs.periodic_equations`: the reference nodes
  `REF_X` and `REF_Y`, the two degrees of freedom of the plane elements, the areas of the
  sides times the thickness). The card of `seeded_plane_stiffness` in the gallery shows the grains
  of its mesh and the von Mises stress of its shear. Tested (`tests/test_plane_meshes.py`:
  the same response as the layer itself, the chunks against the whole, the equations of
  the plane). Public interface:
  `TriMesh.layer`, `CHUNK` (`tests/public_api.txt`).
* The meshes of the plane in the runs (§119): `mesh.plane` (`auto`, the default: the
  columnar grains of seeds of the plane meshed in their plane; `true`: the section of any
  structure at `mesh.z`; `false`: tetrahedra), the files of the triangles
  (`<stem>_tri.<format>`: .vtu, .inp of plane elements with the crystals of the grains,
  .msh, .mesh, their periodic pairs), `homogenization.plane` (`generalized` or `strain`),
  their figures (the triangles in the colours of their grains, the fields of the solutions
  on the deformed triangles: `mesh_figure`, `mesh_section` of a `TriMesh`), and in
  `morphogen homogenize` (`--plane`, `--z`; the meshes of the plane read back). The mesher
  in space takes the columnar grains too (`conforming_mesh`: their exact polygons extruded
  through the box, `TriMesh.extruded(layers=...)`), where it failed on their degenerate
  cells. The exact sections of the tessellations in space by a plane z, the power diagrams
  of the plane of the weights w_k - (z - z_k)^2 (`section_weights`, `planar_pieces(micro,
  z)`, `tessellation_mesh(z=...)`, `exact_section`): `plane_mesh` meshes the sections of
  tessellations exactly. The small features of the polygons removed by default, as the cells
  in space (`tessellation_mesh(min_feature=...)`, a quarter of the size; 0 keeps the exact
  polygons): the short edges between grains collapsed and the thin caps of the cells cut by
  the box moved onto the sides — six times fewer triangles for the grains of seeds, none
  flat, the areas within 0.5 % on average. The problems of the plane factorized up to
  1.5 million unknowns (`PLANE_DIRECT_LIMIT`). The template `seeded_plane_stiffness` (the
  stiffness of the section of a rolled sheet by the finite elements of the plane) and its
  card in the gallery. Tested (`tests/test_plane_meshes.py`). Public interface: `MeshSpec.plane`,
  `MeshSpec.z`, `MeshSpec.on_plane`, `MeshSpec.plane_kwargs`, `HomogenizationSpec.plane`,
  `TriMesh.extruded(layers=...)`, `section_weights`, `exact_section`, `planar_pieces(...,
  z)`, `tessellation_mesh(..., z, min_feature)`, `mesh_figure(..., grains)`,
  `PLANE_DIRECT_LIMIT` (`tests/public_api.txt`).
* The computations of morphoGen as models of VIMSEO (`morphogen.interop.vimseo`, §112; the
  extra `vimseo`, Python 3.10 to 3.12), for the demonstration of the credibility of the
  simulations: a configuration of `morphogen run` with its entries as inputs and its results
  as outputs, the exact densities of the realizations of a model of spheres with the closed
  forms of the model as references, their GEMSEO disciplines and VIMSEO models, the
  reference datasets, the verification of the code and of the solution against its
  discretization. Tested with VIMSEO 0.1.7 on Python 3.12: the code verification of the
  densities against Miles' formulae (1.8 % and 1.0 %, the fluctuation of single
  realizations), the solution verification of an FFT conductivity on four voxel sizes
  (Richardson 2.278, the observed order 0.90).
* Thermo-chemo-elastic fields of a microstructure in time (`coupling.thermochemoelastic`:
  `ThermoChemoElastic`, `thermochemoelastic`, `ThermoChemoMaterial`; §108). The temperature
  (Fourier's law, a volumetric heat capacity, the heat a reaction releases), the activity of
  a species (diffusion, a solubility per phase, a first-order consumption, its drift by the
  gradient of the hydrostatic stress for a partial molar volume) and the quasi-static
  displacement under the thermal and chemical expansions, solved together. Diffusivity and
  rate follow Arrhenius' law. The fields at each output time come with a history: mean
  temperature, uptake, mean stress, displacements of chosen points, curvature of strips, and
  an XDMF time series for ParaView. On voxel images: finite volumes at the voxel centres
  (harmonic conductances, Scharfetter–Gummel fluxes of the drift, exact at Boltzmann's
  equilibrium) and trilinear hexahedra at their corners, with periodic axes free to expand
  or held (`mean_strain`). On meshes of linear tetrahedra: every field linear, with lumped
  capacities. Faces are held at a temperature or an activity (a value or a function of
  time), clamped, or planes of symmetry; the remaining rigid motions are removed by
  statically determinate supports. Implicit Euler or TR-BDF2 steps; fields coupled both ways
  are solved to a fixed point by Aitken's relaxation of the staggered pass. Tested against
  `transient_diffusion` (10⁻¹²), Crank's slabs (Euler first order in time, TR-BDF2 second),
  free and clamped expansions (exact), a periodic laminate (10⁻¹⁰), Timoshenko's bilayer
  (1.5 %), a strip swelling from its upper face against the beam of Crank's profile (3 %),
  the energy of an adiabatic reaction (10⁻⁸), the Arrhenius scaling of times (10⁻¹³), and
  Boltzmann's equilibrium of a species drawn by the stress (10⁻¹²; 10⁻⁷ under strong
  two-way coupling).
* Inverse design of active structures (`optimization.morphing`: `MorphingOptimization`,
  `MorphingTarget`, `MorphingDesign`, `hat_filter`; §109): the layout of two materials that
  expand differently with a stimulus (heat, or a species they absorb), or of one of them and
  voids, in a box of voxels held by supports, for target displacements of points at
  equilibrium, or at output times while the stimulus diffuses from faces held at values (the
  sequential shapes of 4D-printed active composites). SIMP interpolation of elasticity,
  eigenstrain, capacity and conductivity; a density filter cut at the faces of the box;
  Heaviside projection by continuation; optional extrusion along an axis; the method of
  moving asymptotes under volume constraints; exact adjoint sensitivities, run backwards
  through the implicit Euler steps of the diffusion. `bending_target` gives the deflections
  of a strip bent to a curvature, and `verify` solves the black-and-white layout with the
  coupled solver. Tested: sensitivities against finite differences (10⁻⁶), states against
  the coupled solver (10⁻¹²), a strip designed to bend like one whose upper sixth is active
  (5 %, its black-and-white layout too), and a strip designed to bend down then up as two
  materials take up a species at different rates (3 %; its layout 10 %).
* The variability of the tows of woven textiles along their paths (`woven_textile`, §110), as
  the tomography of textile composites measures it (Vanaerschot, Cox, Lomov & Vandepitte
  2013, 2014; Blacklock et al. 2012): the deviations of the centrelines across the tows in
  the plane of the fabric and through its thickness (`deviation_in_plane`,
  `deviation_through`: standard deviations), and the widths and thicknesses of their sections
  (`width_variation`, `thickness_variation`: coefficients of variation; lognormal, of the mean
  of their tow, their logarithms correlated by `size_correlation`). Each is a stationary
  Gaussian process along the direction of the tows, by its spectral representation, of a
  gaussian (smooth paths) or exponential (Markov) covariance (`variation_covariance`) and of
  correlation lengths `correlation_length` (one, or the centrelines and the sections),
  periodic over the box (a window: over the window and six correlation lengths) and
  correlated between the neighbouring tows of a ply or layer by `neighbour_correlation`
  (`rho^d`, periodized on the ring of the tows of a periodic box). The nodes move by the
  deviations, the frames of the sections turn with them, and the half sizes vary linearly
  between the nodes in the point-wise functions of the phases and in the exact meshes. The
  summary gives the deviations, coefficients of variation and correlation realized; the
  theory, the mean areas of the varying sections along the lengthened paths. With the
  variability at zero the textiles are those of before, bit for bit. Tested
  (`tests/test_textile.py`): the standard deviations within 8 % of those prescribed, the
  correlations along the tows within 0.08 of the covariance of their spectrum, the
  coefficients of variation within 10 %, the correlation of the logarithms within 0.06,
  neighbouring tows correlated as the periodized `rho^d` (within 0.12), the mean areas
  against Monte Carlo, the fractions against the theory (2 %).
* The binding depth of the 3D weaves of `woven_textile` (`binder_depth`, the number of weft
  layers that a binder ties together; Boussu, Cristian & Nauman 2015): layer-to-layer angle
  interlocks of any depth (`layers + 2 − D` binders per column; 2, the former weave, by
  default), and orthogonal weaves whose shallower binders tie the layers in turn from one
  column to the next; angle interlocks without stuffer warps (`stuffers: false`). Tested:
  the spans and crimps of the binders in closed form (10⁻⁹), the fractions and unions
  against Monte Carlo, the counts, the phases and the thickness without stuffers.
* The tows as unidirectional composites (`morphogen.homogenization.unidirectional`, §110):
  `composite_cylinders(fibre, matrix, fibre_fraction)` gives the transversely isotropic
  material of aligned fibres in a matrix by the composite cylinder assemblage of Hashin &
  Rosen (1964) and Hashin (1979) (the plane-strain bulk modulus, the coupling, the axial
  modulus and the axial shear modulus exact; the transverse shear modulus of the generalized
  self-consistent scheme of Christensen & Lo, or Hashin's bounds), with the conductivities
  and the thermal expansions (Levin's relation); Hill's universal relations
  (`hill_relations`), the material `{unidirectional: {fibre, matrix, fibre_fraction}}` of
  configurations and `WovenTextile.tow_material`. Tested (`tests/test_unidirectional.py`):
  the limits, Hill's relations (10⁻¹², 10⁻⁶ on an FFT solution), Levin's relation on the FFT
  expansion (10⁻⁵), the FFT stiffness of hexagonal (0.65) and random (0.5) arrays of
  `long_fibers` within 1.6 % and 4.4 % of the closed forms.
* Studies of Code_Aster and cases of OpenFOAM (`morphogen.io.code_aster`,
  `morphogen.io.openfoam`; §111). The study of Code_Aster of the apparent stiffness and
  conductivity of a conforming mesh: the mesh in Code_Aster's format (`.mail`: the groups of
  the phases, of the grains of oriented crystals, of the faces of the box and of their nodes;
  or MED 4.2 written with h5py in the layout of MED's library, its tetrahedra in MED's
  order), the command file (`ELAS`, `ELAS_ORTH`,
  `THER`, `THER_ORTH` by group, the crystals turned by the Bunge angles of their grains
  through `AFFE_CARA_ELEM`, `ANGL_EULER`; the six unit strains and three unit gradients under
  periodic conditions, one `LIAISON_GROUP` per class of the nodes of the upper faces
  translated alike, without a redundant equation at the edges and corners, or kinematic
  ones; `MECA_STATIQUE`, `THER_LINEAIRE`, `POST_ELEM` `INTEGRALE`, a table per loading, of
  14 digits) and the export file of `run_aster`, read back into the apparent tensors and
  their moduli. The polyMesh of OpenFOAM of the voxels of some phases or of the tetrahedra
  of a conforming mesh (the internal faces in upper-triangular order, the patches of the
  box, cyclic pairs along the periodic axes whose translation OpenFOAM computes from their
  faces, the walls, a cell zone per phase) and its cases: the flow through the pores under
  a pressure drop (`simpleFoam`, or `foamRun` with `incompressibleFluid`), Darcy's
  permeability from the flow rate through the outlet, through a tetrahedral mesh on its
  polyhedral dual (`polyDualMesh` in `Allrun`; `dual`; the voxels the robust case of the
  flow through curved interfaces, whose conforming meshes hold slivers that make the dual
  invalid, a warning says so); the diffusion through a phase
  (`laplacianFoam`), its effective diffusivity; the closed pores left out; `checkMesh`
  before the solver. The runs: `morphogen.io.run_code_aster` (`run_aster`, or the program
  that `MORPHOGEN_RUN_ASTER` names) and `morphogen.io.run_openfoam` (in OpenFOAM's
  environment, or after the file that `MORPHOGEN_OPENFOAM` names). The commands
  `morphogen code-aster export|run|read` and `morphogen openfoam export|run|read`,
  `outputs.code_aster`, `outputs.openfoam` and the mesh format `mail`, and the tools
  *Code_Aster study* and *OpenFOAM case* of the tab Solvers & formats. Tested with stand-ins
  of both programs, and with the programs themselves where they are found (Code_Aster
  18.1.11 and OpenFOAM v2412, of conda-forge): the studies give morphoGen's periodic and
  kinematic stiffness and conductivity on the same meshes within 3.4·10⁻¹⁴ (two phases, in
  Code_Aster's format and in MED, linear and quadratic tetrahedra) and 1.4·10⁻¹³ (crystals
  turned by their grains), voids and whole periodic cells too; the cases of voxels, the
  permeability of the finite volumes of a slit in closed form within 1.2·10⁻¹⁰ and the
  effective diffusivity of the finite volumes on the voxels of a random medium within
  1.6·10⁻¹¹, with symmetry planes and cyclic pairs, `checkMesh` passed; the flow through a
  slit of tetrahedra, on their dual, converging to Poiseuille's at the second order; the
  polyMesh closed, ordered, oriented and read back, by VTK's reader where VTK is installed.
  Public interface: the modules `io.code_aster` and `io.openfoam`, their names in `io`
  (`run_code_aster`, `aster_executable`, `run_openfoam` and `openfoam_found` among them),
  `write_openfoam(dual=...)` and `OpenFoamRun.mesh_check` (`tests/public_api.txt`).
* Fixed: the opening and the closing of `morphogen.analysis` took a ball of the size given in
  voxels for their second step on images whose voxels are not of unit size (the opening of
  spheres by a ball of 0.5 on voxels of 0.1875 was their erosion); both steps take the size
  of the voxels. The capacity of balls of a periodic section (an image one voxel thick)
  padded it across its plane into a volume: it is padded in its plane only.

## 1.1.0 — 2026-11-25

Long fibres, woven textiles and grains with subgrains, meshed from their exact geometry;
monodisperse or polydisperse sizes in every model; finite elements on conforming meshes
by default, the voxels as the last resort; the voxels of morphoGen as the arrays of JAX
for spectral phase-field solvers; Microgen, Blender (its rigid bodies, and scenes for its
fluid simulations), the discrete element method (the Gmsh meshes of LMGC90 among its
formats) and the tools of alloy design; the workspace closed in one click, the fields of
every tool drawn in the page, their videos made on demand by the export; installers of a
JupyterLab environment for Windows and macOS.

Changed (the entries below say how): a configuration without a method and without a
`mesh` section computes by finite elements on a light conforming mesh where it ran the
FFT solver (`method: fft` keeps the voxels; without the mesher MMG, the voxels with a
warning), and so do the tools of Mechanics of the interface; the size parameters of a
model are laws once it is built (`PoissonFibers(radius=0.03, volume_fraction=0.1).radius`
is `Constant(0.03)`, its value `.mean()`); no tool of the interface writes a video unless
asked.

* Installers of a ready-to-use environment of notebooks, *morphoGen JupyterLab*
  (`packaging/jupyterlab`), for those who have no Python: a Python of its own (a relocatable
  CPython), morphoGen with its extras, JupyterLab and the usual packages of notebooks, about
  800 MB once installed, without administrator rights and without a connection once
  downloaded. On Windows (x64), `morphoGen-JupyterLab-<version>-windows-x64-setup.exe` (Inno
  Setup 6) installs it for the current user with a shortcut in the Start menu and an
  uninstaller; on macOS, `morphoGen-JupyterLab-<version>-macos-arm64.pkg` (Apple Silicon) and
  `-macos-x86_64.pkg` (Intel) install the application *morphoGen JupyterLab* in the
  Applications folder of the user or of all the users. The shortcut opens JupyterLab on
  `Documents/morphoGen`, made at the first start with a first notebook (the first study of the
  README); the kernel of the notebooks opts out of the power throttling of Windows 11; the
  programs of the packages run from wherever the environment is installed. The workflow
  `jupyterlab.yml` builds the three, installs each one on its system, checks the environment
  it installs (the programs, a realization, the FFT solver, a conforming mesh and the finite
  elements, a plastic loading path, the first notebook through the kernel, the server of
  JupyterLab), removes it, and attaches them to each release (to an earlier one on demand).
* Long fibres along an axis, the unidirectional composites (`long_fibers`,
  [docs/theory.md](docs/theory.md) §95): fibres that cross the box along `axis` and close
  on themselves across its periodic faces (chains of spherocylinders), of a radius per
  fibre (a number, monodisperse; a distribution, polydisperse). Their sections, packed in
  the plane normal to the axis without coming closer than `min_gap`, are a random packing
  (`arrangement: random`: inflated from a fraction of 0.3 by 2 % at a time, the overlaps
  removed at each step, to 0.8, beyond the 0.55 where random sequential addition jams; as
  the random fibre distributions of Melro et al. 2008, Vaughan & McCarthy 2010, Yang et
  al. 2013) or a hexagonal or square array (its spacing from the fraction, a whole number
  of its cells in the box, its sites displaced at random by up to `disorder` spacings).
  `misalignment` (degrees, 0: straight parallel fibres) is the RMS angle of the segments
  to the axis: each fibre waves about its section, the value there of a smooth random
  periodic field of the box of correlation lengths `waviness_length` along the axis and
  `transverse_correlation` across it (by default the same: neighbouring fibres wave
  together, as the misalignments measured in composites do, Yurgartis 1987; 0: each
  fibre on its own), scaled to the angle, then freed of its overlaps in space (the nodes
  moved in their planes by the pushes of the closest points of the segments, smoothed
  along the fibres; `kernels.long_fibres`). The volume fraction counts the contour length
  of tilted fibres, `E[sec a] = 1 + θ²/2 + 5θ⁴/12 + …` (`secant_mean`); the summary gives
  the realized fraction (the exact volume of the chains, their joints included), the RMS
  and largest angles after the corrections, the steps and the smallest gap; non-periodic
  domains are windows of a periodic box enlarged across and along the axis. 110 fibres at
  0.6 and 3° in 0.3 s, 1221 in a box of 80 diameters in 2.9 s; 0.78 at 5°. Tested
  (`tests/test_long_fibres.py`): no two segments of different fibres closer than `r_a +
  r_b + min_gap` (every pair within reach, nearest periodic images, Ericson's closest
  points), the chains closed across the period, straight fibres exactly along the axis,
  the realized fraction against the exact volume of the chains and the voxels (within
  2·10⁻³), the RMS angle exact before the corrections and within 3 % (0.3) and 12 % (0.6)
  after, the covariances of the waviness against those of its modes along the fibres and
  the Gaussian covariance `exp(−d²/2ℓ_t²)` across them, `secant_mean` against quadratures
  of its definition and its series, lognormal radii (Kolmogorov–Smirnov), the arrays
  (spacing, nearest neighbours, exact fraction, largest fractions with a gap), windows,
  fibres along x and y, the theory, the configuration, JSON and seeds; the template
  `long_fibers` and its gallery card.
* Woven textile composites (`woven_textile`, `models.textile`, its kernels
  `kernels.textile`; [docs/theory.md](docs/theory.md) §96): warp and weft yarns (tows)
  swept along the crimped paths of their weave, of elliptic, lenticular or power-ellipse
  sections, in a matrix. 2D weaves by their interlacing matrix (`plain`, `twill [n, m]`,
  `satin [H, s]`, `basket [k]`; `interlacing`), in `layers` plies shifted from one another
  (`ply_shift`, or `random_shift`) with resin between them (`ply_gap`); 3D weaves of
  straight warp and weft layers tied by binders: `orthogonal` (through the thickness in
  the gaps between the wefts, by arcs and vertical lines), `layer_to_layer` and
  `through_thickness` angle interlocks. The levels of the paths at the crossings come from
  the mean thicknesses of the yarns, the paths pass from one level to the next by half
  cosines over a `transition` share of the spacing (`profile`, `Piece`: their arc lengths
  in closed form, by the complete elliptic integral of the second kind), the sizes of the
  yarns are numbers or distributions (a size per yarn). The yarns are a partition of their
  union (a point inside two belongs to the one it is deeper in), and the share of their
  volume where two overlap is in the summary (`interpenetration`, by points drawn in the
  yarns). Each segment of a yarn is a label with the orientation of its material (the axis
  3 along the segment, 1 across the width of its yarn): transversely isotropic or
  orthotropic tows are turned along their paths by the FFT solvers and in the exports to
  AMITEX_FFTP and Abaqus. Phases 0 matrix, 1 warp, 2 weft, 3 binder (or every yarn phase 1,
  `merge_yarns`); the implicit functions of the phases at any point (the textile is meshed
  from them, each element with the label and orientation of its segment); periodic boxes
  hold whole repeats of the weave (an error names the sizes that do), non-periodic ones are windows of
  the infinite fabric; `axis` stacks it along x, y or z. The theory gives the nominal
  fraction of each family (the areas of the sections times the arc lengths), the crimps,
  the thickness of the fabric and, given the fraction of the fibres in the tows, the fibre
  volume fraction. The template `woven_textile` (a 3D orthogonal carbon/epoxy composite,
  its tows turned along their paths: squarish tows, power ellipses of exponent 4, 42 %
  of fibres, the box as thick as the fabric) and its card
  in the gallery; the model in the family *Fibres and layers* of the interface. Tested against
  quadratures (the arc lengths of the pieces), Monte Carlo (the areas of the sections, the
  fractions of the phases against the swept volumes, the union of interpenetrating yarns),
  the exact crimp of the binders of an orthogonal weave, periodicity, windows and the
  stacking axis, and the FFT stiffness of oriented tows against the same tows given in
  the axes of the sample.
* Woven textiles meshed from their exact geometry, solved by finite elements by default
  ([docs/theory.md](docs/theory.md) §96): their conforming meshes take the adaptive
  route of the exact functions of their phases (`exact_route` gives `level_set` to the
  structures whose labels carry orientations and that give them at any point,
  `label_of`; an octree refined on the functions of the sections, cut by MMG, the
  interfaces projected onto them, the faces of a periodic box matching), no longer a
  level set sampled on a background grid, which lost the thin tows (−41 % of their
  volume at the spacing of the default for an orthogonal weave of tows 0.3 thick). Each
  element takes the label of the segment at its centroid and its orientation
  (`mesh.exact.labelled`: `mesh.cell`, `mesh.orientations`; an element whose centroid
  falls across an interface within the Hausdorff distance takes the label of the
  nearest element of its phase), so that the finite elements turn transversely isotropic
  tows along their paths as the FFT solver does: on the same mesh, the stiffness of
  tows turned by the orientations of their elements equals that of the same tows given
  in the axes of the sample (10⁻⁸), and the FFT stiffness on voxels within 6 % (the
  diagonal terms of an orthogonal weave of round tows). The template `woven_textile`
  computes its stiffness by finite elements on that mesh (`mesh: {size: 0.6, hausdorff:
  0.05, order: 1}`, `solver: direct`: 166,000 elements, the volumes of the warps,
  wefts and binders short by 3.0, 2.3 and 4.4 %, about 3 minutes to mesh and 3.5 to
  solve on four busy cores, `morphogen run` of the template in 387 s: C11, C22, C33 of
  38.0, 52.0 and 10.0 GPa; at `size: 0.8, hausdorff: 0.06`, 112,000 elements, the same
  stiffness within 1 %, the binders short by 7.5 %), its card in the gallery likewise.
* Grains with subgrains, or filled with another structure ([docs/theory.md](docs/theory.md)
  §97): the model `nested_grains` (`models.nested_grains`, `NestedGrains` →
  `NestedGrainStructure`) nests the structure of a model (`inner`) in the grains of a
  Voronoi or Laguerre tessellation (`grains`, isotropic or not, with its phases, the
  crystal orientations of its grains and its grain-boundary phase). An inner tessellation
  divides them into subgrains: the germs of the inner tessellation that fall in a grain are
  those of its subgrains, each point of the grain belongs to the nearest of them (in the
  power distance, and the metric of an anisotropic inner tessellation: laths along an axis
  of the crystal), so that every subgrain holds its germ and the sub-boundaries stop at the
  grain boundaries; a grain without a germ is one subgrain. Each subgrain is the crystal
  of its grain turned by an angle drawn from `misorientation` (a number or a distribution,
  2° by default, at most 90°) about a uniformly random axis; the labels are those of the
  subgrains, grain after grain (`parents`, `grain_labels`), their orientations those the
  FFT solvers take with the labels. Any other inner model fills the grains (`filled_phases`,
  all by default) with its phase 1 as a phase of its own (`inner_phase`, the next free one by
  default): lamellar colonies, precipitates in the grains. With `turned` (the default),
  each grain sees the inner structure in the axes of its crystal (`q = g (x − c) + s`, `c`
  its germ, `s` a shift uniform in the box), at its own place of one realization in a
  periodic box: lamellae on planes of the crystals, patterns independent from grain to
  grain; without it, one realization in the axes of the sample cut by the grains. Phases
  and labels at any point (k-d trees of the germs, point queries of the inner structure:
  exact sections and slabs, `viz.sections` draws them point-wise); the functions of the
  phases are those of the grains for subgrains, the signed distances of the voxels of each
  phase for filled grains; JSON and configuration round trips. Theory: the law of the
  angle between two subgrains of one grain, `cos(ψ/2)` uniform between `cos((t_a +
  t_b)/2)` and `cos((t_a − t_b)/2)` for misorientations `t_a`, `t_b` about independent
  uniform axes (`subgrain_disorientation_cdf`, `subgrain_disorientation_moments`: `4t/3`
  on average for a constant `t`), the mean number of germs of subgrains in a grain
  `λ_inner / λ_grains`, the fractions of the phases of filled grains (`f p` for the inner
  phase). The template `nested_grains` (27 grains, about 20 subgrains each, turned by 1 to
  3°) and its card in the gallery; the model among the tessellations of the interface,
  whose grain tools take its subgrains (`cell_of`, `germs`, `orientations`).
  `tests/test_nested_grains.py` (11 tests, 8 s).
* Exact conforming meshes of grains with subgrains, or filled with another structure
  ([docs/theory.md](docs/theory.md) §97), instead of meshes from the grid. Subgrains
  are exact convex pieces (`NestedGrainStructure.subgrain_pieces`, given to the meshes as
  `convex_pieces`): the pieces of the cells of the grains in the box (mapped by an
  anisotropic tessellation, cut by the faces of a window whose preimage is no box), each
  cut by the planes where the power distances of the inner tessellation to two germs of
  its grain are equal (affine in the axes of the sample: `q = g (x − c − t) + s`, `t` the
  periodic image of the grain the piece belongs to, the metric of an anisotropic inner
  tessellation and the weights of a Laguerre one included), and by the planes of the
  grain-boundary phase (the cell shrunk by half its thickness along the faces it shares,
  and the disjoint slabs between). The route of the convex pieces of
  `morphogen.mesh.exact` meshes them with their faces (the faces of neighbouring grains
  overlaid where their subgrains cut them differently), each element with its subgrain
  (`mesh.cell`) and its crystal (`mesh.orientations`, which `pieces_mesh` now carries,
  and `polyhedral_mesh` by default for crystals): the finite elements turn the crystals
  of the subgrains. Filled grains have the functions of their phases at any point: the
  inner phase is the union over the filled grains `B` of the inner structure that `B`
  sees, cut by its cell, `F = min_B max(c_B, φ(q_B))`, continuous across the faces of
  the grains where the patterns change; a filled phase `k` of the grains is
  `max(G_k, −F)`. They take the route of the level sets (`point_cells`: each element
  with the grain of its centre and its crystal), and give the grids their values (the
  signed distances of the voxels stay for the inner structures without point-wise
  functions). Tested (`tests/test_nested_grains.py`): the pieces partition the box (to
  1e-9), each of 6,000 points in a piece of its subgrain and of its phase (turned and
  not, a grain-boundary phase, laths in a window, turned anisotropic grains of a window
  with Laguerre subgrains), the volumes of the subgrains against Monte Carlo; the exact
  mesh of subgrains (every element in the subgrain and phase of its centre, the volume of
  each subgrain that of its pieces to 1e-12, the crystals, matching periodic faces) and
  the bulk modulus of its copper crystals by finite elements (that of the crystal, 1e-6);
  the functions of filled grains (their signs the phases at 20,000 points, no jump
  across the faces of the grains, the grids the same) and their exact mesh (the phases of
  the elements those of their centres, the volumes of the phases those of a voxel image,
  the grains and crystals of the elements).
* `Tessellation.lipschitz`: 2 for a mosaic of several phases (the slope of the function
  of the cells of a phase is the distance of the nearest germs of two phases over the
  mean spacing, 1.97 measured), 1 before: the adaptive meshes of the structures that
  hold a mosaic (combinations, grains filled with another structure) refine the edges
  that an interface can reach.
* Monodisperse or polydisperse objects in every model where objects have a size: a size
  given as a number is one size (the realizations unchanged: a constant law draws no
  random number, and the realizations, summaries and theories of 24 configurations of
  the models below were checked to the last bit against the earlier code), a
  distribution one size per object, drawn once ([docs/theory.md](docs/theory.md) §2,
  §98). The radius became a distribution in `poisson_fibers`, `curved_fibers`,
  `cross_linked_fibers` (`radius`), `bouligand` (`fiber_radius`), `delaunay_network`
  (`radius`), `open_foam`, `strut_lattice` and `random_lattice` (`strut_radius`); the
  models whose sizes were already distributions keep them (`boolean_spheres`,
  `boolean_cylinders`, `penetrable_spheres`, `hardcore_spheres`, `cox_spheres`,
  `hardcore_fibers`, `random_walk_fibers`, `long_fibers`, `dead_leaves`,
  `transparent_dead_leaves`, `osteons`, `laguerre`, `level_set_foam`, `random_laminate`,
  `woven_textile`). The closed forms take the moments of the laws: Poisson fibres
  `p = 1 - exp(-L_V pi E[R^2])`, `S_V = 2 pi E[R] L_V (1 - p)`, the covariance with the
  lens area averaged over the radii, the Choquet capacities of balls with
  `E[(R + r)^2]` and of segments with `E[R^2]` and `E[R]`; curved fibres the volume
  `pi E[R^2] L + 4/3 pi E[R^3]`; cross-linked fibres in contact where their axes come
  closer than `D = r_a + r_b + contact_gap`, the contacts per unit length with `E[D]`
  and the excluded volume of finite fibres with `E[D^2]` and `E[D^3]` (the estimate of
  Toll of `random_walk_fibers` too); plywoods `pi E[R^2] / (s h)`; lattices `pi E[R^2]
  L_V` to first order (`StrutLattice.length_density`). Where a volume fraction sets the
  size (`delaunay_network`, `open_foam`, `strut_lattice`), a distribution of the size
  keeps the shape of its law and its scale is solved on each realization
  (`Distribution.scaled`: the law of `s X`; `radius_scale` in the summary); a number with
  a volume fraction is refused as before. The spheres of the nodes of foams and networks
  take `node_ratio` times the largest radius of the struts that meet there; the radius
  of each strut is kept in the metadata (`radii`), and the frames of beams
  (`network_frame`, `frame_stiffness`) take it. In a periodic box each strut of a lattice
  is drawn once (`periodic_struts`: the struts of opposite faces are one), and a fibre
  family of a plywood normal to `(a / L_u, b / L_v)` is `gcd(a, b)` fibres winding
  around the box, each of one radius (an oblique family is often one fibre: one radius
  per layer then; windows draw one per fibre). The summaries give the mean and standard
  deviation of the sizes drawn (`radius_mean`, `radius_std`, `strut_radius_mean`...).
  `Framework.segments` takes a radius per bar, the random sequential addition of curved
  fibres a radius per fibre, `CrossLinkedFibers.contacts` a radius per fibre (`radii`),
  `BouligandStructure` a radius per fibre of each layer (`radii`, `first`;
  `fibre_radii`). Changed: once a model is built, these size parameters are laws, as
  the other distributions of the models were: `PoissonFibers(radius=0.03).radius` is
  `Constant(0.03)` (its value `.mean()`; the configurations still write the number),
  `StrutLattice.radius` stays the radius of its struts (the mean of the law, or the
  radius solved for the volume fraction). Tests: `tests/test_polydispersity.py` (the
  laws scaled, the moments of `D` against Monte Carlo, the capacities of polydisperse
  Poisson lines against the exact distances of realizations, curved fibres placed
  without overlap by every pair of segments, the contacts of cross-linked fibres against
  an independent computation, the plywoods and lattices on their voxels, the scales
  solved for the fractions, the nodes and frames of the networks).
* The sizes in the interface: every parameter of a model that is a distribution (radii,
  lengths, thicknesses) has its editor in the form of Generate, monodisperse (one size)
  or a law (uniform, log-normal, normal with its optional bounds, discrete sizes and their
  weights), its fields named; a new law starts around the mean of the former, and the
  editor writes the value of the configuration that the form reads as before.
* A cross on each item of the workspace: the chips of its strip close in one click (or a
  middle click), without their menu, and an Undo beside the strip brings the item back for
  a few seconds with its number, title and objects; the session keeps the last sixteen
  items closed (`Session.restore`, `POST /api/workspace/restore`).
* The fields of a loading path in the page, without ParaView: the result of a path of the
  tab Mechanics opens the 3D view of its fields at once (on the voxels of an FFT run too),
  at the field of the form; a slider goes through the steps of the path (their manifest
  `<path>_steps.json`, with their times and driven strains) and Play plays them on one
  colour scale. No video is written unless asked (the option is off by default); Export
  makes one from the steps (a frame each, the mesh deformed by the displacement of each
  step), and writes the files of the steps with their ParaView collection (`steps.pvd`).
* The videos of every tool on demand, as those of the loading paths: Couplings
  (diffusion), Explicit dynamics, Fracture (growth, cohesive zones, polycrystals), Nonlocal
  damage (microfoam, ductile, quasi-brittle) and Stability (frames, morphologies, paths)
  write no video unless their option asks for one (off by default). Their fields go in
  the 3D view step by step instead (`ToolContext.steps`, `<name>_steps.json`: the fields of
  the times of a diffusion, of the steps of a damage, the animations of a dynamic run);
  the frames of the others (the growth of cracks, the opening of cohesive zones, the
  patterns of frames) are kept compressed (`<name>.frames.npz`). The tab Export takes the
  runs of the tools as it takes the simulations: their tables, fields and steps (with
  `steps.pvd`), the view of a field, its video along the steps, the pattern of a
  bifurcation oscillating (from `patterns.vtu`), and the videos of the frames kept (the
  kind "the frames of the run"); the steps of meshes that change (elements deleted,
  cracks) each drawn on its own mesh.
* Finite elements on conforming meshes by default, the voxels as the last resort:
  `homogenization.method: auto` takes the finite elements on the conforming mesh of each
  realization (from its exact geometry wherever the model knows it), with a light mesh
  when the configuration describes none (`config.default_mesh`: 10 elements along the
  shortest edge of the domain, `DEFAULT_ELEMENTS_PER_EDGE`, the interfaces followed
  within an eighth of an element and none below a third of one, `DEFAULT_HAUSDORFF`,
  `DEFAULT_MIN_SIZE`: some times fewer elements than the defaults of `conforming_mesh`;
  its level set sampled at the voxels of a `voxels` section, if any), and the FFT solver
  on voxels only for what only it does
  (`config.voxels_only`: the GPU, the coupled, harmonic and flow properties, the
  phase-field fracture, the reduced models of the NTFA, the window method of RVE studies,
  planar sections of voxels) or without the mesher MMG (with a warning). `morphogen
  homogenize geometry.json` meshes the geometry by default (the same light mesh;
  `--voxels N` for the FFT solver), and the tools of Mechanics (loading paths,
  macroscopic laws, tensors) take finite elements by default. Changed: a configuration
  without a method and without a `mesh` section ran the FFT solver.
* The particles of a phase without its matrix, for the discrete element method
  (`morphogen.io.dem`; [docs/theory.md](docs/theory.md) §100): a closed surface per
  particle cut from the implicit function of the phase (marching cubes), whole across the
  faces of a periodic box (its parts joined across the faces with their translations),
  cut and closed by the faces of a window, the clusters that wind around the box flagged;
  the exact spheres of unions of spheres and segments (a segment, a clump of spheres
  along its axis, `clump_spacing` radii apart at most). STL solids or a file per particle,
  OBJ, VTU, tables of the particles or of the spheres, LAMMPS and LIGGGHTS data files
  (atom style `sphere`, or `hybrid sphere molecular` with the clumps as molecules for
  `fix rigid/small`), YADE columns (`ymport.text`, `ymport.textClumps`), and Gmsh 2.2
  files for LMGC90 (`.msh`): the triangles of each particle, or its tetrahedra
  (`elements="tetrahedra"`, `particle_tetrahedra`: gmsh's mesher fills each surface, its
  triangles kept as their boundary; without gmsh, the fan of the triangles to the centre
  of a star-shaped particle), each particle an elementary entity and a physical group
  (`particle_<k>`, `particle_<k>_cut` if not whole), no blank line, as LMGC90's
  pre-processor reads them (`readMesh`, `separateMeshes`, then `surfacicMeshToRigid3D`,
  `volumicMeshToRigid3D` or `buildMeshedAvatar`). `size` remeshes the surfaces with
  MMG's `mmgs` (`remesh_surfaces`; `mmg_executable(program="mmgs")`) at regular
  triangles of that size, fewer faces for the contacts of polyhedra; the segments of the
  exact primitives are the closed surfaces of their spherocylinders (capsules of their
  volumes), no longer overlapping beads (`particle_spheres` marks the beads of the
  segments, `segment`). `export(..., particles=k)`, `morphogen export --particles k
  [--format yade] [--tetrahedra] [--element-size h]`, and the box *Particles (DEM)* of the
  tab Export (the Gmsh files of triangles or tetrahedra, a triangle size). Tested against
  the volumes, areas and centroids of spheres across the periodic faces, a cap cut by a
  window, a fibre that winds around the box, the matrix of spheres, the files read back
  (the Gmsh files as LMGC90's reader reads them, and by gmsh and meshio).
* Voxel images as the arrays of JAX, PyTorch and CuPy without copies, and phase fields for
  spectral solvers of phase-field equations (`morphogen.interop.arrays`;
  [docs/theory.md](docs/theory.md) §99): a `VoxelImage` hands its array over by DLPack
  (`__dlpack__`, `__dlpack_device__`); buffers aligned on 64 bytes (`aligned`,
  `aligned_empty`), which JAX shares on the CPU (`to_jax` gives an image an aligned buffer
  once, then the image and the JAX array share it; `to_torch`); `from_dlpack` brings a
  field or a channel of a state back as an image on its grid. `phase_field`: the diffuse
  interface `tanh(-d / (sqrt 2 epsilon))` of a phase, `d` the signed distance from the
  point-wise implicit function within `8 epsilon` (over the norm of its gradient, of the
  fourth order, when it is not a distance; `signed_distance`), the equilibrium profile of
  Allen–Cahn's equation for `epsilon = sqrt(nu)` and of Cahn–Hilliard's for
  `sqrt(gamma)`; `spectral_grid` and `state`, the arguments and the layout of the steppers
  of Exponax (the centres of the voxels as their grid: nothing interpolated);
  `realizations`, the initial conditions of ensemble rollouts (`jax.vmap`; the stochastic
  Allen–Cahn ensembles of exponax-spde). Checked by the motion by curvature of a disc of
  morphoGen in Exponax's Allen–Cahn equation, `R² = R0² - 2 nu t` within 1 %, and the
  mass kept by Cahn–Hilliard; JAX and Exponax optional (`tests/test_arrays.py`).
* Microgen (3MAH) with morphoGen (`morphogen.models.microgen`, `morphogen.interop.microgen`;
  [docs/theory.md](docs/theory.md) §101): the model `microgen` names a class of Microgen 2
  that has a field (`Tpms`, `Spinodoid`, `Sphere`, `Box`, `Cylinder`, `Capsule`,
  `Ellipsoid`...) and its `arguments` (a surface function by its name, `center` and
  `orientation` put into the field, the `part` of a TPMS); its realization,
  `MicrogenStructure`, is the inside of the field, voxelized, meshed from its implicit
  function and homogenized like any structure, saved and read back (a spinodoid with the
  seed of the realization). `to_microgen` gives a phase of any realization as a shape of
  Microgen (`from_field`: the point-wise implicit function of the phase, periodic), for
  its meshes, infills and smooth Booleans; `from_microgen` any shape as a structure;
  `to_pyvista` and `from_pyvista` the tetrahedral meshes of both (Microgen's mixed volume
  meshes split into tetrahedra). The template `microgen`, a sheet of Schoen's I-WP
  surface, and its card in the gallery; Microgen optional (the generic tests of the models
  skip it without the package).
* Blender's rigid bodies settle the particles of a phase (`morphogen.interop.blender`;
  [docs/theory.md](docs/theory.md) §102): `write_blender` writes the particles as rigid
  bodies (the exact spheres and spherocylinders of a union of spheres and segments as
  Bullet's spheres and capsules, the other particles as the convex hulls of their
  surfaces), an open box of passive walls, the world (gravity, substeps, iterations,
  friction, restitution, the masses from a density) in `scene.json`, and the script that
  Blender runs (`blender --background`, found on the path or by `MORPHOGEN_BLENDER`, or a
  Python with the module `bpy`); `run_blender` runs it, `read_blender` reads the packing
  back (the spheres and segments moved, or particles given by their surfaces) in the box
  up to its top, and `settle` does the three. `MeshParticles` (`models.mesh_particles`):
  particles given by closed triangle surfaces as a microstructure, voxelized by the parity
  of the crossings of the columns of voxels (`kernels.meshes`, particle by particle,
  through the periodic faces), their implicit function the signed distance of the voxels,
  saved and read back (the kind `mesh_particles`). The
  tab Export offers the scene for Blender among the formats of the particles. Checked
  with Blender 5.3 (`bpy`): hard spheres settle on the floor inside the walls,
  overlapping by less than 1 % of their radius; fibres keep their lengths and radii.
* Microstructures for Blender and its fluids (`morphogen.interop.blender`;
  [docs/theory.md](docs/theory.md) §102): `write_blender_scene` writes the closed surface
  of each phase of a realization (`phase_surfaces`: the boundary of the tetrahedra of the
  phase in the mesh conforming to its interfaces, from the exact geometry when the
  structure has it; the marching cubes of its implicit function without MMG, or with
  `surfaces="implicit"`) as binary STL files (`phase_<k>.stl`, which snappyHexMesh, the
  mesher of OpenFOAM, reads too), `scene.json` and the script that Blender runs
  (`blender --background --python morphogen_blender_scene.py`, or a Python with the
  module `bpy`): a mesh object per phase with its colour, and with `fluid` "gas" or
  "liquid" the fluid of Mantaflow around them, its domain the box of the sample extended
  upstream and downstream along `axis` (walls along the flow, its ends open), the phases
  its obstacles (effectors of collision), an inflow of `velocity` upstream and an outflow
  downstream, `resolution` cells, no gravity, the viscosity of a liquid; `scene.blend`
  saved, the flow baked if asked (`bake`). `read_blender_flow` reads the velocity and the
  density that Mantaflow caches (its `.uni` files, `read_uni`) on the cells of its
  domain, as voxel images for ParaView (`BlenderFlow.images`). `run_blender` runs both
  scenes (and takes the report of a script that went through when Blender fails on its
  way out, with a warning: the module `bpy` of PyPI 5.3 does so on a scene with a fluid,
  its Mantaflow lacking its Python bindings, and bakes nothing). The box *Blender (CFD)*
  of the tab Export archives the scene (the fluid and the axis of the flow chosen, the
  element size of the mesh). Mantaflow is a solver for visual effects: its flows show the
  streams through the pores, not quantitative Darcy flows. Tested: the closed surfaces of
  the phases and their volumes (the exact route of spheres within 3 %, the marching
  cubes of a mosaic of three phases within 0.02 of their fractions, and without MMG), the
  boxes of the fluid along x and z, the grids of Mantaflow written and read back
  (headers, values x fastest, frames, density, images), the export of the interface;
  with Blender 5.3, the scene built (the volumes of the objects, the obstacle, the inflow
  and the outflow, the walls and the open ends) and a flow baked (24 x 16 x 16 cells, six
  frames: nil in the obstacles, along the axis in the pores).
* Alloys from the tools of alloy design to their microstructures and their elasticity
  (`morphogen.interop.alloys`; [docs/theory.md](docs/theory.md) §103): an alloy described
  as a tool of alloy design gives it (Leil-Mat, Thermo-Calc, pycalphad: the composition,
  the phases of its equilibria and of its solidification, the precipitates of its
  ageing, the grain size, the moduli it predicts), in JSON or YAML: its matrix, other
  grains, precipitates and particles, their fractions, radii (a normal law truncated at
  three standard deviations) and elastic constants (isotropic, or cubic crystals turned
  at random). `alloy_scales` gives its scales as configurations of morphoGen, the
  precipitates in a box of about sixty of them, the grains (about sixty-four) with their
  particles as inclusions; `homogenize_alloy` their effective elasticity by FFT, the
  grains on the matrix of the precipitates, within the Hashin–Shtrikman bounds of their
  phases, and the departure from the modulus predicted. `morphogen alloy` writes the
  configurations and homogenizes them; `examples/alloy_design.py`. Leil-Mat gives its
  programming interface on request only: the description is filled from its canvas
  meanwhile.
* Version 1.1.0 (`morphogen.__version__`, `CITATION.cff` with its date of release, the
  recipe of conda-forge, the citation of [docs/citing.md](docs/citing.md)).

## 1.0.2 — 2026-11-19

* The documentation of the laser-architectured sheets reduced to the model:
  [docs/laser.md](docs/laser.md) keeps the choices of a study, the chain of models, its
  checks and its use. The studies, their figures, the literature survey and the scripts
  that ran them (`examples/laser_*.py`) are withdrawn, and the entries of this log on
  them keep the functions they added. The package `morphogen.laser`, its interface, its
  tests and its templates are unchanged.

## 1.0.1 — 2026-11-18

The window of the stand-alone application on Windows, which closed as it opened, and a
Windows installer of the application; the documentation site, its presentation
reorganized.

* The documentation site on GitHub Pages (https://jdirr.github.io/morphoGen/), built and published at each push
  to `main` (`.github/workflows/docs.yml`); the DOI of Zenodo in `CITATION.cff`, the README
  and `docs/citing.md` (all the versions: 10.5281/zenodo.24340315; 1.0.0:
  10.5281/zenodo.24340316).
* The presentation reorganized for a first reading, the content unchanged: the README is a
  short presentation (what morphoGen does, its installation from PyPI, a first study,
  where to read on); the user guide of the documentation has a page for each step of a
  study (models, geometry, measures, apparent properties, behaviours, fracture, the RVE,
  design), and pages for the applications (explicit dynamics), the configurations and the
  command line, the graphical interface and the other codes, which take the feature list
  and the examples of the old README; the keys of the configurations, the commands and
  the files of a run have their page in the reference (`docs/configuration.md`, from the
  architecture notes); the API reference opens on a map of the package, the theory notes
  on a map of their sections by theme, and the page of the laser-architectured sheets on
  the way it reads.
* A Windows installer of the stand-alone application (`packaging/desktop/morphogen.iss`,
  Inno Setup 6): `morphoGen-<version>-setup.exe` installs the bundle of PyInstaller for
  the current user, without administrator rights, with a shortcut in the Start menu and
  an uninstaller, and warns when the computer has no WebView2; its wizard speaks English
  or French. `packaging/desktop/check_installer.py` installs it silently, checks the
  program it installs, its shortcut and its entry among the installed applications, then
  uninstalls it; the workflow `desktop.yml` builds and checks it with the bundles, and
  attaches it to each release (to an earlier one on demand).
* The window of the stand-alone application on Windows: pywebview gave it the PNG icon of
  the package, which Windows Forms refuses, and the window closed as it opened (the check
  `--check` opens none). The package carries its Windows icon (`gui/icon.ico`, from
  `packaging/desktop`), given to the window on Windows; the test of the real window passes
  the icon as the application does, and `check_installer.py --window` opens and closes
  the window of the installed program.

## 1.0.0 — 2026-11-18

The first public release. From this version on, the public interface follows semantic
versioning ([docs/stability.md](docs/stability.md)): the names of
`tests/public_api.txt`, the commands and the keys of the configuration files, the
formats read and written. It gathers the versions below, which were not released: the
models, measures and representative volume elements; the finite element and FFT
solvers, their behaviours and their exports; the cellular automata, combinatorial
design and topology optimization; the GPU, MFront and AMITEX_FFTP; and the
multiphysics, fracture and contact, explicit dynamics with OpenRadioss and the
stand-alone application, textures and orientation maps, scale-rich networks and exact
meshes, and the laser-architectured sheets.

* Version 1.0.0: the classifier `Development Status :: 5 - Production/Stable`, the
  package open to PyPI; `CITATION.cff` with its version and its date of release; the
  recipe of conda-forge at 1.0.0. The workflow `release.yml` uploads the archives of
  each published GitHub release to PyPI by trusted publishing (`docs/release.md`).

## 0.6.0 — unreleased (phase 3b)

The FFT solvers on NVIDIA GPUs; hard-core and curved fibres, rotated and graded lattices;
AMITEX_FFTP workflows; MFront behaviours through MGIS, and MFront sources of morphoGen's
behaviours.

* The words of the laser-architectured sheets: a sheet is treated by the laser, and
  printing is that of additive manufacturing. `treatment_measures` for `print_measures`,
  `treatable` and `make_treatable` for `printable` and `make_printable` (all new in this
  version); the pages, the examples, the tests and the earlier entries of this log
  reworded.
* The explicit dynamics with OpenRadioss and the stand-alone application
  (`morphogen.dynamics`, `morphogen.io.radioss`, `packaging/desktop`; developed apart
  from 06/11, their entry below) merged with the multiphysics, the exact meshes and the
  laser-architectured sheets: the tab *Explicit dynamics* among those of the section
  Simulate, after Couplings; the inputs of OpenRadioss among the solvers of
  `morphogen.io`, beside the orientation maps; the buttons of the dialogs of the system
  beside the menus of the tabs and the search of the header.
* The pure shear test of the rubber strip (`tests/test_phasefield_finite.py`) on fully
  integrated voxels ([docs/theory.md](docs/theory.md) §64). Where its crack runs at a
  fixed stretch, the F-bar voxels of the band, sheared far, turn inside out under
  Newton's corrections from equilibria where every Jacobian is above 0.99, and dividing
  the step only replays the run of the crack: the test stopped at a stretch of 1.51 after
  ten divisions since the second-order prediction of the paths and the sums without BLAS
  changed the rounding of its iterates. Fully integrated (a rubber of Poisson's ratio 0.4
  does not lock), the crack starts at 1.44 and has crossed the cell at 1.51 (1.53 on the
  F-bar voxels when they went through), the area of the section times 1.00 (0.98), in
  about 110 s for 230 to 280 s. The limitation of the F-bar voxels is written in §64 and
  in the module `fracture.phasefield_finite`.
* One interface of the implicit functions, and the exact meshes wherever the model has
  an exact geometry ([docs/theory.md](docs/theory.md) §75, §76, §80, §94). (1) The
  functions of the phases at any point, `Microstructure.phase_level_sets_at(points,
  reach)` (`{k: phi_k}` for every phase), are those of `phase_level_set` on grids at the
  centres of their voxels: `level_set_at` is the function of phase 1, `implicit_at(phase,
  points, band)` one phase of `phase_level_sets_at` (None without point-wise functions),
  for every structure. Given now by the coatings and shells of unions of spheres and
  segments (all their phases, bores included), clustered spheres with their aggregates,
  inclusions over a matrix of any phases and nested structures, tessellations with a
  grain-boundary phase or of several phases, the cells of tilings, the images of these by
  an affine map; checked against the grids within 2·10⁻⁸ of their band for thirteen
  structures. The coatings and shells of structures that are not unions of spheres and
  segments have none (the distance of their voxels): their two-phase shells, exact before
  with the level set of the structure taken for a distance, now mesh on the grid. (2)
  `method: auto` of `conforming_mesh` takes the exact route for them all: partitions of
  several phases (the phase 0 the rest, or the last phase of a partition without it), the
  phases kept by a list (the whole partition meshed, then the phases taken; the cells and
  convex pieces too), the sharp edges of smooth pieces (`adaptive_mesh(sharp_edges=...)`,
  `exact_mesh(sharp_edges=...)`: the octree refined on the pieces, cut piece by piece,
  the nodes of the sharp edges on both surfaces even with 4-node elements; overlaps
  shallower than the Hausdorff distance rounded); tessellations with grain boundaries,
  and those meshed by their phases (`exact_route(micro, grains=False)`, `exact_mesh(...,
  grains=False)`), from the functions of their phases. Several phases are cut at once
  (each by its exact function at the vertices, without remeshing) and remeshed once, as
  on the grid, the lines where three phases meet kept as ridges: a sphere and its
  coating keep their volumes five to seven times better, with elements of twice the
  quality, than cut and remeshed phase by phase, and periodic dead leaves of three phases
  get matching faces in half the time. Two phases are cut and remeshed in one pass as
  before, at once should MMG fail or their periodic faces not match; then the grid, with
  a warning. The octree evaluates the functions within the length of the lines of each
  level, and `curved_mesh` within the size of the elements (dead leaves of three phases,
  elements of 0.3: 20 s of octree for 130 s). On the grid stay transparent dead leaves,
  the coatings of other structures, oriented cells but tessellations, and objects with
  level sets on grids only. (3) One curvature of the 10-node elements: `curved_mesh`,
  with the exact functions of every structure that gives them (the splines of a grid for
  the others), for the meshes of both routes; `adaptive_mesh(order=2)` curves its mesh
  with it (its own curvature of the middles of the edges is gone: `mesh.metadata
  ["curved"]` instead of `"curved_edges"`). The tests that quote the figures of the grid
  for coated spheres, three phases and sharp edges (`tests/test_sharp.py`,
  `tests/test_implicit.py`, `tests/test_curved.py`, the mesh of the tutorial on exact
  geometry) mesh on it (`method="grid"`); `tests/test_exact_phases.py` checks the exact
  route, and the default mesh of Boolean spheres that overlap is now exact
  (`tests/test_conforming.py`).
* The scale-rich networks and the exact meshes merged with the multiphysics, the
  quadratic elements and the sharp edges, developed apart: `method: auto` of the
  conforming meshes keeps the exact route for two-phase structures and the grains of
  tessellations, the 10-node elements of a level set those of its 4-node mesh curved by
  `curved_mesh` (the same mesh at both orders); the phases kept by a list, several
  phases and the sharp edges of smooth pieces (asked, or where two pieces of a phase
  meet) stay on the background grid ([docs/theory.md](docs/theory.md) §94); `method:
  exact` forces the exact route and refuses the phases kept by a list and the sharp
  edges. The sections 59 and 60 of the scale-rich branch are now §93 and §94. The
  tests that quote figures of meshes on the grid (the straight and curved elements of
  a sphere, the mean dilatation, the interface elements of a sphere, beads that
  debond) mesh on it (`method="grid"`). The returns of `grow_grains` in words: their
  names had become a reference of the documentation with two targets.
* A tutorial and a tool of the interface for the orientation maps.
  [docs/tutorial_texture.md](docs/tutorial_texture.md), *From an orientation map to a
  textured polycrystal*, in snippets that the tests run: the map that comes with
  morphoGen, its grains and its texture (pole figures 5.7 times random, 40 % of the
  area near the cube orientation, boundaries 38.4° apart for 40.7° at random), the
  stiffness of its polycrystal (the exact bulk modulus; 188 GPa along the first axis
  for 227 GPa with orientations at random), a tension in finite strains in which the
  lattices turn by 0.7° on average at 2 %, the fibres of a Taylor polycrystal at a
  stretch of 1.5, and a polycrystal of space with the texture of the map. The tool
  *Orientation map (EBSD)* of the tab Statistics of the interface reads an `.ang` or a
  `.ctf` file, gives the numbers of its grains and of its texture, the tables of the
  grains and of their boundaries, the map in the colours of its crystals, its pole
  figures, its inverse pole figure and the disorientations of its boundaries, and adds
  its polycrystal to the workspace.
* The lattices along a loading path ([docs/theory.md](docs/theory.md) §92).
  `simulate_point` gives the orientation of the lattice of a crystal at each step, in
  finite strains and where its twins turn it (`PathResult.orientation`, the columns
  `phi1`, `Phi`, `phi2` of its curves); `PathResult.texture(time)` gives the lattices of
  the crystals of a volume element at a time of a path, with their volumes, from its
  fields; `pipeline.read_path_texture` reads them from the files of the fields, and
  `morphogen texture FIELDS` (`.vti`, `.vtu`, `--lattice`) prints the largest densities
  of their pole figures and of their inverse pole figure and draws them. Checked: a
  crystal in single slip turns as Schmid and Boas found, in tension (`l/l₀ = sin λ₀ /
  sin λ = sin χ₀ / sin χ`: the axis from 40° to 29.63° of the slip direction at a stretch
  of 1.3) and in compression (`h/h₀ = sin φ / sin φ₀`: from 40° to 28.82° of the normal
  of the slip plane), within the elastic strains (2 × 10⁻³); 50 face-centred cubic
  crystals in the same extension (Taylor) take the ⟨111⟩ and ⟨100⟩ fibres in tension
  (3.8 and 1.2 times random, ⟨110⟩ at 0.4) and the ⟨110⟩ fibre in compression (2.2).
* The figures and the command of the textures; the texture of a map in a computed
  polycrystal ([docs/theory.md](docs/theory.md) §92). (1) `morphogen texture MAP` reads
  an orientation map (`.ang`, `.ctf`), finds its grains and prints their numbers phase
  by phase (grains, mean diameters, spread inside the grains, the mean disorientation of
  the boundaries, the largest density of the pole figures); with `-o` it writes the
  tables of the grains and of their boundaries, the map in the colours of its
  orientations and, with matplotlib, the figures; with `--convert` the map in the other
  format, as it is or cleaned (`--clean`). (2) Figures (`morphogen.viz.figures`):
  `pole_figures_figure`, `inverse_pole_figure_figure`, `ipf_map_figure` (the map with
  its grain boundaries and its colour key, `ipf_key`), `disorientation_figure` (against
  Mackenzie's density). (3) `morphogen.texture`: the density of an inverse pole figure
  in multiples of a random texture (`inverse_pole_figure`, `InversePoleFigure`), the
  directions of a crystal in the fundamental sector, their colours and their points on
  the figure (`fundamental_directions`, `direction_colours`, `sector_points`,
  `SECTOR_CORNERS`). (4) `morphogen.io.ebsd.Grains`: `table`, `boundaries` (the pairs of
  neighbouring grains of one phase, the lengths of their boundaries, their
  disorientations) and `summary`. (5) The texture of a map for the cells of a
  tessellation of space: `texture: {type: measured, file: map.ctf, spread: 2.0}` of
  `voronoi` and `laguerre` (`morphogen.orientations.Measured`): each cell takes the
  orientation of a grain of the map, drawn with its area, then scattered by `spread`
  degrees; the file is taken from the folder of the configuration
  (`path_parameters`: keys of mappings too, and in the models that a model holds).
  Checked: the density of the inverse pole figure of one crystal against the kernel
  over the number of equivalent axes (24, 4, 3, 6; 1 and 3 for the hexagonal lattice),
  the boundaries against the sides counted one by one, the orientations of a Voronoi
  polycrystal against the grains of the map.
* Crystallographic textures and orientation maps (EBSD)
  ([docs/theory.md](docs/theory.md) §92): two modules and a model. (1)
  `morphogen.texture`: the proper rotations of the eleven Laue classes (`symmetries`);
  the disorientation of two crystals (`disorientation`: at most 62.80° for cubic
  crystals, 93.84° for hexagonal ones), exact to rounding for crystals that are close,
  and the density of Mackenzie (1958) for random cubic crystals (`mackenzie`: mean
  40.74°); the mean orientation and the spread of the points of a grain
  (`mean_orientation`); the poles of the families of planes and directions of the cube
  and of the hexagonal lattice, their stereographic and equal-area projections, and
  pole figures in multiples of a random texture, each pole spread by an axial von
  Mises–Fisher kernel of a given half-width (`pole_figure`); inverse pole figures and
  their colours (`inverse_poles`, `ipf_colours`); the disorientations of the boundaries
  of an image of grains (`boundary_disorientations`); orientations drawn from a measured
  texture (`sample_texture`). (2) `morphogen.io.ebsd`: orientation maps (`EBSDMap`) read
  and written as `.ang` (EDAX/TSL: square and hexagonal grids) and `.ctf` (Oxford
  Instruments/HKL) files, with the points that were not indexed, the phases and their
  symmetries. The orientations are brought into the axes of the image of the map (the
  first to the right, the second upwards, the third towards the viewer): those of a
  `.ctf` file; those of an `.ang` file turned about the normal according to the setting
  of the acquisition (`setting`: 1 to 4, numbered as by MTEX's reader; `axes="file"`
  keeps the angles of the file). The hexagonal and trigonal crystals of a `.ctf` file,
  whose first axis is along `a*`, are turned by 30° about `c` to the axes of morphoGen
  (the first along `a`, as for its slip and twin systems). The grains of a map
  (`EBSDMap.grains`): the connected sets of neighbouring points less than a threshold
  apart, the small ones dropped, the points without a grain given to the nearest one;
  their mean orientations, spreads, areas and diameters; their polycrystal
  (`Grains.microstructure`). The map that an acquisition would give of a layer of a
  computed polycrystal (`synthetic_map`: scatter of the orientations, points not
  indexed). (3) The model `ebsd`: a window of a map of the size of the domain — its
  first rows and columns, or drawn at random for each realization — as a polycrystal of
  one cell through the thickness, for every solver; in a configuration its `file` is
  taken from the folder of the configuration file (`Model.path_parameters`). A
  demonstration map comes with morphoGen (`io.ebsd.example_map`, written by
  `examples/ebsd_map.py`), with the template `ebsd` (windows of the map homogenized by
  FFT: the bulk modulus of the crystals of nickel, 180.37 GPa, for every window) and
  its card in the gallery. morphoGen has 39 models. Checked against closed forms
  (`tests/test_texture.py`, 28 tests): the disorientations of known rotations (10⁻⁶
  degrees), Mackenzie's density (its integral, its mean, 200,000 random pairs), the
  poles of the cube orientation, the pole figures of a random texture and of one
  crystal, the files of the two systems and their axes, the grains of a noisy map of
  known grains, the stiffness of a map of one crystal (10⁻⁹) and the bulk modulus of
  windows of a map (10⁻⁶).
* The product by nodes, and the preconditioner of a path kept from a step to the next
  ([docs/theory.md](docs/theory.md) §11, *The product by nodes*, *The preconditioner
  along a path*). (1) The products of the Krylov iterations read the matrix by the
  3 × 3 blocks of its nodes, in double precision (`amg.NodeBlocks`, `blocks` of
  `LinearSolver`): the same product to the bit as that of the stored values — the same
  iterations, the same results —, three times faster where the matrix fits the caches
  (2.2 ms for 7.0 ms on 76,398 unknowns) and a fifth of the solves of the large linear
  analyses, which the memory bounds. The blocks are held with the stored values: 70 %
  of the memory of the matrix more. (2) Along a loading path the blocks take the values
  of each tangent at their places, found once (`NodeBlocks.refresh`), and the sweeps of
  the multigrid preconditioner follow (`Smoother.refresh`): the sweeps of the tangent
  as it is at every Newton iteration, where they were those of the first iteration of
  the step. (3) The interpolation and the coarse levels of the preconditioner are kept
  from a step to the next, and built again once the solves take 1.3 times the
  iterations per decade of residual that they took when they were built
  (`LinearSolver.worn`, `solvers.WORN`); the preconditioner stays symmetric positive
  definite whatever their age. `homogenization_viscoplastic` builds two preconditioners
  for its 25 steps and `homogenization_crystal_plasticity` four for 67, with fewer
  linear iterations (1,867 for 1,917; 3,937 for 4,071), the same Newton iterations and
  the same curves to the tolerance of the equilibria. (4) The reduction of the tangent
  sums the values of each place of the reduced matrix on several threads. On four
  threads, back to back: `homogenization_viscoplastic` in 41 s for 61 s,
  `homogenization_crystal_plasticity` in 14 s for 20 s, `homogenization_fatigue` in 21 s
  for 25 s, `homogenization_polycrystal` in 14 s for 18 s (its solves: 7 s for 11 s),
  the solves of `homogenization_gyroid` in 33 s for 42 s and those of
  `homogenization_spheres` in 68 s for 85 s. `LinearSolver.updated` now changes the
  solver that it is called on (its blocks and its sweeps take the new values): that
  solver is not to be used afterwards.
* The sweeps by nodes, the assembly by colours, and the threads of BLAS left asleep
  ([docs/theory.md](docs/theory.md) §11, *By nodes*, *The threads of BLAS*, *The
  assembly by colours*): three changes that leave the iterations and the results as
  they are. (1) Where the unknowns are the components of the displacements of the
  nodes, the Gauss–Seidel sweeps of the multigrid preconditioners work by nodes: the
  nodes are coloured, the copy of the matrix that the sweeps read holds the 3 × 3 blocks
  of their couplings (one column index for nine values), and the three components of a
  node are relaxed one after the other. It is the same sweep as that of the unknowns
  coloured one by one — the greedy colouring gives the components of a node three
  colours that follow each other —, read twice as fast (one application of the
  two-level preconditioner on 76,398 unknowns: 4.9 ms for 9.0 ms). The four kinds of
  boundary conditions work by nodes: the solvers are given the degree of freedom of
  each unknown (`dofs` of `LinearSolver`, `TwoLevelPreconditioner`,
  `SmoothedAggregation` and `amg.Smoother`), and a component that a condition holds
  leaves its place empty. (2) A product of BLAS between two compiled kernels wakes the
  threads of OpenBLAS, which stay busy for a tenth of a second and take the cores of
  the kernels that follow (a product of a sparse matrix: 11.7 ms for 3.7 ms). The dense
  matrix of the coarsest level of a multigrid cycle, the norms of the solves and the
  products of the condensation of a path are no longer products of BLAS
  (`amg.dense_product`, `solvers.times`), and the loading paths and the Krylov solvers
  run with the BLAS on one thread, its threads restored afterwards
  (`settings.blas_on_one_thread`). (3) The tangent, the nodal forces and the couplings
  of the finite element paths are assembled on several threads: the elements are
  coloured so that two elements of a colour have no node in common
  (`fe.element_colours`), and those of a colour are assembled in parallel — the same
  sums for any number of threads (the tangent of 19,025 10-node tetrahedra: 42 ms for
  158 ms). On four threads, measured back to back: `homogenization_viscoplastic` in
  49 s for 93 s, `homogenization_ductile_damage` in 30 s for 59 s,
  `homogenization_crystal_plasticity` in 16 s for 22 s,
  `homogenization_crystal_plasticity_fft` in 74 s for 86 s, `homogenization_rubber` in
  58 s for 67 s; the linear analyses `homogenization_polycrystal` in 15 s for 26 s (its
  solves: 9 s for 18 s), `homogenization_gyroid` in 103 s for 128 s (35 s for 57 s),
  `homogenization_spheres` in 160 s for 198 s (72 s for 107 s). The kernels
  `_fill_tangent`, `_internal_forces` and `_coupling` of `homogenization.nonlinear`,
  their counterparts of `homogenization.finite` and `dilatation.fill_bar` take the
  colours of the elements as arguments.
* The finite element solvers on several threads ([docs/theory.md](docs/theory.md) §11,
  *Linear solvers*). On large systems the Gauss–Seidel sweeps of the multigrid
  preconditioner and the products of the Krylov iterations ran on one thread; they run
  on all of them, with a result that does not depend on their number. The unknowns are
  coloured so that no stored value couples two of a colour, and those of a colour are
  relaxed together: a Gauss–Seidel sweep in the order of the colours
  (`homogenization.amg.Smoother`), which reads a copy of the matrix made for it — rows
  sorted by colour, single precision, 32-bit indices, the parts below and above the
  diagonal apart (the forward sweep from zero reads the first only, the residual that
  follows it the second only). The products share the rows between the threads, each
  row summed as SciPy sums it (`homogenization.amg.product`: the same product to the
  bit). Along a loading path, the reduced tangent `TᵀKT` is filled by the places of
  the stored values of `K`, found once (`homogenization.fe.Reduction`), where two
  products of sparse matrices were made at each linearization (1.3 s for 17.5 s over
  the path of `homogenization_viscoplastic`). On four threads, measured back to back:
  `homogenization_spheres` (368,577 unknowns) in 192 s for 342 s (its solves: 106 s for
  248 s), `homogenization_gyroid` in 124 s for 200 s, `homogenization_polycrystal` in
  24 s for 34 s, `homogenization_viscoplastic` in 86 s for 129 s,
  `homogenization_crystal_plasticity` in 21 s for 32 s. The conjugate gradients take
  the iterations that they took within 1 % (4 % on the crystal path: 4,071 for 3,905),
  and the properties and the curves are the same to the tolerance of the solves. Small
  systems (fewer than 200,000 stored values) keep the sweep in the order of their
  unknowns; the paths on voxel images do not use these solvers.
* Loading paths in fewer iterations ([docs/theory.md](docs/theory.md) §31, *The
  prediction of a step, the secant of the condensed block*). A step starts from a
  second-order prediction (the rates of the last two steps and their change, exact for
  unequal steps) where it would have predicted the last step clearly better than the
  rate of the last step alone; a response that turns (a yield, a crack, twins that
  start) keeps the first order (`prediction = 1` on a solver keeps it everywhere). The
  condensed block of the macroscopic unknowns that the iterative solvers keep from a
  correction to the next is updated by the secants of the corrections that converge
  (BFGS with Powell's damping; Broyden's update of a block that is not symmetric). In
  a steady plastic flow, two corrections a step where three were needed: Newton and
  linear iterations 200 → 170 and 5,510 → 3,905 for `homogenization_crystal_plasticity`
  (42 s for 54 s), 149 → 129 and 3,535 → 2,678 for
  `homogenization_crystal_plasticity_fft` (115 s for 132 s), 133 → 120 and 12,059 →
  9,352 for `homogenization_ductile_damage` (64 s for 73 s), 79 → 69 and 2,225 → 1,899
  for `homogenization_viscoplastic`, 26 divided steps for 34 on
  `homogenization_slip_cleavage`; the paths where cracks or twins start keep their
  iterations and their divided steps. The same curves to the tolerance of the
  equilibria. Measured and not kept: a reference medium of the Fourier preconditioner
  that follows the mean tangent, and a copy of the tangents in single precision for the
  Krylov iterations — neither shortens a solve.
* The integration of the points, two to three times faster for the crystals, with the
  same results to the bit ([docs/theory.md](docs/theory.md) §32). The work arrays of
  the laws (crystals, von Mises, Hill, porous, logarithmic strains) are allocated for
  each chunk of points before the parallel loop over the chunks, which then holds them
  as plain memory: the rows that the kernels took of arrays allocated inside the loop
  counted their references, atomically — a third to a half of the time of a crystal
  point (`work_blocks` and `work_block` of `homogenization.crystal`, `point_chunks` of
  `homogenization.behaviours`). The crystals no longer compute what a point does not
  need: a point that stays elastic is evaluated once, without derivatives; the
  derivatives of the resolved stresses are those of the active systems (finite strains),
  or are kept with the Schmid tensors of the grain (small strains), themselves kept
  from a point to the next (`cp_small(..., key)`); the correction that the line search
  accepts is the next iterate, not evaluated again; no exponential for a system that
  has not slipped, nor for a plane that has not opened. Evaluations of the points of the
  templates replayed on one thread: 7.4 → 2.6 µs per point for the cleavage of a
  polycrystal (21 systems, 88 % of elastic points), 21.5 → 7.7 µs for the nickel
  polycrystal in finite strains (all its points flow), 15.8 → 8.7 µs for the twins of
  magnesium; von Mises points 0.5 → 0.3 µs, Hill 3.2 → 2.4 µs. The templates, on four
  threads, with the same curves to the last digit:
  `homogenization_crystal_plasticity_fft` 148 s for 210 s (its points: 64 s for 123 s),
  `homogenization_cleavage` 36 s for 49 s, `homogenization_slip_cleavage` 35 s for
  60 s, `homogenization_twinning` 46 s for 58 s, `homogenization_cleavage_band` 12 s
  for 18.5 s.
* The alternate scheme of the microdamage, accelerated ([docs/theory.md](docs/theory.md)
  §91), after the accelerated staggered scheme of Storvik et al. (2021): the updates of
  the microdamage extrapolated where their residual falls (Anderson's acceleration of
  depth one) and relaxed where it grows (a crack that starts to run); the equilibria
  between two updates solved to a share of the last change of the microdamage, the last
  one of a step to the tolerance; the condensation of the macroscopic unknowns kept
  from a pass to the next. A bar that breaks across its band: half the equilibria (346
  for 588), and a work 0.10 % above its settled value for 0.36 % with the plain scheme;
  the template `homogenization_cleavage_band`: 25 s for 49 s, a third of the linear
  iterations, the same curve. Options `nonlinear: {microdamage: {tolerance, passes,
  acceleration}}` of the configurations (`acceleration: false`: the plain scheme), and
  `microdamage_options` of the solvers of the paths.
* Cleavage planes regularized by their microdamage ([docs/theory.md](docs/theory.md)
  §91), the micromorphic model of Aslan & Forest (2009): `cleavage: {energy,
  regularization: {width}}` (or `{modulus, length}` with the linear `softening` of the
  planes). The strength of the planes of a point falls with their damage and with a
  field that follows it over a length, the microdamage (`χd − ℓ² Δχd = d`): a plane
  breaks across a band of a width of its own, where its fracture energy is dissipated,
  whatever the grid — no length of elements to give. The loading paths solve the
  equilibrium of a step at a fixed microdamage, then the microdamage of the damage
  reached (in Fourier space on voxel images, by finite elements on meshes), until it
  settles; the planes whose strength has come to nothing are broken through, and the
  band does not widen as the crack opens. Functions `microdamage_band` and
  `microdamage_parameters` (the closed form of the band: its width, its fracture
  energy); field `cleavage_microdamage`, and `cleavage_damage` as the share of its
  strength that a plane has lost; `microdamage_passes` and `microdamage_unsettled` in
  the information of a path. Checked against the closed form of a bar that breaks: its
  softening branch and the fields of its band within 0.5 %, a work within 0.3 % of the
  fracture energy on 64 voxels and that comes to it at the second order of their size
  (−1.2 %, −0.28 %, −0.04 % on 32, 64 and 128). Template
  `homogenization_cleavage_band`; *width of the damage band of the planes* in the
  crystal form of the interface. The local iteration of the crystals drops the systems
  that an active set has brought below their threshold with a reduced step of the
  others, instead of failing (planes with several openings past their strength).
* Cleavage that the slip nucleates ([docs/theory.md](docs/theory.md) §90): `cleavage:
  {coupling: Θ}`, the coupling of plasticity and damage of Aslan, Cordero, Gaubert &
  Forest (2011). The strength of the cleavage planes of a crystal falls with its
  cumulated slip, `Y = Y₀ exp(−w d) exp(−Θ γ_cum)`, and their damage takes `Θ exp(−Θ
  γ_cum) Σ_p (Y₀/w)(1 − exp(−w d_p))` off the resistance of its slip systems (the two
  derive from one energy). A crystal that yields below the strength of its planes no
  longer slips for ever: its plane opens at `γ_cum = ln(Y₀/σ_n)/Θ`; a polycrystal of
  iron breaks at 576 MPa and 1.6 % where it broke at 648 MPa and 2.4 %. 0 by default (a
  constant strength); a coupling that would leave the slip without resistance is
  refused, with its bound. Checked against the flow rules of the two (10⁻⁶), the
  Jacobian and the tangents, cells against the point. `CrystalPlasticity.
  cleavage_coupling`; `kinds_array` has ten columns (`CP_COUPLE`, `CP_WIDTH = 17`), the
  compiled kernels are revised; the crystal form of the interface has the coupling; the
  template `homogenization_slip_cleavage` (a polycrystal that yields, then cleaves).
* Twins and cleavage planes, what their reversal had left
  ([docs/theory.md](docs/theory.md) §89):
  * **Lattices turned grain by grain.** `twin_reorientation: {threshold: 0.2, growth:
    0.0}`: the predominant twin reorientation of Tomé, Lebensohn & Kocks (1991), the
    integration points of a grain in place of the grains of an aggregate. After each
    step, in each grain, the points whose twinned fraction has grown past `threshold +
    growth × turned volume / twinned volume` take the lattice of their predominant
    twin, the most twinned first, while the turned volume is below the twinned one; the
    lattices come back in the same way when the twins shear back. The turned volume of
    a grain is then its twinned volume, to a point: at one threshold for every point
    (a number, as before) a textured magnesium has 65 % of its points turned for a
    twinned volume of 0.47, from 0.30 below the twinned volume of a grain to 0.36 above.
    `crystal.turn_twins`, `crystal.turn_lattice`, `CrystalPlasticity.by_grain` and
    `twin_reorientation_growth`.
  * **A resistance of their own on the way back.** `twin_detwinning: 0.5`: the twins
    shear back at this share of the resistance of the twinning (no twin to nucleate; Lou
    et al. 2007, Hama et al. 2016); `cleavage: {closure: 0.2}`: a plane that is not
    broken through closes at this share of its strength (the reference: all of it).
    `true` is 1, `false` 0.
  * **The twinned fraction is that of the shears of the twins**, which a turn does not
    change: `twin_fraction` of a turned point is the fraction that it had when it was
    turned, less what its twin has undone since (it was 1, less that); when its lattice
    comes back, what is left of its twin goes on shearing back in the lattice of the
    grain: a point that has twinned and untwinned is as it was. The new field and
    curve `twin_reoriented` is the turned volume (1 where the lattice is that of a
    twin).
  * **The cleavage planes of the lattice as it is.** The fields of a path have the
    `orientation` of the lattices where twins turn them, in small strains too (of most
    of the points of an element), and the criterion of Beremin takes the cleavage planes
    of the lattice at each time: the basal plane of a twin of magnesium is 86° off that
    of its grain.
  * Twins whose fractions are within 10⁻⁶ of the largest are as predominant, and the
    first of them turns the lattice: two variants that the symmetry of a loading makes
    equal differed by their round-off, and the points of a homogeneous cell could take
    one or the other.
  * The template `homogenization_detwinning` turns its lattices grain by grain (its
    result figure has the twinned and the turned volumes); the crystal form of the
    interface has the growth of the threshold and the two shares.
  * `kinds_array` has one column more (`CP_SHEAR`: the twinned fraction per unit of
    shear), the turned lattices eleven internal variables (`CP_TURNED`: the twinned
    fraction at the turn); the compiled kernels are revised.
* Cleavage cracks that close, twins that shear back and turn the lattice of the crystal
  ([docs/theory.md](docs/theory.md) §89): what the crystals of `crystal_plasticity` do
  when the stress is reversed.
  * **Closure.** The opening of a cleavage plane follows the sign of its normal stress,
    back to nothing (Aslan & Forest 2009), and the closed plane carries the pressure as
    the sound crystal: a crystal opened by 0.6 % and pushed to −0.4 % carries `E ε`, −521
    MPa, where it carried −1283. The default; `cleavage: {closure: false}` keeps the
    opening. `cleavage: {friction: μ}`: the lips of a plane under pressure rub
    (Coulomb; 0 by default) — in place of the reference, which stops the slips of a
    closed plane: its discontinuity kept a polycrystal with cohesive boundaries from
    converging. `cleavage_opening` is now the opening of the planes at the step, not
    what they have opened.
  * **Detwinning.** The twins shear back under a reversed stress, down to none, at the
    resistance of the twinning (`twin_detwinning`, the default; their second internal
    variable is their net shear). A textured magnesium twinned at 39 % is back at 4 %
    under −89 MPa, where it slipped at −155 MPa with its twins.
  * **The lattice of a twin.** `twin_reorientation: 0.5`: at this twinned fraction the
    lattice of a point becomes that of its predominant twin (Van Houtte 1978; Tomé,
    Lebensohn & Kocks 1991), half a turn about the normal of the twin plane: its
    systems, elasticity and thermal strain are turned, its twins start again, and it
    twins back by the same system (the other twins of the family are held: the
    conjugate variant of a twin of small shear would take the detwinning, and leave the
    lattice 7.4° off). The field and the curve `twin_misorientation`, the
    disorientation of the lattice from that of its grain (86.3° for {10‑12} in
    magnesium, 60° for the cubic twins); the `orientation` of the elements follows.
    `CrystalPlasticity.reorients`, `crystal.disorientation`, `crystal.rotate_stiffness`;
    internal variables more for these crystals (`state_width(..., turns=True)`).
  * The implicit step holds at nothing a system that would go back further (a crack
    closed, a twin undone) instead of shortening its Newton step: several twins undone
    together stalled the line search and fell back on substeps.
  * Templates `homogenization_detwinning` (a textured magnesium pulled and brought
    back, its lattices turning with its twins) and
    `homogenization_crack_closure` (a polycrystal of iron cracked, then pushed: −438
    MPa at −0.2 %, the stress of the sound polycrystal within 1 %); the crystal form of
    the interface has the four options; two snippets in the tutorial *How a
    microstructure breaks*.
  * `CrystalPlasticity.point_fields` takes the slips of the step and the variables of
    the turned lattices (`state_fields` reads them from the states); `kinds_array` has
    the columns of the kinds of the systems (`CP_WIDTH`).
* Fixed: a material point in finite strains whose path came back to no strain and no
  stress with a plastic history (a crack that has closed) did not converge: the floor of
  its tolerance, the round-off of the stiffness times the strain, vanished there; it is
  now that of the gradient `I + H`.
* Contact points at the nodes of the faces, for bodies that meet by their corners and
  edges ([docs/theory.md](docs/theory.md) §82): `points="nodes"` of a `Pair`, an
  `Obstacle`, a `ContactCell` and `compression`; `contact: {points: nodes}` in a
  configuration, `morphogen compress --points nodes`, the field *contact points* of the
  compression tool of the interface. The points inside the faces (the default, `"gauss"`)
  leave the corners and the edges without a point: a cube that stands on a corner went
  through a plane raised against it until the points next to the corner pressed; at the
  nodes (the vertices of the faces; the middles of their edges and their centroids as
  well on faces of six nodes) the plane carries it at once.
  * The weight of a point at a node is the share of its face that the other surface sees
    (the cosine of the angle of their normals): the sides of a block that stands on a
    plane, which share the nodes of the edges of its base, carry nothing — the uniaxial
    stress and the area of the base to 1e-6, where whole weights made the block 4 %
    stiffer. The areas in contact of the results are those of the shares (the state of a
    slave point has an eleventh value, the share; `STATE = 11`).
  * A node of the rim of the slave that lies on the rim of the master (two bodies of the
    same section) still faces it within a twentieth of a face beyond the rim: Newton's
    method did not converge otherwise.
  * The rule is of lower order: between meshes that do not match, the force of two
    blocks pressed together is found to 0.3 % (0.006 % with the points inside the faces).
* A tutorial, *How a microstructure breaks* ([docs/tutorial_fracture.md](docs/tutorial_fracture.md)):
  two layers whose interfaces debond (their strength, their fracture energy), a pore
  under a cyclic load, a crystal and a textured polycrystal of magnesium that twin one
  way, a crystal and a polycrystal of iron that cleave, the probability of cleavage of
  Beremin; its snippets are run by the tests, its figures written from them.
* The compiled functions that hold the kernels of the crystals (`update_points`,
  `update_points_finite`) carry a revision, so that the caches that Numba kept from an
  earlier version are compiled again: without it, a source tree updated in place would
  have run the earlier kernel, the twins and the cleavage planes ignored.
* Fixed: the fatigue of a cycle on a voxel image with pores (`path_fatigue`, the
  configurations with `fatigue` and `method: fft`) stopped on the stresses of the
  voids, which are not defined; these points are now left out, as the phases without a
  criterion.
* The probability of cleavage by the local approach of Beremin
  ([docs/theory.md](docs/theory.md) §88): `morphogen.homogenization.beremin`.
  * `path_beremin(result, Beremin(modulus=22, scale=2600, volume=1.25e-4))` on the fields
    of a loading path: the Weibull stress of the points of the plastic zone
    (`plastic`) at each step and the probability that the weakest link has failed, that
    of the largest Weibull stress reached. The stress that drives a point is its largest
    principal stress or, for the grains of a polycrystal, the largest normal stress of
    their cleavage planes (`planes: cube`, with the orientations of the grains).
  * In a configuration, `homogenization: {beremin: {modulus: ..., scale: ..., volume:
    ..., planes: cube, plastic: 1e-4}}`: the curves `beremin_weibull` and
    `beremin_probability` of the path (and of the RVE studies), the field
    `beremin_stress`. The template `homogenization_beremin`: a ferritic polycrystal
    whose probability of failure rises from nothing at its yield stress to a certainty
    as it hardens.
  * Checked against closed forms: a uniform stress, the weakest link of two volumes,
    crystals turned under a tension, a homogeneous cell loaded and unloaded.
  * In the graphical interface, the box *Cleavage* of the tool *Loading path of a
    morphology*: the modulus, the scale and the reference volume, the stress that drives
    the cleavage (the principal stress, or the planes of the crystals of the grains) and
    the plastic strain of the plastic zone; the curves and a figure of the probability
    and of the Weibull stress along the path.
* Crystals that twin and cleave; hexagonal crystals; cracks through the grains of a
  polycrystal and between them ([docs/theory.md](docs/theory.md) §86–87).
  * **Hexagonal crystals.** The slip families `basal`, `prismatic`, `pyramidal_a`,
    `pyramidal_ca` and `pyramidal2_ca` of a lattice of ratio `c_over_a`, beside the
    cubic ones, in `crystal_plasticity`.
  * **Twins.** `twins: {twin1012: {tau0: 25, ...}}`: systems that shear in one sense
    (the pseudo-slip of Kalidindi), until the fraction `twin_saturation` of the crystal
    has twinned. Families `twin111` (fcc), `twin112` (bcc), `twin1012`, `twin1011`,
    `twin1122` and `twin1121` (hexagonal), their shear computed from their second
    undistorted plane (`twin_systems`). The fields and the curves of a path have the
    `twin_fraction`.
  * **Cleavage.** `cleavage: {planes: cube, strength: 500, energy: 0.05, length: 0.008,
    K: 50, n: 1}`: the planes of a family (`cube`, `dodecahedral`, `octahedral`; `basal`,
    `prismatic`) open at a normal stress and soften, their accommodation systems release
    their shear (the damage systems of Marchal et al. and Aslan & Forest); the fracture
    energy of the planes is dissipated in a band of one element (`energy`, `length`), or
    `softening` is given. Planes that soften faster than the crystal unloads are refused,
    with the bound on the size of the elements. Fields and curves: `cleavage_opening`,
    `cleavage_damage`.
  * **Polycrystals.** With `interfaces: {faces: grains}` the boundaries debond while the
    grains slip, twin and cleave: the crack goes between the grains or through them,
    whichever is weaker. Templates `homogenization_twinning` (a textured magnesium: twins
    in tension along `c`, none in compression, half the flow stress),
    `homogenization_cleavage` (iron at a low temperature on voxels) and
    `homogenization_grain_fracture` (cohesive boundaries and cleavage planes on a mesh).
  * **Textures.** `texture` of the tessellations: a law of the axis `c` of the grains
    (`{type: von_mises_fisher, kappa: 30, axis: [0, 0, 1]}`), the crystals turned at
    random about it (`morphogen.orientations.fibre_orientations`).
  * Checked against closed forms: the shears of the twins, Schmid's law in one sense,
    the saturation, `σ = Y₀ exp(−w δ)` and the work `Y₀ / w` of a plane, the fracture
    energy of a crack band, the consistent tangent. `CrystalPlasticity` has the fields
    `twins`, `cleavage`, `c_over_a` and `twin_saturation`, and `n_slip`, `n_twins`,
    `n_cleavage`, `extended`, `kinds_array`, `point_fields`. The solvers of the CPU only:
    the GPU kernels and the MFront sources refuse these crystals.
  * **Interface.** The crystal of the forms of the mechanics tools (the loading path,
    the material point, the identification) has the hexagonal families and elasticity,
    the ratio c / a, the twins (families, critical stress, saturation) and the cleavage
    planes (strength, fracture energy, size of the elements, viscosity); the twinned
    fraction and the damage of the planes are fields of the videos.
* The graphical interface: interfaces that debond and the fatigue of a cycle, in the
  tool *Loading path of a morphology* (Simulate › Mechanics), each in a box of its form.
  * **Interfaces.** On the conforming mesh (finite elements), "the interfaces debond":
    those of the phases, or of the grains of a tessellation, with their strength, their
    fracture energy (left blank: that of an opening at failure of 2 % of the edge of the
    domain), their friction and a viscosity; the debonded area, the damage and the
    opening of the interfaces join the values and the curves, their fields those of the
    simulation (and of the video of the path).
  * **Fatigue.** A criterion (Crossland, Sines, Dang Van, Findley, Matake), its two
    fatigue limits and the times of the cycle: the utilization, the factor of safety and
    the share of the volume beyond the limit in the values, the fields of the criterion
    with those of the end of the path, and a figure of the share of the volume above
    each utilization.
  * The configuration written beside the simulation (`configuration.yaml`) has the
    `interfaces` and the `fatigue` of the form: `morphogen run` reproduces it.
* Interfaces that debond: a law for each pair of phases, their damage as a field, grain
  boundaries ([docs/theory.md](docs/theory.md) §84).
  * **A law per pair.** `CohesiveCell(mesh, materials, {(0, 1): law, (0, 2): other})`, or
    `interfaces: {laws: {0-1: {...}, 0-2: {...}}}` in a configuration: the interfaces are
    those of these pairs; with `faces: grains`, every boundary of a grain takes the law
    of the two phases that it parts (`1-1`: two grains of one phase). `CohesiveCell.laws`.
  * **Fields.** The fields of a path have `interface_damage` and `interface_opening`, for
    each element those of the interface elements on its faces: in the files of the
    fields and for the figures of a configuration (`color: interface_damage`), as the
    fields of the fatigue of a cycle (`color: fatigue_utilization`).
  * **Grain boundaries.** Checked on a polycrystal of eight grains pulled until a crack
    has crossed it: the work of the stress is the fracture energy times the area of the
    crack (0.14 %), and whole grains (boundaries on the faces of the cell) give the same
    as grains cut by the box. The template `homogenization_intergranular`.
  * **Gallery.** The figure of the result of `homogenization_fatigue`: the utilization
    of a slice through the worst point, and the share of the volume above each
    utilization.
* The damage coupled to plasticity (Lemaitre, Kachanov-Rabotnov) with Hill's criterion:
  the plasticity of the effective stress may be anisotropic (`plasticity: {hill: ...}`
  beside `damage: {law: lemaitre}`), the cumulated plastic strain being that of the norm
  dual to the criterion; checked on a bar of Hill against its closed form, and against
  von Mises for coefficients of 1. It was refused until now.
* Multiaxial high-cycle fatigue of the points of a cell along a cycle of a loading path
  ([docs/theory.md](docs/theory.md) §85).
  * **`homogenization.fatigue`.** The amplitude of a multiaxial cycle (`stress_amplitude`:
    the smallest hypersphere that holds the path of the deviator, its radius
    `sqrt(J2)_a` and its centre, the mean deviator) and five criteria, each calibrated on
    two fatigue limits of a phase (`from_limits`) or given by its coefficients:
    `Crossland`, `Sines`, `DangVan`, and the critical planes of `Findley` and `Matake`
    (`critical_plane`: the normal of the plane, its shear amplitude and its largest
    normal stress). `fatigue(history, criterion)` judges a history of stresses,
    `path_fatigue(result, criterion, cycle=...)` the fields of a path kept at its steps
    (`fields="steps"`): the utilization of each point (its equivalent stress over its
    limit), a criterion per phase, the largest utilization, the factor of safety and the
    share of the volume beyond its limit.
  * **In configurations.** `homogenization.fatigue: {criterion, bending, torsion, path,
    cycle, phases}` (or `criteria`, by phase): the fields of the cycle are kept, the map
    of the utilization, of the amplitude and of the largest pressure is written beside
    the curve (`<name>_<path>_<bc>_fatigue.vtu` or `.vti`), and `fatigue_utilization`,
    `fatigue_safety` and `fatigue_beyond` are results of the run and quantities of the
    statistics of volume elements (`<path>.fatigue_utilization`). `PathResult.scalars`:
    the scalar results of a path that are not curves. The template
    `homogenization_fatigue` (pores in a cast alloy under a tension that alternates) and
    its card.
  * **Checked.** Amplitudes against closed forms (proportional cycles, a tension and a
    torsion a quarter of a cycle apart, triangles of three states); the five criteria at
    the limits that calibrate them, under a mean stress, under a pressure, in any frame;
    a homogeneous cell at the closed form of its criterion by finite elements and by FFT;
    Goodier's stresses at the equator of a spherical pore (the largest utilization to
    3 %).
* Interfaces that debond in periodic cells along loading paths
  ([docs/theory.md](docs/theory.md) §84).
  * **`fracture.CohesiveCell`.** The periodic element of the loading paths with cohesive
    interface elements between its phases, between pairs of them or between the grains
    of a polycrystal: the bilinear mixed-mode law, the exponential one, or lips without
    adhesion, whose broken points touch again in compression and rub (Coulomb's
    friction); periodic or kinematic conditions, small or finite strains, 4- or 10-node
    elements, paths in strains, stresses or both. The curves get `debonded_area`,
    `interface_damage`, `interface_opening` and `interface_dissipation`
    (`PathResult.series`: columns of the curves, quantities of `value()` and of the
    statistics of volume elements); `interface_points()` gives the points of the
    interfaces at the end of a path.
  * **On the torus.** `fracture.open_cell` opens the faces of a periodic mesh on its
    torus: an interface that crosses a face of the cell goes on in the next cell, its two
    lips each linked to their own translates, and an interface that lies on a face of the
    cell joins the two sides a period apart (the stress that it carries through the face
    is part of the macroscopic stress); cells one element thick included.
  * **Everywhere.** `homogenize(..., nonlinear={"interfaces": {...}})` (the law as a
    mapping: `fracture.interface_law`), `homogenization.nonlinear.interfaces` in
    configurations and `morphogen homogenize --interfaces KEY:VALUE,...`; the template
    `homogenization_debonding` (glass beads that debond from their matrix), its card and
    the figure of its result.
  * **Iterations that pass.** A predictor that keeps closed the lips that touched at the
    start of a step; a nonmonotone line search (`nonlinear._line_search` with the largest
    residual of the last iterates), which lets the lips open and close along Newton's
    corrections; an elastic slip of the friction (`slip`: Coulomb's bound reached over a
    length, a stiffness of the stick of `μ p / slip`), without which lips that come to
    touch slide along directions that mean nothing; a `viscosity` of the damage for the
    interfaces that break at once (the error of a path without it says so).
  * **Checked.** A bar with one interface, inside the cell or on its faces, against its
    closed forms: tension, shear, the work that breaks it, the lips pushed back, Coulomb's
    bound and the energy that friction dissipates, lips without adhesion, the exponential
    law, uniaxial stress, finite strains, the recurrence of the viscous damage; the counts
    of the torus; kinematic conditions against `CohesiveProblem` (displacements to 2e-8);
    beads in a matrix between the bonded composite and the matrix with holes.
* Corrected: a loading path that brought an elastic cell back to zero stress did not
  converge, by finite elements or by FFT (a state without stress has no force with which
  to compare its residuals): the scales of the residuals keep a share (1e-6) of the
  largest reached along the path.
* Tests that do not depend on the machine: the factorized solver of the rigidity
  percolation takes 1 to 4 iterations by the machine and its libraries (4 on macOS in the
  CI, where two tests asked for 3 at most); they ask for fewer iterations than the FFT
  solver, and 10 at most. The numbers of tests of the README are those of today.
* A tutorial of contact and damage ([docs/tutorial_contact.md](docs/tutorial_contact.md)):
  a block with a flat pore crushed between two platens, an indenter that reads its
  modulus, the walls of a periodic cell that touch along a loading path, the damage laws
  at a material point against their closed forms and a composite whose matrix fails; its
  snippets run in the tests (about twenty seconds) and draw its figures.
* Damage laws: coupled to plasticity (Lemaitre, Kachanov-Rabotnov), of concrete (Mazars),
  and the unilateral effect ([docs/theory.md](docs/theory.md) §83).
  * **Coupled to plasticity.** `damage: {law: lemaitre, S, s, threshold, closure}` or
    `{law: kachanov, A, r, k, alpha, beta}` beside the `plasticity` of a material
    (`laws.DuctileDamage`): the von Mises elasto-(visco)plasticity, with its isotropic
    and kinematic hardenings and Norton's flow, is that of the effective stress `σ / (1 -
    D)`. Lemaitre's ductile damage `dD = (Y / S)^s dp` (the energy release rate with the
    triaxiality function, the closure of the microcracks in compression); the creep
    damage of Kachanov and Rabotnov `dD/dt = (χ / A)^r (1 - D)^-k` with Hayhurst's
    stress (`rupture_time`). `critical`: the damage at which a point breaks;
    `viscosity`: the time of a delayed damage, which keeps the softening of a cell well
    posed.
  * **Concrete.** `damage: {law: mazars, threshold, tension: [A_t, B_t], compression:
    [A_c, B_c], beta}` (`laws.Mazars`): a damage of tension and one of compression,
    weighted by the shares of the positive strains that tensions and compressions make.
  * **Unilateral effect.** `damage: {split: spectral}` of the quasi-brittle law: the
    negative principal strains keep their stiffness (a crack that closes carries the
    compression).
  * **Everywhere the closed-form laws are.** Finite elements, FFT, extended finite
    elements and the material point, small strains and finite strains (the logarithmic
    strain space); the variable `p` of the results of such a phase is its damage. The
    forms of the behaviours of the interface (the damage of the plastic phase, Mazars'
    law, the stiffness of closed cracks); the template `homogenization_ductile_damage`
    with its card and the figure of its result.
  * **Checked.** Material points against closed forms (a perfectly plastic and a
    hardening bar, the triaxiality, the closure, cycles, the time to rupture and the
    tertiary creep, tension and compression of concrete, the stiffness recovered in
    compression), the tangents against differences, a composite by FFT.
  * **Not offered.** MFront sources, Z-set files and the NTFA of a damaged plastic phase
    (errors that say so); anisotropic damage.
* Ready-made tests of contact: the compression of a sample between two platens and its
  indentation, as functions, commands and tools of the interface (the fifth lot of the
  work on contact, self-contact and friction; [docs/theory.md](docs/theory.md) §82).
  * **Functions** (`morphogen.contact.experiments`). `compression(mesh, materials, strain,
    axis, friction, self_contact, self_friction, steps, kinematics)`: two rigid platens
    on the faces of the box of the sample, on which its whole surface presses and rubs,
    its walls free to close on themselves; the nominal stress against the nominal
    strain, the areas in contact, the energy that friction dissipated
    (`CompressionResult`: `modulus`, `tangent()`, `rows()`). `indentation(mesh, materials,
    depth, indenter="sphere"|"cone", radius, half_angle, point, unload, base)`: the force
    against the depth, the area and the pressures of the contact, and the reduced modulus
    read from the force (Hertz, Sneddon), from the slope of the unloading (Oliver &
    Pharr) and, for a sphere, with the rigid base under the sample
    (`IndentationResult.layer_modulus`).
  * **Closed form.** `contact.layer_correction(radius, depth, thickness, poisson,
    bonded)`: the factor of Hertz's force of a sphere on a layer bonded to a rigid base
    or sliding on it (Dimitriadis et al. 2002).
  * **Commands.** `morphogen compress SAMPLE --strain -0.2 [--friction MU]
    [--no-self-contact]` and `morphogen indent SAMPLE --depth D [--indenter sphere|cone]
    [--radius R] [--half-angle A] [--point U V] [--unload] [--base fixed|smooth]
    [--poisson NU]`, on a geometry (`.json`, meshed with `--size`) or a tetrahedral mesh:
    the curve (`.csv`), the deformed sample with the pressure of its contacts
    (`--fields`), a report (`-o`, `--json`).
  * **Interface.** The tab Contact (after Fracture): *Compression between two platens*
    and *Indentation* of a morphology of the workspace (its solid phases meshed, its
    voids left out), with their curves, figures and files.
  * **Checked.** A block between lubricated platens: the modulus of the uniaxial stress
    with the penalties in series to 2e-4; stiffer with friction; a block with a slot:
    softer until the slot closes on itself, then stiffer. A sphere on a block four to six
    elements thick: the modulus of the block within 5 to 7 % once its base is taken into
    account (the half-space reads 18 to 30 % too much).
* The contact of the walls of the pores in configurations and on the command line, and
  a more robust search of the faces of a surface that touches itself (the fourth lot of
  the work on contact, self-contact and friction; [docs/theory.md](docs/theory.md) §82).
  * **Configurations.** `homogenization: {nonlinear: {contact: {friction: 0.3}}}` (or
    `contact: true`; `stiffness`, `tangential`, `depth`, `augmentations`, `tolerance`):
    the loading paths of a porous sample, on the mesh of its solid, are those of a
    `ContactCell`, in finite strains unless `kinematics: small`;
    `homogenize(..., nonlinear={"contact": ...})`; `morphogen homogenize --contact [MU]`.
    The template `homogenization_contact` (plates parted by gaps and held by pillars,
    compressed until the gaps close: five times stiffer; about two minutes), with its card
    and the figure of its result in the gallery.
  * **Curves.** `PathResult.series`: the other curves that a solver records at each step,
    columns of `rows()`, final values of `components()`, quantities of `value()` and of
    the statistics of volume elements (`rve.properties: [press.contact_area]`). A
    `ContactCell` records there `contact_area`, `contact_pressure`, `contact_penetration`
    and `contact_dissipation` (`contact.periodic.SERIES`), **in place of
    `info["contact"]`** of the third lot.
  * **Fixed: the two sides of a thin ligament.** Where the mesh is finer than the mean
    size of the faces, the two walls of a ligament between pores, back to back, were
    taken for surfaces that went through each other (144 points of 13,746 pressed at rest
    in a cell of three flat pores, whose first step did not converge). A face that the
    point has gone through is left out if the point was already behind it at rest
    (`kernels._across`; `pair_forces` takes the reference positions).
  * **Fixed: walls cut by the faces of a periodic cell.** The edges that such a wall
    shares with its own image were taken for the rim of the surface, and a point over
    one pressed on nothing: the nodes of the images are now those of the cell that they
    are the same as (`pair_forces` takes this identity), for the rims and for the faces
    that share a node.
  * **Changed: a surface faces itself more widely.** The cosine of the normals below 0.6
    (`kernels.FACING_OWN`) instead of -0.25: the walls of a notch that closes touch at
    its bottom before their normals are opposite, and the contact started with the
    penetration that they already had. Two surfaces still face each other below -0.25.
  * **Changed.** `ContactCell` refuses the iterative solvers (`solver="direct"` or
    `"auto"`: the tangent of the contacts is factorized).
  * **Limits found.** A honeycomb crushed beyond half its height creases at the corners of
    its cells, at the scale of a face: not followed (the faces that share a node do not
    press on each other).
* Self-contact in the periodic cells of the nonlinear homogenization: `contact.ContactCell`
  (the third lot of the work on contact, self-contact and friction;
  [docs/theory.md](docs/theory.md) §82).
  * **What.** The periodic element of the loading paths (`NonlinearFEProblem`: strains,
    stresses or both, the behaviours of morphoGen's points, small and finite strains) with
    the contact and the friction of the walls of the pores of the cell (`contact.walls`),
    between themselves and with their images across the faces of the cell:
    `ContactCell(mesh, materials, friction=..., ...).run(path)`; the response has in its
    `info["contact"]` the area in contact, the largest pressure, the deepest penetration
    and the energy that friction dissipated at each step; `slave_points()`.
  * **How.** The faces near the boundary get images translated by the periods, which move
    with the macroscopic gradient; the forces of the contacts work in the macroscopic
    strain by `Σ f ⊗ X`: this is added to the stress of the points in the equilibrium of
    the stress-controlled components, in the macroscopic stress of the results, and in
    the tangent of the macroscopic unknowns. `_PathSolver._extra_stress`: the seam.
  * **Checked.** Layers parted by a gap, compressed across them: nothing until each meets
    the image of the next through the faces of the cell, then the stress of the solid
    with the penalty in series to 1e-6, in uniaxial strain and in uniaxial stress (the
    lateral strains to 1e-5); a cell with a slot and the same medium cut through the slot
    respond alike to 1e-11.
* Contact, self-contact and friction of the surfaces of a mesh, between themselves and with
  rigid obstacles: the package `morphogen.contact` (the second lot of the work on
  contact, self-contact and friction; [docs/theory.md](docs/theory.md) §82).
  * **Surfaces.** `Surface.of(mesh, where, phases)`: free faces of a mesh (three nodes, or
    six on 10-node elements); `Pair(slave, master, friction, stiffness, tangential, depth,
    symmetric)`: two surfaces that may touch, or one against itself (`Pair(surface)`: the
    walls of pores, the struts of a lattice); `Obstacle.plane`, `.sphere`, `.cylinder`,
    `.cone`: rigid obstacles that move with the load factor and rub (platens, indenters,
    dies); `assemble(meshes)`: bodies meshed apart in one mesh.
  * **Method.** The integration points of the slave faces against the nearest point of
    the master surface (curved faces of six nodes included), a penalty or the augmented
    Lagrangian (`augmentations`, `tolerance`), Coulomb's friction by a radial return on a
    slip that rigid motions leave alone; the tangent of each point by central differences.
    A point looks only at faces that face it, that it has not passed by more than a depth
    and over which it is (not beyond the rim of a surface, not beside a face); it stays on
    its face while it hovers over one of its edges.
  * **Solver.** `ContactProblem(mesh, materials, pairs, obstacles, kinematics, bulk,
    interfaces, law)`: the quasi-static solver of the cohesive zones with these forces
    (loadings in stages, a predictor of points that stay closed and stick); results
    (`ContactResult`): the area in contact, the largest pressure and the force of each
    contact at each step, the works (bulk, contact, friction, obstacles), the pressure,
    traction and gap of every slave point, the pressure at the nodes
    (`nodal_pressure`, `write`). Closed forms: `hertz_sphere`, `hertz_cylinder`,
    `sneddon_cone`, `flat_punch`, `mindlin_stick_radius`, `effective_modulus`.
  * **Faster solutions of the structural solver** (cohesive zones, cracks, contact): the
    tangent keeps the full blocks of the dofs of its nodes when its parts are added
    (`cohesive.plus`: the sum of SciPy drops the stored zeros, and the factorization was
    5 to 8 times slower on them), and 10-node meshes take the ordering of matrices of
    symmetric pattern (3 to 5 times faster on them).
  * **Checked.** Two blocks of meshes that do not match transmit the pressure of the
    uniaxial stress to 6e-5 (4-node elements) and 1.4e-4 (10-node), one pass or two; a
    block slid by more than its length along a track takes `T / N` of 0.297 to 0.300 for
    0.3, whichever surface carries the points; Hertz's line contact: the largest pressure
    within 2.2 %, the half-width within 1.6 %, the pressures within 3.8 % (rms); a rigid
    sphere on a block: Hertz's force within 5 % while the contact is small against the
    block; a slot that closes gives the block 0.93 of the stiffness of the solid one; two
    arms that come to rest on each other bend twice as stiff, as two beams.
  * **Limits.** Corners and edges carry no point and enter a face by a few percent of it;
    a first contact by an edge may not converge in small strains (finite strains, the
    default of `ContactProblem`, do); a direct solver: tens of thousands of nodes.
* Friction of the interfaces whose nodes match: cohesive zones and the lips of cracks (the
  first lot of the work on contact, self-contact and friction;
  [docs/theory.md](docs/theory.md) §81).
  * **Law.** `Bilinear`, `Exponential` and `Contact` of `fracture.cohesive` take a
    coefficient of `friction` (Alfano & Sacco 2006): the broken share `d` of a point,
    pressed by the penalty, sticks elastically within Coulomb's cone and slides beyond it
    (a radial return on the slip of the point); the shear traction is `(1 - d) K δ_s + d τ`,
    the sound law while the interface is intact and Coulomb's `μ p` once it is broken.
    `Contact(stiffness, friction, tangential)`: the lips of a crack stick or slide.
    `traction(law, separation, history, slip)` is the law at a point.
  * **Solver.** `CohesiveProblem.run`: the prescribed displacements and the forces may be
    functions of the load factor (a pressure held while a shear grows); `history`: the
    state of the interface at the start (a part already broken); the first tangent of a
    step is that of lips that stay closed and stick where they touched (24 iterations
    become 2 for lips that stick). `CohesiveResult.slip` and `.friction` (the slip of the
    points and the energy that friction dissipated at each), `work["friction"]`, the mean
    slip in the files of the interfaces.
  * **Cracks.** `solve_crack(..., contact=True, friction=μ)` (and `propagate_kinked`): the
    stress intensity factors and the energy release rate count the tractions of the lips
    that rub; with friction the row of faces ahead of the front is opened and bonded, so
    that the lips press and rub up to the front (shared nodes of the front lose the
    friction of the last half element: an error in `sqrt(h)`, 15 % of `K_II` on the test
    below), and `K_I` is zero for lips closed at the front. The integral over the lips is
    taken at points graded towards the front (its auxiliary gradient is singular there).
    `cohesive.lip_tractions`, `cohesive.interior_faces`.
  * **Checked.** A cycle of slip dissipates `4 μ p (D - μ p / K_t)`; two blocks pressed, then
    sheared, stick and then slide at `T / N = μ` (1e-4; the works add up to 1e-6); a broken
    cohesive interface keeps `μ N`; a closed crack slides as under the shear `τ - μ σ`
    (0.6255 for 0.625) and sticks when `μ σ > τ`; an edge crack whose lips rub has `K_II` of
    0.587 and 0.173 of that of the shear alone (0.6, 0.2) and `G` of 0.359 and 0.0391 of it
    (0.36, 0.04).
* The mesher follows the sharp edges of the phases ([docs/theory.md](docs/theory.md) §80).
  * **Why.** The implicit function of a union of spheres or segments is the smallest of
    their distances: it has a fold where two primitives meet, and its surface a sharp
    edge there — the groove between two spheres, the line where two fibres cross, the
    grooves of coatings that overlap. A function sampled on a grid cannot hold such an
    edge: the mesher rounded it over an element, no node lay on it, and the phase kept
    too much volume in its grooves, which no move of the nodes could mend.
  * **The smooth pieces of a structure** (`models.pieces`,
    `Microstructure.implicit_pieces`): the primitives grouped so that two that overlap
    are never together (a colouring of the graph of their overlaps, periodic images
    included: 5 pieces for a Boolean model of spheres at 30 %, 8 at 60 %); the unions
    of spheres and segments, their coatings and their hollow members give theirs.
  * **The mesher** (`conforming_mesh(..., sharp_edges=None)`, `mesh.sharp_edges` in a
    configuration): the box is cut by the surface of each piece in turn, with the
    function of the piece taken exactly at the vertices; the lines where two cuts meet
    are edges of the mesh, given to MMG as ridges and kept through the remeshing
    (`mesh.sharp`, with the two pieces that meet along each: `mesh.sharp_pieces`). By
    default where a structure gives pieces that can meet; `sharp_edges=False` cuts by
    the function of each phase as before. Opposite faces still match in periodic boxes.
  * **Curved elements.** The nodes of the sharp edges, corners and middles, are put on
    the two surfaces that meet there, and on the three where three edges join.
  * **Results.** Two overlapping spheres, 10-node elements of 0.3 of the radius: the
    volume of their union within 0.001 % (0.05 % with the groove rounded), all the nodes
    of the interface but one on their surfaces (46 off before), with 30 % fewer
    elements; three spheres: a node at each of the two points where their surfaces
    meet, to round-off. Random coated spheres (the tutorial on exact geometry): the
    interphase within 0.1 % of its volume (0.9 % too much before), and a bulk modulus
    of 5.691 for a converged value of 5.69 (5.725 before).
  * **What it costs, and what is left.** A cut per piece: the meshes of overlapping
    primitives take 40 to 80 % longer. The cuts being smoother, MMG uses the whole
    Hausdorff distance on the surfaces: 4-node meshes, polyhedra inscribed in the
    phases, lose more volume than before at the same sizes (tighten `mesh.hausdorff`
    for them); the 10-node ones do not. The rims of cylinders with flat caps, the
    overlaps shallower than half the spacing of the background grid, and the sharp
    edges of the structures that do not give their pieces (their functions are those
    of a grid) are rounded as before.
* Isogeometric shape optimization applied: an auxetic sheet, on cells of any lattice
  ([the tutorial](docs/tutorial_iga.md), [docs/theory.md](docs/theory.md)
  §78, §79).
  * **Cells of any lattice.** A periodic cell need not be a box: with the periods of the
    medium (`Multipatch(patches, lattice=...)`, `homogenize(..., lattice=...)`) it is any
    set of patches that tiles the plane by them, the control points of its boundary a
    period apart being one. `cell_with_hole` takes the two vectors of a parallelogram
    (the rhombus of a hexagonal array); `shapes.repeated` puts a cell among its
    neighbours. The same cell cut in other ways gives the same tensor to 10⁻¹³; the
    conductivities of the square and hexagonal arrays of holes are those of the series of
    Perrins, McKenzie & McPhedran (1979) within 0.002 %, and the hexagonal array is
    isotropic.
  * **A sheet with alternating holes** (`shapes.alternating_holes`): elliptic holes on a
    square lattice, elongated in turn along its two axes — the rotating squares. Two
    whole grains on an oblique lattice, sixteen patches, well shaped down to the thinnest
    ligaments. Poisson's ratio goes from 0.27 (circles) through zero to -0.975 as the
    ligaments thin; with circles the stiffness is that of the box with one hole to
    6·10⁻⁷; the FFT solver on an image of the same sheet agrees within 0.7 %.
  * **Images of the patches.** `Multipatch.voxelize` gives the image of an assembly on a
    grid, periodic over a box that is a period of the medium: an optimized shape goes to
    the voxel solvers, the nonlinear behaviours and the exports like any other
    microstructure.
  * **Variables linked by symmetry.** `ShapeDesign.variable(..., symmetry=[...])` with
    the operations of the pattern (`mirror`, `rotation`) moves the images of its control
    points too, and those a period apart: the half-axes of all the holes of the sheet
    are two variables. `optimize_cell(..., linear=(A, b))` adds linear constraints on
    the variables (the width of a ligament). The designs that fold are refused on the
    boundary of the patches as well (`Multipatch.jacobians`); the history of an
    optimization holds the points that its iterations kept; the first step of SLSQP is
    scaled (`move`); a step of MMA that folds is halved.
  * **Results.** The lowest Poisson's ratio at a twentieth of Young's modulus of the
    plain sheet: -0.814 from -0.530 in 8 iterations, the slits at their least width;
    inverse design: the holes found again from their modulus and ratio to 10⁻⁶. A
    tutorial follows the package from a patch to this sheet
    ([docs/tutorial_iga.md](docs/tutorial_iga.md)), its snippets run by the tests.
* Isogeometric analysis and its application to shape optimization: a new package,
  `morphogen.iga` ([docs/theory.md](docs/theory.md) §78, §79).
  * **Geometry.** B-spline functions and their derivatives; NURBS patches (surfaces of
    the plane, solids) and assemblies of patches that share sides (`Patch`,
    `Multipatch`); arcs, rings, a plate with a hole and periodic cells with holes whose
    circles and ellipses are exact (`iga.shapes`); *h*-, *p*- and *k*-refinements that
    keep the geometry, by one operator for any nested pair of spline spaces.
  * **Analysis.** Linear elasticity (plane strain, plane stress, solids) and conduction
    with the functions of the geometry (`iga.Analysis`): tractions on sides, fields,
    error norms. Optimal orders on Lamé's cylinder (3 and 2 with quadratic functions, 4
    and 3 with cubic ones) and on Kirsch's plate with a hole.
  * **Periodic cells.** `iga.homogenize`: the effective stiffness or conductivity of a
    cell whose opposite sides have matching control points. A cell with a circular hole,
    880 unknowns: within 0.05 % of the curved 10-node tetrahedra of the same cell as a
    solid with 158,000.
  * **Shape optimization.** `iga.ShapeDesign`: variables that move control points of a
    coarse design (radial ones with the symmetries of a cell), an analysis on its
    refinement, analytical derivatives of the effective tensor, of the compliance and of
    the area by the energy-momentum tensor (equal to central differences to 10⁻⁹);
    `optimize_cell` and `optimize_structure` with SLSQP or the moving asymptotes. The
    hole of largest bulk modulus of a square cell reaches the bound of Hashin and
    Shtrikman within 0.05 % (the structure of Vigdergauz; a circle is 1.4 % below) in 6
    iterations; the hole of a plate pulled twice as much along one axis is the ellipse
    of half-axes in the ratio 2.008.
* Selective reduced integration of the quadratic elements, for nearly incompressible
  phases. `integration="bbar"` (also `"reduced"`) takes
  the volume change of each 10-node tetrahedron as its mean — one constraint per element
  instead of four — and the rest of its strain at its four points: `B`-bar in small
  strains, `F`-bar in finite strains, with the conjugate stresses and the consistent
  symmetric tangent of their energy, on straight and curved elements
  (`morphogen.homogenization.dilatation`; [docs/theory.md](docs/theory.md) §77). A
  uniform reduced integration is not an option for tetrahedra (one point leaves a
  10-node element 18 deformations without energy), and the four-point rule is already
  the smallest that integrates their stiffness. For the elastic problems and the loading
  paths of a mesh: `homogenize(..., integration="bbar")`, `nonlinear={"integration":
  ...}` for the paths alone, `homogenization: {integration: bbar}` in a configuration,
  `morphogen homogenize --integration bbar`, a choice in the loading-path tools of the
  interface; the default stays `full`, with a warning when a phase has a Poisson's ratio
  of 0.49 or more. Stiff spheres in a matrix of Poisson's ratio 0.4999: the shear moduli
  of the cell are converged within 0.1 % with 4,800 nodes, where the elements integrated
  at every point are 2.4 and 6 % too stiff, and still 0.5 and 1 % with 40,000 nodes.
  Lamé's thick-walled cylinder at the same Poisson's ratio: half the error of the full
  integration at every mesh. The tangents are checked against central differences of the
  forces (10⁻¹⁰), a rubber along a path against its material point.
* Quadratic elements for the finite element analyses. An audit first:
  the 10-node tetrahedra existed for the homogenization (linear and along loading
  paths), the Bloch waves, the bifurcations and the micromorphic media, with straight
  edges; they were the default of two tools of the interface only, and six of the nine
  templates that analyse a mesh used 4-node elements, one of them with a plastic matrix
  (the warning of locking that its run printed). With straight edges they were of little
  use around curved inclusions: the polyhedron that the mesher makes of a sphere loses
  10 % of its volume with elements of the size of its radius, and the moduli converged
  at the rate of that geometry.
  * **Isoparametric 10-node tetrahedra** ([docs/theory.md](docs/theory.md) §76). A
    mid-edge node may leave the middle of its edge: the kernels of the linear problems,
    of the loading paths and of the finite strains take the Jacobian of each integration
    point (`F = I + Σ d_k ⊗ ∇N_k`, the departures `d_k` of the mid-edge nodes), straight
    elements being computed as before, to the bit. `TetMesh.volumes()` integrates the
    Jacobian (the volumes of a curved mesh sum to that of the box to round-off);
    `TetMesh.curved`, `deviations()`, `min_jacobian()`, `straight()`. The loads of the
    static conditions are integrated on the faces with curved edges. Checked on one
    element against the formulas of the reference element (10⁻¹¹), by patch tests under
    the four boundary conditions and along plastic paths in small and finite strains.
  * **Nodes on the interfaces** (`morphogen.mesh.curved_mesh`; the default of
    `conforming_mesh(order=2)`, `curved=False` for straight edges; `mesh.curved` in a
    configuration). The corner and mid-edge nodes of the faces between two phases, and
    of the walls of pores left out of the mesh, are put on the implicit surfaces of the
    microstructure by Newton's method: exactly for the unions of spheres and segments
    and their coatings (their functions at any point, `Microstructure.implicit_at`),
    through cubic splines of the functions sampled eight times finer than the elements
    for the others; within the faces of the box, which keep matching, and as far as the
    elements stay valid: where an element does not bear the moves, its node that moves
    most gives way, and no volume element falls below 0.3 of that of its straight
    tetrahedron (`mesh.min_jacobian()`). `mesh.metadata["curved"]` counts the nodes that
    lie on their surfaces. A sphere keeps its volume within 0.05 % with elements of the
    size of its radius (−10.7 % for the polyhedron), a coated sphere both its volumes
    within 0.06 % (−2.7 %). The conductivity of a simple cubic array of spheres against
    the series of Sangani & Acrivos (1983), elements of three quarters of the radius:
    +0.006 % for the curved elements, −3.2 % for the same elements with straight edges,
    −1.1 % for the 4-node ones; its bulk modulus converged to four digits with 4,800
    nodes, where the straight 10-node elements are still 0.17 % low with 76,000.
  * **What they do not mend** (the notes say it with numbers): the gaps and layers
    thinner than the grid of the mesher (`mesh.background`), which it bridges, and the
    sharp edges that it rounds over about an element — those of the unions of spheres
    and segments and of their coatings are now followed (the entry above); with their
    grooves rounded, random coated spheres kept 0.9 % too much coating with elements of
    half the radius, which the flat faces hid by the 2 % that they lose.
  * **The elements of the analyses.** A configuration that does not give `mesh.order`
    takes 10-node elements for a mesh that the finite elements analyse, 4-node ones for
    a mesh that is only written; `morphogen homogenize --order` defaults to 2; the tab
    Linear homogenization of the interface has a choice of elements, and its tool of the
    linear tensors takes the 10-node ones by default, as the loading paths did. A mesh of
    10-node elements has about eight times the nodes of the same mesh of 4-node ones:
    double `mesh.size` to keep the cost, and give `mesh.background` where the grid of the
    mesher must stay fine (thin gaps and necks).
  * **In the fracture modules** ([docs/theory.md](docs/theory.md) §46, §47). Cohesive
    zones: `insert_interfaces` opens meshes of 10-node tetrahedra (by their corners; each
    lip has its own mid-edge nodes, where they were), with interface elements of six
    nodes a side (`Interfaces.order`, `.weights`), integrated at six points with the
    normal and the area element of each: their faces may be curved. `CohesiveProblem`
    takes them with an elastic bulk in small strains and the neo-Hookean one in finite
    strains (the behaviours of the points keep 4-node elements); `dcb_specimen(...,
    order=2)`. A sphere meshed with elements of its radius: the area of its interface
    elements within 0.03 % (6 % less on flat faces), and the force of a rigid translation
    exactly `K δ` times it. The double cantilever beam with two 10-node elements across
    each arm: the loads of the propagation within 3 % of the corrected beam theory (24 %
    off with 4-node elements of that size, 9 % with 4-node elements of half the size and
    as many nodes). Cracks on conforming meshes: `conforming_work(..., order=2)` takes
    10-node elements whose mid-edge nodes stand at the quarter of the edges that leave
    the front (`quarter_points`; Barsoum 1976), with the consistent loads of faces of six
    nodes (`face_forces`): the energy release rate of an edge crack with elements of 1
    is 0.99 of the handbook value (0.76 with 4-node elements, 0.93 with 10-node ones
    without quarter points). The XFEM and the mixed-mode analyses keep 4-node elements.
  * **Fixed: the faces of periodic meshes of three phases.** The second cut of the
    mesher numbered its vertices as the first one had made them, not alike on opposite
    faces, and MMG split up to six of their triangles along the other diagonal on one
    side: the corner nodes matched, a few edges did not, and the 10-node elements had
    mid-edge nodes without an image (interpolated, with a warning). The vertices are now
    numbered in the order of their coordinates before each cut.
  * **Corrected: the volume of the coatings.** The notes of §75, a test and the tutorial
    on exact geometry compared the coatings of coated spheres with the sum of their
    shells, and read as a loss of the mesh (5 to 9 %) what is mostly the overlap of the
    coatings of spheres closer than twice their thickness (4 % in these cells): against
    the sum of the shells less the lenses that two coated spheres share, the flat meshes
    are 5, 1.5 and 0 % low with elements of 0.8, 0.5 and 0.35.
* A richer tutorial, gallery, documentation and site:
  * **Gallery.** Four templates for what the last versions brought, with their cards:
    `homogenization_rubber` (a filled rubber stretched, unloaded and stretched again in
    finite strains: the eight chains and the softening of Mullins), `fracture_composite`
    (glass beads in a brittle epoxy broken by the phase field), `fracture_rubber` (a
    perforated rubber sheet that tears, the phase field in finite strains) and
    `homogenization_interphase` (three phases meshed conformingly from their implicit
    functions, by finite elements, with the level sets and surfaces of the phases). A
    new family, Fracture. The cards of the templates of `RESULTS` in
    `docs/gallery/make_gallery.py` also show **what the run computes**, drawn from the
    files of the run (`--results`): the curve of a loading path against its strain, its
    stretch or the time, and the damage of a cell that broke — the tension, relaxation
    and unloading of a viscoplastic composite, the relaxation of a viscoelastic one, the
    tension of a polycrystal, the cycles of the rubber, the two cells that break.
  * **Planar sections in the configurations** (`voxels.section: x | y | z`,
    `VoxelImage.layer`): a run studies one layer of voxels through the middle of its
    realizations — a plane strain for the FFT solver, on far fewer voxels —, as the
    interface and the guide to the physics did by hand; the image, the labels, the level
    sets and the measures are those of the layer. The two templates of fracture use it.
  * **Tutorials** ([docs/tutorials.md](docs/tutorials.md)). The tutorial is now five:
    the whole study, with the figures that its snippets save (sections, covariance,
    tension curves); [how large is a representative volume?](docs/tutorial_rve.md) (the
    variance of the volume fraction of Boolean spheres against the volume, with the
    `K = p (1 - p) A₃` of their integral range within 3 %, and of fibres on Poisson
    lines, whose exponent is 0.72 ± 0.09 for the 2/3 of their theory; the volumes that
    a precision asks for, six times the edge for the fibres; the windows of one large
    image; the conductivity and the bulk modulus of a composite from a configuration,
    with its report); [a rubber, from its tests to its tearing](docs/tutorial_rubber.md)
    (four energies fitted to three tests, the softening of Mullins, a filled rubber, a
    notched strip torn at the stretch of Rivlin and Thomas by the phase field in finite
    strains); [exact geometry](docs/tutorial_geometry.md) (the implicit functions of
    three phases, sections, surfaces, a conforming mesh, three solvers on the same
    geometry, files for other codes); and [a study in the graphical
    interface](docs/tutorial_interface.md), with screenshots that a script takes by
    driving the page (`docs/tutorials/make_screenshots.py`). The tests run the snippets
    of the four Python tutorials at reduced sizes (`tests/test_tutorial.py`), and
    `docs/tutorials/make_figures.py` writes their figures by running them as written;
    the whole study also draws the variance of its modulus against the volume, with the
    law fitted to it.
  * **Recipes and questions.** [docs/howto.md](docs/howto.md): twenty short answers to
    "how do I..." (the same realization again, a planar section, a surface for a
    printer, the stiffness tensor, prescribed stresses, the choice of a solver, a
    loading path, a hyperelastic law, a cell that breaks, the size of the RVE from
    results, clusters, Abaqus, AMITEX_FFTP, MFront), as snippets that follow each other
    in one session and that the tests run (`tests/test_howto.py`).
    [docs/faq.md](docs/faq.md): installing and running, units, phases and periodicity,
    the messages of the solvers and what to do about them, memory, the files of a run.
    [docs/glossary.md](docs/glossary.md): the words that have a precise meaning here
    (apparent and effective properties, the representative volume, the integral range,
    the boundary conditions, the samples and their solvers, the phase field...), each
    in a few lines with where to read more.
  * **Guide to the physics.** Two snippets on finite strains, run by the tests: a rubber
    alone and filled with spheres (nominal and true stress, logarithmic strain), and a
    notched strip torn in pure shear near the stretch of Rivlin and Thomas.
  * **Site.** A landing page (what morphoGen does, as cards; a first study; where to
    read what), a logo, the pages in four groups (start, guide, reference, project),
    and three things that the pages do without any library (`docs/_static/site.js`): a
    button that copies each code block, a filter of the cards of the gallery (also
    `gallery.html?filter=fracture`), screenshots enlarged by a click. The horizontal
    padding of the cards of the gallery, which the theme set to zero, is back. The
    continuous integration keeps the site that it builds (the artifact `site` of a run),
    to read it without building it.
  * **Specification.** [docs/specification.md](docs/specification.md) (1.127) records
    the work of 12 to 14 November and what each part delivered (§3.8), and its
    requirement on the documentation (NFR-8) the snippets that the tests run.
    `CONTRIBUTING.md` states the same rule for the pages.
  * **A cut of a mesh** (`viz.figures.mesh_figure`, `mesh_section`): the polygons that
    a plane cuts in the tetrahedra of a mesh, in the colours of their phases or of a
    field per element — the figure of a conforming mesh that a surface does not show.
* Implicit geometry of every model, finite strains in every case, more hyperelastic
  laws, and the menus of the interface:
  * **An implicit function for every phase of every model**
    (`Microstructure.phase_level_set`, `implicit_phases`; `morphogen.models.implicit`,
    [docs/theory.md](docs/theory.md) §75). An audit of the 38 models found 8 without a
    level set at all — cross-linked fibres (whose flag said otherwise), Voronoi and
    Laguerre cells of several phases, dead leaves, transparent dead leaves, fractal
    sponges, osteons, inclusions in a matrix, automata on tilings — and only the phase 1
    of the others described: no structure of three phases could be meshed. Every
    realization now gives, for each of its phases, a function negative in the phase and
    zero on its boundary, exact on the interfaces of the model: distances to the spheres
    of the leaves outside those that cover them; the rings of the osteons; differences
    of power distances to the germs of the cells, and the layer of their boundaries;
    the gauges of the convex cells of a tiling; distances in the maximum norm to the
    cubes of a sponge; the spheres across which a transparent function passes its
    threshold; minima and maxima for inclusions, nested structures, coatings, shells
    and clustered spheres. Their signs are those of the voxel images at every point,
    and they partition the space (57 configurations, two grids each).
  * **Conforming meshes of structures of several phases** (`conforming_mesh`): coated
    inclusions, inclusions in cells of several phases, osteons, leaves of several
    phases, cross-linked fibres... are cut by the functions of their phases one after
    the other (MMG's level sets of several materials) and remeshed, their opposite faces
    matching; `phases` takes the phases to keep. A coated sphere: the volumes of the
    sphere and of its shell within 5 % and 1 % with elements of half the thickness of
    the coating, and the moduli of three materials within 0.8 % of the FFT solver on 64³
    voxels. `conforming_mesh` no longer refuses a model: the cells of one phase without
    grains give a mesh of the box. The configurations write the level set and the
    surface of each phase (`outputs.level_set`, `outputs.surface`), and no longer refuse
    a `mesh` section for the models that had no level set. `GaussianField.values_at`
    and `contains` give the field at any point (the sum of its modes).
  * **More hyperelastic laws** (`morphogen.homogenization.laws`, theory §61). The laws of
    Mooney and Rivlin, of Arruda and Boyce, of Yeoh, of Gent and of Ogden were there;
    added: the polynomial of Rivlin of any coefficients `cij` up to the sixth order
    (`polynomial`; `mooney_rivlin` takes its forms of three, five and nine parameters);
    the eight chains with Cohen's approximant of the inverse Langevin function
    (`eight_chain`: it locks, where the five-term series of `arruda_boyce` does not);
    van der Waals (Kilian); Fung–Demiray; the fibres of Holzapfel, Gasser and Ogden
    (`holzapfel`: families of fibres in a neo-Hookean matrix, their dispersion, tension
    only); the compressible foams of Blatz and Ko and of Ogden, Hill and Storåkers
    (`blatz_ko`, `hyperfoam`: no split, their Poisson's ratio from the bulk modulus); the
    softening of Mullins on any of them (`mullins: [r, m, beta]`, Ogden & Roxburgh).
    `Hyperelasticity.law().nominal_stress(mode, stretch)` gives the uniaxial,
    equibiaxial, planar and shear tests of a rubber (closed forms of the incompressible
    solid, or the law with its bulk modulus), and `fit_hyperelasticity` fits a law to
    such tests (a linear solve for the polynomials). All at a material point, by finite
    elements, on voxels and in the crack solvers, as the others; in the forms of the
    interface. Each against the derivatives of its energy, its closed forms (Blatz–Ko:
    `μ (λ^-1/2 − λ^-3)`), and its parameters recovered from its own stresses.
  * **Finite strains** (theory §63, §74). An audit of the modules: the loading paths
    of the finite elements, of the voxels (CPU and GPU) and of the material points, the
    behaviours (plasticity, crystal and porous plasticity, viscoelasticity, damage,
    hyperelasticity, MFront and learned laws), the cracks in the bulk, the cohesive
    zones, the micromorphic models, the bifurcations along a path and the membrane of
    the laser-treated sheets already take `kinematics="finite"`. Added: the loading
    paths of the **extended finite elements** (`NonlinearXFEMProblem(...,
    kinematics="finite")`, `method: xfem` with `nonlinear: {kinematics: finite}`: total
    Lagrangian, the level set cutting the reference configuration; they were refused), a
    rubber on both sides of oblique interfaces against its material point and two
    rubbers in layers against their exact stretches (1e-8); the tensile test of the
    **volume of a laser-treated sheet** (`volume_tension(..., kinematics="finite")`:
    forces over the initial section, as the membrane, within 1 to 2 % of it up to 8 %
    of strain); and the **phase field of fracture** (`fracture.phasefield_finite`,
    `FinitePhaseFieldProblem`, `phase_field_fracture(..., kinematics="finite")`,
    `fracture` with `nonlinear: {kinematics: finite}` in `homogenize` and the
    configurations, which refused it; theory §64): the rubbers and soft solids that
    stretch much before they break. The energy of the deformation gradient of
    hyperelastic phases (any strain energy of the laws above) or elastic ones (Hencky)
    over the reference configuration, the displacement at fixed damage by the nonlinear
    FFT solver whose points are degraded (the solver of the paths solves a step again
    while the state of a problem settles: the hooks `_settle` and `_rollback` of
    `_PathSolver.run`), the damage problem of the small strains; the splits `none`,
    `volumetric` (the isochoric energy and the expansion) and, for elastic phases,
    `spectral`. A cell stretched to 1.5 against its closed forms (1e-8); at a strain of
    0.1 %, the strength of the small-strain solver within 4 % and its crack; a cracked
    strip of rubber, which breaks at the stretch of Rivlin and Thomas (`W(λ) H = G_c`:
    1.44 to 1.53 for 1.5). In the tool Phase-field fracture of the interface too. Still
    in small strains, by construction: the reduced models of the NTFA (a superposition
    of linear problems), the plate theory of the hygromorphs and the linear elastic
    fracture mechanics of `fracture.lefm` (whose finite-strain counterparts are
    `fracture.mixed` and `fracture.bulk`).
  * **The interface and its menus** (`morphogen gui`). A review of the page with its 11
    tabs of 46 tools found the tools of a tab behind one drop-down list (seven of them
    in Statistics, in Mechanics and in Solvers & formats, seen only once opened), the 38
    models in another, no way to find where an option lives (a law, a file format), and
    a tab named Nonlinear mechanics that also held the linear tensors. Now: a tab shows
    the **menu of its tools** on its left, under the headings of what they do (`tool(...,
    menu=...)`, `gui.tools.menus`; Mechanics: loading paths, behaviours of materials,
    linear elasticity; Statistics: morphology — first —, representative volume,
    reconstruction; Fracture: cracks, cohesive zones, phase field; Solvers & formats:
    AMITEX_FFTP, Abaqus and meshes, behaviours of materials...), and Generate the
    **menu of its models** by family, with a filter; a **search** in the header (`/`)
    finds a tool, a model or an option of a form by the start of its words and opens it
    with the option set and its field in sight ("arruda": the loading path with that
    strain energy; "finite strains": its kinematics; "stl": the surfaces of Export);
    the **address** of the page names the tool or the model shown (`#mechanics/path`,
    `#generate/voronoi`); **?** is a map of the interface whose tabs and tools are
    links; the button that runs a form stays in sight under the long forms and
    `Ctrl + Enter` presses it; the colours follow the system or are set light or dark;
    narrow windows put the menu above the form. The tab Design is named Inverse design,
    the summaries of the models are whole sentences, their documentation is running
    text. The tool Linear tensors also solves by **extended finite elements** (XFEM, on
    a regular grid cut by the level set: no mesh to build), which the interface did
    not offer.
  * **The implicit functions in the interface.** The sections of the structures of
    several phases blend the colours of their phases by the fraction of each phase in
    each pixel (`viz.sections.phase_coverages`, `Section.coverages`, from the implicit
    function of each phase: coatings, inclusions, dead leaves, osteons, cells with
    phases; plain colours beyond `MAX_PHASES` phases). Export and the formats of other
    codes write the function and the surface of each phase
    (`models.implicit.implicit_functions`: `level_set_phase<k>.vti`,
    `surface_phase<k>.stl`, one TIFF or MetaImage per phase) where they wrote those of
    phase 1 alone.
* Three dimensions wherever a module only worked in the plane. An audit of every module
  found the package three-dimensional almost
  everywhere (its planar cases are images one voxel thick), and these parts limited to
  two dimensions or to one, extended here:
  * **Light in three-dimensional photonic crystals** (`morphogen.homogenization.photonic`,
    [docs/theory.md](docs/theory.md) §69): the electromagnetic Bloch waves were the two
    scalar polarizations of a medium invariant along an axis. `band_structure(...,
    waves="electromagnetic")` and `electromagnetic_band_structure` solve the vector waves
    of Maxwell's equations on a voxel image of any morphology by plane waves: two
    transverse amplitudes per plane wave (no spurious gradients), the operators by fast
    Fourier transforms, a block Davidson method preconditioned by the operator of the
    inverse medium, dielectric, anisotropic (permittivity tensors) or magnetic phases
    (the Hermitian pencil of the induction), and the cells that an interface cuts
    averaged as its layers (`coarsen`, `smoothed_inverse`; Meade et al. 1993). Exact
    for the light cones of homogeneous and uniaxial media; second order on the Bragg
    mirror at normal and oblique incidence (1.5e-4 on 80 plane waves); the two scalar
    polarizations together in the plane of a crystal of rods, and the waves that leave
    it; the complete band gap of the diamond lattice of spheres within 1 % of MPB's
    (0.397 to 0.442 c/a for 0.396 to 0.441) in 8 s. `morphogen bands --waves
    electromagnetic [--coarsen N]`. The scalar waves `shear`, `tm` and `te` now refuse a
    cell that varies along z or a wave vector out of its plane (they were solved as a
    three-dimensional scalar problem, which is not that of light), and the default path
    of a cell one voxel thick is Γ–X–M–Γ.
  * **Particles in a volume** (`morphogen.analysis.particles`, theory §70): the watershed
    that split the touching sections of the stereology only took planes.
    `kernels.watershed.flood_volume` and `prominent_maxima_volume` take volumes
    (26-connected voxels, through the faces of a periodic image; the kernels of the
    planes are theirs on one layer, with the same results), and `particle_sizes`
    measures the particles of a phase one by one: its connected parts, those that touch
    or overlap split by the watershed of their distance map, their volumes, equivalent
    radii and centres, and in images that are not periodic the weights of Miles and
    Lantuéjoul for the particles that the frame does not cut. The measure `particles`
    (number per unit volume, mean and coefficient of variation of the radii) goes to
    the configurations, to `morphogen analyze` and to the statistics of representative
    volumes. Two overlapping balls against the volumes of their caps (0.7 %), a packing
    of 155 hard spheres recovered sphere by sphere, its windows (154.5 for 155 per unit
    volume, 49 without the weights).
  * **Diamond and pyrochlore lattices of struts** (theory §71), the counterparts in
    space of the honeycomb and of the kagome lattice, which the plane frames alone had:
    cells `diamond` (four struts per node, bending-dominated, `K = ρ E / 9`) and
    `pyrochlore` (tetrahedra joined by their corners: six struts per node on three
    straight lines, the stiffness of the octet truss per unit density with half its
    struts) of `SpaceFrame.lattice_of` (with the generators of their space group, some of
    whose operations come with a translation: post-bifurcated patterns, Bloch waves), of
    `Framework.lattice` (rigidity) and of the model `strut_lattice`; in the tools
    *Bifurcations of a frame* and *Rigidity of a strut lattice* of the interface.
    **Fixed**: `Framework.lattice("kelvin")` merged the nodes of the Kelvin cell that lie
    at quarters of its edge (9 nodes for 12, bars of zero length), so that the tool
    *Rigidity* was wrong for the Kelvin lattice; `Frame.from_framework` (and
    `network_stiffness(dimension=2)`) projected a framework in space onto the plane x-y
    when none of its bars crossed the box along z: it is now refused.
  * **Laser-treated sheets in their volume** (`morphogen.laser.solid`, theory §74): the
    treated sheets were membranes in plane stress. `case_volume` draws the periodic cell
    of a sheet treated all over as a voxel
    image for the solvers of three-dimensional fields: the hardness that the laser path
    leaves in each layer sorted into classes (`sheet_volume`: a phase per class, with the
    flow law of the mean hardness of its voxels), a gap of void voxels above the sheet
    for its two free faces. `flow_plasticity` turns the flow laws of the sheets into von
    Mises plasticity (Voce's law exactly; Ludwik's and Swift's power laws fitted by a
    linear term and three exponentials, within 1.5 % up to a plastic strain of 0.3);
    `volume_tension` is the tensile test of the volume by the nonlinear FFT solver (the
    strains of the cell across the sheet, which are those of its gap, prescribed: left
    free they made the path singular). A sheet of one hardness follows its flow law
    (7e-8); hard and soft halves side by side or one above the other follow the mean of
    their laws (3e-4), what the membrane assumes; the cell of straight martensite tracks
    in an annealed 0.23 C steel agrees with its membrane within 1.3 % along the tracks
    and 3 % across them, the widths shrinking alike. In small strains, without fracture.
  * **Vertex stars of the icosahedral tiling** (theory §14): `Tiling.vertex_stars` only
    took the planar quasi-periodic tilings (it was "left for later" for the golden
    rhombohedra). The star of a vertex is the set of the rhombohedra around it; its kind:
    the numbers of prolate and oblate cells that meet it by a tip or by another corner,
    and the numbers of cells around each of its edges, which tell apart the stars up to
    the 120 symmetries of the icosahedron (the directions of the edges alone give 9
    kinds). The tiling has 24 stars, of 4 to 12 edges; `ICOSAHEDRAL_STARS` gives their
    frequencies in the infinite tiling as expressions of the golden mean (τ⁻³ for each of
    the three commonest, of 4, 5 and 6 edges; τ⁻⁹ for the star of twenty prolate tips),
    which sum to 1 and give the mean coordination 6 and the numbers of corners per
    vertex exactly, and `window_stars` measures them in the acceptance window of the cut
    and projection (the rhombic triacontahedron), independently of the dual grids that
    build the tilings: within 1.7 standard deviations on 3·10⁷ points. A patch has
    exactly these stars; the cubic approximants have them at frequencies that converge
    with the order (within 0.003 at 5/3, 0.001 at 8/5), a mean coordination of exactly
    6, and a few vertices of other stars (their phason strain).
  * **Cracks of any surface and front** (`morphogen.fracture.surface`, theory §73): the
    tracked cracks were planar with a straight or circular front advancing uniformly, or
    a path in a plane, the same all along a straight front. `SurfaceCrack` is the surface
    swept by the successive positions of a front (rows × stations: a closed front inside
    the body, such as an ellipse, or an open one that leaves it by its ends), with the
    level sets and the axes of the front that the conforming meshes, the opening of the
    lips, `solve_crack`, the G-θ method and the interaction integrals take
    (`SurfaceCrack.ellipse`, `SurfaceCrack.of` a planar or kinked crack).
    `propagate_front` grows it: each point of the front by its own advance, `Δa (G /
    G_max)^(m/2)`, in the direction of the largest hoop stress of its own `K_I` and
    `K_II`. `refine_front` and `crack_mesh(..., front_size=..., front_radius=...)` refine
    the mesh around the front alone (a second pass of MMG with a map of sizes), in which
    the domains of the integrals are then counted. An inclined penny crack against Kassir
    & Sih (`K_I` 3 %, `K_II` 4 %, `K_III` 9 %; the angles of kinking of Erdogan & Sih
    within 1.1°, from −46° to 46° around the front); an elliptical crack against Irwin
    (2 % in the mean); an ellipse grows towards a circle; the inclined crack turns
    towards local symmetry (`K_II` from 0.60 to 0.21 in two advances). The tool **Growth
    of a crack front in 3D** of the tab Fracture of the interface. Not covered: the
    twisting of the front under mode III (it does not break into facets), and the
    toughness of phases and interfaces along a curved front (that of
    `propagate_kinked`).
  * **Thermal oxidation in a microstructure** (`morphogen.coupling.oxidation_field`,
    theory §72): the closed-loop scheme was solved in a thin film and through the
    thickness of a plate. It now runs in every voxel of an image, in one, two or three
    dimensions: each phase with its constants (a polymer that oxidizes, an inert phase
    that lets oxygen through, an impermeable one mapped to None: fibres, fillers),
    oxygen diffusing between the voxels from the exposed faces with its activity
    continuous across the interfaces. Integrated by TR-BDF2 with error-controlled steps
    from the milliseconds of the radicals to thousands of hours; each Newton iteration
    eliminates the local species voxel by voxel and solves the oxygen of all the voxels
    by conjugate gradients (150 steps, 5 s on 16³ voxels, 34 s on 32³). `OxidationField`
    holds the fields of the six species, whose `tracer()` feeds `graded_materials`.
    One voxel against the thin film (6e-4), a plate against its profile (1 % on 240
    voxels, second order), impermeable fibres along the diffusion against the plate
    (4e-6). The tool **Thermal oxidation of a morphology** of the tab Couplings of the
    interface: the amides in the volume, their mean profile from the exposed face
    against a plate of the polymer alone, the oxidized layer.
  * **Sound at any angle** (`homogenization.acoustics`, theory §68): the absorption of a
    layer was that of a plane wave at normal incidence. `absorption(..., incidence=θ)`
    refracts the wave in the layer, whose frame may differ along it and across it
    (`transverse`: the parameters of another axis of the image), and
    `incidence="diffuse"` gives Paris' average over the angles of a diffuse field;
    `surface_absorption` is that of a surface of given impedance. Exact for a layer of
    air and for Paris' closed form (0.951 at ζ = 1.567). In the tool *Flow and sound
    absorption* of the interface: normal, oblique or diffuse incidence.
  * **Light and sound in the interface**: the tool **Band structure of light and sound**
    of the tab Physics (the Bloch-wave tool of the tab Stability takes elastic waves
    only): photonic crystals in three dimensions by plane waves, sonic crystals with
    rigid scatterers, the bands in units of c / L and the band gaps.
  * The template `optimize_auxetic_3d`: a cell of negative Poisson's ratio along its
    three axes by topology optimization (−0.14 on 24³ voxels with cubic symmetry, −0.21
    on a grid twice as fine), beside the planar `optimize_auxetic`.
  * `viz.figures.tiled_figure` (and `morphogen view --tile`) tiles the three middle
    sections of a three-dimensional cell (it tiled its x-y section alone), or the one
    normal to `axis`.
* The flow of a fluid through the pores and the acoustics of porous media
  ([docs/theory.md](docs/theory.md) §67–§68, [docs/physics.md](docs/physics.md) §5), the
  continuation of the multiphysics homogenization below (its acoustic part).
  * `morphogen.homogenization.flow`: the creeping (Stokes) flow through the pores (the
    void phases) of a voxel image by the lattice Boltzmann method with two relaxation
    times (D3Q19, linear equilibrium, bounce-back walls midway between the nodes whatever
    the viscosity), marched to its steady state with Aitken's extrapolation of the
    conserved mass flux; `stokes_flow`, `permeability_tensor` and the property
    `hydraulic_permeability` (the tensor of Darcy's law). Exact for the flow between two
    planes; within 5 % of Sangani & Acrivos (cylinders) and of Zick & Homsy (spheres) on
    24 to 48 voxels per period.
  * `morphogen.homogenization.acoustics`: the parameters of the equivalent fluid of a
    rigid porous frame — porosity, static viscous and thermal permeabilities, tortuosity,
    viscous and thermal characteristic lengths (`acoustic_parameters`, the property
    `acoustic`; the viscous length measured by moving the walls of the pores by one
    voxel) — the model of Johnson, Champoux, Allard and Lafarge (`equivalent_fluid`), the
    absorption of a layer on a rigid wall (`absorption`), the exact responses of slits and
    tubes (`pore_response`), `thermal_permeability` (also the trapping constant of
    diffusion-controlled reactions), `Fluid` and `AIR`.
  * Both properties in the configurations (voxel images, `method: fft`), the command
    line and, by their named results (`hydraulic_permeability.mean`,
    `acoustic.tortuosity`, `acoustic.viscous_length`), the statistics of RVE studies; the
    template `homogenization_acoustic_foam`.
  * Phase field: its elastic problem preconditioned by the Fourier reference scaled by
    the degradation of the nodes (six times fewer iterations across broken voxels, the
    same results; a section of 16² voxels in 14 s instead of 38), and a tool
    **Phase-field fracture of a morphology** of the tab Fracture of the interface (the
    curve of the loading, the strength, the work of fracture and the damage field of the
    volume or of a planar section).
  * The command `morphogen bands`: the band structure of the Bloch waves of a periodic
    cell (an image or a mesh; `--waves elastic`, `acoustic`, `shear`, `tm`, `te`), its
    band gaps, its table (`-o`) and its figure (`--figure`).
  * Examples: `examples/multiphysics.py` (one microstructure, many physics) and
    `examples/acoustic_foam.py` (from a foam to the sound absorption of a layer).
  * The behaviours in closed form in the forms of the interface (material point,
    loading path, macroscopic law): *viscoelastic* (Prony series of the shear and bulk
    moduli), *hyperelastic* (the energy and its parameters, or the neo-Hookean energy of
    the elastic constants) and *damage* (strength, ultimate strain or fracture energy
    with the size of the elements, softening, equivalent strain, delay).
  * A tab **Physics** of the graphical interface (`morphogen.gui.tools.physics`), for a
    morphology of the workspace: *Transport properties* (the five transport phenomena,
    with their tensor and bounds), *Flow and sound absorption* (the permeability, the
    acoustic parameters in SI units and the absorption curve of a layer) and *Dynamic
    moduli* (storage and loss moduli of a composite with a viscoelastic phase over a
    range of frequencies).
* Multiphysics homogenization, the behaviours and the fracture of every solver
  (mechanical, thermal, electrical, electromagnetic, chemical, diffusive,
  optical and acoustic problems, static and dynamic, with their weak and strong couplings;
  the statistics of representative volumes for every property; viscoelasticity,
  hyperelasticity, damage and brittle fracture beside plasticity; extended finite
  elements). The page [docs/physics.md](docs/physics.md) is the guide to all of it for
  researchers in the mechanics and physics of materials (which problem, which solver,
  which name; its snippets run in the tests), and the theory notes hold the formulas and
  the checks (§59–§66). In the order of the changes:
  * **Transport phenomena of a scalar potential** ([docs/theory.md](docs/theory.md) §59).
    `Material(electrical_conductivity=..., permittivity=..., permeability=...,
    diffusivity=...)` beside `conductivity`: one value or, in the axes of a crystal, three
    principal values or a 3 x 3 tensor; a zero coefficient makes a phase inert for that
    physics (an insulator), as a void is for all. The thermal solvers (FFT, finite
    elements) solve them all (`make_problem(sample, materials, "electrical")`), the
    properties `electrical_conductivity`, `permittivity`, `permeability`, `diffusivity` and
    their tensors (`*_tensor`) are those of `homogenize`, of the configurations, of the
    command line, of the bounds (Wiener, Hashin–Shtrikman), of the normalized values and of
    the RVE studies (ensemble and window methods); `Loading(physics=...)`,
    `Loading.gradient(..., physics=...)`, `physics=electrical` in the text of a loading.
  * **Strong couplings** (`morphogen.homogenization.coupled`, §60): piezoelectric,
    piezomagnetic, magneto-electro-elastic, rigid magnetoelectric and thermoelectric
    composites on voxel images, their fields solved together (MINRES or conjugate
    gradients, a block preconditioner in Fourier space, unknowns scaled so that SI units
    are well conditioned). `Material(coupling=Coupling(...))` (or a mapping:
    `{piezoelectric: {e31: ..., e33: ..., e15: ...}}`), the properties `piezoelectric`,
    `piezomagnetic`, `magnetoelectroelastic`, `magnetoelectric`, `thermoelectric` give
    the tensors of the effective law and its constants (`d33`, `g33`, `k_t`, `d_h`, the
    magnetoelectric coefficients of a product property, the figure of merit `ZT`), with
    the effective thermal stresses and pyroelectric coefficients; `laminate` is the exact
    law of layers. Checked to 3.5e-14 on laminates in SI units, against the dense system
    of a small grid, the uncoupled limits and Levin's relation.
  * **Viscoelasticity, hyperelasticity and damage** (`morphogen.homogenization.laws`,
    §61) for the material points, the finite elements, the voxels of the FFT solver (CPU
    and GPU), the cracks and the identification: `Material(viscoelasticity=...)` (a
    generalized Maxwell model: Prony series of the shear and bulk moduli, integrated
    exactly; WLF or Arrhenius shifts), `Material(hyperelasticity=...)` (neo-Hookean,
    Mooney–Rivlin, Yeoh, Arruda–Boyce, Gent, Ogden, Saint-Venant–Kirchhoff, with three
    volumetric energies and analytic tangents; with a viscoelasticity, Simo's finite
    linear viscoelasticity) and `Material(damage=...)` (isotropic damage: energy, Mazars
    or modified von Mises equivalent strains, linear or exponential softening, the crack
    band of Bažant and Oh from a fracture energy, a viscous regularization, a volumetric
    split). Checked against closed forms (relaxation to 3e-12, uniaxial tensions of
    incompressible rubbers to 1e-5, the dissipation of a band independent of the voxels).
  * **Harmonic regimes** (`morphogen.homogenization.harmonic`, §62): the properties
    `dynamic_stiffness` (storage and loss moduli of viscoelastic phases, by the
    correspondence principle), `complex_conductivity` (leaky dielectrics, the interfacial
    polarization of Maxwell and Wagner) at `frequencies`, and `refractive_index`
    (`Material(refractive_index=...)`, complex) at `wavelengths`; complex symmetric laws
    solved by conjugate orthogonal conjugate gradients (MINRES on their real form if the
    method stalls). Checked on laminates, on Dykhne's checkerboard (4e-4) and against the
    same layers followed in time.
  * **Extended finite elements of level sets** (`morphogen.homogenization.xfem`, §63):
    `LevelSetSample.of(micro, shape)` and `method: xfem` solve a morphology on a regular
    grid cut by its interfaces (Moës et al. 2003), without a mesher and without the
    stairs of the voxels: elasticity, every transport phenomenon and loading paths in
    small strains (`XFEMElasticProblem`, `XFEMThermalProblem`, `NonlinearXFEMProblem`).
    Laminates oblique to the grid to round-off; spheres of contrast 10 better on 8 cells
    than on 32 to 64 voxels.
  * **Phase field of brittle fracture** (`morphogen.fracture.phasefield`, §64):
    `homogenize(..., paths=..., fracture={"toughness": ..., "length": ...})` and the
    section `homogenization.fracture` break a voxel image along a loading path (AT1 or
    AT2, spectral or volumetric split, irreversibility by bound constraints or by the
    history field, a toughness per phase, initial cracks). The result is a `PathResult`
    with the damage, the regularized crack surface and the energies.
  * **RVE statistics of every result**: `rve.properties` takes the strength and the work
    of a path (`<path>.strength`, `<path>.work`; `PathResult.strength`, `work`,
    `driven`), the columns of a fracture (`<path>.fracture_energy`) and the named results
    of the couplings, of the harmonic properties, of the poroelastic law and of the
    tensors of expansion (`piezoelectric.e33`, `dynamic_stiffness.shear_loss_1`,
    `poroelastic.biot_coefficient`, `thermal_expansion.A11`). Fixed: the bounds of a
    study failed when a phase was absent from a realization.
  * **Weak couplings** (`morphogen.homogenization.poroelastic`, §65):
    `Material(swelling=...)` and the property `swelling` (hygroscopic and chemical
    expansion); the property `poroelastic` of a porous sample (the drained stiffness, the
    tensor and the modulus of Biot of any solid, heterogeneous or crystalline, and with
    `fluid_bulk_modulus` the undrained stiffness and the tensor of Skempton), by the FFT
    solver, finite elements or extended finite elements.
  * **Scalar Bloch waves** (`morphogen.homogenization.bloch`, §66):
    `band_structure(..., waves="acoustic" | "shear" | "tm" | "te")`,
    `scalar_band_structure`, `wave_coefficients`, `long_wavelength_speeds`: sonic and
    photonic crystals (two-dimensional for the electromagnetic waves), Wood's law.
    Checked against the exact dispersion of layered fluids and the TM gap of alumina rods
    (0.32–0.44, Joannopoulos et al.).
  * **Templates** with their cards in the gallery: `homogenization_transport`,
    `homogenization_piezoelectric`, `homogenization_viscoelastic`,
    `homogenization_poroelastic`, `homogenization_xfem`, `rve_multiphysics`.

  Changed signatures: `Material` (the fields `electrical_conductivity`, `permittivity`,
  `permeability`, `diffusivity`, `refractive_index`, `coupling`, `viscoelasticity`,
  `hyperelasticity`, `damage`, `swelling`; `has`, `inert`, `matrix` and
  `mean_conductivity` take a physics), `Loading` (`physics`), `homogenize`
  (`mean_temperature`, `frequencies`, `wavelengths`, `fracture`, `fluid_bulk_modulus`),
  `band_structure` (`waves`), `HomogenizationSpec` (the same options; `method: xfem`);
  the command `morphogen homogenize` takes `--frequencies`, `--wavelengths` and
  `--fluid-bulk-modulus`. Not covered: the new laws have no MFront, Abaqus or AMITEX
  export and no form in the graphical interface; the couplings, the harmonic regimes and
  the phase field run on voxel images under periodic conditions; the extended finite
  elements are periodic, in small strains along loading paths.
* **Changed default**: conforming meshes are exact by default
  (`method: auto` in `conforming_mesh`, the `mesh` section of configurations and
  `--mesh-method`): the exact mesh below when the microstructure has an exact route, the
  level set on a background grid otherwise (transparent dead leaves, tiling fields,
  oriented cells but tessellations) or, with a warning, should the exact mesh fail or its
  periodic faces not match. Tessellations are meshed from their exact cells
  (`min_feature` 0: their volumes to rounding, at the price of thin elements in the caps
  of the cells on the faces of non-periodic boxes; a positive `min_feature` regularizes
  them as before, `size/4` changing the volumes of the cells by 3 % in the median).
  `method: grid` gives the meshes of previous versions.
* Exact meshes without a background grid ([docs/theory.md](docs/theory.md) §94), for all
  the morphologies in VTK without voxelization (clean geometry,
  clean mesh): `mesh.exact.exact_mesh`, `conforming_mesh(..., method="exact")`, `mesh:
  {method: exact}` in configurations, `morphogen homogenize --mesh-method exact`.
  Tessellations keep their exact cells (`polycrystal_mesh(exact=True)`: not regularized,
  the bends of the facets as MMG's ridges, a Hausdorff distance of a millionth of the
  size: the volumes of the cells to rounding); faceted morphologies are meshed from their
  exact convex pieces (`convex_pieces()`: scale-rich plates and planar networks, fractal
  sponges as greedy boxes; `polyhedral_mesh`): the faces overlaid plane by plane into a
  face-to-face complex, cones remeshed by MMG with a reference per piece, the bends as
  ridges and the same Hausdorff distance (the facets stay planar: the volumes of the
  pieces to rounding), periodic faces meshed alike by `mmg2d`; the others from their exact
  point-wise level set (`Microstructure.level_set_at(points, reach)`, now given by every
  two-phase model with a level set, dead leaves of two phases included, and the slope
  bound `lipschitz`; `adaptive_mesh`): a balanced octree (`mesh.octree`) refined where a
  line of three points crosses the level set twice, misplaces its crossing by more than
  the Hausdorff distance (the parabola through the three values) or bends toward zero
  near it, or where the level set at the zero of the linear interpolation along a
  possible edge of the tetrahedra exceeds that distance (kinks: thin sheets), split into
  Kuhn tetrahedra and cones over fanned faces (conforming, matching opposite faces), cut
  by `mmg3d -ls` in one pass without MMG's detection of ridges (the edges of the box
  given; `ridges=True` keeps the sharp edges of the geometry, at the price of flat
  elements), the nodes of the interface then moved onto the exact zero level set (Newton,
  halved then undone where an element would flip); periodic faces are cut exactly at the
  zeros before MMG and kept (made again on another octree should they not match).
  Partitions of several phases give one level set per phase (`phase_level_sets_at`: dead
  leaves of any number of phases, osteons, inclusions such as cross-linked fibres), cut
  in turn by MMG's multi-material references (phase `k` by half its level set minus the
  least of those of the phases after it, so that no cut runs along the interfaces already
  made), then remeshed once; with `order: 2`, the middles of the edges of the interfaces
  are moved onto them (curved elements, Jacobians kept positive). The interface nodes lie
  on the exact surface to rounding (but where the move would flip an element, and where
  three phases meet), planes exactly; volumes are those of inscribed facets (−0.16 % for
  a sphere at a Hausdorff distance of a three-hundredth of its radius). Not covered:
  transparent dead leaves, tiling fields, and cells with crystal orientations but
  tessellations, meshed from their images. `medit.read_sol`, `write_sol(dimension=2)`;
  `examples/exact_meshes.py` writes a morphology of each route as VTK. Public interface:
  the modules `mesh.exact` and `mesh.octree`, `exact_mesh`, `adaptive_mesh`,
  `polyhedral_mesh` and `exact_route` in `mesh`, `conforming_mesh(method=...)`,
  `polycrystal_mesh(exact=...)`, the `level_set_at`, `phase_level_sets_at` and
  `lipschitz` of the microstructures, `convex_pieces` of fractal sponges and scale-rich
  networks, `GaussianField.values_at` and `contains`.
* Scale-rich networks in three dimensions ([docs/theory.md](docs/theory.md) §93.1):
  `scale_rich_network` with
  `dimension: 3` grows the plates or the beams of Fig. 6a of the paper in the box
  (`models.scale_rich_3d`). Plates (`ScaleRichPlates`): each splits a convex polyhedral
  void into two (faces clipped by Sutherland–Hodgman, the section of the void closing
  each part), so that plates and voids partition the box exactly into closed cells; the
  rim of a plate (the section of its void by its mid-plane) gives its area and its
  neighbours. Beams (`ScaleRichBeams`): cylinders with round ends grown until their axes
  reach another beam or pass within contact of it (open cells; `segments()` as capsules).
  Normals or axes uniform, drawn from an orientation law (`orientation`: layers, vertical
  walls...), or polarized by a pole (radial or tangential plates and beams, shells);
  gradients of nucleation along any axis; frame, voxels, partial volumes, level sets,
  JSON. Theory: the mean area of the first plate in closed form (`first_area`: half the
  volume times the mean inverse distance of two points of the box, 0.941156 in a cube),
  the next ones `c_3 (V/t)^(2/3)` (`c_3` = 1.236, measured), the relative density `1 - (1
  - rho_1 / 3)^3` (within 3.7 % of realizations from 0.1 to 0.97), the jamming thickness
  (`rho_1 = 3`), the tails of areas and degrees 5/2 (measured 2.48–2.50 and 2.53–2.6); a
  mean field of the free paths of beams (`beam_density_estimate`, within 5 % up to 0.2),
  whose density is the volume of their union (2^20 quasi-random points, within
  1.5·10⁻⁴), solved as for plates. Under kinematic conditions (an independent code), the
  elastic anisotropy grows with `alpha` as the paper finds: `E_min / E_max` from 0.73 to
  0.47 for plates at 30 %, from 0.72 to 0.18 for beams at 15 %, following the geometric
  anisotropy (correlations 0.97 and 0.98).
  `density_estimate` takes a dimension; `homogenization.postprocess.spatial_modulus` and
  `sphere_directions` give the Young's modulus along the directions of space (the elastic
  anisotropy `E_min / E_max` of 3D samples). `examples/scale_rich_networks_3d.py`; the
  template `scale_rich_plates`. Public interface: these names added. The sections of the
  viewer sample scale-rich networks, planar or not, on their plane (`viz.sections`).
* Scale-rich networks ([docs/theory.md](docs/theory.md) §93), after the scale-rich
  metamaterials of Both et al. (Nature 659, 345, 2026): the model `scale_rich_network`,
  ligaments `lambda_0 t^-alpha`
  thick grown one after another from points of the voids until they meet, as convex voids
  split by strips (the scheme of the authors' reference implementation: ligaments and
  voids partition the box exactly), extruded along an axis (`ScaleRichStructure`: voxels,
  partial volumes, level set, point queries, JSON); their dual graph (`degrees`, `links`),
  orientation tensor and geometric anisotropy, and power-law tails (`statistics`); the
  initial thickness given, or solved on each realization for a `volume_fraction` (a seed
  per ligament, so that the network moves little with `lambda_0`; regula falsi from the
  theory); uniform, aligned (`angles`, `angle_spread`) or polarized (`pole`, `pole_angle`,
  `polarized_fraction`) ligaments, linear or radial gradients of nucleation, a `frame` that
  makes a sample the cell of a periodic medium. Theory: the mean length of the first
  ligament in closed form (0.946402 in a square, Kellerer 1971), the next ones `c sqrt(A /
  t)` long (`c` = 1.166, measured: Both et al. take 0.946), the relative density `rho_1 (1
  - rho_1 / 4)` (mean field, within 3.5 % of realizations from 0.02 to 0.96) and the
  initial thickness of a density, the jamming thickness (`rho_1 = 2`, the boundary of the
  phase diagram of the paper), checked against realizations; the tails of lengths and
  degrees near 3 as the paper finds. New functions: `analysis.power_law_tail` (Clauset,
  Shalizi & Newman 2009, continuous and discrete) and
  `homogenization.postprocess.directional_modulus` (plane strain or stress). The
  template `scale_rich_network`, its gallery card, and `examples/scale_rich_networks.py`
  (the laws, the density against `alpha`, and an elastic anisotropy `E_min / E_max` that
  follows the geometric one, correlation 0.92, falling with `alpha` at 40 %). Public
  interface: these names added (`tests/public_api.txt`).
* The shear of the periodic cell of laser-architectured sheets:
  `cell_test(free_shear=False)` holds the shear of the cell at zero, the principal axes
  of the mean stretch fixed (the path of the strains prescribed in full);
  `PeriodicCell.build(shape=...)` meshes a cell by the numbers of elements asked.
* Biaxial paths of the periodic cell of laser-architectured sheets:
  `cell_test(ratio=...)` follows linear paths of the true strains (biaxial stretching,
  the stresses in `fields["nominal"]`); `mean_nominal`, the mean nominal stress of a
  membrane; `cell_test(until=...)` and `load_test(until=...)` stop a test on a
  condition; `load_test` follows a nonlinear path of its moved dofs (`direction` a
  function). New: `mean_nominal`; `cell_test(ratio=..., until=...)`;
  `load_test(until=...)` and a callable `direction`.
* Phases of different stiffness and smooth tracks for laser-architectured sheets:
  `Membrane` takes Young's modulus point by point (`design_membrane` one modulus per
  phase); `morphogen.laser.architectures.track_membrane` (smooth tracks over a mesh,
  several treatments, a treatment that changes along a track) and `periodic_lines` (the
  images of tracks over a periodic cell). New: `periodic_lines`, `track_membrane`;
  `Membrane(young=...)` and `design_membrane(young=...)` accept arrays;
  `cell_test(monitors=...)` records more quantities at each step.
* Designs of pixels for laser-architectured sheets (`morphogen.laser.architectures`):
  a design of pixels (`pixelate`, the majority phase of each pixel; `spot_opening`, the
  opening by a square spot), its treatment (`treatment_measures`: connectivity of the
  phases, heat-affected share, raster length, time and switches of the beam) and the
  sheet it makes (`design_membrane`, `sheet_tests`: the periodic cell pulled in several
  directions). New: `design_membrane`, `pixelate`, `sheet_tests`, `spot_opening`,
  `treatment_measures`.
* The processes of the tests, of the command line and of the process pools kept at full
  speed on Windows (tests run in the background were several times slower):
  after a few seconds of computing, Windows 11 moves a process that it deems in the
  background, such as one started by another program, to the efficiency cores at a low clock
  (EcoQoS). `morphogen._platform.full_speed` sets the execution speed of the process to
  never throttled (`SetProcessInformation`; nothing done on the other systems, never
  raises). The processes that a process starts do not inherit the setting, so each one
  makes it: the test processes and the workers of pytest-xdist (`tests/conftest.py`),
  the command line, the processes of `parallel.process_pool` and the build of the site
  (`docs/conf.py`). On the development machine (i9-13900H), throttled: `tests/test_laser.py`
  with `-n auto` takes 27 s instead of 45 (its longest test 6 s instead of 15), and
  `tools/ci_local.sh` 12 min instead of about 26 (the tests of each Python 11 min instead
  of 23 and 24, the site 5 min instead of 9). The throttling depends on the state of the
  desktop: without the setting, a run of `tools/ci_local.sh` at a moment when Windows did
  not throttle took 14 min.
* The strength contrast of the reinforcement of laser-architectured sheets:
  `contrast_phases`, reinforcements between the annealed 0.23 C steel and its laser
  martensite (the Hollomon parameters interpolated: a yield stress 1 to 4.64 times the
  matrix's, a harder state hardening less) under three rules of fracture
  (`FRACTURE_RULES`: `graded`, the critical strain of the martensite over the square of
  the interpolation share, and its bounds `brittle` and `ductile`). New:
  `contrast_phases`, `FRACTURE_RULES`.
* Barlat's Yld2000-2d criterion and crack growth for laser-architectured sheets:
  Yld2000-2d in plane stress
  (`yld2000_from_data`: the eight α fitted exactly to the yield stresses and Lankford
  coefficients at 0°, 45°, 90° and in equibiaxial tension; closest-point return with a
  line search and substeps, its consistent tangent; points of 10 values in
  `Membrane(anisotropy=...)`; `SheetMaterial.anisotropy["criterion"] = "yld2000"`,
  `yield_criterion`; `yld2000_profile`), checked against von Mises, a DC04 fitted to
  1e-8 and its profile in the cell; the growth of a crack by the breaking of elements
  (`load_test(erosion=True)`, `Membrane.alive`, the nonlocal averages among the living
  elements, Newton corrections halved when they turn an element inside out, the
  elements of the front that the iterations turn inside out broken),
  `CompactTension(refine=..., propagate=True)`, `graded(box=...)`; the pattern `spiral`.
  Changed signatures: `Fracture.ratios` (`alive`), `load_test` (`erosion`,
  `max_breaks`), `ct_mesh` (`refine`), `graded` (`box`).
* The rolling texture of a laser-architectured sheet:
  Hill's 1948 criterion in plane stress (`SheetMaterial.anisotropy`, the Lankford
  coefficients of the sheet; return mapping with its consistent tangent, the axes of the
  texture turning with the material; `hill_from_r`), the treated points isotropic or
  keeping a share of the texture (`"memory"`); the periodic cell of a sheet treated all
  over (`morphogen.laser.periodic`: the lattice of a pattern, the cell meshed with its
  nodes matched, the macroscopic gradient among the unknowns, a uniaxial stress in any
  direction; `SheetCase(loading={"kind": "cell"})`), checked against Hill's `r(α)`, the
  rule of mixtures of a laminate and the curve of an untreated sheet; crossed tracks of
  two spacings (`spacing2`), `make_pattern(cover=True)`. Fixed: the gradients of the
  shape functions of turned or sheared quadrilaterals (the transpose of the inverse
  Jacobian; the fillets of the dog-bones only: their tests change by less than 1 %).
  `load_test` keeps its monitors without the fields and maps reduced unknowns (`full`).
  Changed signatures: `Membrane` and `replay` (`anisotropy`), `SheetMaterial`
  (`anisotropy`, `hill`), `crossed` (`spacing2`), `make_pattern` (`cover`),
  `case_measures` (`inside`); new: `hill_from_r`, `ISOTROPIC`, `SheetCase.cell`,
  `PeriodicCell`, `cell_test`, `lattice`, `reduce_basis`.
* Load cases beyond the tensile test and alternative materials for laser-architectured
  sheets (`morphogen.laser.structures`): graded meshes of a compact tension specimen (its
  pins as stars of bars, the compliance of Saxena & Hudak within 2.5 %) and of an
  L-shaped plate, a displacement-controlled test with the initiation of a crack in any
  phase (the nonlocal average now within each phase), the cells of a design over any
  region (`CellProblem(case=...)`, cells without elements left out); Lankford's
  coefficient of the gauge, `r̄`, `Δr` and Hill's 1948 coefficients of a sheet tested in
  several directions, the objective `flow_isotropy`; `PeakSoftening` and the presets
  `22mnb5` and `aa6082_t6`; `Membrane(springs=...)`.
* Cellular design of laser-architectured sheets (`morphogen.laser.cells`), with the
  cellular automata and the topology optimization of SCOLASTIC: the fracture of brittle
  points in the tensile test (a critical plastic strain, local or averaged over a radius,
  `Fracture`; the failure strain among the measures; `SheetMaterial.fracture` by
  hardness); Swift laws and two laws mixed at a point; the phases of Wang et al. (2023);
  `replay`, the history of a test integrated again with other laws (the external work
  within 3e-5), the sensitivities of every switch of a cell (with the weakest
  cross-section); the automata `beso` (BESO swaps or neighbour exchanges, tested by
  finite elements, periodic and symmetric orbits, treatability by a spot, multigrid),
  `hca` and `topology` (MMA); `design_tracks`, the tracks that draw a design, treated by
  the process model; the `cellular` section of `morphogen laser` configurations and the
  template `laser_cells` (its card in the gallery). The martensite of the preset
  `annealed_ms1500` follows the Hollomon law of the paper (uniform elongation 0.062).
  Changed signatures: `Membrane` (`critical`, `fracture_radius`), `tensile_test`
  (`failure_margin`, `record_history`), `TensileResult` (`failure_strain`, `history`),
  `SheetMaterial` (`fracture`).
* Architectured sheets by localized laser treatment (`morphogen.laser`, the command
  `morphogen laser`, the template `laser_corrugated`, [docs/laser.md](docs/laser.md)),
  after Zhige Wang's thesis (PIMM, 2022) and the project SCOLASTIC:
  the moving beams of a laser treatment on a plate, on one face or on both at once (a
  Gaussian fibre beam above, a top-hat diode below), quasi-steady by an integral over the
  delay (images through the thickness, surface losses; separable on grids: a table of
  cycles in 0.04 s) and checked against Rosenthal's and Rykalin's solutions (2e-4) and the
  energy of a cross-section (2e-3), with transient finite volumes of a blank (LOD, Numba)
  within 4 % of it; the metallurgy of C–Mn steels (lever rule with Andrews's `Ac1` and
  `Ac3` plus a superheat, Koistinen–Marburger below Andrews's `Ms`, Maynier's hardnesses,
  Hollomon–Jaffe tempering by additivity), the peak-temperature hardening of AISI 430 and
  melt-pool carburizing, nitriding (Sieverts) and alloying; Ludwik or Voce flow laws
  fitted exactly to `R_p`, `R_m` and `A_g` by Considère, the strength of each hardness
  from the measured states (presets `aisi430`, `annealed_ms1500`, `dp1180`); laser paths
  (straight, corrugated, Fourier, crossed, honeycomb), their length, time, energy, gaps,
  curvature and G-code; the treatment of a path at the Gauss points, track after track,
  with the tempering of earlier tracks; a layered membrane in plane stress with large
  rotations (corotational logarithmic strain increments, radial return with
  `σ33 = 0`, the consistent tangent of the Kirchhoff stress: two or three Newton
  iterations a step, exact homogeneous tension to 2e-4) for the tensile test of
  dog-bone specimens; the optimization of the path (grid, Nelder–Mead, differential
  evolution, genetic, Bayesian with a Gaussian process and the expected improvement)
  for a strength–ductility objective under constraints of strength and process.
  Reproduced: AISI 430 gains nothing from corrugation (too little contrast), and the 430
  fully treated at 15, 20 and 25 kJ/m within 4 % of its measured yield strength, 0.5 %
  of its tensile strength and 0.3 % of its uniform elongation (its hardening fitted to
  the mean hardness of these sheets). Two more load cases: three-point bending of a
  strip by fibre sections (`laser.bending`: the elastic stiffness and the plastic limit
  load exact; a beam on one face against two), and tension in several directions (the
  pattern turned and repeated over the sheet, `directions`, the anisotropy measures and
  an `isotropy` objective). The gallery has cards for the templates `laser_corrugated`
  and `laser_bending`; the template test validates laser configurations; the
  conda-forge recipe imports the new package.
* Explicit dynamics with OpenRadioss, the open-source explicit finite element solver
  (https://github.com/OpenRadioss/OpenRadioss, AGPL-3.0, run as a separate program):
  the crushing, tension, shear and impact of a morphology at the rates of crashes,
  shocks and drops. `morphogen.io.radioss` writes the inputs in the Radioss Block Format
  of the input version 2022, each field in the columns of the Radioss Reference Guide
  (integers of 10 characters, reals of 20): the Starter input `<name>_0000.rad` (units,
  nodes, a part per phase; `/BRICK` for the voxels, with the stabilized brick Isolid 24,
  `/TETRA4` with nodal pressures against locking or `/TETRA10` for conforming meshes, the
  tetrahedra oriented by the natural coordinates of the guide, from node 4 toward nodes
  1, 2, 3, the opposite of VTK's; `/MAT/LAW1` elastic, `/MAT/LAW36` von Mises with the
  linear and Voce hardenings of `Plasticity` tabulated, and Norton's viscosity as yield
  curves at twelve plastic strain rates, `/MAT/LAW2` Johnson–Cook; failure by element
  deletion at a plastic strain; the faces held by `/BCS` and driven by `/IMPVEL` after a
  smooth step of velocity, or the morphology thrown by `/INIVEL` against a fixed
  `/RWALL/PLANE`; the self contact `/INTER/TYPE24` of the outer surface
  `/SURF/PART/EXT`; the reactions of the nodes of the loaded faces, the energies and
  momenta of the parts in the time history) and the Engine input `<name>_0001.rad`
  (`/RUN`, `/TFILE`, `/ANIM` of the displacements, velocities, reaction and contact
  forces, von Mises stresses, plastic strains, energy densities, pressures and statuses,
  `/DT/NODA/CST` for a mass scaling), with the mesh and the manifest that the results are
  read back with; `read_radioss_listing` (cycles, errors, termination),
  `read_radioss_th` (the time histories of `th_to_csv`) and `io.read_vtk_grid` (legacy
  VTK unstructured grids, ASCII or binary, classic or VTK 5.1, the animations of
  `anim_to_vtk`). `morphogen.dynamics` gives the model (`UnitSystem`, the consistent
  units of the inputs, Mg mm s by default; `ExplicitMaterial`, `JohnsonCook`,
  `ExplicitLoading`, `explicit_mesh`, `critical_time_step`, whose estimate gives the
  cycles of a run before it starts), the installation (`find_openradioss`: the path
  given, `MORPHOGEN_OPENRADIOSS`, `OPENRADIOSS_PATH`, `~/morphogen/openradioss`, the
  `PATH`, with the environment its programs need; `install_openradioss`, a release
  downloaded from GitHub for Linux or Windows and unpacked with its permissions, no member
  leaving its folder), the runs (`run_openradioss`: the Starter, the Engine with OpenMP or
  MPI, their progress read in the listing, killed at a stop of `morphogen.stopping`,
  their failures raised with the errors of their listings; `convert_outputs`) and the
  results (`read_explicit_run`: the nominal stress–strain curves of both faces from the
  reactions of the time history, else of the animations, the energies, the fields of
  the animations on the mesh; `summary`: the peak and plateau stresses, the
  densification strain by the energy efficiency, the absorbed energy per volume and
  mass, the gap of the stresses of the faces that tells a quasi-static test, the rebound
  of an impact; `write_frames`, the frames as tetrahedra for the 3D views;
  `explicit_test` does it all). The command `morphogen radioss info|install|export|run|read`,
  the line `openradioss` of `morphogen info`, and the tab Explicit dynamics of the
  interface (section Simulate): a dynamic test in engineering units (MPa, kg/m³, m/s,
  1/s, ms, the lengths of the morphology in mm, µm, cm or m) with its summary, curves (in
  morphoGen's components too, E33 and S33 for a compression along z, so that Comparisons
  draws them with the loading paths), energies, figures, frames in 3D and video, or its
  inputs and the scripts that run them elsewhere; the results of a run made elsewhere;
  the installation found or a release installed. The fields of materials take the
  material of their editor (`Field(default=...)` of the kind `materials`). Tested
  against a stand-in of OpenRadioss (`tests/fake_openradioss.py`), whose programs read
  the inputs and write outputs shaped like OpenRadioss's: the runs, conversions, stops,
  failures and readings, not OpenRadioss itself, which this development could not run.
* The stand-alone application: the interface in a window of its own (`morphogen
  desktop`, the program `morphogen-desktop` of the extra `desktop`, pywebview with the
  web engine of the system, Qt WebEngine on Linux), its server private to it on a free
  port of the local machine (`morphogen.gui.desktop`). The page uses the dialogs of the
  system when it runs there (`window.pywebview.api`): the folders of the workspaces saved
  and opened, the paths of the forms (**Browse…**: `Field(browse="folder" or "file")`,
  the folders of the runs of OpenRadioss, of AMITEX's results and of the databases), the
  files of the results saved where one chooses or opened by their applications, the
  folder of a result in the file manager (**Show its folder**); its menu holds the
  workspace, the folder of the session and the help; closing it asks first when the
  workspace holds items not saved; without pywebview it falls back on the browser.
  `packaging/desktop` freezes it with PyInstaller into a folder that runs without Python
  (`morphoGen.exe`, `morphoGen.app`; its kernels cached by Numba in the user's cache
  folder), which `morphoGen --check` checks without a window; the workflow
  `desktop.yml` builds and checks the bundles of Linux, Windows and macOS on demand. The
  icon is a section of a sheet gyroid. The recipe of conda-forge constrains pywebview
  (the extra `desktop`) and imports `morphogen.dynamics`.
* The rigidity confirmed at the floor of the residual, the tolerance 1e-10 of the voxels
  factorized, the voxels that bear found in near-linear time, and the sites of the square
  and cubic lattices again (left by fffbdd7: the 2D sites at the cut 1e-12; the boxes of
  64³ to 96³ voxels; whether f/nu steepens or the threshold lay low).
  `analysis.percolation`: the voxels factorized iterate by default to 1e-10
  (`SOLVER_TOLS`, 1e-8 for the FFT solver; `rigidity_threshold(solver_tol=None)`,
  `percolation_exponent(tol=None)`), and an energy that bears under 1000 times the cut is
  decided again by a solution at 1e-11, near the floor of the residual
  (`RigidityThreshold.confirmed`): at 1e-8, the tolerance of the 3D measurements of
  fffbdd7, one to three iterations leave floppy images at energies of 1e-13 to 2e-8 whose
  stress averages agree with them, so that they bore by the cut 1e-12 (23 of 32
  realizations of the sites on 32³ to 80³ voxels from fractions lower by 0.0003 to 0.003
  than at 1e-10, where those energies fall to 1e-21 to 1e-14; the medians of 64³ and 80³
  voxels 0.0010 and 0.0013 lower), and at 1e-10 a few keep 1e-12 to 3e-11, their stress
  averages and their perturbations at the round-off agreeing too, which fall under the cut
  or turn unresolved at 1e-11 (39 of the 252 energies confirmed on 8³ to 80³ voxels, 3 of
  25 in 2D; a modulus of 2.0e-12 on 64³ voxels stays to three digits at 1e-11 and 1e-12);
  `RigidityThreshold.unresolved` is now counted (it stayed 0).
  `homogenization.voxel_fe.load_bearing` unites its parts by size with halved paths: 0.1 s
  on 64³ voxels of the sites at the threshold and 0.7 s on 96³, where the plain forest
  grew chains as long as the parts (0.5 s on 48³, 2.5 s on 64³, more than the
  factorization); the docstrings record that SuperLU commits its first guess of the fill,
  20 times the entries of the matrix (37 bytes per entry: 9.1 GiB on 64³ voxels for 0.8
  GiB touched, 31 GiB on 96³; the factors fill 1.5 to 1.8 times the entries in 3D, 2.1 to
  2.4 in 2D), which Windows charges against the commit of the machine, and that a
  factorization failing for memory does not free its arrays (SciPy 1.18): the runs limited
  the commit of each factorization (a job object of the process, between the guesses of
  2.5 and 5 times the entries, which SuperLU meets by halving its own; the same factors),
  recycled their processes and held their dispatch when less than 10 GiB of commit was
  left. Measured on the i9-13900H (on battery, then on mains; nine processes, seven at
  times, the local CI holding the dispatch for 35 min), each realization bisected to its own threshold (its images nested, sites
  u < p), the moduli at fixed fractions from realizations of their own, from the top down:
  sites of the square lattice, 400 realizations of 16² to 128² pixels and 300 of 192² and
  256² (22,404 solutions, 4.5 h of processes): p_half(L) = 0.4397, 0.4564, 0.4547, 0.4559,
  0.4595, 0.4611, 0.4614, 0.4617 and 0.4612, flat from 96², which p_c + a L^(-3/4) does
  not fit (chi2 34 for 7, 0.4635) and the exponent fitted (1.5 +/- 0.2) extrapolates to
  p_r = 0.4617 +/- 0.0004 (0.4617 +/- 0.0003 from 32², the means 0.4620 +/- 0.0003), the
  widths of the realizations' thresholds falling as L^(-0.879 +/- 0.017), nu = 1.14 +/-
  0.02 (central-force rigidity in 2D: 1.21 +/- 0.06, Jacobs & Thorpe 1996), the share
  bearing at p_r about one half from 96²; the first measurement, 0.4627 +/- 0.0016, had
  medians bisected to 0.0035 and nu = 4/3 imposed, the energies deciding as its stress
  averages did. f/nu = 2.26 +/- 0.07 +/- 0.05 (f = 3.01 with nu = 4/3; 200 to 400
  realizations, 1.6 h), the slope rising from 1.55 (16² to 24²) to 2.93 +/- 0.68 (192² to
  256²; chi2 28.5 for 7): 2.46 +/- 0.06 +/- 0.07 from 32², 2.63 +/- 0.07 +/- 0.08 with a
  correction A L^(-f/nu)(1 + B L^(-omega)), omega = 1 (B = -10, chi2 1.9 for 6; 2.51 for
  omega = 2, omega not fixed by the data), 2.15 at the old threshold, its 2.16. Sites of
  the cubic lattice, 1,725 realizations of 8³ to 80³ voxels (200, 300 of 32³, 25 of 80³;
  10,111 solutions, 13.4 h): p_half(L) = 0.1356, 0.1338, 0.1332, 0.1316, 0.1321, 0.1314,
  0.1313, 0.1309 and 0.1308 extrapolate to p_r = 0.1304 +/- 0.0002 (nu = 0.8765, chi2 5.6
  for 7; 0.1300 +/- 0.0003 from 32³, the means 0.1303 +/- 0.0001, the shares of 48³ to 80³
  crossing at 0.1302), the widths giving nu = 0.76 +/- 0.03 from 16³; the share bearing at
  p_r stays at about a third from 8³ to 80³ (0.39 to 0.31 +/- 0.03), while at the old
  0.12904 it falls from 0.38 to 0.11 on 64³ and none of 25 on 80³: that threshold lay
  0.0014 low. f/nu = 4.48 +/- 0.09 +/- 0.09 on 8³ to 80³ (f = 3.93 +/- 0.08 +/- 0.08; 400
  to 20 realizations, 3.8 h; chi2 6.4 for 7), 4.73 +/- 0.17 +/- 0.16 from 16³, 4.27 +/-
  0.43 from 32³, between successive sizes 3.9 to 5.7 without a trend; with a correction
  4.85 +/- 0.21 +/- 0.18 (omega = 1, B = -4.3, chi2 3.9 for 6, not significantly lower),
  4.73 +/- 0.17 for omega = 2, 4.8 +/- 1.1 with omega fitted; at the old 0.12904 the same
  moduli steepen as before (5.04, 5.73 from 16³, 6.15 from 24³, chi2 19.6 for 7). The
  class: in 3D f/nu lies 1.4 to 2.0 combined errors above the bond-bending 4.3 and 18
  above the 2.1 of central forces on the bcc lattice (the abstracts of Arbabi & Sahimi
  1993, part I): the exponent is that of bond bending within two errors, not of central
  forces; in 2D it lies 3 to 8 below the bond-bending 2.97, still rising at 256²; the
  thresholds, distinct from the connectivity's, and the nu of the widths (1.14 and 0.76
  against 4/3 and 0.8765) make the voxels' rigidity a percolation of its own, as for
  central forces, with the 3D f/nu of bond bending: neither class. Theory §55 (the
  criterion and its tolerance, the commit of the solver, the 2D and 3D thresholds, the
  exponents and their table, the class, what the laptop leaves out), the docstrings,
  tests/public_api.txt. Checked: tests/test_voxel_fe.py (the parts against the connected
  components of their graph, random and a chain visited from its far end),
  tests/test_percolation.py (a floppy image of 32³ voxels that bears at 1e-8 with its
  stress average and does not once confirmed; the rule with its confirmations; the default
  tolerances), with tests/test_public_api.py 132 passed, and tests/test_gui_tools.py -k
  percolation 5 passed. About 4 h 35 min of computing (26.7 h of processes, 23.3 h kept:
  the runs at 1e-8 and before the confirmations were done again). Left: 96³ voxels (35 to
  60 s per solution with two processes, 7.5 GiB each) and more realizations of 64³ and 80³
  voxels (20 to 100 for the moduli, means known to 36 to 68 %), which would tell whether
  the correction that the 3D fits allow (4.48 to 4.85) is there; 384² and 512² pixels,
  where the 2D slope and median would show whether f/nu reaches 2.97 and whether nu = 1.14
  holds; the discs and spheres again at the tolerance 1e-10; a 3D central-force f/nu and
  nu read from a source for the voxels' ratio.
* GMRES recycling from the first restart (GCRO-DR), the stalls of finer grids, and
  `tangent: auto` by phase (the Left of the deflated restarts). `solvers.gmres` and
  `gpu.gmres` take a `recycle` (`solvers.Recycling`, one per path of the voxels) and a
  `stagnation`: the first solve of a path that restarts leaves the recycler the harmonic
  Ritz vectors U of its last deflated restart and their orthonormal images C = A M^-1 U,
  which the Arnoldi relation of that restart gives without a product; each later solve is
  GCRO-DR (Parks, de Sturler, Mackey, Johnson & Maiti 2006; `solvers._gcro_dr`, written
  once for the CPU and the GPU through their operations on vectors): its residual
  projected out of C, the Krylov vectors of (I - C C^T) A M^-1 orthogonalized against C
  and themselves in the same Gram-Schmidt passes, the residual minimized over [U, V] (the
  least squares problem of GMRES with a diagonal first block, the same rotations), the
  harmonic Ritz vectors of [U, V] with respect to its images kept at each full cycle (G^T
  G z = theta G^T W^T V z) and combined in place with their images and the next residual
  (72 vectors at most instead of 51 during a solve, 40 kept between the solves). The
  columns of Y reuse the images of their tangent; a new tangent computes them again (20
  products) if the last solve filled a cycle (`Recycling.needed`) and drops the space
  otherwise, so that a path whose solves never restart is that of GMRES-DR to the bit;
  `info["recycled"]` counts the solves recycled. The 16³ stalls: the 31 solves that ran to
  their 1,320 iterations above their tolerance (6 of them above ten times it, the 6 stalls
  reported) were all in the third step of the Rousselier composite (0.4 to 0.6 % of
  strain), whose 25 Newton iterations lowered the residual by a fifth before it was
  divided, 41,000 of its 55,869 products. Its preconditioned tangent has dozens of
  eigenvalues of both signs near zero (34 harmonic Ritz values of negative real part in
  400 Arnoldi iterations, about 10 on 8³ voxels), its symmetric part as many, and no
  Krylov method solves it within its iterations: GMRES without restart reaches the 1e-2 of
  a correction after 1,535 iterations and not the 3e-4 of a column of Y in 3,000,
  GMRES-DR(80, 30) and (100, 40) stagnate at 1.3e-2 and 1.2e-2 (not the restart length),
  MINRES on the symmetrized tangent takes 2,600 iterations to 1e-2 and its solution leaves
  11 % of the residual of the full tangent (a fallback to it would not pay), six other
  reference media change MINRES by -2 to +16 % and leave GMRES-DR at 1.6e-2 to 2.0e-2 (not
  the preconditioner: the indefiniteness lies in the softening voxels), recycling does not
  help; CG and MINRES stall there too (14 solves at 1,320), that path leaving the step
  after 8 Newton iterations at a failed integration. Along the paths GMRES therefore stops
  a solve whose residual has not halved over its last 8 cycles
  (`solvers.GMRES_STAGNATION`: on the 16³ composites the 35 solves that would end at
  `maxiter` stop after about 300 iterations, and none of the 95 that converge after
  restarts, up to 528 iterations; 6 cycles would stop three of these, 3 cycles eight), an
  inexact correction, and a step whose GMRES corrections stop above their tolerance at 3
  Newton iterations in a row is divided (`nonlinear._STALLED`; CG, MINRES and LU
  unchanged). `auto` decides by the largest asymmetry of the phases
  (`nonlinear.asymmetry(tangent, wts, groups)`, per phase on the GPU too), which a small
  learned phase no longer dilutes: 0.21 for the learned perceptrons and recurrent networks
  at any fraction, 3.9e-3 to 6.2e-3 for the matrices of morphoGen's composites and 1.6e-2
  for the latent hardening (one phase), so that the 2e-2 decision stays that of
  morphoGen's laws; the recurrent network at 0.6 % of 8³ voxels keeps its full tangents
  (48 Newton iterations instead of the 61 of the symmetrized ones, which its whole
  volume's 1.3e-2 gave it; one voxel of 5³, 0.8 %: 43 against 60). Measured (8 threads,
  the machine shared with another job; GPU RTX 3000 Ada): the Rousselier composite at a
  high triaxiality on 8³ voxels 8,678 products instead of 11,064 (9,763 for CG and
  MINRES), the same 89 Newton iterations and curve, 14.1 s instead of 17.2 s on the CPU
  and 40.1 s instead of 50.2 s on the GPU (the same products); on 16³ voxels 14,947
  instead of 55,869 (35,244), 127 and 128 Newton iterations, 9 divided steps, the curve to
  6e-7, 106 s instead of 819 s on the CPU and 76 s instead of 508 s on the GPU (the same
  products; 54,380 for GMRES-DR there; without dividing the steps 22,291, and 22,273
  without recycling; recycling alone 47,395); the localizing GTN composite on 8³ voxels
  3,312 instead of 4,263 (53 Newton iterations instead of 61, the same solution), 6.1 s
  instead of 7.2 s, GPU 16.0 s instead of 17.7 s; on 16³ 8,887 instead of 13,655 (13,430
  for CG and MINRES, the same curve there), 82 and 83 Newton iterations, 5 divided steps,
  63 s instead of 87 s, GPU 101 s instead of 133 s; the Rousselier composite in four
  increments, which stalls on coarser grids, 7,501 instead of 19,028 on 6³ voxels (21,293
  for CG and MINRES, the same divided steps and stresses to 1e-7; 8,522 without recycling,
  12,740 with recycling alone) and 13,948 instead of 24,692 on 8³ (28,672); 6³ in ten
  increments 3,380 instead of 3,792; recycling only between the solves of a same tangent
  8,510 and 3,775 on 8³. The learned laws, whose solves never restart, keep their products
  and times (8³ voxels through auto: the perceptron and the recurrent network 892 and 928
  products, 7.4 and 8.2 s against 7.9 and 8.1 s on the CPU, 13.2 and 14.0 s against 13.0
  and 14.5 s on the GPU). Checked against fffbdd7 to the bit on the CPU and the GPU: the
  symmetric cases (voxels in small and finite strains and full integration, finite
  elements periodic, kubc, finite strains and multigrid, material points, a crystal and
  the softening GTN composite forced onto the symmetrized tangents) and the paths whose
  solves never restart (Armstrong-Frederick composites and slip hardening with full
  tangents, small and finite strains; the learned perceptron and recurrent network and the
  Armstrong-Frederick and GTN composites through auto). Tests (test_nonsymmetric.py):
  GCRO-DR against GMRES-DR along 8 indefinite nonsymmetric systems (the same solutions to
  2e-10, 556 products against 901, the first solve to the bit, the images reused by a
  solve of the same operator, a sequence that never restarts to the bit with the space
  empty, a solve that fills no cycle dropping it; on the GPU the same iterations), GMRES
  stopped where it stagnates on a system with 30 negative and 30 small positive
  eigenvalues of 300 (290 iterations at 0.14 of the residual instead of 2,000 at 0.13; a
  converging solve to the bit; the GPU), the measure per phase against its definition and
  a learned voxel at 0.8 % of 5³ voxels kept full, the Rousselier composite with recycling
  on 6³ voxels (3,367 iterations, 23 solves recycled; on the GPU the same, slow) and in
  four increments (slow); test_nonsymmetric.py and test_gpu.py 35 passed (the slow ones
  included), test_fft_nonlinear.py, test_hill.py, test_learning.py, test_ntfa.py,
  test_porous.py and test_public_api.py 71 passed (not slow), and in WSL (MFront and MGIS)
  test_mfront.py, test_learning_mfront.py and test_nonsymmetric.py 39 passed. Theory 31
  and 34, architecture, the public interface. Left: the GMRES of the finite elements
  (multigrid) neither recycles nor was measured on softening points; the conjugate
  gradients and MINRES of the symmetric paths are not stopped where they stagnate
  (unchanged to the bit: 14 solves at 1,320 iterations on the 16³ Rousselier composite); a
  stalling step is divided only at its third stagnated correction (about 900 products on
  16³ voxels), no earlier sign of its indefinite tangents read from the first cycles; the
  images of the recycled vectors by new tangents about pay for themselves on the
  corrections (the columns of Y gain most); the window of 8 cycles measured on two
  composites (a dense system of 400 with 40 negative and 40 small eigenvalues converges
  after a plateau of 1,300 iterations, which it would stop); the times measured on a
  machine shared with another job (the base of 16³ took 316 s in an earlier run).
* The shifts of the window method in the RVE tool of the interface, and the areas of
  planar windows on the axes of the figures (the "Left" of 5c047a5: the RVE tool did not
  offer the shifts, and the axes of the figures said volume for the areas). The tool "RVE
  size" of the tab Statistics (`gui/tools/statistics.py`) gains the field "shifts per
  axis" (`shifts`, shown for the window method, for its volume fraction and its stiffness
  by FFT alike; 1 by default): more than 1 passes `rve.shifts` to the pipeline (the
  tilings shifted around the periodic realization, the stiffness from the same solutions),
  1 leaves the configuration as it was (no `rve.shifts`: run.json and the results
  unchanged); refused before anything is generated, for a count that is not positive or on
  a morphology whose domain is not periodic (its realization then not periodic), with the
  messages of `rve.window_statistics` and `rve.window_tensors`: their refusals are now one
  check that the tool shares (`rve.scaling._shift_count(shifts, periodic)`), their
  messages unchanged. The tables of the mean stiffness of the largest windows count them
  as "C the mean of n windows x T tilings". The figures of the reports (`rve.report.plot`,
  which the RVE tool shows) draw the windows of a planar image against their areas: "area
  V" for the variance and the mean, "window area V" for the tensors, where cubes keep
  "volume V" and "window volume V" (the RVE sizes already in V_RVE^(1/2)). README,
  architecture. Checked: the tool with 2 shifts on a periodic Boolean model of spheres (a
  realization of 8³ voxels, windows of 2 and 4): results.csv counts 64 x 8 and 8 x 8
  windows, its volume fraction (means and variances) equal to the bit to that of
  window_statistics with 2 shifts on the same realization (generated again from its
  run.json), its mean stiffness that of window_tensors with 2 shifts to 1e-12, titled "C
  the mean of 8 windows x 8 tilings"; by default and with 1, the volume fraction of
  window_statistics without shifts to the bit, no column tilings, the configuration
  without rve.shifts, results.csv, rve_report.json and rve_report.md equal byte for byte;
  the refusals (counts 0 and -2, 2 shifts on a morphology that is not periodic, the
  stiffness too, before any output) with the very messages of the API, the windows of that
  morphology without shifts still studied; the axes of the figures of a planar image of
  32² pixels (area V, window area V, V_RVE^(1/2), nothing of a volume) and of a cube of 8³
  voxels (volume V, window volume V, V_RVE^(1/3)), read as they are saved; in the browser,
  the field shown for the window method and 2 shifts on the default spheres (96³ voxels)
  studied in 7 s, the windows counted 512 x 8, 64 x 8 and 8 x 8. Left: the realizations of
  the RVE tool stay cubes (a planar morphology is studied in 3D, its figures in volumes;
  planar windows through the API).
* The rigidity decided by the energies, the voxels that bear, the bounds of the bisection
  tried last, and the 3D threshold, the discs and the spheres again (left by b266c34: near
  the cut on large planar boxes the stress averages of floppy images strayed to −6e-7; 40³
  voxels and more; the 3D threshold with the shift 1e-12; the discs and the spheres).
  `analysis.percolation`: the modulus of a realization is the energy of its solution,
  ⟨σ:ε⟩ (stationary: an upper bound whose error is of second order, where the stress
  average's is of first), and `rigidity_threshold` decides by it: a realization does not
  bear at or under the cut; above it, it bears when its stress average agrees with its
  energy within half of it, otherwise (`RigidityThreshold.unresolved`) when two more
  solutions with the element matrices perturbed at their round-off (2.2e-16 of their
  entries) leave the energy within 1 % (`solver="fe"`), or when its stress average exceeds
  the cut (`"fft"`, whose iterations stop short of the round-off); the default cut is the
  solver's (`CUTS`: 1e-8 for the FFT solver, 1e-12 for the voxels factorized; `cut=None`);
  for a family whose phase grows or shrinks with the fraction the bisection tries its
  bounds last, only when the bracket reaches them (the same brackets for a monotone
  share); `percolation_exponent` takes the energies too.
  `homogenization.voxel_fe.load_bearing(mask, axis)` keeps the voxels that bear along an
  axis, exactly: the biconnected components of the graph of the voxels and their nodes
  (Hopcroft & Tarjan, compiled) merged where they share a voxel are parts meeting at
  single nodes in a tree, and those through which no cycle winds around the box along the
  axis bear nothing (the parts hinged at an edge, which may bear as bars, stay); the
  voxels factorized assemble them alone, void around. `morphogen percolation --cut`
  (default the solver's), "unresolved" in the JSON and the text; the value "unresolved" of
  the tool "Percolation threshold", whose field "cut" defaults to none. Measured on the
  i9-13900H (on mains, then on battery at the end of the runs): 1,920 solutions of the realizations of the 2D threshold (96² to 256² pixels at
  its five fractions near p½, tolerance 1e-10): the stress averages of the 927 floppy ones
  lie between −6.1e-7 and −8e-14 and stay there at the floor of the residual (−1.30e-7 at
  1e-10 and at 5e-12 on 128² pixels, its mechanisms at 2e5 for a unit strain: the
  round-off of the element matrices strains them), and four perturbations at the round-off
  of the 89 doubtful ones scatter the floppy energies by 17 to 50 % and those that bear by
  1.4e-3 at most, their stress averages within 30 % of them; the energy decides as the
  stress average in 1,919 of the 1,920 (the other bears at 1.02e-8, stable), the one
  floppy energy above the cut (2e-7 on 256², mechanisms at 7e5) is unresolved and floppy
  by the round-off, and four rigid images of 256² have moduli under 1e-8 (1.3e-10 to 8e-9,
  resolved within 6 %). In 3D the resolved moduli spread without a gap from 1e-4 down to
  1e-14 (137 of the 375 resolved on 40³ voxels near p½ between 1e-14 and 1e-8), the floppy
  energies under 1.5e-11. One process: the factorization of the sites of the cubic lattice
  at p_r takes 0.38 s on 32³ voxels, 1.5 s on 48³ (0.12 GiB) and 4.1 s on 64³ (0.28 GiB),
  15 to 25 % less on the voxels that bear (79 to 82 % kept), but 2.8 s at 0.15, 12.8 s at
  0.17 and 122 s (1.28 GiB) at 0.2096 on 48³, the upper bound whose images made the 30 s
  per solution of b266c34; minimum degree on Aᵀ + A with SymmetricMode, the current
  ordering, fills 1.2 million entries on 24³ in 0.45 s against 3.8 million in 1.2 s by
  COLAMD, 1.8 million in 1.0 s by minimum degree on AᵀA and 13.7 million in 12.8 s by a
  nested dissection by planes of nodes; SuperLU's relax and panel_size change nothing; the
  factors of one image as the preconditioner of the next of the bisection (48³, 345 to 727
  voxels apart) do not converge in 500 to 2,500 iterations (34 to 123 s, against 0.7 to
  3.7 s for new factors). Sites of the cubic lattice (192 realizations of 8³ to 48³ voxels
  between 0.1046 and 0.1606, tolerance 0.0015, 16 processes, 12 min): p½(L) = 0.13916 to
  0.13041 extrapolate to p_r = 0.12904 ± 0.00029 at the cut 1e-12 (0.12861 ± 0.00031 with
  ν = 1, 0.12913 ± 0.00045 without 8³ and 12³, ν fitted 0.88 ± 0.29) and 0.13018 ± 0.00032
  at 1e-8, whose p½ are a step higher from 32³; f = 4.58 ± 0.19 ± 0.07 at 0.12904 (128
  realizations, 16.5 min; f/ν = 5.23 ± 0.22 ± 0.08, df/dp_c ≈ −240), 4.84 ± 0.39 ± 0.07
  without 8³ and 12³ and 5.68 ± 0.55 ± 0.20 on 24³ to 48³ as the share of the realizations
  that bear at p_r falls from 33 % to 16 %, which a threshold 0.001 higher would undo, so
  that no correction to scaling can be told from the threshold's error; the boxes of 64³
  voxels (4 s per factorization alone, 34 s with ten processes sharing the memory
  bandwidth) were stopped after their first fraction. Boolean discs of 4 pixels of radius
  (nested in the fraction, 64 realizations of 32² to 256² pixels): p_r = 0.608 ± 0.006,
  between their thresholds through 8 and 4 neighbours (0.596 ± 0.003, 0.624 ± 0.005 for
  the same family by the fire), f = 3.20 ± 0.19 ± 0.24; Boolean spheres of 3 voxels (24³
  to 48³): p_r = 0.202 ± 0.004, its p½(L) within 0.007 of those through 26 neighbours
  (0.1815 ± 0.0025; 0.236 ± 0.004 through 6), f = 4.65 ± 0.59 ± 0.05: their grains meet at
  several nodes, which brace them (the floppy and rigid moduli six decades apart, 73 to 98
  % of the images decided by their connectivity), and their exponents are those of the
  sites within the errors, the voxels not resolving the necks narrower than a voxel; the
  abstracts of Halperin, Feng & Sen (1985) and of Feng, Halperin & Sen (1987) give no
  exponent value (the former: the Swiss-cheese exponents of the conductivity, the elastic
  constants and the permeability significantly larger than the lattices', except the
  conductivity in 2D). Theory §55 (the criterion, the solver of the voxels, the 2D and 3D
  thresholds, the discs and spheres, the table of f, what the laptop leaves out), the
  docstrings, docs/architecture.md, tests/public_api.txt. Checked: tests/test_voxel_fe.py
  (the voxels that bear on chains, a hanging pixel and block, a column and a hinged voxel;
  their moduli and conductivities those of the whole images, planar and 3D),
  tests/test_percolation.py (the decisions of chains of pins, ball joints and hinges, in a
  line or zigzagging, by both solvers; the rule on unresolved energies; the bisection with
  its bounds last, rising and falling, and its refusal; the hinged chains in 7 fractions;
  "unresolved" in the command), tests/test_gui_tools.py (the value "unresolved"): with
  tests/test_public_api.py, 148 passed. About 1 h 45 min of computing, the CPU mostly full
  (the 2D survey 30 min, its tight solutions of 256² pixels stopped after a few minutes;
  11 min of 64³ voxels stopped). Left: the boxes of 64³ to 96³ voxels (about an hour each,
  the memory bandwidth shared), which with the distribution of the moduli at the threshold
  would tell whether f/ν steepens or the threshold of the energies lies 0.001 low, and so
  the class of the voxels' rigidity; the 2D sites again at the cut 1e-12; more
  realizations of the discs and spheres (their thresholds within 0.006 and 0.004).
* GMRES with deflated restarts, and `tangent: auto` decided by the measured asymmetry of
  the tangents (the Left of the nonsymmetric tangents). The restarts of `solvers.gmres`
  and `gpu.gmres` are deflated (Morgan 2002, GMRES-DR): the next cycle starts from the
  harmonic Ritz vectors of the 20 harmonic Ritz values of least modulus
  (`solvers.GMRES_DEFLATE`; `deflate=0` gives the plain restarts) and from the residual,
  their combinations written over the first vectors of the basis in place (the sums in the
  order of the basis whatever the threads, one thread per component on the GPU), so that
  the memory stays that of the 51 vectors of the plain restarts; the full first block of
  the Hessenberg matrix is factorized by Householder reflections, the residual computed
  again after deflated cycles (a plain cycle follows if it is above the tolerance), and a
  solve that does not restart is the plain one to the bit. The Gram-Schmidt kernels of
  GMRES are swept by the threads from two blocks of 4096 values on (they were from 65,536
  values; the same sums to the bit): an iteration on 24³ voxels from 15.7 to 9.3 ms
  (plain) and 10.4 ms (deflated) against 6.5 ms for the conjugate gradients. On the
  softening points of the Rousselier composite at a high triaxiality (8³ voxels) GMRES
  takes 11,064 products instead of 41,068, no solve stalling above its tolerance (12 did),
  against 9,763 for the conjugate gradients and MINRES on the symmetrized tangents, for
  the same Newton iterations (89) and curve (13,309 keeping 10 vectors, 10,049 keeping
  30): 21 s instead of 48 s on the CPU (12 s for the conjugate gradients), and on the GPU
  the same 11,064 products instead of 47,072, 45 s instead of 175 s (29 s); on 6³ voxels
  3,792 against 6,919 and 4,454; on the localizing GTN composite 4,263 against 4,448. On
  16³ voxels, where the solves take about 150 iterations, the deflation barely helps
  (55,869 against 56,871, six solves still stalling, 54,804 with 100 vectors keeping 40;
  GPU 54,311 against 63,470) against 35,244 for the conjugate gradients and MINRES, which
  no restarted method reaches there. GCRO-DR (Parks, de Sturler, Mackey, Johnson & Maiti
  2006), prototyped, which recycles the deflated space from a solve to the next across the
  Newton iterations, saved a further 8 % of the products per solve on the Rousselier
  composite and 17 to 25 % on the GTN one, for 20 products by each new tangent and 40
  vectors kept between the solves: not taken, the solves of the other laws not restarting
  (fewer than 50 iterations on 8³ voxels). With laws that are not all symmetric, `tangent:
  auto` now measures at each linearization of the Krylov solvers (voxels on the CPU and
  the GPU, finite elements with multigrid; LU keeps the full tangents, its iterations
  exact Newton ones) the asymmetry a = |C - C^T|/|C| of the block-diagonal operator of the
  tangents of all the points (`nonlinear.asymmetry`, Frobenius norms weighted by the
  points), and symmetrizes the tangents below 2e-2 (`nonlinear.ASYMMETRY`; to the bit
  those of `tangent: symmetric` in small strains), GMRES solving the others; the outputs
  report the largest a and the linearizations solved with full tangents (`asymmetry` and
  `full_tangents` in `info`). The threshold from the iteration counts on 8³ voxels:
  morphoGen's own laws stay below it (1.2e-4 for the slip hardening, 1.8e-3
  Armstrong-Frederick, 2.8e-3 GTN, 3.7e-3 Rousselier, 1.6e-2 for Franciosi's strong latent
  hardening) and their full tangents save no Newton iterations there (30, 37, 40 and 89
  against 30, 37, 41 and 87; the latent hardening 153 with 11 divided steps against 126
  with 6; the GTN composite 61 against 50), while the learned perceptrons and recurrent
  networks reach 0.14 to 0.21 (0.10 and 0.11 in finite strains) and their full tangents
  take a third of the iterations (47 against 148, 43 against 124, 48 against 152; 55
  against 136 and 162 in finite strains); diluted among von Mises voxels the recurrent
  network keeps fewer iterations with its full tangent down to the smallest fraction tried
  (47 or 48 against 74, 66, 63 and 61 at 7, 3.5, 1.6 and 0.6 % of the voxels, a = 4.5e-2,
  3.1e-2, 2.1e-2 and 1.3e-2), so that below about 1.5 % of the volume it takes the
  symmetrized tangent and a quarter more iterations. Point by point, |C_p - C_p^T|/|C_p|
  (the measure first tried) does not separate them: 0.14 for GTN points near their
  softening, 0.18 for the latent hardening, 0.20 to 0.35 for the learned laws. The
  localizing GTN composite bifurcates at its third step (0.6 %): from the same converged
  state both tangents reach a solution of the same increment (residuals 2.5e-7 and 2.6e-7
  of the nodal forces; 4.4e-9 and 3.6e-9 at a tolerance of 1e-8, the paths parting at the
  same step towards the same solutions), their fluctuations 60 % apart and their stresses
  1.1 %; the symmetric part of the tangent of the fluctuations is positive definite at the
  solution of the symmetric path (its least eigenvalue 1.0e-3 of the largest) and has a
  negative eigenvalue at that of the full one (-3.0e-3): the symmetric path finds the
  stable solution, GMRES on the full tangents an unstable equilibrium, each path then
  keeping its branch (694 and 715 MPa at 2 %); documented, the solver unchanged, `auto`
  taking the symmetric path there now. The Schur complement of the macroscopic block
  refreshed at every linearization with full tangents saves 18 to 30 % of the Newton
  iterations (the learned laws 47, 43 and 48 to 35; Armstrong-Frederick 37 to 27; GTN 40
  to 33; slip hardening 30 to 21; latent hardening 153 to 121) for 36 to 91 % more
  products and more time in every case: not kept. Checked against 16d2750 to the bit: the
  symmetric cases (voxels in small and finite strains and full integration, finite
  elements periodic, kubc, finite strains and multigrid, material points, a crystal and
  the softening GTN composite forced onto the symmetrized tangents, MINRES included), on
  the CPU and the GPU; the full tangents with GMRES (Armstrong-Frederick composites in
  small and finite strains, slip hardening) and the learned recurrent network and
  perceptron through `auto`, on the CPU and the GPU (their solves do not restart); the
  plain GMRES with the threads from two blocks on, to the bit of 16d2750's on 12³, 16³ and
  24³ voxels. Tests (test_nonsymmetric.py): the deflated restarts against the plain ones
  on an indefinite nonsymmetric system (115 against 305 products with 50 vectors, 116
  against 357 with 30, the same solution; without restart the plain GMRES to the bit; the
  GPU the same iterations), the decisions of `auto` (the measure against its definition,
  an Armstrong-Frederick composite at 1.3e-3 symmetrized to the bit of `symmetric`, the
  recurrent network at 0.14 full to the bit of `full`, LU not measured, multigrid
  symmetrized), its measure and decisions on the GPU those of the CPU (the GPU's
  symmetrization that of its kernels, to the bit), the Rousselier composite with full
  tangents on 6³ voxels (no solve above its tolerance, fewer than 4,500 iterations; on the
  GPU the same 3,774 iterations, marked slow); the tests that reached GMRES through `auto`
  on morphoGen's laws now force `tangent: full`; in WSL (MFront and MGIS) test_mfront.py,
  test_learning_mfront.py and test_nonsymmetric.py, 35 passed. Theory 31 and 34,
  architecture, the public interface. Left: the learned laws below about 1.5 % of the
  volume solved on their symmetrized tangents; the restarted GMRES on the softening points
  of finer grids (16³: 1.6 times the products of MINRES, solves stalling, deflated or not)
  and GCRO-DR's recycling; why the full tangents of the latent hardening divide more steps
  on voxels; a cheaper update of the macroscopic coupling than its solves; the times
  measured on a machine shared with other jobs (the 8³ ones on a quiet one).
* The voxels assembled and factorized, a solver fast near the rigidity threshold, and the
  elastic exponent f in 3D (left by a972517: near p_r the FFT solutions take 45 L
  iterations in 2D and do not converge in 3D, the Fourier preconditioner of a homogeneous
  medium spreading the displacements over the void while the clusters pinned at their
  corners turn at vanishing energies). `homogenization.voxel_fe`: `VoxelFEElasticProblem`
  and `VoxelFEThermalProblem` extend the FFT problems (the same trilinear voxels, loads,
  averages and fields) and solve them with the matrix assembled on the nodes of the voxels
  of the materials (`voxel_stiffness`; the two nodes of an axis one voxel thick folded
  into one, as the periodic grid makes them), by conjugate gradients preconditioned by the
  sparse LU factorization (SuperLU, minimum-degree ordering, diagonal pivots) of the
  matrix shifted by `SHIFT` = 1e-12 of its mean diagonal, restarted from the true residual
  every `CYCLE` = 100 iterations: the floating and floppy clusters make the matrix
  singular (at least 400 mechanisms on 24³ voxels at p_r), but the loads are orthogonal to
  its null space, which the shifted factorization maps onto itself, so that the shift only
  preconditions (at 1e-10 one image of 24³ voxels in 558 stalled at a residual of 5e-7
  after 2,200 iterations, its modulus off by 0.1 %; 1e-12 solves it in 3 to 10).
  `rigidity_threshold(..., solver="fft"|"fe")` and `percolation_exponent(...,
  solver=...)`, recorded in `RigidityThreshold.solver` and `PercolationExponent.solver`;
  `morphogen percolation --solver fft|fe` (in the JSON and the text), the field "solver"
  of the tool "Percolation threshold"; the FFT solver stays the default. Compared on the
  same images at p_r (the laptop on mains, the GPU at 420 to 465 MHz in P5 on these small
  solutions, 1,530 to 1,830 MHz under load): 64² pixels, FFT 1,310 to 2,665 iterations in
  3.9 to 7.8 s on the CPU and 2.8 to 6.9 s on the GPU, FE 1 to 3 iterations in 0.08 to
  0.15 s; 128², FFT 7,000 to 7,400 (most at the bound, residual 3e-5) in 52 to 58 s (CPU)
  and 16 to 17.5 s (GPU), FE 2 to 4 in 0.6 s; 12³ voxels of the cubic sites, FFT 957 to
  1,600 (half at the bound, residuals up to 2e-4) in 0.9 to 2 s (CPU) and 1.9 to 5.6 s
  (GPU), FE 1 to 2 in 0.04 s; 16³, FFT at the bound of 1,800 for all (residuals 2e-6 to
  3.5e-5, moduli off by up to 16 %) in 2.7 to 5.3 s, FE 0.09 s; 24³ 0.35 s, 32³ 1.3 to 2
  s, 40³ 7 s, 48³ 30 s (0.24 GiB of factors), 256² 4 s, 512² 71 s; floppy images that the
  unconverged FFT solutions put above the cut (1.5e-8 on 12³, 7.3e-8 on 128²) are at 2e-15
  and 2e-11. The smoothed-aggregation multigrid of the finite elements (the translations
  for near-null modes) does not converge on these matrices (64² at p_r: 4,200 iterations,
  residual 6e-3). Not the default, as it is not better everywhere: on a dense phase (sites
  at 0.5) the FFT solver takes 53 iterations and 0.14 s on 16³ voxels against 12.8 s, and
  73 iterations, 0.81 s on the CPU and 0.27 s on the GPU, on 32³, where the factorization
  ran more than 8 min with 3.6 GB before it was stopped; with a contrast every voxel is
  assembled (24³ voxels at p_r with 1e-8: more than 6 min, where the FFT solver stops at
  its bound, 2,200 iterations, in 4.3 s on the GPU). Measured with it (rng.random(shape) <
  p; seeds 2026 and 2027 in 2D, 3026 and 3027 in 3D; ten processes on the CPU, beside the
  local CI): the sites of the square lattice over 16² to 256² pixels (96
  realizations, tolerance 0.0035, solver_tol 1e-10), p½(L) = 0.4409, 0.4566, 0.4597,
  0.4566, 0.4597, 0.4597, 0.4628, 0.4628, 0.4597, extrapolate to p_r = 0.4627 ± 0.0016 (nu
  = 4/3; 0.4623 ± 0.0015 with nu = 1.21, 0.4618 ± 0.0019 without 16² and 24²; 15.5 min, at
  most 34 iterations; the 32 realizations of 16² to 64² by the FFT solver gave 0.4647 ±
  0.0056), and at it 64 realizations of the same sizes, void around, give f = 2.88 ± 0.12
  ± 0.23 (statistical, then from the threshold; f/nu = 2.16 ± 0.09 ± 0.17; df/dp_c ≈ −146;
  2.98 ± 0.16 ± 0.28 without 16², 3.13 ± 0.17 ± 0.34 without 16² and 24²; 3 min); the
  sites of the cubic lattice over 8³ to 32³ voxels (96 realizations, between 0.0976 and
  0.2096, tolerance 0.0035; the shift then 1e-10, the one stall decided by its modulus to
  0.1 %), p½(L) = 0.1370, 0.1317, 0.1335, 0.1317, 0.1317, extrapolate to p_r = 0.1307 ±
  0.0009 (0.1304 ± 0.0011 with nu = 1, 0.1312 ± 0.0012 without 8³; 11 min; the run,
  stopped in its boxes of 40³, logged the medians of 8³ to 32³, fitted afterwards by the
  fit of rigidity_threshold), between the thresholds of the same realizations through 26
  and 18 neighbours, 0.0975 ± 0.0008 and 0.1383 ± 0.0009 (percolation_scaling, 16³ to 64³,
  96 realizations), far below that through the faces, 0.3116: the corners alone, ball
  joints, hold the voxels (Maxwell's count balances at 0.1, 0.3 in 2D); at it 64
  realizations give f = 3.23 ± 0.30 ± 0.25 (f/nu = 3.69 ± 0.34 ± 0.29 with nu = 0.8765;
  df/dp_c ≈ −270; 3.33 ± 0.47 ± 0.32 without 8³, 4.04 ± 0.52 ± 0.34, f/nu = 4.6, without
  8³ and 12³; 1 min). Against the values read in the abstracts, bond bending f/nu about
  4.3 in 3D (Arbabi & Sahimi 1993, part I) and f ≈ 3.75 (Sahimi & Arbabi 1993, part II,
  physical gels), central forces on the bcc lattice f/nu ≈ 2.1 (part I): in 3D the voxels'
  3.69 lies 1.4 combined errors below bond bending and 3.5 above the bcc central forces,
  moving towards bond bending as the smallest boxes are left out; in 2D 2.16 lies 4.1
  combined errors below the bond-bending 2.97 (f = 3.96, Sahimi 1994, as quoted before,
  not read again). As far as the data go, the thresholds, distinct from those of the
  connectivity, take the voxels out of the bond-bending class, and the 3D exponent puts
  them nearer bond bending than the central forces of the bcc lattice; the 2D exponent
  lies below bond bending, with no central-force value read to compare. Near the cut on
  the largest planar boxes, rigid moduli come down to 1.3e-8 from 96² while the stress
  averages of floppy images, whose soft modes take displacements of 1e5 for a unit strain,
  stray to −6e-7 (their energies, quadratic in the error, do not: 4e-9 for one of 128² at
  −1.3e-7); 27 of the 28 moduli solved between 1e-9 and 1e-6 on 96² to 256² bear by their
  energy as by their stress average, to 1 %, the other lies at the cut. Theory §55 (the
  solver of the voxels, the thresholds and exponents in 2D and 3D, Sahimi & Arbabi 1993
  part II), the docstrings, docs/architecture.md, README, tests/public_api.txt. Checked:
  tests/test_voxel_fe.py (the matrix against the operator of the FFT solvers on the nodes
  of the materials, elastic and thermal, planar and one or two voxels along an axis; the
  stiffness and conductivity tensors of the FFT solvers to 1e-8, planar and 3D, the other
  phase soft or void, three spacings; sites of the square and cubic lattices at p_r,
  floppy at round-off and bearing to 1e-6 of the FFT solution, in 2 iterations against 300
  to 900; uniform and void images, the shift refused out of (0, 1)),
  tests/test_percolation.py (rigidity_threshold and percolation_exponent with solver="fe"
  deciding and measuring as the FFT solver, void and at a contrast, in at most 3
  iterations where it takes more than 100; the chains of pins; the solver refused;
  --solver fe in the command, JSON and text), tests/test_gui_tools.py (the tool with the
  field "solver"): with tests/test_public_api.py, tests/test_fft.py and the percolation
  tests of the tool, 127 passed. About 2 h of computing in all:
  40 min of boxes of 40³ voxels stopped before their end, with the shift 1e-10 that
  stalled, and the local CI running alongside. Left: 40³ voxels and more in 3D,
  the thresholds within 0.0005 (df/dp_c ≈ −146 and −270) and a few hundred realizations,
  to settle the class of the voxels' rigidity; the energy rather than the stress average
  as the estimate of the moduli on large boxes, and a cut that comes down with its
  round-off; the 3D threshold again with the shift 1e-12; the discs and the spheres.
* Shifted windows on periodic images, and the edges of planar windows (the "Left" of the
  window method: the windows tiled the image from its origin only, and the reports printed
  V^(1/3) also for the areas of the squares of a planar image). `rve.window_statistics`
  and `rve.window_tensors` take `shifts` (1 by default: the tiling from the origin, the
  results unchanged to the bit) and, on a periodic image (`periodic`: that of a
  `VoxelImage`; an array needs `periodic=True`), add the tilings shifted by the offsets k
  w // shifts voxels along each axis that the windows tile, all their combinations
  (shifts^3 tilings of cubes, shifts^2 of squares, fewer for windows of fewer voxels),
  wrapping around the image; the tensors from the solutions of the unit loads already
  computed (no new solve: the Numba kernel of the window sums takes the offset of its
  tiling and wraps its voxels, each tiling one more pass over the image, 2 to 3 % of a
  solve of 32^3 to 64^3 voxels at a tolerance of 1e-6). The estimators (`rve.scaling`,
  theory §9): each tiling a translate of the first on the torus, so that by stationarity
  its windows have the joint law of the unshifted ones; the mean over all the windows of
  all the tilings (the image mean when its edges are multiples of w); D^2(V) the variance
  pooled within the T tilings, sum_t sum_(k in t) (Z_k - Zbar_t)^2 / sum_t (n_t - 1), the
  mean of their unbiased variances: of the expectation of the variance of one tiling
  (unbiased when its windows are uncorrelated, V >> A3) and of a smaller scatter; not the
  variance of the T n windows about their mean with T n - 1 degrees of freedom, lower by T
  (n - 1) / (T n - 1) (the overlapping windows know the image mean no better than one
  tiling); the count behind the uncertainties n, the windows of one tiling (the
  independent ones): the standard error D(V) / sqrt(n) of the mean, which the shifts leave
  unchanged, the relative errors, the Student tests of the components of the tensors and
  the weights (n - 1) / 2 of the fits (conservative for the variance), the fits and the
  RVE sizes from these variances. `SizeStatistics` and `TensorStatistics` gain `tilings`
  (the number of tilings of each size: the column `tilings` after `n` in their rows,
  `rve_<tensor>.csv` and `rve_report.json`) and `dimension` (3, or 2 for the squares of a
  planar image) with `edges()`; `WindowTensors` gains `dimension`, `offsets` (those of the
  tilings of each size, its `tensors` then tiling by tiling) and `tilings`;
  `StudyReport.dimension`; the reports count the windows as n x T, with a note. The
  pipeline's window method takes `rve.shifts` (a positive integer; refused for the
  ensemble method and on a domain that is not periodic): `results.csv` gains `tilings`.
  Wherever an edge is printed, a planar window's is the root of its area and a cube's
  V^(1/3): the heads of the report's tables (L^2, L^3), the edges of the statistics, of
  the RVE sizes (with a note) and of their bootstrap intervals, of the discrepancy,
  `rve_table(..., dimension=2)`, `rve_<tensor>.csv`, `rve_report.json` (with `dimension:
  2`), the figure of the RVE sizes (V_RVE^(1/2)) and the RVE tool of the interface (L² for
  a square). Theory §9, README, architecture. Checked: the mean over all the windows of
  all the shifted tilings of a periodic 16^3 image equal to the image mean exactly (dyadic
  window means), each tiling's too, the pooled variance the mean of the variances of the
  tilings rolled by hand (1e-12) and the variance of all the windows lower by T (n - 1) /
  (T n - 1), the distinct offsets (3 shifts of windows of 2 voxels: 8 tilings, of 4: 27),
  squares across a planar image; on independent voxels (D^2(V) = p (1 - p) / w^3 exactly),
  16 images of 32^3 and windows of 4^3, the shifted variance (64 tilings) within 3 errors
  sqrt(2 / (n - 1)) of one tiling's on every image, both unbiased on average, of a scatter
  0.35 times that of one tiling there (0.55 on 200 images, 0.70 with 2 shifts); every
  window of every shifted tiling of a laminate normal to z f1 A1^T C1 A1 + f0 A0^T C0 A0
  to 1e-9 (the layers wrapping; the stiffness of orthotropic phases with distinct shear
  moduli, and the conductivity), its statistics (n of one tiling, the pooled standard
  deviations by hand to 1e-12); on a random 8^3 image the six unit loads solved once each
  with or without shifts (the solves counted), the first tiling equal to the bit to the
  unshifted windows, each tiling of the box of the box's tensor to 1e-14, the variances of
  the diagonal within the error of one tiling; the pipeline with `rve.shifts: 2`
  (results.csv, its volume fraction that of window_statistics on the same realization, the
  mean stiffness the apparent one to 1e-8, the report, rve_stiffness.csv, the refusals);
  the edges of a planar study in the report, rve_report.json, rve_stiffness.csv, rve_table
  and the interface's tool; by default the outputs of window_statistics, window_tensors,
  their statistics and the reports of the API and of the pipeline (results.csv,
  rve_report.json, .md, .tex, rve_stiffness.csv, rve_conductivity_tensor.csv) equal byte
  for byte to those of 16d2750 (but the edges of planar windows), the earlier tests
  unchanged. Left: the RVE tool of the interface does not offer the shifts (its
  realizations are cubes); each tiling is one more pass of the kernel (4 shifts: 64 passes
  per size), the windows at every position without summed-area tables; the effective
  degrees of freedom of the pooled variance (between n - 1 and T (n - 1)) are not
  estimated, the fits and the tests keeping n - 1; windows that do not divide the image
  leave each tiling its remainder (the mean of all then not exactly the image mean); the
  samples of the ensemble method stay cubes (no dimension for `ensemble_statistics`), and
  the axes of the figures say volume for the areas.
* Nonsymmetric consistent tangents in morphoGen's own solvers (the Left of the learned
  laws and of the MFront exports): the laws whose tangents are not symmetric keep them
  whole in the volume elements, and the Newton corrections are solved by restarted GMRES.
  The solver: `solvers.gmres` and `gpu.gmres`, preconditioned on the right by the
  preconditioners of the conjugate gradients (the Fourier inverse of the reference medium
  on voxels, the multigrid of the symmetric part by finite elements, where LU factorizes
  the full tangent as it is), classical Gram-Schmidt passed again where it cancels most of
  the vector (its dot products in one sweep of the basis, summed by blocks that do not
  depend on the threads; one reduction over the rows of the basis on the GPU, without
  cuBLAS), restarted after 50 iterations (`solvers.GMRES_RESTART`: 51 vectors, 1.2 kB per
  voxel). GMRES rather than BiCGSTAB: one product by the tangent per iteration and the
  least residual over the Krylov space, about as many products as the conjugate gradients
  on the symmetrized operators of morphoGen's laws (2,235, 1,129 and 1,250 against 2,414,
  1,174 and 1,338: GTN and Armstrong-Frederick composites, a polycrystal with slip
  hardening, 8³ voxels) where BiCGSTAB took 1.3 to 1.4 times more than GMRES (3,145,
  1,498, 1,793; 1,061 against 819 for a learned recurrent network), no breakdown, a
  monotone residual: the nonsymmetric counterpart of MINRES. When: the option `tangent`
  (`auto`, `symmetric`, `full`) of `NonlinearFFTProblem`, `NonlinearFEProblem`,
  `simulate_point` and of the configurations (`homogenization.nonlinear.tangent`); `auto`
  decides from the laws (`behaviour_table`, `Behaviours.symmetric`, the problem's
  `symmetrize`): symmetric for elasticity, von Mises and Hill without Armstrong-Frederick
  terms, crystals in small strains without isotropic hardening, the learned potentials and
  standard models (`LearnedLaw.symmetric`) and the MFront behaviours that declare
  `@ProvidesSymmetricTangentOperator` (`mfront.declared_symmetric`, `MFrontLaw.symmetric`:
  the trait `isConsistentTangentOperatorSymmetric` of the tridimensional hypothesis in the
  header MFront writes beside the library, which the generic interface does not export;
  without the header, not symmetric); not symmetric (measured on the full tangents of two
  steps, of their largest entries) the Armstrong-Frederick hardening (1.5e-2), the
  isotropic hardening of slip systems (5e-4) and the crystals of finite strains (1e-3),
  porous plasticity (4e-3 for GTN, 2e-2 for Rousselier), the learned perceptrons and
  recurrent networks (7 to 33 %). With full tangents the kernels of the CPU and of the GPU
  skip their symmetrization (the flags `symmetric` of `update_points`,
  `update_points_finite` and `_update`, `sym` of the point kernels and of `behaviours.cu`;
  in finite strains that of the tangent of the logarithmic strain and that of dP/dF),
  MGIS's tangents come whole (`MFrontPoints`, `_tangent_out`, `_dpdf_out` with
  `symmetric`), the condensation takes K_Ew (the integral of the stress increments of the
  fluctuation, `macro_response` of the three discretizations) for K_wE^T; at a material
  point Newton's method takes the exact tangents of the laws that are not symmetric and of
  the MFront laws (MGIS's were symmetrized), and `path_bifurcation` keeps the symmetrized
  tangents (its eigenvalues are those of the symmetric part). Symmetric laws keep the
  conjugate gradients and their results to the bit: checked against 50f11ec (curves,
  Newton and linear iterations on voxels in small and finite strains and full integration,
  by finite elements periodic, kubc, finite strains and multigrid, at a material point,
  and a crystal forced onto `tangent: symmetric`), and in the tests `auto` against
  `symmetric` to the bit. Checked (tests/test_nonsymmetric.py): a recurrent network (6
  states, trained in the test: 20 epochs on 8 walks) beside stiff voxels takes 47 Newton
  and 819 linear iterations against 148 and 2,631 with its symmetrized tangent (a
  perceptron 43 and 757 against 124 and 2,561), the same curve; on a homogeneous image and
  by finite elements its symmetrized tangent does not converge where the full one follows
  the material point to 7e-7; crystals with Franciosi's latent hardening along a biaxial
  path that turns: 44 and 47 Newton iterations against 72 and 79 at material points, 35
  against 51 and 53 against 64 by finite elements (LU: exact Newton iterations), the same
  curves to 1e-6; the asymmetry of each law and the symmetrized tangents their symmetric
  parts (to the bit in small strains); the operator, its coupling and K_Ew of the voxels
  and of the elements against finite differences (GTN, small and finite strains); the full
  tangents of every law on the GPU against the CPU to 1e-10, and paths
  (Armstrong-Frederick composites, small and finite strains, the recurrent network) to
  1e-6; GMRES on dense nonsymmetric systems, on the CPU and the GPU; multigrid and GMRES
  by finite elements against LU; the conversion of nonsymmetric MGIS tangents whose
  entries all differ, TFEL's shears (12, 13, 23) against morphoGen's (23, 31, 12), turned;
  the option in configurations; and in WSL (MFront 5.1, MGIS 3.1: test_mfront.py,
  test_learning_mfront.py and test_nonsymmetric.py, 31 tests) the declared symmetry of the
  twins (the elastic laws symmetric, the implicit twins not) and of the learned laws, the
  tensions of the four learned models through MGIS at a material point now converging with
  the iterations of morphoGen's learned points (they did not for the perceptron and the
  recurrent network), and on voxels beside a stiff phase the same curves and Newton
  iterations as morphoGen's learned points (their symmetric parts, forced, taking more or
  failing). test_paths_with_softening_points forces `tangent: symmetric` (it tests
  MINRES). Left: on voxels the small asymmetries of morphoGen's own laws do not change the
  Newton iterations at the default tolerance (the macroscopic unknowns corrected by
  quasi-Newton iterations and the corrections inexact: 30, 40, 37 and 89 against 30, 41,
  37 and 87 for a polycrystal, GTN, Armstrong-Frederick and Rousselier composites); on
  softening points the restarts make GMRES 2 to 5.5 times costlier than the conjugate
  gradients and MINRES (Rousselier at a high triaxiality: 19,149 products with 120
  vectors, 54,025 with 30, against 9,763), where `tangent: symmetric` is the better
  choice, and a localizing GTN composite reaches other localized solutions (3 % apart); a
  deflated or flexible restart, and the Schur complement of the macroscopic block
  refreshed at every iteration for full tangents.
* The rigidity threshold, where the elastic moduli of a phase by the FFT solver vanish,
  and the elastic exponent f measured there (left by 91d8987: the voxels joined at corners
  and edges turn about them, so that the moduli vanish between the thresholds through 8
  and 4 neighbours, 26 and 6 in 3D, which no connectivity gives).
  `analysis.rigidity_threshold(generate, sizes, lower, upper, cut=1e-8, contrast=0, ...)`
  returns a `RigidityThreshold` (a `PercolationScaling` with, per size, the decisions and
  moduli of the realizations at the fractions tried; the cut, the largest modulus solved
  under it and the smallest above, the solutions and the images decided without one, the
  iterations): a realization bears when its modulus C_xxxx of a uniaxial strain along the
  axis, the other phases void, exceeds the cut (1e-8 of the Young's modulus of the phase);
  the fraction at which half the realizations of each size bear is bisected as for the
  connectivity (common random numbers: the median of the realizations' own thresholds; the
  realizations of percolation_scaling from the same seed) and extrapolated as p_c + a
  L^(-1/nu). The criterion is chosen over a set multiple of the contrast and over the
  crossing of the curves L^(f/nu) <C>: void around, the moduli of the realizations that do
  not bear vanish to the round-off of the solver and those that bear sit 3.6 to 7.6
  decades above (sites of the square lattice: floppy below 2e-11 up to 48² pixels and
  2.2e-10 on 64², rigid above 4.7e-4 on 16² and 8.0e-7 on 64²), whereas with a contrast δ
  those that do not bear take δ times an enhancement rising to 650 (32²) and 4,400 (64²)
  near their own rigidity while those that just do bear as little as 2e-6 (32²), which
  leaves no multiple of δ = 1e-8 between the two; the crossing presupposes f and averages
  moduli spread over decades. The images are decided without the solver where they can be
  (no component wrapping around the box through 26 neighbours, 8 in 2D: floppy; a
  component joined by faces, 6 or 4, wrapping: rigid; an image containing a bearing one of
  the same realization, or contained in a floppy one, the moduli growing with the
  stiffness of the voxels): 64 % of the images on 16² pixels, 43 % on 64². The elastic
  solutions near the threshold take up to 45 L iterations (1,703 on 48², 2,885 on 64², the
  bound 5,800 reached on 96²), beyond the solver's default 20 L + 1000: the moduli take 50
  L + 1000 in rigidity_threshold and in percolation_exponent (physics elastic; the
  conductivity keeps the solver's). The bisection of percolation_threshold and the
  realizations of percolation_exponent are factored out (`_bisect`, `_samples`),
  unchanged. `morphogen percolation CONFIG --sizes ... --rigidity [--cut C]`, and
  `--exponent elastic` without `--threshold` now extrapolates the threshold by the moduli
  first, no longer through 26 neighbours (the JSON says `threshold_criterion`);
  `--rigidity` is refused without sizes, along any axis and with `--exponent thermal`. The
  fields "rigidity threshold" and "cut" of the tool "Percolation threshold" (the shares of
  the realizations that bear and their moduli against the fraction, the finite-size
  scaling; the threshold of the exponent f when none is given); theory §55 (the rigidity
  threshold, the elastic exponent; Jacobs & Thorpe 1995 and 1996, Arbabi & Sahimi 1993,
  Kantor & Webman 1984), docs/architecture.md, README. Measured on the sites of the square
  lattice (rng.random((L, L, 1)) < p): the rigidity threshold over 16² to 64² pixels (32
  realizations, tolerance 0.0035, seed 2026; six processes on the GPU, 1 to 9 min per
  size), p½(L) = 0.4409, 0.4534, 0.4597, 0.4597, 0.4566, extrapolates to p_r = 0.4647 ±
  0.0056 (nu = 4/3; 0.4635 ± 0.0050 with the nu = 1.21 of central-force rigidity
  percolation, 0.4587 ± 0.0057 without 16²), a third of the way from 0.4073 (8 neighbours)
  to 0.5927 (4), the widths of the realizations' thresholds shrinking from 0.052 to 0.014
  as about L^-1; at it, 48 realizations of 16² to 64² pixels, void around (seed 2027),
  give f = 2.47 ± 0.21 ± 0.34 (statistical, then from the threshold; f/nu = 1.85 ± 0.16 ±
  0.26 with nu = 4/3, f = 2.24 with nu = 1.21, 2.18 ± 0.36 ± 0.37 without 16²; df/dp_c ≈
  −62, about 2.8 at the threshold without 16²; residuals up to ± 0.3, the means scattering
  by 16 to 25 %), 3.7 combined errors below the 3.96 (f/nu = 2.97) of the bond-bending
  lattices. As far as the data go, the threshold, distinct from that of the connectivity,
  takes the voxel model out of the bond-bending class, whose every connected cluster is
  rigid, into a rigidity percolation of central forces (the corners pins; ν = 1.21 ± 0.06
  in 2D, Jacobs & Thorpe 1996), and the exponent, below the bond-bending one, points the
  same way less firmly; its value for that class is not settled by these sizes. The
  contrast at p_r (24², 16 realizations, the same images): 1e-8 raises the mean by 4e-5 of
  void's, 1e-6 by 0.4 % and 1e-4 by 30 % (the floppy realizations at δ times 14 to 58),
  the iterations alike (at most 461, 460 and 433 against 518 void). The exponent ran on
  the CPU (three or four threads per process, 3 to 12 min per size): the GPU, saturated by
  twelve processes and then throttled to 210 MHz, ran slower than the CPU on these boxes;
  in all about 80 min of computing, the threshold of 96²
  pixels alone running 40 min before it was stopped. Checked: tests/test_percolation.py
  (chains of pixels joined at corners, a zigzag of pins floppy at round-off and a diagonal
  bearing as 0.2747/L exactly: the moduli vanish below 0.6 though the pixels connect from
  0.3, one solution of each per realization and size, the diagonal's elastic exponent 1, a
  contrast with a cut above its enhancement; sites of the square lattice bearing between
  the fractions where the same realizations join through 8 and through 4 neighbours, the
  decisions growing with the fraction, the exponent at the threshold with its standard
  error; the errors; `morphogen percolation --rigidity`, alone and before `--exponent
  elastic`, the refusals), tests/test_gui_tools.py (the tool on discs of a planar
  morphology: the figures, the values, the exponent f at the same threshold, sizes
  required). Left: f on boxes beyond 64 sites, with the threshold within 0.002 and a few
  hundred realizations, to tell which class the voxels' rigidity belongs to; the discs and
  spheres, and 3D, where the solutions near the rigidity threshold converge slower still
  (sites of the cubic lattice on 12³ voxels stop at 1,240 iterations with residuals up to
  3e-4); a solver that converges in fewer than 45 L iterations near the threshold (a
  multigrid, or a deflation of the soft modes), on which the cost rests.
* The window method for tensors: the stiffness C_ij and compliance S_ij, and the
  conductivity and its resistivity, of the windows of one large periodic realization (the
  window method measured the volume fraction alone, and the conductivity tensors of the
  studies were not averaged: the "Left" of the stiffness of the RVE studies).
  `rve.window_tensors` (`rve.windows`, `WindowTensors`) solves the realization once per
  unit macroscopic load by the FFT solver (six unit strains; the three in-plane ones E11,
  E22, E12 across a planar image, its plane stiffness in Voigt 11 22 12; three unit
  gradients) and takes as the tensor of a window the matrix of the window means of the
  mutual energy densities <eps(k) : C : eps(l)>_w (<grad T(k) . K grad T(l)>_w),
  integrated at the Gauss points of its voxels as the solver integrates the energy (a
  Numba kernel, by chunks of 65,536 windows x planes): symmetric to the bit (its upper
  triangle mirrored) and positive semi-definite (the energy of every combination of the
  loads); its mean over the windows tiling the box the box's mean energy to round-off and,
  by Hill's lemma, the box's apparent tensor of the mean stresses (`stiffness_tensor`,
  `conductivity_tensor`) to the tolerance of the solver (the energy is stationary: its
  error of the second order in that of the fields); its diagonal the energy estimate of
  Dirrenberger et al. (2014, eqs. 64, 66). Not the window means of the stresses under the
  unit strains (Kanit et al. 2003): they tile as exactly but are neither symmetric nor
  positive (16 % off symmetry on random voxels of a contrast of 10, indefinite there and
  in windows of 2 voxels at 1000, measured), and in a laminate have no scatter across the
  layers whatever the window (the tractions continuous); nor <sigma>_w <eps>_w^-1, neither
  symmetric nor tiling (the reasons in the module documentation and theory §9).
  `WindowTensors.statistics` gives the `TensorStatistics` of the windows as realizations
  (`method="windows"`): per size the mean Cbar and the standard deviation of each
  component, S = Cbar^-1, <C^-1> and the standard deviations of the compliances, each
  component a series with its RVE sizes (none for the couplings that the largest windows
  do not distinguish from zero), from their moments size by size
  (`study._tensor_statistics`, which `tensor_statistics` now calls too, the realizations
  grouped by volume; the compliances inverted at once with the guard of
  `postprocess.compliance`: a window within a pore has none, NaN). The conductivity tensor
  joins the stiffness in `TENSOR_PROPERTIES`, for both methods (the columns
  `conductivity_tensor.<bc>.C11` ... `C33`, `HomogenizationSpec.tensors`, `rve.properties:
  conductivity_tensor`; `CONDUCTIVITY_NOTATION`: the matrix K_ij in the axes 1 2 3, its
  inverse the resistivity; `tensor_notation`, `TENSOR_WORDS`). The pipeline's window
  method (`rve.method: windows`) takes a homogenization computing `stiffness` and/or
  `conductivity_tensor` by FFT (scalar properties, loadings, paths, fields and finite
  elements refused) and `rve.properties` naming them (by default, with the volume
  fraction): `results.csv` gains per window size the components of the mean tensors
  (`<tensor>.periodic.C11` ...), of their inverses (`S11` ...) and their standard
  deviations (`C11_std` ...), `run.json` the apparent tensors of the realization
  (`apparent`), the report the tables of each tensor counting the windows ("Windows per
  volume" for the series), `rve_<tensor>.csv` and the figures (the definitions in
  `rve_report.json` and the notes of the tables state the windows and their mutual
  energies); `window_statistics` takes squares across a planar image, measured by their
  areas; the RVE size tool of the interface offers the stiffness by the window method
  (FFT) and its tables at the largest windows. There is no `morphogen rve` command:
  `morphogen run` runs these studies from their configuration. Checked: on an 8^3 box of
  two orthotropic phases whose shear moduli differ (G23, G31, G12 = 7, 11, 17 and 40, 25,
  60), the windows tiling the box give its tensor to 1e-14 and its apparent stiffness and
  conductivity to 1e-15 at a tolerance of 1e-12, the tensors symmetric to the bit and
  positive, their statistics equal to the bit to those of `tensor_statistics` of the
  windows taken as realizations; a laminate of these phases normal to z: every window f1
  A1^T C1 A1 + f0 A0^T C0 A0 to 1e-9 (C44 = C~44^2 <1/G23>_w, C55 = C~55^2 <1/G31>_w, C66
  = <G12>_w; the Backus C44 and C55 harmonic means, C66 arithmetic: exchanged slots fail),
  its conductivity likewise; a planar image (the plane stiffness, the plane part of the
  stiffness of the image); pores (NaN compliances of the windows within them, a finite S);
  the pipeline on a 8^3 realization of spheres (results.csv, run.json, report,
  rve_<tensor>.csv; the tensors alone), the refused configurations, the interface's tool;
  an ensemble study of the conductivity tensor by hand; the ensemble tests of the
  stiffness unchanged. Left: the reports print the edges V^(1/3) of cubes, also for the
  areas of a planar image; the windows tile from the origin only (no shifted or
  overlapping windows of the periodic image); on the GPU the windows are integrated on the
  CPU (the unit fields copied back); the scalar moduli (bulk, shear) of the windows are
  not series of their own; `morphogen fit` cannot redo the window analysis from
  `results.csv` (one row per size); no template.
* The mean stiffness C̄_ij and compliance S_ij of each volume of the RVE studies, with the
  standard deviation of each component, their convergence with the size for each boundary
  condition and the RVE size of each component (the RVE reports gave the statistics of
  scalar series only). When the
  realizations have their apparent stiffness (the columns `stiffness.<bc>.C11` ... `C66`
  of `results.csv`), `rve.tensors_from_rows` gives a `TensorStatistics` per boundary
  condition (`rve.tensor_statistics` from tensors and volumes): per volume, the
  realizations, the mean stiffness C̄ and the standard deviation of each of its
  components, the compliance S = C̄⁻¹, the inverse of the mean stiffness so that S C̄ = I,
  not the mean of the compliances of the realizations ⟨C⁻¹⟩ (kept as `mean_compliance`:
  larger, the inverse being operator convex, by a term of the order of the relative
  variance; under SUBC its inverse is Huet's lower bound, ⟨C⟩ under KUBC the upper one),
  and the standard deviation of each component of the compliances C⁻¹ of the realizations;
  each component of C and of C⁻¹ analysed as a series (D²(V) = K V^-γ, γ imposed when the
  study imposes it, the RVE sizes for each (ε_rel, n), no bootstrap), the RVE sizes kept
  for the diagonal components and for the others whose mean the realizations of the
  largest volume distinguish from zero (Student's t test at the level of the study: a
  coupling that vanishes on average, C14 of an orthotropic medium in its axes, has no
  relative precision). Every output states the notation: morphoGen's Voigt order 11, 22,
  33, 23, 31, 12 with engineering shears (`postprocess.NOTATION`, `matrix_entries`). The
  places: `StudyReport.tensors` and `analyze_study(tensors=...)`; `rve_report.json` gains
  `tensors.stiffness.<bc>` (`notation`, the `definitions` of its matrices, per volume
  `stiffness`, `stiffness_std`, `compliance`, `compliance_std` and `mean_compliance`, and
  `components` `C11` ... `S66` with their statistics, fits and RVE sizes); the report
  (Markdown and LaTeX) gains per condition the tables "Mean stiffness Cbar_ij +/- 2 D" and
  "Compliance S_ij = Cbar^-1 +/- 2 D" (every component of the upper triangle against the
  volumes, with the realizations) and "RVE sizes per component" (Zbar, γ, K and the edge
  for each (ε_rel, n)); the figure `rve_stiffness_<bc>.png` (the components of C̄ and S ±
  2 D against the volume, and their relative scatter D / |Zbar| with the fitted laws); the
  table `rve_stiffness.csv` (one row per condition and volume: `n`, `C11` ... `C66`, `S11`
  ... `S66`, `C11_std` ... `S66_std`); `rve.properties` accepts `stiffness` and
  `stiffness.<bc>`, there by default when the homogenization computes the stiffness
  (`HomogenizationSpec.tensors`); `pipeline.build_report` adds the tensors of the rows
  (all of them, or those named), so `morphogen run`, `merge` and `fit` (`--property
  stiffness`) print their tables; the RVE size tool of the interface offers the property
  stiffness (six loadings) and shows, before the tables and figures of the report, those
  of C̄ and S at the largest volume (`elastic_tables`). The LaTeX tables write every
  exponent in math mode (^-1, ^-gamma, ^(1/gamma), not only ^3). Checked on an orthotropic
  material of three distinct shear moduli (G23 = 7, G31 = 11, G12 = 17, distinct Young's
  moduli and Poisson's ratios): synthetic realizations whose variance decreases as 1/V
  (per volume, the mean equals the average of the tensors of the realizations, the
  standard deviations those of the components, S C̄ = I, ⟨C⁻¹⟩ and the standard deviations
  of the compliances; C̄44, C̄55, C̄66 and S44, S55, S66 in their slots within 1 %; γ ≈ 1
  for C11; RVE sizes for the orthotropic components, none for the vanishing couplings; the
  CSV, the JSON, the Markdown and LaTeX tables), an ensemble study on voxels (FFT) of two
  orthotropic phases of contrast 3 (per volume, the mean C of `rve_report.json` and
  `rve_stiffness.csv` is the average of the tensors of `results.csv`, S C̄ = I, C̄44 <
  C̄55 < C̄66 and S44 > S55 > S66 with the ratios of 7, 11 and 17 within 10 %, `morphogen
  fit --property stiffness`), and the interface (S C = I in its tables of the largest
  volume). Left: the window method measures the volume fraction alone, so it has no
  tensors; the discrepancy between boundary conditions stays that of the scalar series,
  not of each component; the conductivity tensors of the studies are not averaged. After
  review: the couplings of exactly symmetric realizations (laminates: 1e-17 of the
  diagonal, of one sign, the round-off of the solver) passed Student's test and got RVE
  sizes up to 1e57; a component under 1e-12 of the largest diagonal one is zero.
* The material of version 1.0 brought up to the state of the package (phase 4 of the
  specification, §3.6 and §10: the parts that are not the author's; publishing stays his,
  `docs/release.md`). `paper/paper.md`, the JOSS draft, rewritten for what morphoGen does
  at 0.6, dated 1 November 2026, its authors and affiliations unchanged, in the six
  sections that JOSS's current guidelines require and 1029 words (they ask for 750 to
  1750): the 38 models (straight, curved and cross-linked fibres, anisotropic versions,
  automata on quasi-periodic tilings), their measures against the closed forms up to the
  percolation thresholds, their anisotropy and critical exponents (the burning of
  Herrmann, Hong & Stanley 1984, finite-size scaling), finite elements and FFT on the CPU
  and the GPU (CuPy), the trilinear voxels of Schneider, Merkert & Kabel (2017) and the
  Newton–Krylov iterations of Gélébart & Mondon-Cancel (2013), von Mises, Hill, crystal
  (Méric & Cailletaud 1991) and porous plasticity in both solvers, MFront behaviours
  through MGIS, Z-mat files and AMITEX_FFTP inputs, the RVE statistics, the macroscopic
  laws identified and the learned behaviours (up to a generalized standard material whose
  free energy is an input-convex network, exported to MFront), about 1200 tests; the
  research impact statement adds the RVE studies of polycrystals (Yang et al. 2019) and of
  nanolayered films (Bironeau et al. 2016). `paper/paper.bib`: 11 references added (32 in
  all), each from the theory notes' list (`docs/theory.md` and §14 of the specification)
  and its DOI checked on Crossref (the PMLR and LearningSys papers by their URL; the books
  and the Journal de Mécanique have none); the 12 earlier ones that the list gives agree
  with it, the 9 others (the software: DREAM.3D, PoreSpy, PuMA, microgen, Z-set, Numba,
  NumPy, SciPy; and Moulinec & Suquet 1998) stay as checked against Crossref and Zenodo on
  2026-10-28, and Moulinec & Suquet (1998), cited by the theory notes without an entry,
  now has its entry in their references. The API reference: its pages are written at each
  build by `docs/conf.py`, one per module of the public interface, 172 pages — the modules
  of `tests/public_api.txt`, the newer `learning` (its eight modules), `automata.tilings`,
  `analysis.percolation`, `homogenization.networks` and `models.affine` included, and the
  two modules that define no public name (`homogenization.tensors`,
  `learning.logarithmic`) — so that none was missing. `packaging/conda-forge/recipe.yaml`:
  the version of `_version.py` (0.6.0 rather than 1.0.0), the extras of `pyproject.toml`
  as constraints (imageio-ffmpeg >=0.4 for `video` and cupy >=14 for `gpu` added; for
  `mesh`, the package mmg of conda-forge as before, pymmg not being there), the import of
  the 15 packages of the public interface (every public module imports without the
  optional dependencies) and the smoke test of `tools/ci_local.sh` (`morphogen template
  boolean_spheres`, `morphogen run`, its `results.csv`: 38 s with the optional
  dependencies blocked, the kernels compiled), its description brought up to date.
  `CITATION.cff`: its abstract brought up to date (fibre networks, quasi-periodic tilings,
  percolation, the GPU, the behaviours, the macroscopic laws identified or learned, Neper
  and DAMASK) and the keywords of `pyproject.toml`; version 0.6.0, without `date-released`
  before a release. `docs/release.md`: the step Branch done on 2026-11-01 (pull request #2
  merged the development line up to 91d8987 into `main`, merge commit 9a6b660), the
  author's steps unchanged. Checked: `tests/test_release.py` (the dependencies,
  constraints, entry point, summary, licence and imports of the recipe against
  `pyproject.toml` and `morphogen._public`, the version of `CITATION.cff`, every citation
  of the paper in its bibliography and every entry cited, their braces, JOSS's six
  sections and length), the documentation built with warnings as errors. Left, the author's
  (`docs/release.md`): the rights and the licence with the technology-transfer office, the
  ORCID identifiers, the repository made public; then version 1.0 (`_version.py`, the
  change log, `version` and `date-released` of `CITATION.cff`, the classifiers), the tag,
  Zenodo's DOI, PyPI, the pull request to conda-forge with the version and sha256 of the
  source archive, the documentation site; for the paper, the research impact statement,
  the acknowledgements and the AI usage disclosure completed by the author, and its
  submission six months after the repository is public.
* The stiffness C_ij and the compliance S_ij = C⁻¹ in every output of elastic properties,
  beside the engineering constants (until now most outputs gave C, or only the constants).
  `homogenization.postprocess` gains `compliance(stiffness)` (6 x 6, or 3 x 3 for plane
  problems, a plain error otherwise), `notation(stiffness)` and the constants `NOTATION`
  and `PLANE_NOTATION` that the outputs state (morphoGen's Voigt order 11 22 33 23 31 12
  with engineering shears, 11 22 12 in the plane; TFEL and Abaqus ordering their shears 12
  13 23; S in the same notation), `VOIGT_ORDER`, `PLANE_ORDER` and `matrix_entries(matrix,
  prefix)` (the upper triangle as `C11` ... `C66`). No computation changes: S is the
  inverse of the C already computed. The places: `homogenize` adds `compliance` to its
  `stiffness.<bc>` results; the results of runs and RVE studies (`results.csv`, the
  records, so the database) add the columns `stiffness.<bc>.S11` ... `S66` beside `C11`
  ... `C66`, and `run.json` states their notation (`stiffness_notation`); the Ashby charts
  label them ("Compliance S44 (periodic; Voigt 11 22 33 23 31 12, engineering shears)", a
  stress `S11` of a loading left alone); `morphogen homogenize` prints C, S and the
  engineering constants of a stiffness (its JSON: `tensor`, `compliance`, `notation`);
  `morphogen frame-stiffness` prints S beside C (JSON: `compliance`, `notation`);
  `morphogen amitex read` prints both, adds `compliance` and `notation` to its JSON and
  `S11` ... `S66` to its CSV row; `morphogen optimize` prints C and S of the thresholded
  design and `optimization.json` gains `compliance`, `binary_compliance` and `notation`
  for the elastic designs (None for a singular or incomplete tensor: that of the densities
  of a planar cell has its in-plane entries only); the summaries of the network models
  with `frame_stiffness: true` add `frame_C11` ... and `frame_S11` ... (8 digits). In the
  interface, a shared `elastic_tables` writes the tables of C and S (with their .csv, the
  notation and units in the titles: those of the moduli and their inverse, per unit area
  for plane frames): Linear tensors, Stiffness of the frame of beams (and `compliance`,
  `notation` in `frame_stiffness.json`), the AMITEX results (apparent stiffness and
  compliance), Identification of a behaviour and Macroscopic law of a morphology (the
  elasticity of the law identified), Learning (the tangent of the learned law at the
  natural state, against the elasticity of the reference at a material point), Generate
  (the frame of the network models), the topology optimization (the thresholded design);
  the layers of the hygromorphs add their plane-stress compliance S_xx ... beside Q (and
  the shear couplings Q_xs, Q_ys of homogenized layers). Theory §27 writes the orthotropic
  S, the order of TFEL and Abaqus and the Mandel forms W C W and W⁻¹ S W⁻¹. Checked on an
  orthotropic material of three distinct shear moduli (G23 = 7, G31 = 11, G12 = 17,
  distinct Young's moduli and Poisson's ratios, so that an exchange of the shear
  components shows): S C = I, S11 = 1/E1, S12 = −ν12/E1 = −ν21/E2, S44 = 1/G23, S55 =
  1/G31, S66 = 1/G12 for `compliance`, the apparent stiffness of a homogeneous image
  (FFT), the columns of a run and the notation of its run.json, the JSON and text of
  `morphogen homogenize`; S C = I in the tables of Linear tensors (a copper polycrystal),
  of the frames (3D and plane), of Generate, of AMITEX, of the learned tangent (within 5 %
  of the reference), of the hygromorph layers, of the topology optimization of the
  interface (against their .csv), in `frame_stiffness.json`, the JSON of `frame-stiffness`
  and `optimization.json`, the CSV row of `amitex read`, the summaries of the open foams.
  Left: the main Simulate of the interface computes the bulk or the shear modulus of one
  loading, which has no tensor to invert (the six loadings are those of Linear tensors);
  the RVE reports give the statistics of scalar series, so S_ij as a series only when
  asked for (`stiffness.periodic.S44`), not the mean C and S of each volume; the material
  inputs written for other codes (MFront, Z-mat, Abaqus, AMITEX) are unchanged. After
  review: a stiffness singular to round-off (layers of void, nothing stiff across them)
  printed the inverse of its round-off as its compliance (S33 -2.9e15, and S11 shown as
  0); when its smallest eigenvalue is within 10 machine epsilons of its largest, or
  negative, the compliance is NaN (a stiffness only badly scaled, as a design stiff
  along its fibres whose plane is as stiff as its void, keeps it), an exactly singular
  stiffness raising as before.
* The workflow's macOS job, run again on 1 November, found two tests that held on Linux
  and Windows only. macOS has no `SC_AVPHYS_PAGES`, so that the interface knew no free
  memory and let any computation through (`check_memory`): it reads the free,
  speculative and inactive pages of `vm_stat` (psutil's measure). The test of the
  networks of `ln C` compared to the bit `S = Jᵀ T` computed by the kernel of the step
  and by NumPy's `einsum` in `predict`, which differ by round-off with Accelerate on
  arm64; it compares them to 10⁻¹² of the stress, as the recurrent network's steps.
* The critical exponents of the effective properties at the percolation threshold (the
  next step of the percolation tools: how the conductivity and the elastic moduli
  of the percolating phase vanish, K ∝ (p − p_c)^t and E ∝ (p − p_c)^f, measured with
  morphoGen's FFT solvers). `analysis.percolation.percolation_exponent(generate, sizes,
  threshold)` (and `PercolationExponent`) takes the finite-size scaling at the threshold,
  ⟨K⟩ ∝ L^(−t/ν), the method of the literature (Derrida & Vannimenus 1982; Gingold & Lobb
  1990), rather than a sweep, whose window between the size of the boxes and the end of
  the critical region is narrow on a laptop's images: for each of at least 3 box sizes (a
  sized family, as for `percolation_scaling`), realizations seeded by size and number are
  drawn at the threshold, their phase conducting (K = 1) or bearing (E = 1, ν = 0.3) and
  the other phases at `contrast` times its properties (10⁻⁸ by default, 0: void), and the
  periodic FFT solver (`device="auto"`: the GPU when there is one) gives the conductivity
  along the axis or the modulus C_xxxx of a uniaxial strain along it (one solution; all
  the moduli share f); ln⟨K⟩ is fitted against ln L weighted by the inverse variances of
  the logarithms, t = ν × slope (ν = 0.8765 in 3D, 4/3 in 2D), its standard error from the
  weights scaled by the root of the reduced chi-square when it exceeds 1 (means whose
  realizations are all alike, a deterministic family or boxes where none carries, fitted
  unweighted); `threshold_stderr` computes the same images at p_c ± σ (common random
  numbers) and `threshold_error` is half the change of t between them; `fractions` add a
  sweep in ln(p − p_c) on every size for comparison. Checked first: the trilinear voxels
  join through the nodes they share, so that on 8³ voxels a staircase touching at corners
  conducts (2.6·10⁻³ against 1/64 for a channel; 7.8·10⁻³ through edges) and bears a
  uniaxial strain (7.2·10⁻⁴; 3.9·10⁻³): the conductivity percolates through 26 neighbours
  (8 in 2D), whose threshold is the one to give, and the elastic moduli vanish where the
  voxels joined by ball joints (corners) and hinges (edges) become rigid, between the
  thresholds through 26 and 6 neighbours, which no connectivity gives; the contrast 10⁻⁸
  leaves the iterations those of voids (308 to 400 on 64³ voxels of spheres, 150 to 200 on
  256² pixels of discs) and shifts the means at p_c by about δ (L/ℓ)^((t+s)/ν), below 10⁻³
  up to 40 grains across in 3D and 300 in 2D. Thresholds of the voxel images (the fire, 49
  realizations, tolerance 0.002, extrapolated): periodic Boolean discs of 4 pixels of
  radius (64² to 512²) 0.600 ± 0.005 through 8 neighbours and 0.633 ± 0.006 through 4
  (continuum 0.6763), the matrix around such discs 0.367 ± 0.006 through 8 (continuum
  0.3237), spheres of 3 voxels of radius (24³ to 96³) 0.186 ± 0.004 through 26 and 0.245 ±
  0.004 through 6 (continuum 0.2895): the corner contacts lower them well below the
  continuum. Measured on the GPU (statistical error, then the threshold's): the discs (the
  inverted Swiss cheese in 2D; 48 realizations) t = 1.13 ± 0.16 ± 0.21, and ± 0.04 at the
  mean of seven crossings of squares, unbiased in 2D (0.6001 ± 0.0011), against 1.30;
  their matrix (the 2D Swiss cheese) 1.80 ± 0.15 ± 0.24, and 1.58 ± 0.14 ± 0.13 at the
  mean of its crossings (0.3715 ± 0.0028), against 1.30 (unchanged in the 2D continuum;
  Halperin, Feng & Sen 1985); spheres (the inverted Swiss cheese in 3D; 32 realizations)
  2.37 ± 0.36 ± 0.32 against 2.0: within 0.6 to 1.8 combined errors, measured to 0.2 to
  0.5 only, the threshold weighing most (dt/dp_c ≈ −40 in 2D, −80 in 3D: t within 0.05
  wants p_c within 0.001 and 0.0006), then the scatter of the means at p_c (relative
  standard errors 14 to 24 % in 2D, 35 to 54 % in 3D); the sweep of the discs over 0.62 to
  0.75 rises with the box, 0.79 ± 0.09 on 64² to 1.26 ± 0.04 ± 0.10 on 512² pixels. The
  runs took 7, 5 and 11.5 min. `morphogen percolation CONFIG --sizes ... --exponent
  thermal|elastic` (the threshold extrapolated first through 26 neighbours wrapping around
  the box, refused when it leaves the bounds within its error, or `--threshold` and
  `--threshold-stderr`; `--fractions`, `--contrast`, `--device`) and the field "critical
  exponent" of the tool "Percolation threshold" (the means of the boxes against their size
  on logarithmic axes, with the power law); theory §55. Checked:
  `tests/test_percolation.py` (a channel conducting exactly as L⁻¹ in 2D and L⁻² in 3D and
  bearing as L⁻¹, layers swept with exponent 1, sites of the square lattice through 8
  neighbours within the error of 1.30, the errors, the command), `tests/test_gui_tools.py`
  (the tool, a threshold too uncertain refused). Left: the elastic exponent f, whose
  threshold (the rigidity of voxels joined at their nodes, between those through 8 and 4
  neighbours) is to be found from the moduli themselves, and whose first run (discs, 64²
  to 256², 24 realizations) did not end in 10 minutes; the 3D Swiss cheese, whose matrix
  percolates at 3.4 % in the continuum with necks no voxel resolves; thresholds to 0.001
  and a few hundred realizations per size to bring t within 0.05; percolation_scaling with
  the amplitude fixed at 0 for squares in 2D.
* The learned laws as MFront behaviours (the next step of the learning module): a
  law learned from the responses of a microstructure now goes to every solver that MFront
  serves (Abaqus, Cast3M, Code_Aster, AMITEX_FFTP), as morphoGen's own behaviours do.
  `morphogen.io.mfront_source`, `morphogen mfront` and the AMITEX and Abaqus exports take
  a material whose `learned` names a trained model (new module
  `morphogen.learning.mfront`: `learned_source`, `learned_model`,
  `learned_strain_measure`, `MEASURES`) and write a behaviour of MFront's `Default` DSL
  whose constants are the trained parameters (their shortest decimal forms, which read
  back to the bit) and whose integrator is morphoGen's C++ port of the model
  (`learning/learned.cpp`, a section per kind, line by line from the NumPy and Numba
  functions: the perceptron and its Jacobian, the input-convex potential and its Hessian,
  the gated recurrent unit and its tangent at the state of the start of the step, the
  Newton solve of the incremental potential of `Standard` with Armijo's backtracking and
  its Schur complement). Small strains: the law of the strain. Finite strains:
  `@StrainMeasure Hencky` for the laws of the logarithmic strain (MFront's strain is `½ ln
  C` and its stress the `T` conjugate to it, the model's own `T`, and MFront turns `∂T/∂H`
  to the operator of the interface), `@StrainMeasure GreenLagrange` for the laws of `E`.
  The state of the recurrent network and the internal variables of the standard model are
  auxiliary state variables (`LearnedState`, zero at the natural state; the strain at the
  start of a step is MFront's `eto`); the thermal strain `α (T - 293.15)` of the material
  (a tensor in its axes, shears too) is taken out of the strain the model sees, out of `½
  ln C` in finite strains (the split of the Hencky points); a law in the axes of its
  material is an orthotropic behaviour (`@OrthotropicBehaviour`, and Abaqus's axes in its
  first six state variables in finite strains), a law in the axes of the sample an
  isotropic one; TFEL's components (11, 22, 33, then √2 times 12, 13, 23) in the
  behaviour, morphoGen's Voigt order (11, 22, 33, 23, 31, 12, engineering shears) in the
  model, converted in the integrator as for the porous kernels; the tangents of the
  potential and of the standard model declared symmetric. Refused: an expanding law of `E`
  in finite strains (its Hencky split `E_m = ½ (exp(2 H_m) - I)` is not written) and a law
  in the axes of the sample with an anisotropic expansion. `mfront_state_size`,
  `mfront_symmetry` and `mfront_strain_measure` answer for the learned laws;
  `test_learned_laws_elsewhere` now expects a source where it expected a refusal. Measured
  (`tests/test_learning_mfront.py`, in WSL with MFront 5.1 and MGIS 3.1; the four models
  as small networks, their parameters moved off their start, set up on an orthotropic
  crystal of shear moduli 60, 40 and 20 GPa): compiled and integrated by MGIS, the laws
  follow morphoGen's integration of the same laws at a material point along paths of the
  six components to 4.5e-17 to 5.6e-16 of the stress in small strains, 4.8e-15 to 1.9e-14
  through MFront's Hencky strain and 9e-16 to 6.2e-15 through its Green-Lagrange strain,
  heated (by 60 K and 600 K) and turned by (30°, 50°, 70°); in tensions with free
  components the potential and the standard model take the same Newton iterations
  (stresses within 3e-16 to 2.8e-13); the tangents are the central differences of the
  stresses in MGIS's components to 2.6e-12 to 1.3e-10, heated and from states of the
  models off zero; the linear law of the crystal answers TFEL's slots 12, 13 and 23 with 2
  G12, 2 G13 and 2 G23 (40, 80 and 120 GPa); called as AMITEX (umat, each zone of a
  polycrystal) and Abaqus (abaqus, the *ORIENTATION of the sections, in finite strains the
  axes in the state variables) call them, the learned phases of the exports give
  morphoGen's curves to 2.6e-16 to 1.2e-15 and 4e-13 in small strains, 8.8e-15 and 9e-13
  in finite strains (Abaqus's axes written with twelve digits). Found on the way: the
  tangents of these moved perceptrons and recurrent networks are 7 to 33 % off symmetric,
  so that Newton's method with the symmetric part that MGIS through morphoGen gives its
  solvers does not solve their free components at a material point, where morphoGen's
  learned points keep the exact tangent. Left for later: the Hencky split of the expanding
  laws of `E` in their sources (the divided differences of the exponential and their
  second derivatives), and the nonsymmetric tangents of the MFront laws in morphoGen's own
  solvers. After review: in Abaqus inputs of small strains, a learned law in the axes
  of the sample (`oriented: false`) took the *ORIENTATION of its grains' sections, so
  that Abaqus turned the strains it runs on (0.35 of the stress on a grain turned by
  (30°, 40°, 50°)); its phases take none, as in finite strains (2×10⁻¹⁶ of morphoGen's).
* The gallery shows the thin networks as the surface of their phase 1 in 3D, which the
  faces of their boxes barely cut: the
  Delaunay fibre network, the trabecular bone (an open foam), the open-cell foam of
  distance fields and the strut lattices (octet, graded, random and hierarchical) join the
  vascular tree and the aggregates in `SURFACE` of `docs/gallery/make_gallery.py`; seven
  images rewritten, 41 to 67 kB each, the gallery about 1 MB in all.
* A gallery of examples and a tutorial in the documentation.
  `docs/gallery.md` shows the 64 templates in 14 families (point processes and
  Boolean models; dead leaves, Gaussian fields and reconstructions; fibres and layers;
  tessellations; foams and lattices; TPMS; automata and tilings; growth; bio-inspired;
  linear homogenization; nonlinear behaviours; RVE studies; topology optimization;
  couplings and the whole chain), a card each in a grid (a `container` per card, styled by
  `docs/_static/gallery.css`; a card whose configuration is unfolded takes the whole
  width): the image of the template, its name and a line of description, the optional
  parts it needs (MMG, scikit-image, VTK and ffmpeg, read from its sections), the commands
  that reproduce it (`morphogen template NAME -o NAME.yaml`, then `morphogen run
  NAME.yaml`, with `--seed 1` for the four random templates without a seed, `bouligand`,
  `homogenization_mixed_loading`, `osteons` and `wood`, or `morphogen optimize`) and its
  configuration, folded (a `literalinclude` of the template itself, read at each build).
  `docs/gallery/make_gallery.py` writes the page from its table of families and the
  images, `docs/_static/gallery/<template>.png`: the first realization that `morphogen
  run` makes (the realization 0 of the seed of the template, at the largest size of an
  ensemble RVE study, the size of a window study; a template without a seed whose model is
  random seen with seed 1, the others checked deterministic) or the design of an
  optimization template; three faces of its box in axonometry (top, front and right,
  shaded, their edges drawn), each the section of the microstructure computed in its plane
  by `morphogen.viz.sections`, as the interface draws them (anti-aliased interfaces of the
  level set, the colours of the phases, a colour per grain of a polycrystal, per cell of a
  mosaic of one phase); the x-y section of a planar structure; and for the vascular tree
  and the aggregates, which the faces barely cut, the surface of their phase 1 in 3D
  (`viz.video.turntable_frames`, VTK off screen, z up). Supersampled twice, quantized to
  128 colours with their transparency by Pillow: 560 × 476 pixels, 2.7 to 57 kB, 0.90 MB
  in all. `docs/tutorial.md` follows SiC particles (a Boolean model of spheres at 20 %) in
  an aluminium alloy through a whole study: the installation; the morphology from the
  command line (a configuration from the template, `morphogen run`, `morphogen view`),
  from Python (the same realization, `derive_seed(1, 0)`) and in the interface (Generate,
  the workspace and the tabs that take it); its volume fraction, specific surface and
  covariance against the closed forms, and its percolation threshold (0.27 on 64³ voxels
  of a box of 16 radii, 0.2895 in the continuum); its apparent moduli by FFT (38.1 GPa in
  shear, within the Hashin–Shtrikman bounds 36.7 and 45.9 GPa) and the RVE of its shear
  modulus (16 realizations of three volumes, `K V^-1` fitted, the volumes of 1 and 5 %
  with 1 and 10 realizations); a tension to 1 % with a von Mises matrix (Voce hardening),
  twice the stress of the matrix alone; the MFront source of the matrix, an Abaqus input
  with its materials and the inputs of AMITEX_FFTP; a learned generalized standard
  material trained on the homogenized responses of its voxels along 32 random strain
  paths, with its gap to the voxels along the tension. README.md (the introduction, the
  quick start and the documentation) and `docs/index.md` (the toctree Guide: the tutorial
  after the installation, the gallery after the guide) link both; `conf.py` publishes the
  style sheet alone (`html_static_path` of the file: the images are copied once, with the
  page). Checked: `tests/test_gallery.py` (every template has its card and its image,
  every card its template, each card its own image, file and commands; the images are PNG
  files below 100 kB, 5 MB in all; the page is the one the script writes),
  `tests/test_tutorial.py` (the Python snippets run in one session at smaller sizes, each
  replaced text found once in the tutorial, with the threshold within its bracket, the
  shear modulus between Reuss and Voigt and the files written; the configuration runs with
  `morphogen run`; each `morphogen` command of the tutorial parses), the documentation
  built with warnings as errors. Timing: the images in about five minutes (the
  optimizations 33 and 43 s, the window study of the automaton 27 s, the reconstruction 21
  s, fibres packed by forces 12 to 19 s), the snippets of the tutorial in a few seconds
  each once the kernels are compiled (the first run of the nonlinear solvers compiles
  theirs, about a minute), the test of the snippets 2 s (slow tier). Left: the fields of
  the homogenization templates (a von Mises stress rather than the morphology) and the
  curves of the loading paths in their cards; a page per template; the tutorial as a
  notebook. After review: `nacre`, random (its stack of layers at a random height),
  passed the check of determinism on 24³ voxels, so that its card's command did not make
  its image; the check is made on 64³ voxels, which finds the five random templates and
  no other. In the tutorial: the buttons and tabs as the interface names them (Generate,
  Nonlinear mechanics), the AMITEX results read by the name of their export
  (`--name composite`: the command line writes a second export into the same folder),
  and the ratio of the composite's stress to the matrix's resolved: 2.05 on the 16³
  voxels of the snippet, 1.71, 1.57 and 1.42 on 24³, 32³ and 48³.
* Ban et al.'s replicas and a drawing of the frame in the interface and on the command
  line (the two things the tool Stiffness of the frame of beams still
  lacked; left by the previous step). `networks.softening` takes a `callback(done, total)`
  followed replica by replica and checks its law and coefficients of variation before
  solving; `Softening` keeps its study (`mode`, `law`, `replicas`, `axis`) and gives
  `scatter()` (σ_E²/E₀²), `predicted_scatter()` (v CV²) and `rows()`, a row per CV: the
  plain mean of E/E₀ and its standard error σ/√n, the mean corrected by the control
  variate and its standard error, 1 − s CV², σ_E/E₀, the scatter with its standard error
  for normal draws, σ²√(2/(n − 1)), and v CV². `networks.beam_forces(frame, axis=0, *,
  strain=None)` solves a uniaxial stress of 1 along an axis (or a macroscopic strain) and
  gives the strain, the axial force N = EA δ/l of each beam (positive in tension; δ the
  extension of its flexible part, which rigid ends do not change) and its energy per unit
  volume in stretching, bending, torsion and shear (`PARTS`), which sum to E·CE/2 and
  split as `FrameStiffness.split`. `viz.figures.frame_figure(frame, values=None, *, scale,
  label, title)` draws the beams in the cell (each cut where it crosses a face, its pieces
  brought back by the periods), in an orthographic 3D view or in the plane of a planar
  network, coloured by a value per beam on a linear, symmetric (about zero, up to the 99th
  percentile of the magnitudes: tension red, compression blue) or log scale (three decades
  below the largest), the strongest widest and on top, the beams rasterized in the vector
  files beyond `VECTOR_BEAMS` (2000) pieces; `viz.figures.softening_figure(studies)` draws
  the means of the replicas against CV with their error bars (corrected and plain) under 1
  − s CV², and their scatter against v CV² with the diagonal. `morphogen frame-stiffness`
  takes `--replicas N` (per CV, for each mode of `--softening`), `--cv CV ...` (default
  0.1 0.2 0.3), `--distribution normal|lognormal|bimodal`, `--replica-seed S` (default 1),
  `--table CSV` and `--figure FILE`, prints the table of the replicas and keeps it in its
  JSON output under `replicas`. In the interface, the tool takes a box Drawing of the
  frame (the colour of the beams: their axial force, their stretching or bending energy,
  or the beams alone; the axis of the uniaxial stress) and, in its box Softening of
  non-identical beams, the replicas per CV (0, the default: the coefficients alone), the
  coefficients of variation, the distribution and the seed; it draws the frame (the figure
  `frame`), gives the share of the beams storing half the energy, the largest tension and
  compression and, for the replicas, run in the thread of the tools with their progress
  replica by replica, their table (`replicas.csv`), their figure and their rows in
  `frame_stiffness.json`. Checked, to round-off (4·10⁻¹⁶ to 4·10⁻¹⁴): the forces of
  `beam_forces` against the dense `axial_forces` of a kagome and an octet lattice under
  random strains, the energies of the beams summing to 1/2E and split as `split`, the
  stretching energy N²l/2EA of each beam with rigid ends and Timoshenko beams, the virial
  of a truss giving the stress back; the pieces of the drawn beams as long as the beams
  and inside the cell; the command and the tool against `softening` of the same frame and
  seed (10⁻⁹), their tables, figures and files, the progress, the errors (replicas without
  a mode or fewer than two, a table without replicas, a bimodal CV of 1 or more, a plane
  network drawn along z). Measured: half the energy of a uniaxial stress in 14.7 % of the
  beams of an open foam of 200 cells (bending-dominated), 9.6 % of Ban's Delaunay network
  and 11.7 % of a planar one of 2000 points; a replica costs one factorization, 0.05 s for
  1154 fibres, 1.5 s for the 2690 beams of the foam (1.1·10⁷ nonzeros in SuperLU's factors
  for 8067 dofs), and its drawing about 2.5 s (its forces, then its PNG and SVG). Left: a
  faster factorization of large frames (a fill-reducing ordering such as nested
  dissection, or the factors of the identical beams as a preconditioner of the replicas);
  the drawing on the command line; the scaled frames of Ban's cases (`scaled_frame`) in
  the interface.
* One thermal convention for the learned laws and the Hencky points in finite strains
  (after the finding of the logarithmic laws that heated, a learned law
  gives `J_th = exp(3α ΔT)` times the stress of the Hencky points, 1.1 % at 300 K: the
  learned laws aligned on the Hencky points). The Hencky points take the thermal strain out of
  the logarithmic strain, `T = ℂ : (½ ln C - α ΔT)` (the expansion in the axes of the
  sample, turned with the grains), their stress conjugate to the total `½ ln C` and their
  free energy per unit reference volume, as MFront's logarithmic strain framework; the
  learned laws took the multiplicative split `F = F_m F_th`, `F_th = exp(α ΔT)`, with the
  free energy per unit volume of the heated configuration (`S = J_th A S_m A`). Now every
  learned model (perceptron, potential, recurrent network, standard model), of `ln C` or
  of `E`, splits as the Hencky points do: it sees the Green-Lagrange strain of the
  mechanical logarithmic strain, `E_m = ½ (exp(ln C - 2 α ΔT) - I)` (a law of `ln C` sees
  `H_m = ½ ln C - α ΔT` itself), and its stress gives `T = J_m⁻ᵀ S_m` and `∂T/∂H = J_m⁻ᵀ
  (∂S_m/∂E_m - T : ∂²H/∂E²|E_m) J_m⁻¹`, then `S = Jᵀ T` and `∂S/∂E = Jᵀ (∂T/∂H) J + T :
  ∂²H/∂E²` by the kernels of `learning/logarithmic.py` (`thermal_split` and
  `thermal_conjugate` replace `thermal_stretch` in `morphogen.learning.behaviour`, at a
  material point, by finite elements and on voxels alike). In `E` this is the consistent
  equivalent of the Hencky points' split, the only strain whose logarithmic strain is
  theirs, so that a law of `E` of Hencky's elasticity gives their stresses: the
  multiplicative split without `J_th` for an isotropic expansion, off the Hencky points
  for an anisotropic one (2.8e-5 of the stress along a heated gradient of a turned
  crystal, 8e-16 with `E_m`). The tangent is symmetrized when the model's is symmetric to
  the bit, a heated point fails where `C` is not positive definite (as a Hencky point
  does), and small strains (`σ = law(ε - α ΔT)`) are unchanged; §57 of the theory states
  both conventions and the derivation, §33 that of the Hencky points. Measured (`α =
  1.2e-5 /K`, or the expansions 1, 2 and 3e-5 /K of an orthotropic crystal turned by (30°,
  50°, 70°); 300 K): a law that is Hencky's elasticity (the potential of `ln C` started on
  Hencky's data, its network silenced, and a law of `E` computing `S = Jᵀ ℂ H`) gives the
  stresses of the Hencky points under constraint, along a heated gradient of the six
  components and along a heated tension to 1.3e-16 to 8.5e-15 (before: 1.1 % isotropic,
  1.8 % for the turned crystal) and their strains to 3e-15; the linear law of `ln C` of
  the thermal test, whose `J_th` factor it encoded, their stresses to 1.7e-14; heated
  free, no stress (1e-15 of `E α ΔT`) and the logarithmic strain the thermal strain to
  1.6e-15; homogeneous cells on voxels (the crystal turned by its grain) and by finite
  elements give the material point to 2e-10 and 6e-16; every model in both strains, its
  parameters moved, turned with the three expansions and heated, has `∂P/∂F` the central
  differences of `P` to 3.4e-9 to 7.1e-9 and, the potential and the standard model, no
  stress at the free stretch `exp(ΔT gᵀ α g)` (4e-14 of the constrained stress); unheated
  or without an expansion in finite strains, and in small strains with or without one, the
  stresses, tangents and states of the four models in both strains are those of 8f5d53c to
  the bit. The split costs 10 µs a point (a step of 100 000 points: 32 against 23 µs a
  point for a potential 16-16 of `ln C`, 27 against 17 µs for a recurrent network of 16
  states). Left for later: a compiled kernel of the split (NumPy around four passes of the
  kernels of `ln C` now), and the crystals of finite strains, which still take their
  thermal strain out of the elastic Green-Lagrange strain. Where `C` is not positive
  definite, the heated standard model of `E` divided by zero instead of failing
  (`ZeroDivisionError` at a material point compressed to `F11 = 0`, where the Hencky points
  divide the step): the models are no longer called at points whose strain is not finite,
  their stresses NaN (failed points), the other points unchanged.
* The other learned laws in the logarithmic strain (the second technical follow-up of the
  learning module, after the finding that at large strains the measures, not the
  models, limit the learned laws): `FeedForward`, `Potential` and `Recurrent` take the
  option `strain` of `Standard` in finite strains, `logarithmic` by default for the new
  models (the space of morphoGen's Hencky points, where Hencky's elasticity is linear and
  isotropic plasticity is that of small strains, the default of `Standard`, both strains
  agreeing at small strains, and what the measures below decide) or `green-lagrange`. In
  `ln C` they see `H = ½ ln C` (the recurrent network `H` at the end of each step and its
  increment) and give the stress `T` conjugate to it, whence `S = Jᵀ T` and `∂S/∂E = Jᵀ
  (∂T/∂H) J + T : ∂²H/∂E²` by the kernels of `Standard`, now shared in
  `morphogen.learning.logarithmic`; the potential `ψ(H)` gives `S = ∂ψ(H(E))/∂E`, its
  tangent symmetric to the bit, and represents Hencky's elasticity by its quadratic term;
  the training carries the gaps of `S` to `T` through `J` (in the compiled backpropagation
  through time of the recurrent network), the linear terms starting at the fit of `T = J⁻ᵀ
  S` to `H`. `STRAINS` moves to `morphogen.learning.networks` and the property
  `logarithmic` to `Model` (both still reachable from `morphogen.learning.standard` and
  `Standard`); `strain` is accepted for every kind in the `model` section of `morphogen
  learn`, and the field *strain of the law* of the tool *Learn a behaviour* is shown for
  every model in finite strains. The files written before the option load as the laws of
  `E` they were, and `strain="green-lagrange"` is the code of before to the bit: 118
  arrays (the losses, parameters, stresses, tangents and states of the steps and the
  material points of the five models in small and finite strains) equal to those of
  221737b, whose files load and run to the bit. Measured, whatever the parameters along
  gradients to 15 %: the training gradients against central differences to 1.7e-9, 2.4e-7
  and 1.5e-9 (perceptron, potential, recurrent network), the tangents `∂S/∂E` to 1.6e-9,
  1.6e-9 and 1.7e-9 of the derivatives of the stresses, `S = ∂ψ/∂E` to 1.3e-9, `∂P/∂F` at
  the points to 4.6e-9, a superposed rotation turning `P` and `∂P/∂F` to 3e-15 along two
  steps, the small strain models recovered to the first order of the gradient (3.5 %, 4.0
  % and 2.8 % at scale 1, a hundred times less at 1e-2), a turned potential of `ln C`
  equivariant to 4e-15, at a material point, on voxels and by finite elements to 9e-8 and
  5e-9, free heating `F = exp(α ΔT) I`. Von Mises plasticity at large strains (the Hencky
  points, `H = 2 GPa`, the walks of the logarithmic strains to 10 %, Green-Lagrange
  strains to 22 %, 300 epochs, rotated): the logarithmic strain fixes the recurrent
  network, 11.2 % on the validation paths (11.7 % on its training paths) against 64 % (63
  %) in `E`, as in small strains (10.9 %); a second seed 10.7 % (63 % in `E`), 1000 epochs
  8.3 % (`E` stays at 63 %); along the unseen cycle ±10 %, shear cycle in 23 and tension
  then shear 11 %, 10 % and 12 % (the standard model of `ln C` 19 %, 11 % and 29 %; the
  network of `E` finds no lateral strains along the first two, 34 % along the third), its
  lateral strains 2.2 %, 6.8 % and 1.2 % (the standard model 0.4 to 0.8 %), on voxels as
  at a material point (5e-7), trained in 6 s against 71 s for the standard model. Hencky's
  elasticity: the linear start of a potential of `ln C` is Hencky's law to 4e-15; trained,
  0.017 % on the validation paths (4.2 % in `E`) and 0.022 % with strains to 25 % (16 % in
  `E`; the perceptron 0.025 % against 30 %), a uniaxial tension to 10 % at 0.022 % (4.5 %
  in `E`). Found on the way: heated under constraint, the linear law of `ln C` (Hencky's
  own) gives `J_th = exp(3α ΔT)` times the stresses of the Hencky points (to 2e-14; 1.1 %
  at `α ΔT = 3.6e-3`): the free energy of a learned law is per unit volume of the heated
  configuration (`S = J_th A S_m A`), that of the Hencky points per unit reference volume;
  and a small recurrent network (6 states, 20 epochs on 8 paths) does not converge on
  voxels in either strain, its tangent symmetrized there (trained on the walks above, it
  does). Left for later: the tangents of the recurrent laws, not symmetric, in volume
  elements (a nonsymmetric Krylov solver), one convention of the heated free energy for
  the learned laws and the Hencky points, a dissipation potential of the state and deeper
  input-convex free energies.
* The stiffness of the frames of beams of fibre networks and open foams in the interface
  and on the command line (until now from the API only).
  `networks.network_frame(micro, *, young, poisson, shear_coefficient, rigid_joints)`
  makes the frame of a realization of `delaunay_network`, `open_foam` or `random_lattice`:
  the framework of its metadata, the radius of its fibres from its summary (`RADII`:
  `radius`, `strut_radius`; solved for a volume fraction) or from its model, a plane
  `Frame` when the framework lies in a plane (the planar Delaunay networks), a plain
  refusal for a microstructure without a framework; `FrameStiffness.by_axis()` gives,
  along each axis, the Young's modulus, the Poisson's ratio (the contraction along the
  next axis), the ratio to the affine estimate, the energy split and the non-affinity of a
  uniaxial stress, whose means `summary` now takes. `morphogen frame-stiffness FILE` takes
  a configuration (realized as the realization `--realization R` of its batch run,
  `--seed`) or the geometry (.json) of a realization, `--beams euler-bernoulli|timoshenko`
  (Cowper's coefficient, or `--shear-coefficient K`), `--rigid-joints C` (radii),
  `--young`, `--poisson`, `--softening axial|bending|modulus|area ...` along `--axis` (Ban
  et al.'s s and v) and `--json`, and prints the Voigt stiffness (per unit area in the
  plane), the engineering constants, a table of the uniaxial stresses along the axes and
  their means (the keys of the summaries of the models). In the interface, the tool
  Stiffness of the frame of beams of the tab Mechanics takes the items of the new trait
  `framework` (the networks of the workspace; other morphologies are not offered, and
  refused plainly by the API), with the same options in two boxes (Beams; Softening of
  non-identical beams, folded), and gives the values (E/E_s, Poisson's ratio, E/E affine,
  stretching energy, non-affinity, fibre volume fraction, E/(E_s φ), the engineering
  constants, the counts), the tables of the stiffness and of the uniaxial stresses, the
  softening table, the directional Young's modulus (`_young_figure` now draws plane
  stiffnesses too), a sentence and `frame_stiffness.json`, in the thread of the tools with
  its progress; Generate shows the frame of the summaries of the network models
  (`frame_stiffness: true`) in a table of its own, labelled (`api.FRAME_SUMMARY`,
  `api.frame_summary`), and its E/E_s in the card of the morphology. Checked: the command
  and the tool against `network_stiffness` of the same framework within 10⁻¹²
  (Euler–Bernoulli; Timoshenko of Cowper's coefficient with rigid joints at E_s = 2, ν =
  0.25; a given κ = 0.9; a planar network per unit area), the summary of the first
  realization against that of the model with `frame_stiffness: true` (8 digits) and that
  of `--realization 1` against the second, the softening coefficients against
  `softening_coefficients` (10⁻¹⁰), the errors (a model without a framework, a shear
  coefficient of Euler–Bernoulli beams, the z axis of a plane network, a seed for a
  geometry, negative rigid joints), the form (its trait, boxes and conditions), the run
  kept in the workspace, the labelled summary of Generate. Timing: the trabecular-bone
  template (64 cells at 15 %, 802 Timoshenko beams with rigid joints and the softening of
  the modulus) in 6 s, most of it the realization. Left: the softening replicas
  (`softening`, several CV and laws) and the scaled frames of Ban's cases in the
  interface; a figure of the frame itself; the comparison with the FFT solver of the same
  realization from the tool (by hand: Linear tensors).
* The frames of beams against the continuum, rigid joints, and Ban et al.'s case 9 looked
  at again (Ban, Barocas, Shephard & Picu 2016). `SpaceFrame.from_framework`,
  `Frame.from_framework` and `network_stiffness` take `rigid_joints`, the length of a
  rigid end of each beam at its joints in radii of the beam (one value or one per bar; the
  attribute `rigid` of both frames, m x 2, None by default): the beam is flexible over l'
  = l − a₁ − a₂ and its ends move with the rigid arms of its nodes, u + θ × a e, its
  stiffness Tᵀ K(l') T (Euler–Bernoulli or Timoshenko with Φ = 12EI/κGA l'²); the affine
  estimate takes EA L²/l'; the ends of the shortest beams are scaled down to leave
  `FLEXIBLE_LIMIT` (0.1) of them flexible; only the sparse effective stiffness (and the
  softening coefficients) take them, the dense paths (stiffness, axial forces, buckling,
  Bloch waves) refuse such frames. Checked: the triangular lattice with rigid ends in
  closed form, E = (2/√3) E_s t (l/l') 3(1 + τ)/(3 + τ), τ = t²(l/l')²/(1 + Φ'), within
  4.4·10⁻¹⁶ (Euler–Bernoulli and κ = 5/6); random frames in space and in the plane against
  their beams split in three with end segments 10⁷ times stiffer, within 2.4·10⁻⁸ to
  5.2·10⁻⁷; the cubic lattice stiffer by l/l' at its affine estimate. Frames against the
  FFT solver on the voxel image of the same realization (open foams of 10 Lloyd-relaxed
  cells, a Delaunay network of 12 points, E_s = 1, ν = 0.3, pores void): the FFT modulus
  converges by five voxels across a strut (+1.45 % from 64³ to 128³, 0.02 % from 128³ to
  160³ at r = 0.03); the ratio frame/FFT is 0.978, 0.909, 0.891 and 0.878 with
  Euler–Bernoulli beams and 0.926, 0.796, 0.739 and 0.691 with Timoshenko beams for foams
  at image fractions 0.019, 0.062, 0.105 and 0.155 (the first at three voxels across, its
  ratios a few per cent high), 0.934 and 0.933 for the Delaunay network at 0.123: the
  joints of the continuum, where the struts merge, not the shear (real in the continuum: 6
  to 20 % of the energy of the foams), make the departure. Rigid ends of the node radius
  overshoot by 42 to 94 %; the length that matches is 0.18 (about 0.2–0.25 corrected),
  0.34, 0.36 and 0.37 radii for Timoshenko foams and 0.56 for the Delaunay network, and
  Timoshenko beams rigid over a third of their radius follow the continuum within 4 % for
  the foams from 6 to 15 % (1.069 at 2 %) and 3 % for the Delaunay network: a fit, not a
  rule from the geometry of the joints. Case 9: their coordination of 21.12 is reached by
  no Delaunay triangulation of Poisson points (15.54 periodic; 13.4 for 150 points in a
  cube without images, the hull missing neighbours) but by counting as fibres the pieces
  of a periodic network cut by the cube (20.8, 21.3 and 20.5 for three realizations of 150
  points; 21.2 for 120), so that their network is ours (their springs agree in s,
  0.35–0.38 against 0.354–0.367, and in scatter); a cube cut out of the periodic network
  with their boundary conditions (cut ends of x = 0 and x = L held in x₁, lateral faces
  free) softens more, s = 0.196, 0.196 and 0.190 against 0.178, 0.174 and 0.173 in the
  periodic cell; their coefficient of variation is σ/⟨k⟩ as ours; their Fig. 4 read again
  at 400 dpi gives s ≈ 0.12 (seven points from 0.106 to 0.135, median 0.122) and v ≈
  4.9·10⁻⁴, a scatter 4.5 times below ours (2.2·10⁻³) while that of their springs is ours;
  no shear stiffness gives both (drawing EA alone at 1.5 times the shear stiffness of the
  area gives s = 0.12 but v three times theirs, and case 4 would fall from 0.46 to 0.26).
  Left: case 9, a factor 1.45 not settled (the paper gives neither the shear stiffness of
  its sections nor the number of its fibres); the comparison with the FFT on Delaunay
  networks at small fractions and on more realizations (thin fibres at three to four
  voxels across take the GPU more than 20 min a run); a rule for the rigid length from the
  angles of the struts at each joint; the frame stiffness and rigid joints in the GUI and
  the CLI. Since: `rigid_joints` 0 gives the frame without rigid ends (its zero lengths
  made the dense paths and the buckling refuse it).
* Thermodynamically consistent learned laws in finite strains (the first technical
  follow-up of the learning module): `morphogen.learning.Standard` now learns
  from finite-strain data (the Green-Lagrange strain `E` and the second Piola-Kirchhoff
  stress `S`), its free energy `ψ(H, α)` of an objective Lagrangian strain chosen by
  `strain`: `logarithmic` (the default) `H = ½ ln C`, the space of morphoGen's Hencky
  points (Miehe, Apel & Lambrecht 2002), with `S = Jᵀ T` (`T = ∂ψ/∂H`, `J = ∂H/∂E` from
  the divided differences of `½ ln` in the eigenbasis of `C`) and the tangent `Jᵀ (∂T/∂H)
  J + T : ∂²H/∂E²` (the second divided differences, Daleckii-Krein, exact for equal
  stretches), or `green-lagrange` (`S = ∂ψ/∂E`, a tangent positive definite in `E`);
  `Standard.logarithmic`, `standard.STRAINS`, `strain` in the `model` section of
  `morphogen learn` and the field *strain of the free energy* of the tool *Learn a
  behaviour* (a standard model in finite strains). Either way the dissipation of a step is
  the same `-∂ψ/∂α·Δα ≥ 0` (`S : Ė = T : Ḣ`), `S` derives from `ψ`, the tangent is
  symmetric and the law, a function of `C`, objective; the training carries the gaps of
  `S` to `T` through `J`, which the parameters do not change. It is integrated as the
  other learned laws in finite strains: at a material point, by finite elements and on
  voxels with `P = F S`, turned in the axes of its material, with the thermal stretch
  `F_th = exp(α ΔT)`. Newton's method of a step no longer stalls at a residual of 1e-6
  where Armijo's test met the round-off of the incremental potential (it backtracks only
  while the decrease it predicts is above 1e-12 of the potential; 2 of 1024 validation
  steps stalled before, in small strains too in principle). Measured: the training
  gradients against central differences to 1e-6 (both strains), and whatever the
  parameters along gradients to 15 %: no negative dissipation, `S = ∂ψ/∂E` to 2.5e-9, `D
  Δt = A·Δα` to 3.4e-9, `∂S/∂E` to 1.8e-9 and `∂P/∂F` to 5.1e-9 of central differences,
  the tangent symmetric to the bit (indefinite at 34 of 90 steps in `ln C` through its
  geometric term, positive definite in `E`), a superposed rotation turning `P` and `∂P/∂F`
  to 3.3e-15 along two steps, the small strain model of the same parameters recovered to
  the first order of the gradient (3.1e-2, 3.1e-4 and 3.1e-6 at scales 1, 1e-2 and 1e-4 of
  a 1 % path), a tension to 8 % at a material point, on voxels and by finite elements to
  1e-6, the turned law equivariant to 1e-12, free heating `F = exp(α ΔT) I`; von Mises
  plasticity at large strains (Hencky points, `σ_y = 200 MPa`, `H = 2 GPa`, 128 walks of
  the logarithmic strains 11, 22, 33 and the simple shear 12 to 10 %, Green-Lagrange
  strains to 22 %, 300 epochs, rotated): the standard model of `ln C` at 11.4 % on the
  validation paths and 19 %, 11 % and 29 % along an unseen tension-compression cycle of
  ±10 %, a shear cycle in 23 and a turn (lateral strains within 0.4 to 0.8 %), its
  dissipation nonnegative at all 1024 validation steps, 180 MPa along the cycle between
  the reference's dissipation (77 MPa) and plastic work (225 MPa); the recurrent network
  and the standard model of `E` do not fit their own training paths (63 % and 62 %; along
  the cycle and the shear the material point finds no lateral strains with the recurrent
  law, 34 % along the turn; the standard model of `E` 56 %, 12 % and 36 %, its lateral
  strains 10 to 48 % off), whereas the same walks in small strains are learned to 10.9 %
  (GRU) and 10.6 % (standard): at large strains the measures, not the models, limit the
  learned laws; it trains in 70 s against 6 s for the GRU, and 1000 epochs bring it to 8.8
  %. Left for later: the other learned laws (the recurrent network, the potential) in the
  logarithmic strain, longer trainings at large strains (300 epochs leave the standard
  model under-fitted, its best epoch the last), a dissipation potential of the state and
  deeper input-convex free energies.
* Timoshenko beams in the periodic frames, and Ban et al.'s cases 4 and 9 (Ban, Barocas,
  Shephard & Picu 2016). `Frame` and `SpaceFrame` take, as an option, a transverse shear stiffness
  κGA per beam (the attribute `shear`; `shear_coefficient` in `from_framework`,
  `lattice_of` and `network_stiffness`: a value, one per beam, or `"cowper"`, the new
  `cowper_coefficient`, 6(1 + ν)/(7 + 6ν) for circles and 10(1 + ν)/(12 + 11ν) for
  rectangles): `beam_matrices` and `space_beam_matrices` then give the exact two-node
  Timoshenko stiffness with Φ = 12EI/(κGA L²) in each bending plane and the consistent
  geometric stiffness of its interpolation (Engesser's load as the elements refine);
  without it the beams stay those of Euler–Bernoulli and every result is unchanged (the
  same arrays to the last bit at κGA = ∞). The sparse path splits the bending into a
  uniform curvature and the rotation of the chord, whose energy goes to bending and shear
  as 1 : Φ (`FrameStiffness.shear`, the part `shear` of `split`); `softening_coefficients`
  adds the curvature of the beam stiffness itself (k_a'' = −2k_aΦ/(1 + Φ)²) for the modes
  `bending` and the new `area` (EA and κGA together, the area of a general section), and
  `softening` rebuilds the beams of each replica. Checked: the matrices against the energy
  of the interpolation within 4·10⁻¹⁶; a cantilever (PL³/3EI + PL/κGA, rotation PL²/2EI)
  and a fixed–fixed beam (PL³/192EI + PL/4κGA) to round-off, in both planes of a strut in
  space too; buckling within 2·10⁻⁴ and 1.2·10⁻⁵ of Engesser's P_E/(1 + P_E/κGA) with 16
  and 64 elements; Φ = 4.8·10⁻¹³ within 1.8·10⁻¹³ of Euler–Bernoulli; the honeycomb with
  the shear compliance of its walls, E = 4/√3 E_s t³/(1 + Φ + 3t²), ν = (1 + Φ − t²)/(1 +
  Φ + 3t²), energy split 1 : Φ : 3t², and the triangular lattice with t²/(1 + Φ) for t²,
  within 2.4·10⁻¹³ (derived; Gibson & Ashby's thick-wall correction not compared); the
  struts in space against the dense path within 1.1·10⁻¹⁵; the derivatives of the four
  modes against finite differences (10⁻⁵, 2·10⁻³) at Φ of order 10 to 100. Ban et al.'s
  general sections set A, I and J apart, so that their EA = 10⁻³(EA)₀ is an area A =
  10⁻³A₀, and Abaqus gives a general section the shear stiffness kGA of its area (k = 1)
  unless set apart: Timoshenko beams of that shear stiffness (Φ of median 531 in the
  Voronoi network, 59 in the Delaunay one) with the area drawn at random give s = 0.464
  for case 4 (0.464 ± 0.004 over four realizations; replicas 0.00472 ± 0.00006 at CV = 0.1
  for s CV² = 0.00464) against about 0.45 read on their Fig. 4a, which settles it, and s =
  0.171 for case 9 (0.175 ± 0.001 over four realizations, 0.170 with 400 points; replicas
  0.00172 ± 0.00003 for 0.00171) against about 0.11: the order of magnitude is recovered
  (Euler–Bernoulli: 0.0075) but a factor 1.5 remains, possibly their denser networks (Z =
  21.12 against 15.6), free lateral faces or mesh of linear elements. Drawing EA alone
  gives 0.26 and 0.14, keeping the shear of the full section 0.112 and 0.0080
  (Euler–Bernoulli); cases 1 and 11 are unchanged (0.458, 0.354). Thick struts (r = 0.08)
  lose 28 % of the modulus of the Kelvin lattice to shear, 7 % of the bcc one and 1 % of
  the octet. Left: the remaining factor 1.5 of case 9 (networks of coordination 21, free
  faces); the frames against the FFT solver on a voxelized open foam; Abaqus' linear beam
  elements as such (their slenderness compensation of the shear stiffness); Timoshenko
  beams in the GUI stability tool and the summaries of the network models.
* Periodic approximants of the quasi-periodic tilings, so that they fill periodic boxes
  for the FFT solvers and periodic finite elements (a follow-up of the cellular automata
  on other lattices): `approximant` (an order k >= 1, or one per
  axis) for `lattice: penrose`, `ammann_beenker` and `icosahedral` in
  `cellular_automaton`, `activator_inhibitor` and `reaction_diffusion`, and
  `make_tiling(..., approximant=k)`, `approximant_grid`, `approximant_period`. They are
  the rational approximants of the cut-and-project, a phason strain that keeps the tiles:
  the dual method on grids of integer vectors, τ replaced by F_{k+1}/F_k (Fibonacci) in
  the pentagrid and in the axes of the icosahedron (the cubic approximants 1/1, 2/1, 3/2,
  5/3...), the silver mean 1 + √2 by P_{k+1}/P_k (Pell) in the four Ammann-Beenker grids;
  periodic in rectangles of √5 τ^k × 2 sin 72° τ^k edges, squares of side (1 + √2)^k and
  cubes of side 2τ^(k+1)/√(2 + τ), with |det g| cells of each family per period (2L_{2k+1}
  thick and 2L_{2k} thin rhombi, H_{2k} squares and 2P_{2k} rhombi, 4F_{3k+2} prolate and
  4F_{3k+1} oblate rhombohedra), the vertices numbered on the torus by an exact integer
  reduction modulo the periods. The box holds the nearest whole number of periods along
  each axis, fitted to it like the unit cells of the periodic tilings (refused under half
  a period; the tiles exact when the box is a multiple of the period, affine images
  otherwise, whose shared edges and faces now give the fluxes); non-periodic domains take
  windows of a larger approximant; realizations rebuild their tiling from its
  specification. Tori too small for their cells are refused with their remedy (a larger
  box, i.e. more periods, or a higher order), detected in general before the adjacency is
  deduplicated: a cell meeting its own image, two cells meeting through several images
  (shared vertices that are not those of one common vertex, side, edge or face), or a face
  shared through another image than the nearest. Checks: every cell has edges ±e_j and the
  area or volume of a tile (1e-12), the cells fill the box (1e-12), 3,000 random points
  each in exactly one cell; every edge or face shared by two cells across the faces of the
  box (4 or 6 face neighbours), the tiling translated by a period is itself (1e-9),
  rasterized images without seam; Euler's relation on the torus V - E + F = 0, and V - E +
  F - C = 0 with exactly V = C, E = F = 3C (1/1 to 5/3, 256 to 2,440 cells); the
  degenerate tori refused exactly when their neighbourhoods would be wrong (over the
  orders 1 and 2 along each axis and up to three periods: one period along an axis of
  order 1, and of the cubic 1/1 and 2/1, whose 2/1 keeps its 6 face neighbours but has 46
  pairs of cells sharing two vertices through two images, taken for an edge), the others
  with the face and edge neighbourhoods of their shared sides, edges and faces; tile
  ratios tending to τ, √2 and τ, their errors divided by about τ^4, λ^4 and τ^6 per order;
  Ammann-Beenker approximants with the six vertex stars and no other at every order, at
  frequencies within 2.9 %, 0.087 %, 2.6e-5 and 7.5e-7 at orders 3 to 6; Penrose
  approximants with de Bruijn's eight stars plus exactly 8 defect vertices per period (3.3
  % of the vertices at order 4, 0.026 % at order 9), the eight within 1-3 % at order 6 and
  0.15-0.3 % at order 8; the automata, activator-inhibitor and Gray-Scott models on planar
  approximants, prisms and cubic approximants in periodic boxes; a Gray-Scott pattern on
  two periods of the order-4 Penrose approximant through the FFT solver, apparent bulk
  modulus 1.884 between the Hashin-Shtrikman bounds 1.001 and 3.212, its energy equal
  within 2e-13; built in 0.11 s for the order-9 Penrose approximant (30,254 tiles), 0.56 s
  for the 13/8 cubic approximant (43,784 rhombohedra). Left for later: the vertex stars of
  the icosahedral tiling and of its approximants, approximants in `fit_rule`, `morphogen
  fit-rule` and the Design tool (which keep free patches), and non-rectangular periods
  (the rhombic or body-centred cells of some approximants).
* The stiffness of random fibre networks and open foams by periodic frames of beams, and
  the softening of networks of non-identical beams (Ban et al. 2016, with Bosco et al. 2017
  and Shahsavari & Picu 2015). `SpaceFrame.from_framework` and
  `Frame.from_framework` (in the plane) turn the framework of a realization
  (`delaunay_network`, `random_lattice`, and `open_foam`, whose Voronoi framework
  `models.foam.periodic_framework` is now kept in its metadata) into a frame of
  Euler–Bernoulli beams of circular sections with rigid joints (a radius, modulus and
  Poisson's ratio per fibre); the new `homogenization.networks` solves its periodic
  problem on the sparse stiffness (SuperLU) for the 6 unit strains (3 in the plane):
  `frame_stiffness` (the method `effective_stiffness` of both frames) and
  `network_stiffness` return a `FrameStiffness`, the Voigt stiffness (order 11, 22, 33,
  23, 31, 12, engineering shears: `engineering_constants`; Mandel W C W), its parts stored
  in stretching, bending and torsion (`split` of any strain), the affine estimate
  φE_s⟨n⊗n⊗n⊗n⟩ (`affinity`) and the non-affinity Σ|w_j − w_i|²/(|E|²Σl²) over the beams
  (`nonaffinity`, after Head et al. 2003 and Hatami-Marbini & Picu 2008); the three models
  add them to their summary with `frame_stiffness: true`. Checked: the four lattices of
  struts against the dense `macroscopic_stiffness` within 1.4·10⁻¹⁵ (0.004 s against 0.43
  s for the octet), and from their frameworks within 4.4·10⁻¹⁵; the honeycomb (Masters &
  Evans: E = 4/√3 E_s t³/(1 + 3t²), ν = (1 − t²)/(1 + 3t²)) and the triangular frame (E =
  2/√3 E_s t·3(1 + t²)/(3 + t²), derived) within 1.2·10⁻¹⁴; a slender Delaunay network
  against its truss from the compatibility matrix within 6.4·10⁻¹⁵. Measured over four
  realizations: slender Delaunay networks are stretching-dominated but do not tend to the
  affine estimate: 0.821 ± 0.002 of it in space (E = 0.1364 φE_s for an estimate of
  0.1662, about 1/6), with its Poisson's ratio 0.253 (1/4), and 0.756 ± 0.002 in the plane
  (ν = 0.374 for 1/3), the non-affine relaxation of a disordered network of central
  forces; Poisson–Voronoi open foams are bending-dominated, E = 1.27 ± 0.02 E_s φ² as φ →
  0 (Gibson & Ashby: about 1; Kelvin foams of circular struts 0.60), ν → 1/2. Networks of
  non-identical beams (Ban et al.): `softening_coefficients` gives exactly the
  coefficients of their eqs. 22 and 25, (E_0 − ⟨E⟩)/E_0 = s CV² and σ_E²/E_0² = v CV² (one
  solve per beam, the free lateral strains relaxed; the derivatives within 10⁻⁵ of finite
  differences), and `softening` replicas of normal, lognormal or bimodal stiffnesses
  (`fibre_scales`, `scaled_frame`; modes axial, bending or modulus) with a control
  variate: on their reference (r = 0.02 L, (EI)₀/(EA)₀ = 10⁻⁴ L²), the bending-dominated
  Voronoi network (their case 1) gives s = 0.458, the replicas 0.00465 ± 0.00006 at CV =
  0.1 for s CV² = 0.00458, against s ≈ 0.45–0.55 read on their Fig. 4a, and Delaunay
  springs (case 11) s = 0.354 (0.378 bimodal at CV = 0.5) against 0.34–0.39; the scatter v
  within a factor of two of their Fig. 4b; with the axial stiffness lowered to 10⁻³(EA)₀
  (cases 4 and 9), our Euler–Bernoulli beams soften far less (s = 0.11 and 0.0075 against
  about 0.45 and 0.11), possibly because their Timoshenko beams of general sections lost
  their shear stiffness with the area. Left: the comparison with the FFT solver on
  voxelized foams; Timoshenko beams (transverse shear) to settle cases 4 and 9; the
  tension–compression asymmetry and the nonlinear range of Ban et al.; the stiffness of
  the frames of `cross_linked_fibers` (cross-links along the fibres) and of the
  random-walk fibres; a tool of the interface and a command line.
* Thermodynamically consistent learned laws (an item the learning module left for later):
  `morphogen.learning.Standard`, a learned generalized standard material in small
  strains (`kind: standard` and `internal` in the `model` section of `morphogen learn`,
  `--model standard`, and among the models of the tool *Learn a behaviour*). A few learned
  internal variables α, a learned free energy ψ(ε, α) convex in (ε, α) jointly (an
  input-convex network of softplus ridges and a quadratic term, zero with its gradient at
  the natural state) and a learned dissipation potential, convex, nonnegative and zero at
  zero: a rate-independent part positively homogeneous of degree 1 beyond a smoothing of
  the increments (smoothed ramps of smoothed norms and linear forms, `plastic`, the
  default) and a viscous quadratic part (`rate`). The state of a step is the minimizer of
  the incremental potential ψ(ε_{n+1}, α) + Δt φ((α − α_n)/Δt), by Newton's method on its
  gradient with a backtracking line search, and σ = ∂ψ/∂ε: by construction the dissipation
  of every step is nonnegative and the consistent tangent (the Schur complement of the
  implicit function theorem) symmetric and positive definite, whatever the parameters. The
  training goes through the minimizations by implicit differentiation (the adjoint of a
  step solved with the Hessian of the incremental potential, carried back to the previous
  state through the dissipation), its gradients written out and compiled by Numba;
  `Standard.history` gives the internal variables, the free energy and the dissipation of
  each step, `Standard.energy` the free energy. It is integrated as a `learned` material
  like the other models: at a material point, by finite elements, on voxels, through the
  seam of the MFront points. Measured: the gradients against central differences to 1e-6
  (rate-independent, viscoplastic and viscoelastic models); a model of parameters moved
  off their start, along paths 1.5 times beyond its data, dissipates at every step, its
  tangent symmetric to the bit, positive definite and the derivative of its stress to
  1e-6, its stress and dissipation those of its free energy to 1e-6; a standard linear
  solid is represented to 1.2e-15 (one Newton iteration a step) and learned from 18 paths
  at random rates to 0.11 % on the validation paths and 0.43 % along an unseen relaxation;
  von Mises plasticity with hardening on the data of the recurrent network: 2.3 % on the
  validation paths (the GRU 4.1 %), 7.4 %, 4.8 % and 8.4 % on an unseen
  tension–compression cycle, a shear cycle in a direction never driven and a turned path
  (the GRU 7.9 %, 11 % and 12 %), the lateral strains 5 to 15 times closer, no negative
  dissipation in 1024 validation steps (the least 7.7e-5 MPa), a symmetric tangent of
  eigenvalues 2.6 to 500 GPa (the GRU's asymmetric by 4 to 5 %), the work of the cycle the
  reference's (11.11 MPa) and its cumulated dissipation, 8.72 MPa, between the reference's
  dissipation σ_y p (6.03 MPa) and plastic work σ_y p + H p²/2 (10.57 MPa), stress data
  fixing the work but not its split between stored and dissipated energies; it trains in
  63 s against 8 s for the GRU (a Newton solve at each step). Left for later: standard
  models in finite strains, a dissipation potential of the state (isotropic hardening
  explicit in the state of a step) and deeper input-convex free energies. Since: `hidden`
  and `internal` must be positive integers (a list from `--model standard` raised a
  `TypeError`) and `smoothing` finite.
* Learned behaviours, the follow-ups: a tool of the interface, laws turned with
  their material, thermal strains. The tab Learning (`gui/tools/learning.py`, tool
  `learn`) learns a behaviour from a material point (the behaviours of Mechanics, a
  material of the configurations, a behaviour of the workspace) or from the homogenized
  responses of a morphology (FFT on its voxels, its grains turned), its fields of paths,
  model and training generated from the sections of `morphogen learn` with their defaults;
  the training runs in the thread of the tools, reports the losses of each epoch and stops
  between two; it draws the learning curves and the learned law along a validation path
  (turned by an orientation for a crystal) against the reference, writes `learned.npz`,
  the data, `material.json` and the `learning.yaml` that learns it again, and keeps the
  law in the workspace as a behaviour (B) that names its model in its folder (saved and
  opened), which the loading paths give to a phase and the material point of Mechanics
  turns by an orientation. Learned laws run in the axes of their material
  (`LearnedBehaviour(model, oriented=True)`, `learned: {model, oriented}`): the
  orientation of the point (`simulate_point(orientation=...)`, the grains of polycrystals
  on voxels and meshes) turns ε (E in finite strains) in and σ, the tangent (S) back, in
  morphoGen's Voigt order 23, 31, 12 at every interface (the learned points never pass
  TFEL's 12, 13, 23); `train(rotate="cubic")` turns the paths of a cubic crystal by the 24
  rotations of the cube; `learned_material(expansion=...)`, `learned_law(expansion=...)`:
  the stress-free thermal strain α ΔT taken out before the law (F_th = exp(α ΔT) in finite
  strains, S = J_th A S_m A), the expansion of a reference material carried over
  (`reference_elasticity`), `responder`, `compare_path` and `path_errors` with an
  orientation; the material point stops its Newton iterations at the round-off of the
  stresses of the strain (zero-stress solutions such as a free heating). Measured: an
  orthotropic stiffness of shear moduli 60, 40, 20 GPa learned and turned by (30°, 50°,
  70°) gives the crystal turned to 1.2·10⁻¹⁵ (tangent 7·10⁻¹⁶; an exchange of the shears
  23 and 12 would show as 13 %); equivariance to below 10⁻¹² (a recurrent law along steps,
  a potential in finite strains with dP/dF); a copper polycrystal on voxels with the
  learned crystal against its crystals to 4·10⁻¹¹ (not oriented: 67 GPa instead of 131
  GPa); Méric-Cailletaud crystal plasticity learned at a point in its axes (recurrent 32,
  256 walks) against the crystal law on a polycrystal of 8 grains in tension to 1 %: 10.8
  % without rotations (validation 10.4 %), 6.7 % with the rotations of the cube
  (validation 5.2 %), 5.9 % with 512 walks (validation 3.9 %), within the error of the law
  at a point; thermal: free and constrained heating and a heated tension to 10⁻¹⁵, F =
  exp(α ΔT) in finite strains as the Hencky points, a turned crystal of three expansions
  to 10⁻¹⁵; through the tool, isotropic elasticity 0.16 % (validation) and 0.09 %
  (tension), a learned matrix of a morphology 0.04 % from the elastic one, von Mises
  plasticity 6.0 % and 13.5 % along a cycle in 5 s. Left for later: thermodynamically
  consistent recurrent laws, the temperature as an input (morphoGen's behaviours depend on
  it through their expansion only; MFront data at several temperatures would need it), the
  symmetry groups of other lattices in the training, and training on the GPU.
* Quasi-periodic tilings and reaction-diffusion on tilings, the follow-ups of the automata
  on other lattices: `lattice: ammann_beenker`, the squares and 45° rhombi of the octagonal
  Ammann-Beenker tiling by de Bruijn's dual method on four grids (the cut-and-project of
  Z^4), and `lattice: icosahedral`, the prolate and oblate golden rhombohedra of the
  icosahedral tiling by the dual method on six grids of planes normal to the fivefold axes
  of the icosahedron (the cut-and-project of Z^6), both built like the Penrose rhombi
  (`_dual_cells`, exact integer vertices, singular grids refused, offsets drawn from the
  seed and kept in the realization) for `cellular_automaton`, `activator_inhibitor`,
  `reaction_diffusion`, `fit_rule` and the Design tool, on non-periodic domains;
  `morphogen fit-rule --tiling T` searches the rules on any tiling and writes its
  configuration (non-periodic for quasi-periodic tilings). `reaction_diffusion` takes
  `lattice` too: Gray-Scott on the cells of any tiling with a finite-volume Laplacian,
  two-point fluxes A/d across each face over the cell's area or volume
  (`Tiling.face_measures`, `Tiling.diffusion_weights`, `gray_scott_tiling`,
  `stable_substeps`; `tile_edge`; `gray_scott(initial=...)`), the unit steps split into
  stable explicit substeps (two on Penrose rhombi, triangular prisms and rhombohedra),
  `volume_fraction` by volume. Checks: Ammann-Beenker (80 × 80, 6,749 tiles) the six
  vertex stars and no other, at the frequencies λ^-1, 2λ^-2, 2λ^-3, 2λ^-4, λ^-5, λ^-4 of
  their regions of the octagonal window (λ = 1 + √2, computed exactly) within 3.5 %,
  rhombi/squares = √2 and equal areas within 0.012 % (smooth window), the tiles clipped to
  the box summing to its area within 4e-14, V - E + F = 1; icosahedral (32^3, 41,230
  cells) every point of 4,000 in exactly one cell, every inner face shared by two cells,
  V/C = 1.0000, E/C = 2.9998, F/C = 2.9999 (V = C from the triacontahedral window, Euler),
  V - E + F - C = 1, prolate/oblate = τ within 0.7 % (0.04 % on 56^3), rasterized volumes
  within 2 % at 10 voxels; on the square tiling the finite-volume Gray-Scott equals the
  grid's scheme within 2.5e-14 over 4,000 steps in 2-D and 3-D; on Voronoi tilings the
  face pyramids fill the cells to 1e-12 and plane waves decay at -k^2 within 1 %; on
  rhombi and rhombohedra the homogenized diffusivity of the graph is that of space within
  0.5-2 % (a linear field 6.5-14 % off); labyrinth wavelengths 11.72 cells on the grid,
  11.63 on hexagons, 11.68 on triangles, 11.86 on Penrose and 11.68 on Ammann-Beenker
  tiles (spots within 2 % on the periodic tilings, 4-8 % wider on the quasi-periodic
  ones). Left for later: the vertex stars of the icosahedral tiling, consistent fluxes on
  rhombi (DDFV or mimetic schemes), periodic approximants of the quasi-periodic tilings
  for periodic domains, and the reproduction of the grid scheme to the bit (its stencil
  sums in another order).
* Covariances of oriented fibres, the initial placement and mixed laws of Altendorf,
  Jeulin & Willot, and Delaunay fibre networks (Peyrega et al. 2009, 2011, 2013, Altendorf
  et al. 2014 and Ban et al. 2016). `PoissonFibers.covariance(h, direction)` and
  `BooleanCylinders.covariance(h, direction)` give the exact covariance of Poisson fibres
  (lines of any law of directions: the new `orientation` of `poisson_fibers`) and of
  Boolean cylinders and spherocylinders along any direction, for every orientation law,
  from Q(h) = q² exp(θ E[K(h)]): the covariogram (l − |h·u|)+ lens(|h × u|) of cylinders
  (Peyrega et al. 2009, eq. 15) and the integral along the axis of the lens of the
  sections of spherocylinders (`theory.cylinder_covariogram`,
  `theory.spherocylinder_covariogram`, `theory.lens_area`, `theory.oriented_covariance`;
  ∫K = V² to 4·10⁻⁶), averaged over the axes by the quadratures of the laws
  (`OrientationDistribution.quadrature`, `orientations.angle_weights`, `half_sphere`);
  `BooleanCylinders.mean_covariogram` and `integral_range` (in its theory; E[V²]/E[V] as p
  → 0, to 5·10⁻⁶). Measured on realizations, along x, y, z and the mean fibre axis (1, 1,
  1): Boolean spherocylinders of a β law within 0.0026 (four boxes of 120³ voxels),
  Poisson lines of a girdle and a von Mises–Fisher law within 0.0021 (twelve windows),
  where the directions differ by 0.02; the curves of the interface draw the closed form
  along each axis. New orientation laws for every fibre model: `{type: von_mises_fisher,
  kappa, axis}` (`VonMisesFisher`) and `{type: mixture, laws, weights}` (`Mixture`, the
  mixed β laws of the glass fibres of Altendorf et al. 2014). `random_walk_fibers` takes
  `placement_trials`, the initial placement of least overlap of the 2014 paper (each fibre
  at the best of that many uniform positions, the one of least overlap with those placed
  before it; `kernels.force_biased.place`; the strategy is not described in the papers:
  our choice): with 30 trials the packing takes 31 steps instead of 266 for
  the template, 508 instead of 1842 for isotropic fibres at 0.6, 1416 instead of 5104 at
  0.7 (β = 0.1), so that a budget of 2000 steps reaches 0.7 instead of 0.65.
  `delaunay_network` (`DelaunayNetwork`, next to the open foams): fibres on the edges of
  the periodic Delaunay triangulation (`models.foam.periodic_delaunay`) of Poisson,
  hard-core or relaxed points, in space or in a plane, its framework in the metadata; mean
  coordination 15.532 ± 0.007 for 2 + 48π²/35 = 15.535 in space, 6 in the plane, mean
  fibre length 1.1332 ± 0.0009 λ^(-1/2) for Miles' 32/(9π) = 1.1318 (1.286 λ^(-1/3) in
  space, measured); template `delaunay_network`, in the interface under "Fibres and
  layers". Left: the strategy of the ICS 2011 paper itself, should it differ; the
  covariance of `cross_linked_fibers` (its fibres are those of the models above) and of
  curved fibres; a closed form of the mean Delaunay edge in space; beam computations on
  the Delaunay frameworks (Ban et al.'s moduli). Since: `placement_trials` above 1 needs
  `packing: true` (unpacked fibres overlap as a Boolean model, whose fraction the
  placement raised, 0.346 for 0.3) and places nothing when there are no fibres; in a box
  that is not periodic, the fibres of a Delaunay network leaving it by a face enter it
  again by the opposite one, the periodic network cut by the box (the fraction near the
  faces was 0.12 for 0.20); the periodic triangulation takes the images of as many
  periods as its margin spans, so that thin plates are triangulated (5 × 5 × 1 with 200
  points was refused); `node_ratio` must be finite; the curves draw the capacity of
  segments along x, where they are measured.
* Behaviours learned from data: `morphogen.learning` and `morphogen learn
  CONFIG`. Data: random strain paths (random walks or independent strains of chosen
  components, with amplitude, volumetric ratio, free components and strain rate)
  integrated at a material point or by the path solvers of a microstructure (its
  homogenized responses), in small strains or as Green-Lagrange strains and second
  Piola-Kirchhoff stresses in finite strains, split into training and validation paths and
  normalized. Models, their gradients written out (no automatic differentiation): a
  perceptron σ(ε); a learned convex potential (an input-convex network and a quadratic
  term: σ = ∂ψ/∂ε, a symmetric positive semi-definite tangent, ψ(0) = σ(0) = 0); a gated
  recurrent unit over the strain increments with learned internal variables, its
  backpropagation through time compiled by Numba. Training: Adam on mini-batches with
  clipping, decay, early stopping and random rotations of the paths for isotropic
  behaviours, reproducible from the seeds; `.npz` files with their JSON metadata.
  Integration: a material with `learned: model.npz` has its points integrated by the model
  along the seam of the MFront points (`homogenization.nonlinear.external_points`), at a
  material point, by finite elements and on voxels, with P = F S in finite strains.
  Measured: isotropic elasticity within 0.08 to 0.2 %; Hencky's elasticity in finite
  strains 4.3 % (a linear fit 25 %); von Mises plasticity with hardening 4.1 % on
  validation paths and 8 to 12 % on an unseen tension-compression cycle, a shear in a
  direction never driven and a turning path, against 51 to 110 % for a feed-forward
  baseline, which cannot represent hysteresis. Left for later: thermodynamically
  consistent recurrent laws (a learned free energy and dissipation), thermal and
  anisotropic learned laws, training on the GPU and a tool of the interface. Learned laws
  go since where the other inelastic behaviours go: the point bulk of the crack solvers,
  files relative to the configuration, an explicit refusal by the MFront sources.
* Fibres drawn into contact, and entangled cross-linked fibres (25 papers surveyed
  in the theory notes). `random_walk_fibers` takes the contact packing of Keilmann,
  Redenbach & Willot (Comput. Mater. Sci. 2026: `interaction_distance`,
  `contact_distance`, `contact_weight`, `contact_threshold`): the pairs of fibres within
  the interaction distance are shortlisted and drawn together by contact forces against
  the springs, the conflicting pairs released, then the end step runs; the summary gives
  the contacts per fibre before and after and the clots, the theory Toll's estimate
  (template `contact_fibers`; aspect ratio 10 at 20 %: 4.40 contacts per fibre after the
  packing alone, 6.44 after the contact packing, against Toll's 5.35, no overlap left).
  `cross_linked_fibers`, the volume elements of Chatti et al. (2018, 2020) and Mezeix et
  al.: straight fibres, infinite on Poisson lines of any orientation law or finite Boolean
  spherocylinders, in contact where their axes are closer than a diameter, a fraction of
  the contacts bonded by cross-links (a third phase); the theory gives the exact spacing
  of the contacts 1/(2D L_V E|sin γ|) (Mezeix's d/2f; Poisson lines: 0.02542 contacts per
  µm against 0.02538) and the contacts per finite fibre θ V_ex (3.61 against 3.54). The
  Gaussian girdle law `{type: girdle, sigma, axis}` joins the orientation laws of every
  fibre model, sigma at most 20 (isotropic within 0.4 %; its rejection sampler slowed
  without bound). Template `cross_linked_fibers`. With `end_step: shrink`, the contacts of
  the summary are counted on the fibres returned, their radii reduced.
* Anisotropic versions of the morphological models: one field, the same in the
  configuration, the CLI
  and every GUI form, `anisotropy: {stretch: [sx, sy, sz], euler: [phi1, Phi, phi2]}` (or
  3 or 6 numbers, or an elongation along z), maps the realization of the isotropic model
  by stretches along turned principal axes, normalized to preserve volumes
  (`morphogen.models.affine`: `Anisotropy`, `AffineStructure`). Boolean, hard-core,
  cherry-pit and Cox spheres become aligned ellipsoids; Voronoi and Laguerre tessellations
  are computed in the stretched space and mapped back; dead leaves and transparent dead
  leaves become leaves of ellipsoids; nucleation and growth gets ellipsoidal fronts;
  aggregation, TPMS and (random) strut lattices get stretched periods; the Gaussian
  excursion puts the metric in its covariance, for any principal axes and in periodic
  boxes too; the open foam's elongation takes the common schema. Periodic boxes take
  principal axes along their axes (an exact map of grids); turned axes need a non-periodic
  domain and are refused otherwise; `theory()` keeps the characteristics a
  volume-preserving map keeps. Without anisotropy every model is unchanged to the bit (15
  models compared with the previous sources). Checks: with stretches 1:1:2 the correlation
  lengths along the axes are in the ratios 1:1:2 within 4 % and the mean chords within 5 %
  (dead leaves apart, whose slivers are thinner than the voxels), the volume fraction kept
  within 0.03, the longest axis of the fabric tensor along the turned axis (cosine above
  0.99). The models anisotropic by construction (level-set foams, honeycombs, nacre,
  Bouligand structures, osteons, laminates), the combinations, the deterministic frames
  and the branching networks are left as they are.
* Bending fibres packed by forces: `random_walk_fibers`, the model of Altendorf &
  Jeulin (Phys. Rev. E 2011): random walks whose steps follow the multivariate von
  Mises-Fisher law between the main direction of their fibre (`kappa1`) and the previous
  step (`kappa2`), turned so that their end-to-end vectors follow the main directions,
  then packed without overlap by the paper's force-biased algorithm
  (`kernels.force_biased`, Numba with linked cells: repulsions of overlapping segments,
  springs on the segments and open angular springs on the joints, `recovery`, `max_steps`,
  `end_step: repulsion | shrink`, `min_gap`, `lumen`); chains of spherocylinders in a
  periodic box, which fibres may outgrow. Checks: the step law (Kolmogorov-Smirnov), the
  paper's estimators of kappa2, the tortuosity against a small-angle prediction (within 6
  %), no overlap after packing (minimal clearance −2.4e-15 r over the periodic images),
  the realized fraction and its image (0.3561 both); 0.51 for isotropic fibres of aspect
  ratio 11, 0.64 nearly aligned. Schladitz's β law of orientations, `{type: beta, beta,
  axis}`, for every fibre model. Template `random_walk_fibers`.
* Cellular automata on other lattices: `lattice: triangular | hexagonal | penrose | bcc | fcc` for
  `cellular_automaton` and `activator_inhibitor`, in `fit_rule` and in the Design tool;
  the default `square` is unchanged to the bit (the CSR engine on the square tiling equals
  the grid automaton for six rules, and seven realizations keep their SHA-256 digests).
  The cells are triangles, hexagons, Penrose rhombi (de Bruijn's pentagrid,
  quasi-periodic, non-periodic domains only), the prisms stacking them in 3-D, and the
  truncated octahedra of BCC and the rhombic dodecahedra of FCC, each of the area or
  volume of the square or cube of edge `cell_size`. Neighbourhoods are CSR graph balls
  across faces, edges or vertices, or Euclidean balls; majority votes are fractions, the
  cell's own vote doubled when its neighbours are odd in number; the conserved rule ranks
  local fractions; the activator-inhibitor weights are balanced cell by cell; sequential
  updates follow colourings. Realizations are `TilingField`s, rasterized cell by cell at
  any resolution, so that every analysis and solver applies (`morphogen.automata.tilings`:
  `make_tiling`, `Tiling`). Checks: the neighbour counts and shells of each lattice, the
  volumes of the Voronoi cells, the rasterized cells against their areas and volumes, the
  Penrose ratio of thick to thin rhombi (1.622 against the golden ratio) and de Bruijn's
  eight vertex stars with their frequencies, the growth from one seed (a hexagon on the
  hexagonal and triangular lattices, a cuboctahedron on FCC, a rhombic dodecahedron on
  BCC). A 3-D quasi-periodic lattice (icosahedral) is left for later. Refused since: a
  singular pentagrid (three of its lines through one point, as with offsets all 0), whose
  rhombi overlapped, and a Euclidean radius reaching no centre (1 on the hexagonal, BCC
  and FCC lattices, whose nearest centres lie 1.075 to 1.122 cell sizes apart), which left
  every cell unchanged.
* The anisotropy of percolation, in 2-D and 3-D. The percolation tools
  take planar images (one voxel along z): along x or y, never trivially z, through 4 or 8
  neighbours (`connectivity` 4 and 8; 6 is 4 there, 18 and 26 are 8), by the labels and
  the forest fire; `family_from_config(..., sized=True)` grows a planar configuration in
  its plane, and `percolation_scaling` takes ν = 4/3 for planar images (`nu=None` by
  default, 0.8765 in 3-D; `morphogen percolation --nu` alike). Checked on the square
  lattice: the exact duality of 8 and 4 neighbours on 300 rectangles, p_c = 0.592746 and
  0.407254 within 0.002 on 128² and 256² sites, 2:1 rectangles extrapolated within 0.002
  with ν = 4/3. Along any direction, `burn_along` lights the fire across the core of the
  box (the equilateral cylinder, a square in 2-D, of the inscribed ball, congruent for
  every direction); `directional_percolation` gives the crossings, chemical distances and
  geometric tortuosities of one image along the lattice directions (4 in 2-D, 13 in 3-D),
  sampled ones or vectors (and the measure `directional_percolation`);
  `percolation_anisotropy` gives the thresholds of a family along them: each realization
  bisected along every direction (common random numbers), the medians, the paired
  deviations and ratio with their standard errors, the threshold tensor (eigenvalues,
  principal directions). Isotropic lattices, discs and spheres: deviations within 1.7
  paired standard errors; sites of 2 × 1 voxels: lower along x (−6.8 σ), higher along y
  (+6.4 σ), the diagonals alike; layers separated by walls: never across; aligned fibres:
  0.099 along x against 0.40–0.42 across. `morphogen percolation --directions
  axes|lattice|N` and the field "anisotropy" of the tool "Percolation threshold" (a rose
  in 2-D, bars in 3-D). New public names: `burn_along`, `directional_percolation`,
  `DirectionalPercolation`, `percolation_anisotropy`, `PercolationAnisotropy`, `NU_2D`,
  `NU_3D`; changed: `percolation_scaling(nu=None)`; the measure `connectivity` of a planar
  image no longer has z columns.
* Every porous plasticity in the MFront sources: the cases that MFront's brick does not
  match exactly, refused until
  now, hold morphoGen's own integration of a point, as the crystals of finite strains do:
  the C port of the kernels (`behaviours.cu`, the GPU's, compiled as host C++ in the
  source), its `porous_update` with the row of parameters of the porous point and its
  internal variables, in a behaviour of the `Default` DSL (the strain turned from TFEL's
  components to morphoGen's, the stress and the consistent tangent of the kernel back),
  in small strains and in the space of the logarithmic strain (`@StrainMeasure Hencky`),
  with morphoGen's thermal strain α (T + dT − 293.15). So are exported Rousselier's
  criterion, the elliptic one, Chu & Needleman's Gaussian nucleation, the exponential
  one, the thresholds of the nucleation, the other tables of f*, the coalescence with q3 >
  q1² and the kinematic hardening of porous plasticity (GTN and elliptic); GTN keeps the
  brick where it is morphoGen's (f* = f or the coalescence, Plastic or Norton, the
  constant nucleation from p = 0); an anisotropic expansion stays refused. Their state
  variables: the elastic strain (small strains), the kernel's (10, 6 per backstress), p
  and the porosity (`mfront_state_size`, the AMITEX and Abaqus inputs); the abaqus
  interface takes them expanding in finite strains (they subtract their thermal strain:
  `io.mfront.mfront_brick` tells the brick's laws, still refused there). Compiled by MFront
  5.1 and integrated by MGIS along triaxial paths that grow the porosity (by 0.04 to
  0.13), Rousselier's criterion with the Gaussian nucleation and, viscous, the exponential
  one from thresholds, the elliptic criterion with Armstrong-Frederick and Prager
  backstresses, GTN with a table of f*, a threshold of the constant nucleation and a
  backstress, and with q3 > q1² and two backstresses, expanding, in small and finite
  strains, follow morphoGen's kernel to the roundoff (9·10⁻¹⁶ of the stress scale in
  small strains, 2 to 6·10⁻¹⁴ in finite strains; the tests ask 10⁻¹²), with the same Newton
  iterations in a tension whose free components need the tangent; an AMITEX phase
  (Rousselier, umat) and an Abaqus one (GTN with a table of f*, abaqus, expanding too)
  within 9·10⁻¹⁶.
* Finite-size scaling of percolation thresholds.
  `analysis.percolation_scaling(generate, sizes, lower, upper, *,
  nu=0.8765, fit_nu=False, ...)` bisects the threshold p½(L) of boxes of several sizes,
  each seeded from the seed and its size, and fits p½ = p_c + a L^(−1/ν) by least squares
  weighted by L^(2/ν), with ν given (0.8765, Ballesteros et al. 1999) or fitted. It
  returns a `PercolationScaling`: the sizes, thresholds and brackets, p_c and its standard
  error (from the residuals and the brackets), a, ν and its error, the residuals, and the
  results of each size. `family_from_config(..., sized=True)` draws the model of a
  configuration in cubes `size` voxels per edge at its voxel size, the domain scaled.
  `morphogen percolation --sizes 16,24,32,48 [--nu] [--fit-nu] [--json]` and the field
  "box sizes" of the tool "Percolation threshold" run it; the tool plots p½ against
  L^(−1/ν) with the fit and tabulates the boxes. On the sites of the cubic lattice (16 to
  48, the three pairs of faces of an open cube, forest fire, 99 realizations, 2.4 s), the
  extrapolation lands within 0.0015 of p_c = 0.3116, where the largest box is off by
  0.008. The measure `connectivity` gives periodic images `wraps_x`, `wraps_y` and
  `wraps_z`, the wrapping of the phase around the box, from `analysis.wrapping`; the
  wrapping logic of the labels moved to `analysis.connectivity`, with the same results.
* The forest-fire algorithm of percolation (the components were labelled until now by
  `scipy.ndimage.label`,
  joined by a union-find on periodic images). `analysis.burn(image, phase=1, axis="x", *,
  connectivity=6, periodic=None, full=False)` burns the phase as Herrmann, Hong & Stanley
  (1984): its voxels on the face at 0 normal to the axis catch fire at time 0, and the fire
  spreads at each step to the unburnt phase neighbours (6, 18 or 26), the transverse
  directions wrapping on periodic images; a compiled breadth-first search over a queue of
  flat indices, which stops at the end of the step at which the opposite face is reached
  unless `full`. `analysis.Fire` gives whether the phase percolates, that step (the
  chemical distance between the faces), the size of the front at each step and, with
  `full`, the burning time of each voxel (-1: unburnt). `percolation_threshold(...,
  method="fire")`, `morphogen percolation --method fire` (the JSON output names the
  method) and the field "algorithm" of the tool "Percolation threshold" take it as the
  criterion: face to face along the axis, periodic or open across it; it decides as the
  labels on open boundaries, while on periodic images their wrapping implies it but not
  conversely (0.3084 against 0.3162 on 48³ site lattices, seed 0). On 128³ voxels it takes
  9 ms against 47 ms for the labels at the threshold, 30 ms against 53 ms at 0.40. Tests:
  on 200 small site lattices (6, 18 or 26 neighbours, the three axes) the fire against the
  labels (open) and scipy's components of the graph wrapping transversally (periodic),
  which the wrapping implies; the exact times of a channel, a staircase, a diagonal and a
  periodic corridor, those of the whole clusters against a pure-Python breadth-first
  search; the threshold of the simple cubic lattice, the command and the interface by the
  fire.
* Percolation thresholds in the interface and the command line:
  `analysis.percolation.family_from_config(config, parameter="volume_fraction")` turns the
  `model`, `domain` and `voxels` sections of a run configuration (a file, a mapping or a
  `RunConfig`, validated by `load_config`) into the family `generate(fraction, rng)` of
  `percolation_threshold`: the model with the parameter set to the fraction, drawn in the
  domain and voxelized on the grid of the configuration as `pipeline.run` voxelizes it; a
  model without the parameter is refused with the names of those it has, a configuration
  without a voxel grid too. `morphogen percolation CONFIG [--parameter P] [--lower L]
  [--upper U] [--phase N] [--axis x|y|z|any|all] [--connectivity 6|18|26]
  [--periodic|--no-periodic] [--realizations N] [--tol T] [--seed S] [--json]` prints the
  threshold, its bracket and the probabilities tried (bounds 0.05 and 0.95 by default, the
  seed of the configuration unless `--seed`, the periodicity of its domain unless
  `--periodic` or `--no-periodic`); the tool "Percolation threshold" of the tab Statistics
  estimates that of the model of a morphology of the workspace (the parameter, the bounds,
  the voxels per edge, the phase, the axis, the neighbours, the boundaries, the
  realizations, the tolerance, the seed; in the boxes Family, Criterion and Bisection),
  reports its progress image by image and stops there when asked, and shows the threshold,
  its bracket, the curve of the probabilities against the fraction and their table.
  Checked: the family of a configuration (the fraction set on its images, the refusals),
  the command on Boolean spheres of 24³ voxels (3 realizations: the bracket, the
  probabilities on either side of 1/2, the same threshold again from the seed of the
  configuration; a missing parameter, no crossing and no voxel grid reported as errors)
  and the tool through the harness of the interface.
* Percolation thresholds, which no tool computed: `connected_components` labels the
  components of a phase (6, 18 or
  26 neighbours) and tells whether one joins the faces normal to each axis (the measures
  `percolates_<axis>`), and nothing estimated the threshold of a family.
  `analysis.percolation_threshold(generate, lower, upper, *, phase, axis, connectivity,
  periodic, realizations, tol, seed)` bisects, down to `tol`, the fraction at which the
  probability that `generate(fraction, rng)` percolates crosses 1/2 (rising or falling),
  over `realizations` images whose generators are seeded alike at every fraction (common
  random numbers: for a family growing with the fraction the estimate is the median of the
  thresholds of the realizations), and returns a frozen `PercolationThreshold` (the
  estimate, its bracket, the fractions tried and their probabilities, the number of
  realizations); the criterion is one axis, any or all, through 6, 18 or 26 neighbours,
  joining the faces inside the box or, on periodic images, wrapping around the box (the
  components joined across the faces by a union-find that keeps the shifts of their copies,
  as Newman & Ziff 2001: an infinite cluster of the tiled medium). Both names join the
  public interface. Checked (`tests/test_percolation.py`, 29 tests, 2.4 s): the exact
  brackets of the step of a slab, wrapping against joining the faces (a staircase joins
  them without wrapping, a helix wraps without joining them, diagonals through edges and
  corners), site percolation of the simple cubic lattice on 48³ voxels with 9
  realizations, 0.3162 along x, open or periodic, for 0.3116 through faces (Lorenz & Ziff
  1998) and 0.0998 for 0.0976 through 26 neighbours (Kurzawski & Malarz 2012), above p_c
  by the finite-size shift of the crossing along one axis (theory, §55), and Boolean
  spheres, 0.27 for 0.2895 in the continuum (Rintoul & Torquato 1997), spheres 4 voxels in
  radius touching on the image when closer than about a voxel (0.28 with 6 voxels).
* Kinematic hardening of porous plasticity: `PorousPlasticity(kinematic=...)`,
  pairs (C, D) of Armstrong-Frederick terms (D = 0: Prager), four at most, for GTN and the
  elliptic criterion, the potentials homogeneous of degree 1: the potential of σ - X
  (Besson & Guillemer-Neel 2003; Besson, Cailletaud, Chaboche & Forest 2010, chap. 7),
  normality and the work equivalence on σ - X, Ẋ = 2/3 C ε̇p - D X ṗ with the matrix strain
  rate ṗ, the backstresses with their hydrostatic part (tr ε̇p is the growth of the voids:
  the surface moves along the mean stress too, a hydrostatic Bauschinger effect).
  Rousselier's criterion (its constant σ1: not homogeneous) takes none; MFront 5.1's brick
  refuses kinematic hardening rules with a porosity evolution ("not supported when coupled
  with a porosity evolution"), and so does the MFront export, with that reason. The update
  stays the Newton iteration on four unknowns along the invariants of the relative trial
  stress σ_tr - Σ θ X_n (θ = 1 / (1 + D Δp), the implicit recovery), whose direction Δp
  sets, the moduli 3 μ + c and K + 2c/9 (c = Σ θ C): the analytical Jacobian takes the
  dependence of J*, m* and c on Δp, the consistent tangent the turn of the direction with
  Δp (symmetrized); the backstresses are internal variables of the points after the matrix
  stress (6 each, `porous.state_width`), in small and finite (logarithmic) strains and on
  the GPU (the CUDA port, its rows of parameters 62 wide). Z-mat's `porous_plastic` reads
  and writes the bricks of gen_evp, `*kinematic nonlinear` C D and `*kinematic linear` C
  (named X1, X2...; Z-set itself was not at hand); the form of the tab Mechanics shows the
  terms for porous materials, the identification reaches them (C1, D1...), the NTFA refuses
  porous phases (it reduced them as elastic). Checks: dense, GTN and the elliptic
  criterion are the von Mises kernel with the same terms along a turning cycle (stresses
  within 8·10⁻¹³ of R0, backstresses 1.4·10⁻¹³ of R0, tangents 4·10⁻¹³ of E); with f0 = 0.01
  a tension-compression cycle yields in reverse at 153 MPa after 438 MPa (438 MPa with the
  isotropic hardening of the same saturation); under a hydrostatic strain Prager's
  backstress has the mean 2/9 C tr εp, the mean stress yields on the shifted surface within
  10⁻¹⁵ and hardens (953 MPa against 728 MPa, softening, without it), the voids grow less
  (0.0162 against 0.0175); the tangents are central differences within 2·10⁻⁹ (dP/dF:
  10⁻⁵), the Jacobian and its trial derivatives within 10⁻⁵; the GPU gives the points of
  the CPU within 10⁻¹² (stresses, internal variables) and 10⁻¹⁰ (tangents) of their scales.
* Porous plasticity in the MFront sources: the materials of Gurson-Tvergaard-Needleman are
  written with MFront's brick `StandardElastoViscoPlasticity` and its porous criterion
  `GursonTvergaardNeedleman1982` (q1, q2, q3; the effective porosity f, or Tvergaard &
  Needleman's coalescence as MFront's `f_c` and `f_r`, `f_r` that of morphoGen's slope of
  f* when q3 is not q1², MFront's f* reaching (q1 - sqrt(q1² - q3)) / q3 at `f_r`), the
  porosity in the implicit system of the brick (`standard implicit scheme`, θ = 1:
  morphoGen's backward Euler), the Plastic and Norton flows with linear and Voce
  hardening, the constant nucleation from p = 0 (MFront's power law of exponent 0), in
  small and finite strains (Hencky) and with thermal expansion; the source sets the
  porosity to f0 while p = 0, so that the natural state of its internal variables is zero
  (9, 10 with a nucleation). Refused with their reasons: Rousselier's criterion (MFront's
  `RousselierTanguyBesson2002` is homogeneous of degree 1, the matrix stress in place of
  morphoGen's constant σ1), the elliptic one (none in the brick), Chu & Needleman's
  Gaussian nucleation (MFront integrates the Gaussian exactly over a step, morphoGen by
  backward Euler), the exponential one (none), a threshold of the nucleation (MFront's
  power law integrates from it), other tables of the effective porosity, the coalescence
  with q3 > q1², an anisotropic expansion. Compiled by MFront 5.1 and integrated by MGIS
  along triaxial paths that grow the porosity (from 0.01 to 0.055 in small strains, to
  0.13 in finite strains), the laws follow morphoGen's kernel within 10⁻¹⁰ of the stress
  scale (2 to 7·10⁻¹¹: rate-independent with Voce terms, Norton with the coalescence, the
  nucleation with q3 = 2, expanding, finite strains); AMITEX and Abaqus inputs take the
  porous phases (their internal variables those of MFront's examples), whose laws,
  called as AMITEX (umat) and Abaqus (abaqus, expanding too) call them, follow the kernel
  within 5·10⁻¹¹ (7·10⁻¹¹ expanding, in Abaqus). MFront's porosity is bounded otherwise
  (0.985 of its upper bound, the material broken beyond 0.984 of it; morphoGen's f* below
  0.995 / q1).
* Hill's anisotropic plasticity: the
  plasticity of a material takes Hill's criterion in Z-set's form, `sqrt(3/2 s : M : s)`
  with `M` diagonal on the components of the deviator (`hill: {11: …, 22: …, 12: …}`, 1
  where not given: von Mises), with the hardenings of von Mises (isotropic, Armstrong-
  Frederick, Norton). Its update is the closest point projection (Newton's method on the
  plastic strain increment and `dp`, Mandel's notation, a backtracking line search) with
  its consistent tangent; the finite element and FFT solvers, small and finite strains
  and the fracture tools take it, the GPU too (a CUDA port of the update, compared with the
  Numba one point by point); the NTFA refuses it. The MFront sources of morphoGen's
  materials write it as an orthotropic behaviour in the axes of the sample (MFront's
  criterion `Hill` with TFEL's F ... N), for MGIS, AMITEX_FFTP and Abaqus: in WSL, the
  compiled law, and the law called as AMITEX and Abaqus call it, follow the native
  kernel to 1e-8 along the six components of a path.
  `Plasticity.yield_stresses` gives its yield stresses along the axes and in the shears.
  The identification fits its coefficients (`hill11` … `hill31`, from von Mises's), Z-mat
  files with `*criterion hill` run on the native kernels (MFront's brick only with the
  models that need it) and `write_zmat` writes it; the forms of Mechanics take its six
  coefficients. Checks: von Mises's radial return to 1e-12 with unit coefficients, the
  closed-form yield stresses and Lankford ratio, the tangent against finite differences,
  MFront's Hill to 1e-8, the coefficients identified (theory notes, §54). The example
  `examples/hill_of_a_laminate.py` identifies the law of a laminate of two plastic
  phases on four of its responses (its shear across the layers: `h31 ≈ 2`) and writes
  it for Z-set.
* The macroscopic law of a morphology in one go (`identification.macroscopic_law`; the tool
  "Macroscopic law of a morphology" of the tab Mechanics): its responses under a tension
  along each axis and a shear in each plane chosen (nonlinear FFT on its voxels in the
  interface), a starting law estimated on them (`law_start`: E and ν of the first
  increment, the yield stress and the linear hardening of the curve), then the von Mises
  or Hill law identified on them together (Hill: the coefficients of the loadings run but
  the first; with cycles, the loadings reversed to the opposite strain and a kinematic
  hardening in the law: the Bauschinger effect of the morphology; by FFT on its voxels or
  finite elements on its conforming mesh, in small or finite strains); Gurson's law of a
  porous morphology (the matrix of its phase, its voids as the porosity, q1 and q2 on
  loadings of several triaxialities: `confined11`... a tension whose lateral strains are
  held, `hydrostatic`; `loading_path`; the example `examples/gurson_of_pores.py`); the
  law joins the workspace and is written as a Z-mat file (`.mat` files listed by the
  page). Checks: a Hill material taken as the morphology gives back its nine
  parameters; layers of two plastic phases find their shear across the layers weaker.
* Faster behaviours of the points: the
  updates of Hill's points and of the porous points take their work arrays once per
  block of points instead of at each call; Hill's Newton iteration evaluates its system
  once per iteration (680 ns per point become 408, 200 000 points on 20 threads); at the
  points of the line search, GTN's matrix stress is bracketed from the current point's;
  the Newton iteration of the porous points and their consistent tangent take the
  analytical Jacobian of the four equations instead of differences (the second
  derivatives of the matrix stress: implicit ones for GTN, closed forms for Rousselier
  and the elliptic criterion), checked against central differences: a porous increment
  in the middle of a loading, 2.8 µs per point become 0.5, the same points failing in
  large steps, the stresses within 1e-12 of the former ones. The
  example `examples/behaviour_benchmark.py` measures each behaviour in small and finite
  strains (README, Performance).
* Porous plasticity on the GPU: the FFT solvers on the GPU integrate the porous points
  (GTN, Rousselier, the
  elliptic criterion, small and finite strains) with a CUDA port of their update, its
  inner iteration of the matrix stress and its analytical Jacobian included, instead of
  refusing them (`device="auto"` fell back to the CPU). Point by point the port follows
  the Numba kernel to 1e-12 of the largest stresses and internal variables and 1e-10 of
  the tangents (GTN with its nucleation and coalescence, viscous GTN, Rousselier, the
  elliptic criterion), and the path of a porous matrix around an inclusion that of the
  CPU. The rows of parameters of the GPU are those of the porous points (`ROW_WIDTH`),
  and the porosity of the points comes back to the host at the end of each step.
* Crystal plasticity 1.4 to 1.7 times faster (the CPU and the GPU): the exponentials
  `exp(-b v_r)` of the isotropic hardening, which
  depend on the system `r` and the `b` of the family of `s`, are computed once per family
  at each evaluation of the residual and of its Jacobian instead of once per pair of
  systems (12 instead of 144 for the octahedral systems), the same results to the bit. A
  point of an fcc crystal takes 1.4 µs instead of 2.3 in small strains and 2.6 instead
  of 3.6 in finite strains (200 000 points, 20 threads); on the GPU, whose integration of
  the crystals bounded the paths, the paths of a polycrystal gain 1.6 to 2.8 times instead
  of 1.4 to 1.9. `examples/gpu_benchmark.py` times a porous path too (`--porous`: stiff
  spheres in a GTN matrix pulled to 1 %): 2.4 to 3.5 times faster on the GPU from 32³ to
  64³ voxels, the same stresses, porosities and iterations. Held lateral strains (high
  triaxialities) made the conjugate gradients of some of its Newton iterations stall at
  the symmetrized tangent of the softening points, on both devices (MINRES, below).
* Thermal expansion in the MFront sources of small strains (an open point of the exports
  since 2026-10-23): `thermal_expansion` of the
  Hooke potential of the brick (von Mises; Hill's criterion also takes three principal
  values in the axes of the sample) and `@ComputeThermalExpansion` of the crystals (one
  coefficient or three principal values in the axes of the crystal), with MFront's
  reference temperatures, 293.15 K, the natural temperature of morphoGen's MFront
  behaviours. Compiled and integrated by MGIS, a von Mises cycle, Hill's criterion with
  an orthotropic expansion and a turned cubic crystal with one follow the native kernels
  within 1e-8 while the temperature changes. In finite strains MFront's coefficient is a
  mean one (its Hencky brick stretches by `1 + alpha dT`: `ln(1 + alpha dT)`, measured to
  eight digits), morphoGen's that of the logarithmic strain `alpha dT`: the brick gets the
  mean coefficient `(exp(alpha theta) - 1) / theta` by its series, and von Mises and Hill
  then follow the finite kernels within 1e-11 over 600 K. The crystals and Hencky's
  elasticity of finite strains compute morphoGen's thermal strain in their own sources (a
  turned crystal, isotropic and cubic Hencky elasticity within 1e-8 over 600 K). Refused:
  an expansion tensor with shear components in the axes of the behaviour, an anisotropic
  expansion of a von Mises material, and the expanding phases written as MFront laws in
  the AMITEX_FFTP inputs, which carry no change of temperature. The Abaqus inputs start
  their nodes at 293.15 K when a material expands (`*INITIAL CONDITIONS,
  TYPE=TEMPERATURE`; their steps give absolute temperatures; CalculiX heats an
  isotropic and a turned anisotropic grain with their `*EXPANSION` by 100 K from there
  to morphoGen's stresses): called as Abaqus calls
  them, the expanding laws follow the kernels within 1e-8 in small strains (von Mises,
  Hill, a turned crystal) and within 1e-10 in finite strains (Hencky's elasticity, a
  turned crystal). They refuse the brick's expanding laws of finite strains: MFront's
  abaqus interface has no stress-free expansion in finite strains. The steps of the tool
  Abaqus input of the interface keep the nodes there (`*TEMPERATURE`, 293.15 K, in each:
  its paths have no change of temperature), which CalculiX asks of each step: it stopped
  on the first otherwise ("no final temperature defined").
* The FFT tools of the tab Mechanics (loading paths, the macroscopic law of a morphology,
  the full-field paths of the NTFA) take a device:
  the CPU, the GPU or `auto`; the loading paths keep it in the configuration of the run.
  The GPU integrates every behaviour of morphoGen's kernels now (von Mises, Hill, porous
  plasticity, crystals; MFront points go to the CPU), so that only the Simulate tab used
  it before. On the GPU, a path of a plastic matrix around voids gives the curve of the
  CPU to 1e-8.
* MINRES after the conjugate gradients (on the
  stall found): softening points (a GTN matrix with a weak hardening at high stress
  triaxialities) make the symmetrized tangent indefinite, the conjugate gradients met a
  direction of negative curvature and stopped, their iterate leaving residuals up to 78
  times the right-hand side (`FFT solver stopped at relative residual ...`).
  `solvers.minres` and `gpu.minres` (Paige & Saunders, preconditioned) take over from
  that iterate or from zero; `conjugate_gradient` hands them the system, so that the
  FFT paths (CPU and GPU) and the iterative solvers of the finite elements gain them. On
  8³ voxels of stiff spheres in GTN with a nucleation, under a uniaxial strain, no
  warning, no divided step (two before), 50 Newton iterations instead of 55; on 16³, 84
  instead of 91 with five divisions instead of six, for twice the linear iterations.
  The GPU's solver reports a stalled solve as the CPU's does. The reference medium of
  the preconditioner stays the means of the extreme elastic moduli, the best of five
  tried; the divided steps of such paths come from Newton corrections tens of times the
  converged strains at the softening points (a localization of the damage, which the
  micromorphic damage regularizes), which ten halvings of the line search instead of six
  would cut by one of five for twice the linear iterations (theory notes, §34). The
  import of Z-set files in the interface no longer says that porous laws run on the CPU.
* The tangent `dP/dF` of the points of finite strains (elastic, von Mises, Hill, porous;
  CPU and GPU) takes `F dS` for the six symmetric directions of `dC` and combines them
  for the nine of `dF`, and the divided differences of the logarithm their 6 and 10
  distinct values: 3 to 12 % faster (an elastic point 254 ns instead of 289; Jacobi's
  eigen decomposition and the algebra of the tangent keep most of the cost of the
  logarithmic strain).
* Local CI: the disk caches of Numba of the package are cleared when a kept environment
  takes a new wheel. Numba keys them on the contents of each source file, not on its
  date, so that a kernel whose file did not change kept the compiled code of the kernels
  it calls in other files: the CIs of the faster porous kernels ran the solvers'
  `update_points` and `update_points_finite` with an older porous kernel (the new ones
  had been checked on fresh caches beside them).
* The list of the public interface counts as the data of a module only what it assigns,
  not the constants it imports (80 names left out, which a change of imports would
  have shown as a change of the interface).

* Preparation of version 1.0 (roadmap phase 4),
  without publishing anything: a documentation site built from `docs/` (Sphinx, MyST and
  Furo: the theory notes, Z-mat, the specification and architecture, the README in parts,
  the change log, and an API reference written at each build from the docstrings,
  `pip install ".[docs]"` then `python -m sphinx -b html docs site`; `.readthedocs.yaml`
  for Read the Docs), whose docstrings no longer hold reST errors (`|x|` read as
  substitutions, a malformed table of the models); docstrings for the 44 public functions
  that had none. The public interface (`docs/stability.md`, `morphogen._public`): semantic
  versioning from 1.0, the names of the `__all__` of the packages and of the modules of the
  API reference, the commands, configuration keys and files written; `FutureWarning` for a
  minor version before an incompatible change; `tests/test_public_api.py` keeps the list of
  the public names with the parameters of the functions and classes
  (`tests/public_api.txt`, written again by `python tests/test_public_api.py`). The
  citation file `CITATION.cff`, `docs/citing.md`, a draft paper for JOSS (`paper/`), a
  conda-forge recipe (`packaging/conda-forge/recipe.yaml`), the steps of a release
  (`docs/release.md`: the rights to confirm first, then the version, PyPI, the DOI of
  Zenodo, conda-forge, the site; JOSS six months after the repository is public) and a
  contribution guide (`CONTRIBUTING.md`).
* Fix: the list of the public interface writes a path default as `Path('...')`, the same
  on every system. Kept on Windows, it held `WindowsPath('morphogen_output')` (the
  directory of `OutputSpec`), and the test of the public interface failed on Linux and
  macOS, where the default reads `PosixPath`.
* Tests in parallel: `pytest-xdist` in the extra `test` (`python -m pytest -n auto`), the
  test processes each on their share of the cores (`tests/conftest.py`;
  `MORPHOGEN_TEST_THREADS`), the longest files first. The local CI
  (`tools/ci_local.sh`) runs the Python versions at once, the tests of each in parallel
  (`--dist loadfile`, `CI_LOCAL_WORKERS` processes per version), builds the documentation
  site with its warnings as errors and lints `docs/`; its environments are kept from one run
  to the next while `pyproject.toml` and the extra packages do not change, for a week at
  most (`CI_LOCAL_FRESH=1` makes new ones). The tests of one Python take about 10 minutes
  on 10 processes of the development machine instead of about 50 in one. The GitHub workflow
  runs its tests in parallel and builds the site.
* Two tiers of tests: the slowest (marker `slow`, the numerical
  checks of more than about 15 s, two thirds of the time) run under one Python version in
  the CIs, the last given to `tools/ci_local.sh` (`CI_LOCAL_SLOW`; it gets three times
  the cores of each other version) and the job Ubuntu with Python 3.14 of the GitHub
  workflow; the others take the fast tier (`-m "not slow"`, also a quick check in
  development). The tests of the interface run on coarser meshes, fewer load steps and
  smaller images where the tests of the library check the same physics (the growth of
  cracks through morphologies, laminates and polycrystals, the ductile damage ahead of a
  notch, the transient diffusion), their assertions unchanged: those seven tests take a
  third less time.
* The GPU kernels compile after Gmsh has opened a file in the same process (Windows): the
  probe of the GPU keeps NVRTC's library of builtins loaded, which NVRTC otherwise opened
  by its name at each compilation, in vain once Gmsh had narrowed the search path of the
  libraries (the tests of the GPU failed after those of Gmsh in the same process, as the
  parallel runs may order them).

* Mathematical morphology (`analysis.random_sets`, `analysis.topology`), after the
  appendix of J. Dirrenberger's thesis: Choquet capacities `T(K)` of balls,
  segments and pairs of points along grid directions and cubes (`choquet_capacity`, on
  the torus or by minus sampling), dilations, erosions, openings and closings by them,
  spherical and linear contact distributions; the morphological covariance
  `E[min(Z(x), Z(x + h))]` of grey-level images; the integral of mean curvature from
  the Euler characteristic of plane sections (`Minkowski.mean_curvature`), the four
  Minkowski densities of the excursion sets of a grey-level image
  (`minkowski_curves`); Betti numbers and connectivity density; measures `minkowski`
  and `betti`. Closed forms: `choquet_capacity` of Boolean spheres and Poisson fibres,
  Miles' `M_V` and `χ_V` of Boolean spheres, Adler's `M_V` and `χ_V` of Gaussian
  excursions (theory notes, §39).
* The mean intercept length and its fabric tensor (`analysis.anisotropy`:
  `mean_intercept_length`, `fabric_tensor`, `Fabric` with the principal MILs, their axes,
  ratios and degree of anisotropy; measure `anisotropy`), the anisotropy descriptor of
  bone morphometry (BoneJ) and of the copper foam of Degdeg & Dirrenberger (§39); the
  first minima of the covariance along the axes (`covariance_minima`, `HoleEffect`,
  measure `covariance_minima`), their second measure of the aspect of the cells.
* The camera of the 3D views and videos (`elevation`, `azimuth`) in `viz.video` and in the
  visualization section.
* Remeshed crack propagation in inelastic solids (`fracture.ductile`): the state of the
  points (internal variables, the dissipated part of their
  work, stresses) and the displacement transferred between the meshes of successive cracks,
  by the nearest element or a projection on the nodes, the lips apart; the equilibrium of
  the transferred state regained at the load of the transfer, dissipating nothing
  (`CohesiveProblem.run(initial=, start=)`, `solve_crack(initial=)`); ductile growth by the
  driving force at the tip or by the porosity of GTN points near it (the local approach),
  the direction of kinking, and the resistance curve (`propagate_ductile`).
* Crack growth in remeshed polycrystals (`crack_mesh(grains=...)`, `with_grains`): the
  crack meshes carry the grains of a tessellation and their
  orientations, the state of the points moves grain by grain (located and projected within
  each grain: the slips and `Fp⁻¹` of a grain never mixed with a neighbour's), the kinking
  by the isotropic Voigt-Reuss-Hill estimates of a crystal; the elastic bulk of small
  strains turns its crystals by their grains (a polycrystal's `G` was 0.314 against 0.588
  by the points). A crack grows through twelve columnar grains of crystal plasticity in
  small and finite strains, deflected by up to 11° near a grain boundary.
* Cracks in inelastic solids, in small and finite strains (`fracture.bulk`): the
  behaviours of morphoGen's points as the bulk of the cohesive and crack
  solvers (von Mises elasto-viscoplasticity, porous plasticity for ductile damage: GTN,
  Rousselier, crystal plasticity with the orientations of the grains, which
  `insert_interfaces` keeps, MFront laws; `bulk="points"`, automatic
  for inelastic phases and anisotropic ones in finite strains), the works of the loads,
  the bulk and the interfaces, the time of the viscous laws (`duration`); the driving
  forces of dissipative solids (`release_rate(energy="work" | "free" | "tip")`): the J of
  the deformation theory, within 3 % of the release of the work of the loads, and the
  driving force at the tip with the configurational forces of the internal variables,
  independent of the domain within 1 %; cohesive cracks in elasto-viscoplastic plates (the
  R-curve of bulk plasticity) and the intergranular cracking of polycrystals in crystal
  plasticity.
* Porous elasto-(visco)plasticity (`homogenization.porous`): Gurson-Tvergaard-Needleman
  (effective
  porosity of a table, Tvergaard and Needleman's coalescence), Rousselier and its modified
  criterion, Abouaf's elliptic criterion of compaction (C and F tables of the porosity);
  Norton or rate-independent flow of the matrix with its isotropic hardening, the growth
  of the porosity and Chu and Needleman's nucleation; implicit integration along the
  invariants (Aravas) with a consistent tangent, in small and finite strains, on the CPU
  (material points, finite elements, FFT; the porosity in the curves, phases and fields);
  Z-mat's `porous_plastic` files translated natively. Checked against GTN's hydrostatic
  yield (10⁻¹²), the von Mises limit, the closed form of nucleation, the yield of
  compaction along the densification (10⁻⁸), the tangents and a superposed rotation.
* Micromorphic continua (`morphogen.micromorphic`): mixed
  tetrahedra (quadratic displacements, linear generalized degrees of freedom, node-wise
  sparse assembly, Newton's method with an optional reuse of the factors); the microfoam
  model of Dillard, Forest & Ienny (2006: the anisotropic compressible plasticity of
  nickel foams with its parameters, a full micromorphic microdeformation, small strains),
  reproducing the article's plastic contraction ratios, the one-dimensional
  characteristic lengths of its Appendix A and its hole size effect (stress concentration
  1.25 at a hole of 1 mm, 1.2 in the article); a micromorphic damage model (the implicit
  gradient model of Peerlings et al. as a scalar micromorphic variable, small and finite
  strains) whose bands of damage and softening curves do not depend on the mesh;
  `examples/microfoam.py`.
* Diagonal mirrors in the mirrored conforming meshes (`mirrored_mesh(diagonals=...)`,
  `point_group`): the groups of the square (4/mmm) and of the cube (m-3m), whose chamber
  (a union of background tetrahedra) is meshed by MMG and mapped by every operation;
  invariant to 10⁻¹⁶, the whole groups detected (64 operations for a 2 × 2 cell of holes,
  384 for a 2 × 2 × 2 cubic cell). Square arrays of holes in equibiaxial compression keep
  the 4-fold rotation: their first bifurcation, the alternating ellipses in a
  one-dimensional irrep, at 4.1 % on each axis against 9.3 % in uniaxial stress; cubic
  cells of spherical holes buckle linearly in three-dimensional irreps.
* Cells of several primitive cells split by their translations in the post-bifurcation
  (`post_bifurcation(bloch=..., patterns=...)`, `wave_vectors`, `Bifurcation.wave_vectors`):
  the wave vectors (Γ, X, M, R) of the characters of the translations, the sparse bases
  of their fluctuations, the little co-groups of their stars and the irreps they induce
  (Clifford's theory, Frobenius reciprocity), solved one block at a time without dense
  bases of the whole cell; the patterns of the critical irreps in the coordinates of the
  arms of their stars. The loads, multiplicities and patterns of the dense bases within
  2·10⁻¹⁴ on a lattice of bars of 2 × 2 × 2; the cubic cell of spherical holes of
  2 × 2 × 2 (384 operations, 60 696 dofs) in 4 min: under hydrostatic compression it
  buckles first at 7.55 % on each axis in the star of M, doubling the period along two
  axes (8.15 % for the single cell); in uniaxial strain, at 15.9 % in the wave vector M
  normal to the load (16.5 % for the single cell).
* Micromorphic continua in finite strains and ductile damage (`morphogen.micromorphic`): the
  microfoam model in finite strains (`kinematics="finite"`, `micromorphic.finite`: total
  Lagrangian, the compressible plasticity in the logarithmic strains, the objective
  measures `χ⁻¹ F` and `χ⁻¹ Grad χ` of Forest & Sievert with the moduli of small strains;
  tangents within 2·10⁻⁹ of finite differences, forces turned by a superposed rotation
  within 5·10⁻¹⁵, the forces of small strains within 7·10⁻⁷); the nickel foam compressed
  to half its height densifies to 0.656 of its volume along TD and 0.962 along RD;
  micromorphic ductile damage (`MicromorphicDuctile`, `DuctileProblem`: von Mises
  elasto-(visco)plasticity of the effective stress with linear and Voce hardening and
  Norton's overstress, its damage driven by the micromorphic cumulated plastic strain
  `χ - ℓ²Δχ = p`, after Engelen, Geers & Baaijens; small and finite strains), whose ductile
  bands and softening curves do not depend on the mesh (within 5 % and 3.4 %) and which
  cracks a bar (`d → d_max`) and a notched plate (two bands at about 45° from the notch
  across the ligament, the curves of two meshes within 2.2 %); the `predictor` of the
  Newton solver (the free degrees of freedom of a step from the last tangent) and the
  `duration` of viscous paths.
* The graphical interface opens every model with its example configuration (the
  parameters and the domain of its template), instead of empty forms that the models
  with a choice of parameters (`tpms`: `level` or `volume_fraction`; `hardcore_spheres`:
  `volume_fraction` or `number`) refused; every model generates from it.
* Fix: the forms of the interface read numbers typed in a JSON field ("0.012, 0.012")
  as a list; the example tube radii of `hierarchical_lattice` went to the model as text,
  which it refused.
* Fix: the page shows the images that tools give as data URLs as they are. It ended every
  figure of a result by `?inline`, which broke these URLs: the sections of the morphologies
  made by Reconstruction (Statistics) and Automaton rule (Inverse design) were broken
  images.
* The graphical interface reorganized as a workflow: a workspace of the session keeps
  every morphology, simulation and optimized design
  under a key, and each tab takes its input from it — Generate, then Simulate (a
  morphology: its conforming mesh of its level set, or its voxels), then Export (the
  figures, videos and files of a morphology or a simulation, drawn on demand, with an
  archive of them all), and Optimize (from a void ball, noise, a morphology or a
  simulation, the design joining the workspace as a morphology whose level set is the
  densities about 1/2). The views of the morphologies are sections of their level sets
  (`morphogen.viz.sections`: anti-aliased, up to 2048 pixels per edge without a voxel
  image of the volume), with the measures of their level sets; voxel images are made on
  demand, up to 2048 voxels per edge within the free memory. The formation of the
  models grown by a process is shown in Generate with the share of the cells changed at
  each step; "Design space" became "Tile assemblies", with what it computes. The videos
  of a growing or oscillating load moved the mesh but kept the colours of the full load:
  the fields now follow the load (`field_scaling`: the components linearly, norms by
  `|s|`, energies by `s²`). `CellOptimization(initial=...)` takes the densities of a
  microstructure (`initial_density`). A generated morphology is the realization 0 of a
  run with its seed, so that the configuration of its simulation reproduces it.
* Sections computed on their planes (`morphogen.viz.sections`): a grid one cell thick
  whose cell centres are those of the slice of the voxel grid of the same pixels gives
  the phase, the level set, the anti-aliased coverage of phase 1 (`clip(1/2 - φ/(|D_u φ|
  + |D_v φ|), 0, 1)`) and the labels at up to 2048 x 2048 pixels without the volume
  (0.2 to 1.4 s for a TPMS, hard-core spheres or a Voronoi tessellation), every model
  of the registry sampled on the plane; `level_set_measures`: the volume fraction by a
  smoothed Heaviside and the specific surface by marching cubes of the level set (a
  sphere within 0.03 %). The separable synthesis of Gaussian fields contracts the axes
  of fewest voxels first (a plane of 2048² pixels: 81 MiB instead of 1 GiB).
* The interface: tools built from a registry (`morphogen.gui.tools`: forms of typed
  fields, conditional fields, items of the workspace and the materials of their phases,
  jobs with their progress; a module of tools appears by itself), tabs without numbers
  in the order of the work, any of them taking the items of the workspace made by any
  other; the tab Statistics: RVE sizes by the ensemble or the window method (volume
  fraction or apparent moduli, the tables and figures of the report), morphological
  measures against the theory of the model, correlation functions and distributions
  with the closed forms of the model. The warnings of the libraries (deprecations while
  Numba compiles) are no longer shown with the results.
* The tabs Mechanics (loading paths of a morphology with every behaviour of morphoGen,
  in small or finite strains, by FFT or finite elements; the NTFA; a material point; the
  linear tensors and their bounds), Fracture (stress intensity factors, crack growth,
  cohesive zones, cracks in polycrystals) and Solvers & formats (AMITEX_FFTP, Abaqus,
  Gmsh, Medit, FEniCSx, Neper, DAMASK, TIFF and MetaImage files, MFront sources, Z-set
  import); Statistics adds stereology (Saltykov's unfolding); every tool returns an
  archive of its files; the export draws every field of a simulation (plastic strains,
  porosity); the simulations of the session go to a results database for the Ashby
  charts.
* The tabs Nonlocal damage (microfoam, micromorphic ductile damage with the triaxiality,
  micromorphic damage of quasi-brittle solids, on notched plates, bars or morphologies)
  and Couplings (transient diffusion into a morphology, thermal oxidation of an epoxy
  plate, hygromorph multilayers whose layers are homogenized from morphologies).
* The menus of the interface regrouped in four sections with their tabs (Create,
  Characterize, Simulate, Export & compare; `tab(section=...)` for the modules of
  tools); the items sent on by the buttons of their usual next steps or a menu of the
  other tools by section, which sets the forms to take them (the specimen of a crack or
  of nonlocal damage switched to the morphology sent); the items of the workspace
  removed from it; the help panel by section. The item sent to a tab follows the tools
  chosen in it; compact forms (short fields two by two, their explanations on one line,
  the long explanations of the tools and models shortened, the rest on demand); the
  long forms in boxes (`tool(groups=...)`: specimen, material, loading, solver, output,
  the details seldom changed folded; a box whose fields are all hidden is hidden). Each
  run of a tool joins the workspace as a result (`GET /api/result`), titled by the items
  it took, shown again in its tab with its form filled with its parameters (the strip
  of the workspace showing the morphologies and simulations first, then the four
  newest runs, the earlier ones on demand), saved and
  opened with the workspace (the items it took renamed with it); the tab shown is kept
  in the address of the page; the forms and the menus that send items on offer only the
  items a tool takes (`Field(needs=...)`, the traits of the items: the realization of a
  model for RVE studies, a tessellation for the cracks of polycrystals); the models of
  Generate in a menu by family
  (`MODEL_FAMILIES`: random sets and particles, fibres and layers, tessellations and
  foams, lattices and TPMS, bio-inspired, growth and patterns, assemblies).
* The long computations of the interface stopped from the page (**Stop** beside their
  progress, `POST /api/job/stop`) at their next checkpoint (`morphogen.stopping`:
  `stoppable`, `checkpoint`, `Stopped`, a `BaseException` that the fallbacks on another
  method let through): each iteration of the conjugate gradients (CPU and GPU), each
  property and loading of an apparent homogenization, the end of the mesh of a
  realization, each realization of a run (those its processes have not started
  cancelled), each progress a tool reports. A stopped tool or simulation keeps nothing in the workspace, a stopped
  optimization the design of its last iteration (`CellOptimization.run`: a callback
  returning True stops it there). A link to a tab (`#fracture`) shows it.
* Fix: the number of an item removed from the workspace of the interface was given again
  to the next item (two #3).
* The behaviours of materials kept in the workspace of the interface (B): a material
  point keeps its behaviour
  (calibrated on the points of a test), an import of Z-set the behaviour it translates,
  as the configuration of its material with its own elasticity (its MFront source in its
  folder, saved and opened with the workspace; `api.behaviour_config`,
  `behaviour_material`, `behaviour_summary`); the loading paths of morphologies, the
  NTFA (von Mises phases), the material point, the bulk of a cracked plate or one phase of
  a cracked morphology (inelastic phases of morphologies in the tab Fracture), the inputs
  of AMITEX_FFTP and Abaqus (for one phase) and the MFront sources take it ("a behaviour
  of the workspace", or an optional field: `Field(none=...)`); a behaviour is sent on to
  Mechanics and Fracture from its menu, shown again in its tab, and to the tools whose
  input it is (the MFront sources, its Z-set file) before those where it is an option of
  one phase (the inputs of AMITEX_FFTP and Abaqus). The material of the
  fracture forms asks its behaviour first, its E and ν only when they are used.
* The tab Comparisons of the interface (Export & compare): the curves of several runs of
  the workspace on one figure (`compare_curves`: loading paths of morphologies and
  material points, the strain component that varies most or one chosen, or the time,
  against a stress, the plastic strain or the porosity, true measures in finite strains),
  with the points of a test and the gap of each run to them, their final and extreme
  values in a table; the forms take several items (`Field("items")`, check boxes of the
  items of the workspace with the traits needed: `curves`, the runs of a loading path or
  a material point; a run sent to it is checked among the others), every item has its
  traits, and a run is sent on to the tools that take runs. A tool of the same name as
  another's is refused (the comparison first shadowed the correlation functions).
* The items of the workspace of the interface renamed from their menu (they keep their
  number; `POST /api/workspace/rename`).
* The materials of morphoGen written as material files of Z-mat
  (`homogenization.zmat.write_zmat`), read back as the same materials: `linear_elastic`
  (isotropic, cubic or anisotropic elasticity, each component named) or `gen_evp` of one
  von Mises potential (Norton or rate-independent flow, the isotropic hardening
  `constant`, `linear`, `nonlinear`, `linear_nonlinear` or `nonlinear_sum`, one brick per
  kinematic term) with an isotropic thermal strain, or `porous_plastic` (Gurson with the
  table of its coalescence, Rousselier or modified Rousselier, the elliptic criterion
  with tables, the nucleation, the initial porosity in `*initialize_variable`; an
  identification of `q1` lets `q3 = q1²` follow it), the numbers in full (or to
  `digits` significant digits); in the tab
  Solvers & formats, "Z-set file of a behaviour" writes a behaviour of the workspace (one
  identified on the response of a morphology goes back to Z-set). The coverage of Z-mat
  lists its `****optimize` as done elsewhere (the identification).
* Identification of the parameters of a behaviour on reference responses
  (`homogenization.identification.identify`, as Zopt does for Zsim): the elastic
  constants and the parameters of von Mises plasticity (`yield`, `hardening`, `Q`, `b`,
  `C`, `D`, `K`, `n`, and the next terms `Q2`...) such that the material point under the
  path of the reference gives its stresses over time in the least-squares sense (the
  logarithms of the parameters within `span` decades of their starting values, SciPy's
  trust-region method, a stop between material points); several references followed
  together (`identify_references` with `Reference`s), each by its weight whatever its
  number of points: a tension and a cycle in shear tell the isotropic hardening from the
  kinematic one. In the tab Mechanics ("Identification of a behaviour"), on the curves of
  runs of the workspace, each under its own path (the macroscopic law of a morphology),
  with their weights, or on the points of a test along the path of the form; one panel
  per reference, the behaviour identified joining the workspace (theory notes, §53).
  Porous behaviours too: the parameters of their matrix, the initial `porosity`,
  Gurson's `q1`, `q2`, `q3`, the coalescence `fc`, `fF`, Rousselier's `sigma1`, `D`, the
  elliptic `C`, `F` and the nucleation (the Gurson model of a porous morphology under
  several triaxialities, `f0` held at the volume fraction of the voids: freed, it trades
  off against `q1`); the form starts from von Mises or porous values, or from a behaviour
  of the workspace, and compares the stress components of each run that vary (the lateral
  stresses of a uniaxial strain tell the triaxiality).
* A tool of the interface run on several items one after another ("Run on several items"
  below its form: its parameters for each morphology or behaviour checked, each run kept
  in the workspace, the others not run once one is stopped or fails), and their curves
  compared in one click.
* The 3D view of the items of the workspace in the interface (`gui/viewer.js`: WebGL 2,
  no library, the page working offline; "View in 3D" in the menu of each item, below the
  results of Generate and Linear homogenization): orbit, pan and zoom, a section plane
  along x, y or z filled with the phases or the field inside, the box, the phases shown
  or hidden, colour maps and scales of the fields, the magnified displacement. Its
  surfaces (`morphogen.viz.surfaces`, `GET /api/surface`): the marching cubes of the
  level set cut by the box, the exact facets of Voronoi and Laguerre polycrystals, the
  faces of voxels of cells, grains and FFT results, the faces of FE meshes with their
  fields (element fields averaged at the nodes on each side of an interface) and
  displacements; at most 300 000 triangles (the finest grid within the budget, vertex
  clustering for large meshes); the area of the surface of hard-core spheres within 0.2 %
  of the measures of the level set. `io.vtk.read_vti(names=)` reads the arrays asked; the results of tools with a mesh of fields (cracked
  bodies, damaged specimens, bifurcated patterns) drawn too, deformed; the colour scale of
  a field from its first to its last percentile by default, so that a singular peak (a
  crack tip) does not darken the rest.
* Fix: the 3D view of a run of Cohesive zones took the wedges of its interfaces
  (`cohesive_interfaces.vtu`, the last of its meshes by name) and refused them, not a
  mesh of tetrahedra; the meshes of a result leave out the cohesive interfaces
  (`*_interfaces.vtu`) as they leave out the meshes alone, and the view draws the bulk.
* Cracks through any morphology: the crack meshes carry its phases with their materials and the
  sides of the crack apart (`crack_mesh(morphology=, voids=, conforming=)`,
  `TetMesh.side`, `TetMesh.select`), as staircases (the phase at the centroid of each
  element) or conforming to the level set of a two-phase morphology (`phase_mesh`: MMG
  cuts the grid along it once, then each crack cuts that mesh and remeshes it in its
  multi-material mode; `cut_mesh`, `with_phases`, `phases_at`), its void phases removed
  (`without_voids`; small islands dropped, a body broken apart refused); `solve_crack`,
  `propagate_kinked` and `propagate_ductile` on them with a material per phase (elastic
  or the inelastic points, small and finite strains), the state transferred material by
  material (grain and material together in polycrystals), the fronts run through the
  pores they reach (`through_voids`). Checked by equal materials (the homogeneous plate
  to 10⁻⁹; K_I within 0.17 % on conforming meshes) and the kinking towards a pore
  (+14°, Kirsch's estimate +15.7°) and a soft cylinder, away from a stiff one, at three
  mesh sizes (theory notes, §51). The tab Fracture grows cracks through the morphologies
  of the workspace (E and ν or void for each phase, a slab in plane strain, conforming
  interfaces; the phase at each tip, the path over the phases), its factors take pores
  and conforming interfaces, its cohesive zones leave the void phases out.
* The toughness of the phases and of their interfaces in crack growth
  (`fracture.toughness`: `Toughness(phases, interfaces)`): each advance is the candidate
  of the largest G(θ)/Γ(θ), the kink of the maximum hoop stress in the bulk (the toughest
  solid phase it enters) or a chord along an interface met within an increment (both
  senses within 90° of the crack; junctions of interfaces stopped at and branched),
  G(θ) by the first-order kinked factors of Cotterell & Rice; the critical load of each
  crack at a fixed load, the advance once the largest ratio reaches 1 by load steps (J_tip
  against the toughness of the phase in inelastic bulks); `propagate_kinked(toughness=)`,
  `propagate_ductile(toughness=)` with a dict or a `Toughness` (a number keeps today's
  growth exactly). Checked by He & Hutchinson's deflection of a crack normal to an
  interface (deflected at Γ_i/Γ = 0.15, through at 0.4; trial cracks on their own meshes
  give G_d/G_p = 0.244 for short kinks), a tough disc doubling the load to enter it, and
  staircase paths through layers and brick-and-mortar structures with weak boundaries
  (theory notes, §51). The tab Fracture takes a toughness per phase and an interface
  toughness; the path figure marks the deflections and a figure gives the critical loads.
* Fix: the side of a point nearest a kink of a kinked crack came from the normal of one
  of its segments, right for kinks below 90° only: a crack deflected by 90° after a
  slightly tilted segment was opened through the body a second time (area 26.2 for 6.75,
  G = -240); it now comes from the sum of the normals of the two segments.
* Fix: the material at the front of a crack was that of the element whose centroid is
  nearest the front, an inclusion near a refined interface (K_I four times too large in
  one case); it is now that of the elements containing the front. The interaction
  integrals and J_tip leave out the other materials and the pore walls beyond one element
  from the front (the inhomogeneity term of Kolednik et al. is not included); the far J
  keeps the whole domain.
* The workspace of the interface saved to a folder and opened in another session (its
  morphologies with their microstructures, its simulations and results with their files,
  their links renamed); the video of the formation of a structure in the export.
* The tab Design (the automaton rule of a morphology by a genetic search; the rigidity
  of strut lattices along their dilution, after Maxwell and Calladine); Statistics adds
  the RVE fit of pasted tables of results and reconstructions from the two-point
  function; FFT simulations on the GPU from the tab Simulate.
* The tab Stability: post-bifurcation of lattice frames and of morphologies (mirrored
  conforming meshes or voxels; irreps, critical loads, patterns, wave vectors, the
  Bloch-wave stability), bifurcation along nonlinear paths with the post-bifurcated
  branch (the square array of holes: 6.08 % and its alternating ellipses), and the
  Bloch-wave band structures of morphologies with their gaps.
* Fix: the periodic conditions of porous meshes whose opposite faces differ (voxel
  meshes of a pore phase) tied the nodes of an upper face facing a pore of the lower one
  to its nearest element: the cell was too stiff (a lattice of bars shifted by two voxels
  lost 1.9 % of its bulk modulus) and its stiffness did not commute with its symmetries
  (the post-bifurcation analyses stopped). Such nodes now stay free, merged with a node
  at the same place through an edge or corner of the box. Results of homogenization,
  Bloch bands and bifurcations of such meshes change accordingly.
* Fix: the conforming meshes of porous media kept phase 1, assumed solid, so that pores
  of phase 1 in a matrix were meshed alone; `conforming_mesh(phases="matrix")` keeps
  phase 0, chosen by the configurations when the void is phase 1.
* The triaxiality in the micromorphic ductile damage (`MicromorphicDuctile`): the damage
  driven by the plastic strain
  weighted by the stress triaxiality, `ė = w(η) ṗ` with `w(1/3) = 1`, after Rice & Tracey
  (`triaxiality="rice_tracey"`), Johnson & Cook (`"johnson_cook"`) or Lemaitre
  (`"lemaitre"`), `"none"` by default; the consistent tangents with `∂η/∂ε` (within 3·10⁻⁸
  of finite differences), small and finite strains. A notched plate in plane strain damages
  ahead of the notch and breaks across the ligament (mode I), where the plastic strain alone
  spreads the damage and leaves two thirds of the force; the force curves of two meshes
  within 2 % (`examples/ductile_triaxiality.py`).
* Conforming meshes of symmetric cells (`mesh.mirrored_mesh`): the irreducible part of a cell
  symmetric about its mid-planes meshed by MMG conforming to its level set, then mirrored,
  so that the mesh has the symmetries exactly (2·10⁻¹⁶) and matching opposite faces;
  copies side by side for cells of several primitive cells; `conforming_mesh(periodic=(...))`
  with matching faces along chosen axes; `ImplicitBody`, a microstructure of any level-set
  function. The post-bifurcation analyses detect the groups of these meshes exactly (32
  operations for a 2 × 2 array of holes, none on an unstructured mesh of the whole box):
  Bertoldi's specimen 1 in uniaxial strain bifurcates at 4.60, 4.18, 3.35 and 2.99 % as the
  elements shrink from 0.08 to 0.04 of the cell, towards 2.3–2.6 %, against 2.25 % for
  Sperling et al. (the voxel value of 1.47 % had thin staircase ligaments and the lateral
  stress free).
* Mixed-mode cracks in 3D, in small and finite strains (`fracture.mixed`): the G-θ method
  (Eshelby's tensor of the reference configuration, exact for a
  neo-Hookean body; the mean or a profile along the front), the factors of modes I, II
  and III by interaction integrals (objective in finite strains; the tractions of lips in
  contact; the curvature terms of curved fronts), the frictionless contact of the lips
  (`Contact`), kinked cracks remeshed by MMG with their front on edges (`snap_front`),
  kinking by the largest hoop stress and fatigue propagation (`propagate_kinked`); dead
  loads and a value per node of the prescribed displacements in the cohesive solver.
  Checked against the handbook (mode I, 2.4 % at a/h = 18), the antiplane strip (K_III
  within 0.9 %), a penny crack in shear (Kassir & Sih: K_II and K_III along the front
  within 4-7 %), the energy of the body, the superposition of the modes, a superposed
  rotation (10⁻⁶) and the local symmetry of kinked paths.
* Crack propagation in 3D (`fracture.lefm`): planar cracks by
  level sets (straight and circular fronts); conforming meshes remeshed by MMG at each
  step with the crack faces opened (as Z-cracks); XFEM on linear tetrahedra (shifted jump
  and branch enrichments, cut elements integrated on sub-tetrahedra); the energy release
  rate by compliance; propagation by Paris' law or Griffith's criterion; an edge crack
  within 2-4 % of the handbook by XFEM.
* Cohesive zone elements (`morphogen.fracture`): bilinear
  mixed-mode (Benzeggagh-Kenane) and exponential (Ortiz-Pandolfi) laws, irreversible,
  with contact; interface elements between linear tetrahedra in small and finite strains
  (current frame, neo-Hookean bulk); interfaces inserted between phases or along chosen
  faces by duplicated nodes; a quasi-static Newton solver with viscous regularization;
  calibration (cohesive zone length, Turon's strength) and the double cantilever beam
  against the corrected beam theory (Linke & Lammering 2023).
* Z-mat material files (the behaviour bricks of Z-set): `homogenization.zmat` reads the
  gen_evp and linear_elastic behaviours of Z-mat and translates them to morphoGen's kernels
  (von Mises elasto-(visco)plasticity) or to MFront's brick StandardElastoViscoPlasticity
  (Hill's criterion in TFEL's convention, user-defined viscoplastic flows, power-law
  hardening, softening, Burlet-Cailletaud, several potentials); materials `{zmat: file}`,
  `morphogen zmat`, the coverage of Z-mat's library (docs/zmat.md); checked against MFront
  (MGIS) and Hill's analytical yield stresses in tension and shear.
* Post-bifurcation along nonlinear paths in finite strains (`path_bifurcation`): the
  periodic element followed along a loading path with its nonlinear
  behaviours (Hencky elasticity, von Mises elasto-(visco)plasticity, crystals, MFront),
  the consistent tangent of each converged step block-diagonalized in the isotypic
  subspaces of the group of the cell restricted to the path (the macroscopic components
  of prescribed stress condensed, the mechanisms of the natural state left out), the
  first block that loses its positive definiteness (time, irrep, multiplicity, the
  softest patterns of its axial classes) and the switch to the bifurcated branch by an
  equivariant imperfection; a `monitor` of the converged steps in
  `NonlinearFEProblem.run`. Checked: slender columns tend to their linear buckling
  (within 0.7 % at a critical strain of 0.25 %), a square array of holes turns into
  alternating ellipses, with the plateau and the negative Poisson's ratio of its
  bifurcated branch, an elastoplastic column bifurcates at its yield.
* Post-bifurcation in 3D, after Azulay: frames of struts in space (`SpaceFrame`: cubic,
  body-centred cubic, octet and Kelvin lattices of 3D Euler-Bernoulli beams with torsion;
  stiffness, axial forces, buckling, Bloch waves, apparent stiffness checked against the
  scaling laws of lattice materials, relative density), analysed by `post_bifurcation`
  with their O_h symmetry and centring translations; patterns drawn and animated in 3D;
  Schoenflies symbols of the point groups in space.
* Graphical interface: a tab "Simulate & export" runs the whole chain on one realization
  from a form (model, mesh or voxels, materials, apparent modulus, views of the fields on
  the deformed mesh, a video, every file and the configuration to download), and redraws
  the view and the video without solving again; diverging colormaps centred on zero.
* The whole chain in one configuration (2026-10-24): a
  `visualization` section draws figures and videos of the first realizations (sections
  and orthoslices of the image, views, turntables and sweeps of the mesh coloured by
  phase, the fields of properties, loadings and loading paths on the deformed mesh or
  the voxels, curves), with colour bars drawn beside the frames; `outputs.tables` writes
  the tables as `.txt` and `.xlsx`; loading paths keep their fields at every step
  (`fields="steps"`); the dependencies are checked before a run; the template
  `workflow`; `morphogen export` converts images, meshes with their fields, geometries
  and tables to other formats, figures and videos.
* Exports and videos (2026-10-24): `morphogen.export(obj, path)`
  chooses the writer from the object and the suffix (images, meshes, microstructures,
  band structures, bifurcation analyses, loading paths, frames, tables, figures);
  tables as `.csv`, `.txt` and `.xlsx` (`io.tables`, no dependency); figures as
  bitmaps, vector graphics and TikZ (`viz.export.tikz`: pgfplots code of matplotlib
  figures); videos (`viz.video`: `.mp4` by ffmpeg, `.gif`, PNG frames; once, loop,
  bounce) of sections, orthoslices, VTK turntables, cross-section sweeps and fields
  with a magnified displacement.
* Fix: on a machine without a display, EGL or OSMesa, VTK aborted the process (a
  segmentation fault) at the first 3D view. A subprocess now checks that VTK can render
  (`viz.video.vtk_can_render`): the workflow names the missing display before a run, a
  scene raises an error, and the tests of the 3D views skip.
* The post-bifurcation analysis of continua:
  `post_bifurcation` of tetrahedral meshes and voxel images, the initial-stress matrix
  of tetrahedra under the periodic prestress (`initial_stress_matrix`), the symmetries
  detected on meshes (`mesh_symmetries`), mechanisms left out, VTK output of the
  patterns; isotypic bases without dense projectors and the group and its fixed-point
  subspaces computed by arrays (a group of 384 elements in seconds). A column agrees
  with Timoshenko's load within 3 to 6 %.
* Post-bifurcated patterns by group theory, in the manner of R. Azulay:
  `morphogen.symmetry` (symmetry groups of periodic cells, real irreps from the
  regular representation, the axial isotropy subgroups of the equivariant branching
  lemma), `homogenization.frames` (periodic plane frames of Euler-Bernoulli beams, axial
  forces of a macroscopic strain, linear buckling, the Bloch-wave stability that selects
  the cell) and `homogenization.bifurcation`
  (`post_bifurcation`: the patterns that minimize the elastic energy in their fixed-point
  and isotypic subspaces, the first bifurcation of a loading and its irrep, figures). The
  honeycomb reproduces Table 3 of Azulay & Combescure and the triple first bifurcation in
  T2 of the thesis (§45).
* Bloch waves in periodic cells (`homogenization.bloch`: `band_structure`,
  `BandStructure` with its band gaps and figure, `brillouin_path`, `voxel_tet_mesh`,
  `mass_matrix`): the frequencies of the elastic waves of tetrahedral meshes or voxel
  images along the edges of the Brillouin zone, within 0.6 % of the folded acoustic
  branches of a homogeneous cell and 0.4 % of Rytov's bands of a bilayer (§44).
* Seeds of level-set foams calibrated on sections (`models.calibrate_seeds`,
  `SeedCalibration`): the scale of the seeds, and with `fit_shape` their shape factors,
  until the mean radii of the sections of the pores match measured ones (Gauss–Newton
  steps with the Jacobian of ellipsoids), the pores being larger than their seeds (§42).
* Stereology (`analysis.stereology`), the chain of Degdeg & Dirrenberger: the sections
  of a phase by the planes of the grid (`section_sizes`: periodic, or with the
  Miles–Lantuéjoul weights of the frame; a watershed from prominent maxima to split
  touching sections), Saltykov's unfolding of the size distribution of spheres or of
  oriented ellipsoids of a common shape from one or several directions
  (`unfold_sections`, `Unfolding.distribution` for the seeds of a foam), the
  consistency of a shape with the mean section radii (`shape_consistency`,
  `consistent_shape`); `kernels.watershed` (§39).
* Orthotropic hygromorph layers (`Layer(stiffness=Q, beta_xy=...)`, 3 x 3 laminate
  matrices, twist with `deformation(..., full=True)`) and `Layer.from_microstructure`:
  plane-stress stiffness, swelling and through-thickness diffusivity of a layer by the
  periodic homogenization of its microstructure.
* Oxidation stresses with FEniCSx (`oxidation_stresses`), as Ernault et al. (2019): the
  amide profile of the diffusion-limited oxidation at the depth of each element, its
  modulus and shrinkage from a `PropertyLaw`, the elastic response of the free sample;
  within 5 % of the free laminated plate.
* Diffusion and swelling of tetrahedral meshes with FEniCSx (`coupling.fenicsx`,
  `swelling`, `strip_curvature`): moisture entering from exposed faces, the free
  swelling of the sample at each output time (3-2-1 supports), the curvature of strips;
  a bilayer strip reaches Timoshenko's curvature within 0.3 %. `kinematics="finite"`:
  large rotations and strains (`F = F_e F_s`, Saint Venant-Kirchhoff in `F_e`, Newton
  with continuation); `arc_curvature`: a long strip curls into an arc whose chord
  shortens as geometry requires (within 1 %).
* Kinematic hardening in the NTFA: the Armstrong-Frederick (and Prager) terms of the
  phases become reduced backstresses, integrated with the reduced variables; exact for
  uniform fields, a composite along cycles within 5 %.
* Transient diffusion in configurations: a `diffusion` section (phases, exposed faces,
  activity history, times, fields) writes the uptake curve, the mean concentration of
  each phase and the concentration fields of each realization; template `diffusion`.
  The steps beyond 4096 voxels go through conjugate gradients (a 24^3 run: 9 s instead
  of 130 s with one factorization per length of step).
* The NTFA in configurations and the pipeline: `homogenization: ntfa` (training paths,
  modes, a file for the model, the full field too for comparison) trains the model on
  the voxel image and follows the `paths` with it (`<name>.ntfa` curves);
  `homogenize(..., ntfa=...)`; the mean stress and strain of every phase along the
  reduced paths; template `homogenization_ntfa`.
* Hollow fibres as in PuMA: `lumen` (inner radius over radius) of `hardcore_fibers` and
  `curved_fibers`, the walls a difference of unions (exact without overlaps), the
  lumens in phase 0; `solid_fraction` in the theory.
* Nonuniform transformation field analysis (`homogenization.ntfa`), after Michel & Suquet
  (2003, 2004): `train_ntfa` reduces a periodic voxel image of
  von Mises phases with isotropic hardening (plastic or Norton viscous): modes by POD of
  the plastic strain snapshots of full-field paths (normalized as Michel & Suquet's),
  elastic influence fields of the unit strains and of the modes by FFT, the affine
  relations of the macroscopic and reduced stresses; `NTFAModel.run` integrates the
  reduced laws along loading paths (backward Euler, Newton with an analytic Jacobian,
  strain, stress or mixed control); `local_fields`, `save`, `load`. A stiff sphere in a
  plastic matrix: within 2.5 to 7.7 % of the full fields, in milliseconds per path.
  Theory notes, §43.
* Fix: the merit function of the line search of the loading paths divided by zero when
  a scale was below 1e-150 (its square underflowed); the squares are now bounded.
* Foams as level sets of distance fields (`level_set_foam`),
  after the paper of N. Degdeg and J. Dirrenberger (Sonon et al. 2012; Kilingar et al.
  2019): seeds of aligned polydisperse ellipsoids, densely packed (the affine image of a
  Lubachevsky-Stillinger packing of spheres) or by random sequential addition; the
  level-set function `O_p - t O_s - K_c O_k` of the four smallest normalized distances:
  Plateau-border struts on the edges of the tessellation, the curvature of their
  cross-sections, their thickening towards the nodes, walls of closed cells; `t` given
  or solved for a solid fraction; `level_set` for conforming meshes. The copper foam of
  the paper comes out with ligaments of 74 um (72.8 in their model, 71.1 measured).
  Template `level_set_foam`; theory notes, §42.
* FEniCSx solver (`homogenization.fenicsx`): `solver="fenicsx"`
  (configurations, `--solver fenicsx`) assembles and solves the finite element problems
  of tetrahedral meshes with dolfinx and dolfinx_mpc (UFL weak forms, periodic
  multi-point constraints, the constrained degrees of freedom of morphoGen's solver,
  MUMPS or CG + GAMG); its fields go through morphoGen's post-processing, so that both
  solvers give the same apparent properties (within 1e-8, P1 and P2, every boundary
  condition, oriented crystals). Optional, from conda-forge, Linux or WSL. Theory notes,
  §41; decision D21.
* Couplings of transport, chemistry and mechanics (`morphogen.coupling`), after the
  papers of J. Dirrenberger with E. Ernault, C. Vibert, G. Mou and A. Le Duigou:
  `transient_diffusion` (voxel images, diffusivities and solubilities per phase,
  exposed faces, consumption; `slab_uptake`, `slab_profile`); `ClosedLoopOxidation`
  (thin films and diffusion-limited plates, Ernault et al.'s constants for DGEBA/IPDA
  at 110 °C); `PropertyLaw` and `graded_materials` (properties of an ageing tracer,
  binned into phases for every solver and export, eigenstrains as expansions);
  `Hydrolysis` (Ekenstam, critical DP); `Multilayer` and `Layer` (moisture through a
  hygromorph and the curvatures of the free plate; `timoshenko_curvature`). Theory
  notes, §40.
* Curved fibres as in PuMA: `curved_fibers` takes
  `shape: worm_like | arc | helix` (circular arcs of radii of curvature drawn from
  `curvature_radius`, in random planes about the tangents at their middles; crimped
  fibres, helices of `helix_radius` and `helix_pitch`) and `overlap: false` (fibres
  without overlap by random sequential addition of the chains of segments, `min_gap`,
  `max_attempts`; `kernels.rsa.random_sequential_addition_polylines`). Template
  `curved_arcs`.
* GPU backend (`morphogen.gpu`, `homogenization.gpu`; `device: cpu | gpu | auto` in
  `homogenization`, `--device` for `morphogen homogenize`, `device=` of `make_problem`,
  `homogenize`, the FFT problems and `NonlinearFFTProblem`): the FFT solvers of voxel
  images, linear and along loading paths, run their conjugate gradients on an NVIDIA GPU
  through CuPy, in double precision. The stiffness operators and the tangent operators
  of the voxels (B-bar, F-bar, full integration) are CUDA kernels compiled at run time
  and applied without assembly; the Fourier preconditioner uses cuFFT, its inverse
  symbol computed on the GPU. Along loading paths the integration points stay on the
  GPU: their measures, the integration of their behaviours (`behaviours.cu`, CUDA ports
  of the Numba kernels of von Mises plasticity, crystal plasticity and the logarithmic
  strain space, one thread per point, in launches whose size follows their duration),
  the conjugate stresses of B-bar and F-bar and the nodal forces; the Newton iterations
  stay on the CPU. The same solutions as on the CPU; on a laptop (RTX 3000 Ada), linear
  solves 5 to 7 times faster from 64³ to 192³ voxels, crystal plasticity paths 1.4 to
  1.9 times (theory notes, §35; `examples/gpu_benchmark.py`, `--paths` alone for the
  paths). Optional extra `[gpu]`: CuPy, the CUDA runtime, NVRTC and cuFFT (about 270 MB,
  no CUDA toolkit); `morphogen info` shows the GPU. Decision D3 of the specification
  (CuPy rather than PyTorch; the behaviours ported to CUDA C).
* The Fourier preconditioner of the CPU inverts its 3 × 3 symbols with a compiled kernel
  (cofactors) instead of NumPy's batched inversion.
* Fix: the measures of a voxel turned inside out, with full integration in finite
  strains, no longer write past the array of the gradients (CPU).
* `hardcore_fibers`: non-overlapping straight fibres (spherocylinders) by random
  sequential addition: constant or random radii and lengths, isotropic, aligned or planar
  axes, a minimal gap between their surfaces; the nearest periodic images through cell
  lists, non-periodic samples cut from an enlarged periodic box, an explicit error when
  the process jams; the exact fraction and specific surface in their theory.
* `curved_fibers`: Boolean model of curved fibres, worm-like chains of straight segments
  whose tangents decorrelate as exp(−s/ℓp) along the contour (von Mises–Fisher turns);
  the intensity, length density, volume fraction and mean square end-to-end distance in
  their theory.
* `strut_lattice`: rotated lattices (`rotation`: Bunge angles) and functionally graded
  lattices (`radius_gradient`, `gradient_axis`: radii interpolated linearly along an
  axis), in non-periodic domains; the mean and local densities of graded lattices in
  their theory.
* Templates `hardcore_fibers`, `curved_fibers` and `graded_lattice`.
* AMITEX_FFTP workflows (`morphogen.io.amitex`; `outputs.amitex` in configurations,
  `morphogen amitex export|read`): the inputs of runs of an image (material and zone
  numbers in AMITEX's legacy VTK, big-endian; the material file with `elasiso` for
  isotropic phases and voids and `elasaniso` for crystals, the first two axes of each
  grain as zone values in binary files; the reference medium of morphoGen's FFT
  solver; loading files of the six unit strains, of mixed loadings and of loading
  paths, in small or finite strains; the algorithm file; an MPI script and a manifest),
  and their results read back in morphoGen's orders: the averages of the cell
  (`.std`), of the materials (`.mstd`) and of the zones (`.zstd`), in small and finite
  strains, the apparent stiffness and moduli of the unit strains (a CSV row for
  `morphogen fit`). The materials are those of linear elasticity in small strains
  (AMITEX has no native law for morphoGen's plasticity: `elastic_only` exports their
  elasticity, for an MFront law to replace). The formats follow AMITEX's documentation
  and its official interface (py4amitex); AMITEX itself was not run.
* MFront behaviours (`homogenization.mfront`; the key `mfront` of materials, in Python,
  configurations and materials files): any standard behaviour written in MFront (TFEL)
  in place of morphoGen's own, integrated point by point by MGIS with its consistent
  tangent along loading paths, by finite elements, on voxels and at a material point, in
  small strains (strain-based behaviours; TFEL's components, whose shears are ordered
  12, 13, 23 with factors √2, converted to morphoGen's usual Voigt order 23, 31, 12 with
  engineering strains) and in finite strains (finite strain behaviours, loaded for the first
  Piola–Kirchhoff stress and its derivative). Crystals are turned to the axes of their
  lattice by the orientation of each grain. Material properties, parameters, a reference
  temperature to which the paths add their change, other external state variables. The
  internal state variables live in morphoGen's arrays, so that Newton's iterations and
  divided steps work unchanged; the Numba and CUDA kernels skip the MFront points, which
  MGIS integrates on the CPU (also on the GPU path). `.mfront` sources (relative to the
  configuration or materials file) are compiled on first use with `mfront --obuild
  --interface=generic` and cached by their SHA-256. Validated against MFront twins of
  von Mises elasto-viscoplasticity (small strains and the logarithmic strain space) and
  of Méric–Cailletaud crystal plasticity: the same curves and Newton iterations to
  10⁻¹⁰–10⁻¹⁶, at a material point, by finite elements and on voxels, on the CPU and the
  GPU (theory notes, §36); the paths take 2 (von Mises) to 5 times (crystals, whose twin
  has a numerical Jacobian) the time of morphoGen's kernels. MGIS and MFront are
  optional, from conda-forge (`conda install
  -c conda-forge mgis mfront cxx-compiler`; WSL on Windows); `morphogen info` shows them.
  Decision D20 of the specification.
* MFront sources of morphoGen's behaviours (`morphogen.io.mfront`: `write_mfront`,
  `mfront_source`; `morphogen mfront`): von Mises elasto-(visco)plasticity as the brick
  `StandardElastoViscoPlasticity` (Norton or rate-independent flow, linear and Voce
  isotropic terms, Armstrong-Frederick and Prager kinematic terms; in finite strains,
  in the space of the logarithmic strain), Méric-Cailletaud crystal plasticity in small
  strains as an implicit scheme with morphoGen's slip systems (their Schmid tensors in
  TFEL's components: 11, 22, 33, then sqrt(2) times 12, 13, 23), parameters of each
  system and interaction matrix, the elasticity as orthotropic constants in MFront's
  order. Compiled, they give morphoGen's curves through MGIS to 10⁻¹⁰–10⁻¹³ (theory
  notes, §37).
* AMITEX exports of nonlinear materials: their MFront laws, compiled by the script of the
  export with the umat interface, with the coefficients that the interface imposes (the
  axes of each grain per zone for crystals) and their internal variables, along loading
  paths (the unit strains of the stiffness need linear materials; `elastic_only` keeps
  the elasticity only). Checked by calling the compiled laws as AMITEX calls them;
  AMITEX itself was not run.
* Fix: the grains of a material with an MFront behaviour keep their orientations
  whatever its elastic parameters, so that an orthotropic MFront law (a crystal) of a
  material with isotropic elastic constants is turned to the axes of its grains (it was
  integrated in the axes of the sample).
* The Abaqus inputs of conforming meshes define the materials of the phases
  (`write_inp_tet(..., materials={phase: Material})`, and the `.inp` outputs of runs with
  a homogenization section): `*ELASTIC` (isotropic, or the 21 constants of
  `TYPE=ANISOTROPIC` in Abaqus's order of the components, 11, 22, 33, 12, 13, 23, in the
  axes of the crystal), `*EXPANSION`, `*DENSITY`, and the nonlinear materials as MFront
  laws of the abaqus interface: `ABAQUSBEHAVIOUR_<behaviour>_3D` with `*DEPVAR` and
  `*USER MATERIAL` (the one constant that the interface asks of laws without material
  properties), their sources in `<stem>_mfront` and the commands in the header of the
  input. The names of the exported behaviours have no underscore (`<stem>Phase<p>`), as
  the dispatcher of MFront in Abaqus splits the names of the materials at them. Checked
  with CalculiX, which reads Abaqus inputs (an isotropic grain and a turned grain of 21
  distinct anisotropic constants), and by calling the MFront laws as Abaqus calls them
  (10⁻¹¹ to 10⁻¹²); Abaqus itself was not run.
* The Abaqus inputs of voxel meshes define the materials of the phases too
  (`write_inp(..., materials=..., orientations=...)`, and the voxel `.inp` outputs of runs
  with a homogenization section): with the orientations of the labels, the sets of the
  grains, their `*ORIENTATION` and a section per grain; checked with CalculiX on C3D8
  elements.
* Crystal plasticity in finite strains as an MFront source
  (`write_mfront(..., kinematics="finite")`, `morphogen mfront --kinematics finite`):
  morphoGen's own kernel (the C port of the GPU, compiled for the host) in a behaviour of
  the `DefaultFiniteStrain` DSL, its internal variables an array of auxiliary variables,
  the first Piola-Kirchhoff stress and its consistent derivative of the kernel; the curves
  and the Newton iterations of morphoGen's kernel through MGIS (10⁻¹⁰ to 10⁻¹²). The
  constants of the C ports are shared by the GPU and the sources
  (`behaviours.kernel_constants`).
* The user's own MFront laws (materials with `mfront`, given by their source) in AMITEX
  and Abaqus inputs: their sources compiled for the interface of each code, the
  coefficients that the umat interface imposes (merged with the material properties of
  their glossary names) then their other material properties, their material properties
  as the constants of Abaqus (a material per phase, `ABAQUSBEHAVIOUR_<B>_3D_PHASE<p>`),
  their internal variables, their parameters in `<behaviour>-parameters.txt`; checked
  against MFront's examples and by calling the compiled laws as AMITEX and Abaqus call
  them, against morphoGen's integration of the same laws through MGIS.
* Finite strains in the AMITEX and Abaqus exports (`write_amitex(...,
  kinematics="finite")`, `write_inp`/`write_inp_tet(..., kinematics="finite")`,
  `morphogen amitex export --kinematics finite`, and the outputs of runs whose paths
  are in finite strains, which `outputs.amitex` now accepts): every phase but the voids
  an MFront law of finite strains, Hencky's elasticity for the elastic ones
  (`write_mfront(elastic, kinematics="finite")`; `morphogen mfront --kinematics
  finite` writes them too). AMITEX: `Small_Perturbations` false, the nine components of
  the paths, the functions `umat<behaviour>_malls` of the laws of Hencky's strain
  measure. Abaqus (`NLGEOM=YES`): no orientation in the sections, the axes of the
  orthotropic laws in their first six state variables
  (`@AbaqusOrthotropyManagementPolicy[abaqus] MFront` in morphoGen's sources) given by
  `*INITIAL CONDITIONS, TYPE=SOLUTION`. The user's laws of finite strains (written in
  the deformation gradient, or with Hencky's or Green-Lagrange's strain measure) too.
  Checked by calling the compiled laws as AMITEX and Abaqus call them in finite
  strains, against morphoGen's finite strains along a path whose principal axes turn
  (10⁻¹⁵ to 3·10⁻¹¹) and against MGIS for the user's laws.
* Dead leaves (`dead_leaves`): families of leaves of their own sizes (`phase_radius`;
  `phase_fractions` the fractions of space of their leaves), spheroidal leaves (`shape:
  spheroid`, `aspect_ratio`, `orientation`), a finite coverage over a background
  (`coverage`, `background`: the last cell of the labels); their exact bivariate laws
  (`bivariate`, `covariance`, `same_leaf_probability`), the fractions with the
  background and the integral ranges. A parameter kind `distributions`.
* Dense packings of hard spheres (`hardcore_spheres`, `method: lubachevsky_stillinger`,
  `growth_rate`): event-driven molecular dynamics of growing spheres, beyond the RSA
  limit, jamming near 0.64 for equal spheres (`kernels.packing`).
* Fix: the initial random sequential addition of the dense packings places the largest
  spheres first. Drawn late, a large sphere could find no room left, and the packing
  stopped there (3 seeds in 12 of the template `dense_packing`, its own included).
* Fix: the random sequential addition of `hardcore_spheres` (and of the cores of
  `penetrable_spheres`) places the largest spheres first too (`largest_first`). In the
  order of their draw, the radii of the template `hardcore_spheres` (uniform between 0.8
  and 1.2, 0.05 apart) jammed in 2 % of its realizations at its fraction of 0.3, 7 % in
  cubes of half its edge and most from 0.32, so that an RVE study of it in the interface
  (20, 10 and 5 realizations of edges 10, 20 and 30) failed four times in five; the
  largest first, they still fill 0.41. Equal spheres are placed as before.
* Penetrable spheres with impenetrable cores (`penetrable_spheres`): Torquato's
  cherry-pit model, the cores by RSA or grown.
* Stochastic reconstruction (`reconstruction`, Yeong & Torquato 1998): a lattice
  annealed to the two-point probability and the lineal-path function of a Debye medium,
  of a model or of an image file (`kernels.annealing`).
* Torquato's descriptors (`analysis.correlation`): the lineal-path, two-point cluster and
  pore-size functions, the radial distribution function and the structure factor of
  point patterns.
* Templates `sequential_dead_leaves`, `dense_packing`, `penetrable_spheres`,
  `reconstruction`.
* Transparent dead leaves (`transparent_dead_leaves`, Jeulin 1997): the random function
  of leaves letting `1 - transparency` of what they cover show through, of uniform or
  Bernoulli values, and its excursion above a threshold (or a volume fraction, from the
  law of its series); its exact mean, variance and covariance.
* Stealthy hyperuniform germs of Voronoi tessellations (`germs: stealthy`,
  `stealthiness`): ground states of the collective coordinates (`kernels.collective`),
  whose structure factor vanishes up to a wave number; cells about ten times more
  regular than those of Poisson germs.
* Templates `transparent_dead_leaves`, `hyperuniform_cells`.

## 0.5.0 — unreleased (phase 2c)

Whole periodic grains, the periodic element with mixed macroscopic loadings,
thermo-elasticity, nonlinear behaviours along loading paths (von Mises and crystal
plasticity, small and finite strains, on meshes and voxels), post-processing, a database
of results with Ashby charts, nested and bio-inspired models.

* Whole periodic grains (`cells: whole`, the default for periodic tessellations;
  `--cells`): the cells of periodic polycrystals are meshed whole, around their germs,
  instead of being cut by the faces of the box; the boundary of their union is made of
  translated facets meshed alike, so that periodic conditions link nodes to nodes.
  Meshes record their outer faces (`TetMesh.outer`) and derive periodic links, matching
  faces and exports from them; KUBC and SUBC apply on their boundary with its normals.
* The periodic element (`Loading.mixed`, `Loading.uniaxial`, `Loading.from_spec`,
  `Loading.parse`; `homogenization.loadings` in configurations, `--loading` on the
  command line): any mix of prescribed macroscopic strain and stress components
  (gradient and flux in conduction), solved exactly by condensation of the macroscopic
  unknowns from cached unit solutions, for every boundary condition and both solvers.
* Nonlinear behaviours along loading paths (`homogenization.behaviours`, `paths`,
  `nonlinear`): materials may yield (`plasticity`: von Mises elasto-viscoplasticity with
  Norton flow, linear and Voce isotropic hardening, Prager and Armstrong–Frederick
  kinematic hardening, or rate-independent plasticity), integrated implicitly with the
  consistent tangent. Loading paths (`LoadingPath`; `homogenization.paths` in
  configurations, `--path`/`--paths` on the command line) prescribe each component as a
  strain or a stress in time, with a change of temperature; they are solved by finite
  elements under periodic and kinematic conditions, with the macroscopic strain
  components as degrees of freedom of the Newton iterations (the periodic element; the
  macroscopic block condensed at each iteration), inexact multigrid solutions and step
  division (`homogenization.nonlinear`). Outputs: macroscopic curves, phase averages,
  element fields of stress, plastic strain and cumulated plastic strain. A material
  point simulator (`simulate_point`). Template `homogenization_viscoplastic`.
* Crystal plasticity of Méric and Cailletaud for polycrystals (`CrystalPlasticity`,
  `homogenization.crystal`; `crystal_plasticity` of a material): octahedral, cube and
  body-centred cubic slip families, Norton flow, nonlinear isotropic hardening coupled by
  an interaction matrix (uniform, self/latent, Franciosi's six interactions or full) and
  nonlinear kinematic hardening per system, the slip systems turned by the orientation
  of each grain. Implicit integration with the flow rule written as stresses (Newton from
  below on an active set, robust under large overstresses), substeps as a fallback, warm
  starts and the consistent tangent. Template `homogenization_crystal_plasticity`.
* Finite strains for every nonlinear behaviour (`kinematics: finite` in
  `homogenization.nonlinear`, `--finite`, `simulate_point(kinematics="finite")`;
  `homogenization.finite`): total Lagrangian finite elements; elasticity (Hencky) and von
  Mises plasticity in the logarithmic strain space; multiplicative crystal plasticity
  with lattice rotations. The macroscopic gradient is upper triangular: its components
  are nominal strains and simple shears, conjugate to nominal stresses, and a tension
  with free faces is a uniaxial Cauchy stress. Outputs add the Cauchy stresses (`TS`)
  and logarithmic strains (`LE`) of the curves, the mean Cauchy stress of the phases,
  and fields of Cauchy stress, logarithmic strain, lattice orientations and nodal
  displacements.
* Newton iterations of the loading paths with a backtracking line search, the material
  point with extrapolated steps; the fields of the paths add the displacement of the
  nodes (point data of the `.vtu`).
* The nonlinear FFT solver of voxel images (`NonlinearFFTProblem`; `method: fft` with
  `homogenization.paths`, and images on the command line): loading paths with von Mises
  and crystal plasticity, in small and finite strains, the macroscopic strain components
  as degrees of freedom of the Newton iterations, which the finite elements and the
  voxels now share (`nonlinear._PathSolver`). Each correction is solved by conjugate
  gradients preconditioned in Fourier space, the tangent applied without assembly
  (Newton–Krylov). The voxels average their volumetric strain (`integration: bbar`:
  B-bar in small strains, F-bar in finite strains, in variational form, so that the
  tangent stays symmetric): fully integrated, they lock under plastic flow (twice the
  flow stress of a plastic composite on 10³ voxels). Fields as `.vti` images, one array
  per component. Template `homogenization_crystal_plasticity_fft` (20 grains, 24³
  voxels, 5 % in finite strains, 90 s).
* Fix: crystals with isotropic elasticity and slip systems (or an anisotropic
  expansion) now receive the grains and orientations of their image in the pipeline and
  on the command line (FFT).
* RVE statistics of nonlinear responses: `rve.properties` names quantities of the
  curves of the loading paths, at the end of a path or of one of its steps
  (`tension.S11@3`; `PathQuantity`, `PathResult.value`), analysed as the apparent
  properties in the reports of the runs and by `morphogen fit`. Template
  `rve_crystal_plasticity_fft` (Voronoi polycrystals with crystal plasticity, FFT, after
  Yang et al. 2019). Fix: the summaries of the loading paths in `homogenization.csv` no
  longer break the analysis of studies with paths.
* Local continuous integration (`tools/ci_local.sh`): the jobs of the workflow for the
  system of the development machine, Linux, macOS or Windows with Git Bash (lint,
  format, the wheel in fresh environments for Python 3.12 and 3.14, the tests against
  it, the command-line smoke test), for when GitHub Actions is not available; uv is
  optional (venv and pip otherwise, with versions or paths of interpreters), and
  `CI_LOCAL_PACKAGES` adds packages to the environments of the tests (`"vtk gmsh"`:
  the checks of the exported files by these independent readers).
* Thermo-elasticity: thermal expansion of the materials (scalar, transversely isotropic
  or tensor, turned by the grain orientations), uniform changes of temperature in
  loadings, the apparent thermal expansion tensor (`thermal_expansion`), Levin's exact
  relation and its Hashin–Shtrikman bounds; densities of the materials.
* Post-processing (`homogenization.postprocess`): engineering constants, anisotropy
  indices (universal index, Zener ratio, distance to isotropy), macroscopic values of
  porous media, solid fraction and effective density, relative, normalized and specific
  values in every results row.
* Results database and Ashby charts (`morphogen.results`; `outputs.database`,
  `run --db`, `morphogen db`, `morphogen ashby`): every configuration tested and the
  results of its realizations in one SQLite file, filtered by conditions on any key,
  exported to CSV; normalized Ashby charts with families, envelopes and the scaling
  laws of cellular solids, light and dark, palette validated for colour-blind readers;
  an Ashby tab in the graphical interface (with a table view and dark mode).
* Nested models: model parameters may be models. `cox_spheres` (particles in the phase
  1 of another model: Cox models of nanocomposites, hard-core or not, with an aggregate
  phase), `coating` (coatings, shells and hollow members, exact for unions of spheres and
  segments), `hierarchical_lattice` (frames of generation G of solid or hollow tubes,
  Rayneau-Kirkhope et al. 2012), `fractal_sponge` (Menger, Vicsek, Cantor dust, any
  iterated function system), `operation: nested` in combinations, hollow struts in
  `strut_lattice` (radius solved below the peak of the fraction; exact for cubic cells;
  voxel estimates offset from the symmetries of the cells).
* Bio-inspired models: `nacre` (brick-and-mortar stacks of polygonal or staggered
  tablets), `bouligand` (twisted plywoods, periodic fibre families), `honeycomb`
  (hexagonal, square or random prismatic cells; softwood with radial files, growth rings
  and rays between the files, uniseriate or multiseriate), `osteons` (Haversian bone as dead leaves of osteons, exact phase fractions),
  `open_foam` (open-cell foams and trabecular bone on the edges of relaxed, elongated
  periodic Voronoi cells, at a relative density), `branching` (vascular trees grown by
  space colonization, Murray's law).
* Templates: `cox_nanocomposite`, `coated_spheres`, `hierarchical_lattice`,
  `menger_sponge`, `nacre`, `bouligand`, `wood`, `osteons`, `trabecular_bone`,
  `vascular_tree`, `homogenization_mixed_loading`.
* Fixes: periodic Euclidean distance transforms padded by the coating thickness only
  (30 times faster); tests read their files as UTF-8 (Windows).

## 0.4.0 — unreleased (phase 2b)

Polycrystals in finite elements, anisotropic crystals, parallel and cluster runs,
exports for other codes, morphological measures, combinations and further models.

* Exact cells of Voronoi and Laguerre tessellations (`mesh.polyhedra`): power diagrams
  clipped to the box or on the torus, pieces sharing their facets exactly.
* Conforming meshes of polycrystals (`mesh.polycrystal`, `conforming_mesh` of
  tessellations): features smaller than `min_feature` removed first (thin caps on the
  faces of the box flattened or merged into a neighbour, clusters of short edges
  contracted, periodic images moved alike), then MMG remeshes the grains as
  subdomains; periodic faces are triangulated in the plane (constrained Delaunay
  triangulation with exact orientation tests, then `mmg2d`) and match node for node.
  Meshes carry the grain of each element and the orientations of the grains, kept by
  the VTU, Medit, Gmsh and Abaqus/CalculiX files (`*ELSET CELL_<id>`,
  `*ORIENTATION`, `*SOLID SECTION`).
* Anisotropic crystals (`Material.cubic`, `.hexagonal`, `.anisotropic`; `cubic`,
  `hexagonal`, `stiffness` in configurations): stiffness and conductivity turned by the
  orientation of each grain, in the finite element and FFT solvers; Voigt, Reuss and
  Hill averages; Hashin–Shtrikman bounds of polycrystals of cubic crystals. The bulk
  modulus of cubic polycrystals is exact for every boundary condition.
* Parallel and cluster runs (`morphogen.parallel`): `run -j` (processes sharing a
  thread budget), `--resume`, `--task-id auto` (array index of the scheduler), `--mpi`;
  `morphogen merge`, `status` and `submit` (job arrays and a dependent merging job for
  SLURM, PBS Pro, Torque, SGE, OAR and LSF); one record per realization; results
  independent of the splitting.
* Exports: periodic constraint equations for Abaqus/CalculiX (`*EQUATION`, reference
  nodes, conforming and voxel meshes), DAMASK grids and material files, Neper seeds,
  weights, orientations and command, FEniCSx XDMF meshes with phase, grain and face
  tags, XDMF images, and voxelization slab by slab into memory-mapped `.xdmf` or
  `.mhd` files for volumes larger than memory.
* Measures (`analysis.morphology`; `chords`, `thickness`, `euler`, `tortuosity` in
  configurations): chord lengths, granulometry by openings and local thickness (pore
  sizes), Euler characteristic and Minkowski densities, geodesic distances and
  geometric tortuosity.
* Combinations of models (`combination`): union, intersection, difference and
  complement with combined level sets, and inclusions overlaid on a matrix (a
  polycrystal keeps its grains and orientations); theory of set operations on
  independent sets.
* Grain-boundary phases of tessellations (`boundary_thickness`, `boundary_phase`), from
  exact distances to the faces of the cells; grains in a boundary phase have a level
  set and are meshed through it.
* Models `random_laminate` (layers of random thicknesses with holes, nanolayered films;
  renewal theory of the variance along the stack) and `dead_leaves` (perfect
  simulation, covariance and integral range); Kelvin cells for `strut_lattice`; germs
  of hard-sphere packings for `laguerre`.
* Models declare the phases their realizations can contain (`active_phases`), checked
  against the materials of configurations.
* Templates `homogenization_polycrystal`, `homogenization_polycrystal_inclusions`,
  `random_laminate`, `dead_leaves`; example `examples/polycrystal.py`.
* Specification 1.4 (§3.3, FR-H10, FR-H11, FR-B5, decisions D8, D11, D12).

## 0.3.0 — unreleased (phase 3)

Cellular automata, lattice growth processes, combinatorial design, topology
optimization, visualization and a graphical interface.

* Cellular automata (`morphogen.automata`): von Neumann, Moore, edge and Euclidean
  neighbourhoods in 2D and 3D; Life-like `B/S` rules (with ranges), majority, Vichniac's
  annealing rule, majority at conserved volume fraction, probabilistic rules, Young's
  activator–inhibitor rule; synchronous, asynchronous and sequential updates; fixed or
  periodic boundaries; history of the volume fraction and of the interface density.
* Gray–Scott reaction–diffusion (spots, worms, labyrinths, holes).
* Growth processes (`automata.growth`):
  * nucleation and growth with Johnson–Mehl–Avrami–Kolmogorov kinetics, by an
    event-driven automaton with exact Euclidean (or faceted, oriented) growth fronts,
    site saturation or continuous nucleation, phantom nuclei;
  * normal grain growth by the Monte Carlo Potts model (parallel, reproducible);
  * diffusion-limited, Eden and ballistic aggregation of clusters and deposits.
* Lattice microstructures (`LatticeField`): exact on their lattice, interpolated and
  smoothed elsewhere, with a level set for conforming meshes, grain labels and
  orientations, exact JSON storage.
* Models `cellular_automaton`, `activator_inhibitor`, `reaction_diffusion`,
  `nucleation_growth` (Avrami theory: exponent, transformed fraction, grain density)
  and `aggregation` (fractal dimension); non-periodic domains are windows of larger
  periodic simulations.
* Measures: `fractal_dimension` (box counting); `cells` accepts the labels of grain
  structures (negative labels are ignored).
* Combinatorial design (`morphogen.design`):
  * tile sets with labelled faces (equal or opposite matching), rotations, geometry of
    assemblies; built-in polarized blocks (combinatorial metamaterials after Coulais et
    al. 2016) and pipe networks with controlled node degrees;
  * compatible assemblies by wave function collapse with backtracking (bit sets,
    Numba), with boundary labels and per-cell restrictions; exact counting and exact
    uniform sampling by transfer matrices; exact count of polarized-block assemblies by
    a gauge reduction (6, 450, 151 206, 145 456 074 for edges 1 to 4);
  * periodic pin-jointed frameworks: lattices, random dilution to a mean coordination,
    node perturbation, Maxwell–Calladine rigidity (mechanisms, states of self-stress,
    floppy macroscopic strains);
  * simulated annealing, genetic algorithm, and the inverse design of automaton rules
    from target morphological descriptors (`fit_rule`).
* Models `tile_assembly` and `random_lattice`.
* Topology optimization of periodic cells by inverse homogenization
  (`morphogen.optimization`):
  * homogenized tensors of graded voxel cells and their exact sensitivities (mutual
    energies) with the FFT-preconditioned solver, warm-started;
  * SIMP interpolation, periodic density filter, Heaviside projection with
    continuation, orthotropic and cubic symmetry;
  * objectives: bulk and shear moduli, Poisson's ratio (auxetics, with a stiffness
    constraint), conductivity, target tensor; 3D or planar (prismatic) cells;
  * methods: optimality criteria, method of moving asymptotes (Svanberg), BESO and the
    hybrid cellular automaton (Tovar et al. 2006);
  * designs are lattice microstructures (smooth level set from the densities), ready
    for meshing and finite element validation;
  * black-and-white designs are checked for hinges (voxels touching only along an edge
    or at a corner, which trilinear elements make stiff) and re-evaluated on a finer
    grid (`morphogen optimize` does both).
* Visualization (`morphogen.viz`): sections, periodic tilings, 3D surfaces, histories,
  transformation kinetics and animated GIFs of the formation of structures
  (matplotlib); colour maps and PNG images without any dependency.
* Graphical interface in the web browser (`morphogen gui`), served locally with the
  standard library: forms built from the models' parameters, sections and measures
  with downloads, animations of automata and growth processes, live topology
  optimization of cells, counts and assemblies of tile sets.
* Models with a formation history (automata, reaction–diffusion, nucleation and
  growth, aggregation) give it with `model.evolution(domain, seed, frames)`.
* Command line: `morphogen fit-rule` finds the automaton rule whose structures match an
  image and writes the configuration that generates them; `morphogen optimize` runs a
  topology optimization from a configuration file; `morphogen view` draws figures and
  animations; `morphogen gui` opens the interface.
* Solvers: the conjugate gradients compute their dot products in a single thread instead
  of with BLAS, whose threads competed with those of the compiled kernels: FFT solves are
  six to seven times faster on grids such as 64² and 32³ voxels, 1.6 times on 64³.
* Templates: `cellular_automaton`, `activator_inhibitor`, `reaction_diffusion`,
  `nucleation_growth`, `aggregation`, `homogenization_crystallization` (finite elements
  on a partially crystallized structure), `rve_cellular_automaton`, `tile_assembly`,
  `random_lattice`, `optimize_auxetic`, `optimize_bulk`, `optimize_conductivity`.

## 0.2.0 — unreleased (phase 2a)

Homogenization and statistical analysis of RVE studies (specification 1.2).

* Conforming tetrahedral meshes of every model with a level-set function
  (`mesh.conforming_mesh`), with MMG from the optional `pymmg` wheels:
  * the interface follows the morphology within a Hausdorff distance;
  * opposite faces match for periodic microstructures;
  * porous media can be meshed without their pores;
  * 4-node or 10-node tetrahedra (`TetMesh`);
  * exports to Abaqus/CalculiX, Gmsh 2.2, VTK, Medit and meshio formats, readers for
    `.vtu` and `.mesh`.
* Finite element solver on these meshes (`homogenization.fe`):
  * kinematic (KUBC), mixed (MBC, Dirrenberger et al. 2014), periodic and static
    (SUBC) boundary conditions;
  * linear elasticity and conduction of isotropic phases;
  * volume averages with zero stress in the pores, energy estimates, per-phase
    statistics, local fields.
* FFT solver of periodic voxel images (`homogenization.fft`): trilinear elements and a
  Fourier preconditioner; pores without an ersatz material.
* Linear solvers: sparse direct, and conjugate gradients with smoothed-aggregation
  algebraic multigrid (`homogenization.amg`, Numba), two-level for 10-node elements.
* Apparent properties (`homogenize`): bulk and shear moduli, conductivity, stiffness
  and conductivity tensors, isotropy criteria; Voigt, Reuss and Hashin–Shtrikman
  bounds.
* Statistical analysis after Dirrenberger et al. (2014) (`rve.study`, `rve.report`):
  * per property and boundary condition: intervals, relative errors, scaling law with
    free and imposed exponents, RVE sizes and object counts, bootstrap confidence
    intervals, isotropy;
  * discrepancy between boundary conditions;
  * Markdown, LaTeX and JSON reports, figures.
* Configurations: `mesh` section, `homogenization` section (method `fe` or `fft`),
  `outputs.conforming_mesh`; `homogenization.csv` results; RVE studies of apparent
  properties, split across tasks and merged by `morphogen fit`.
* Command line: `morphogen homogenize` (geometry, mesh or image), and `morphogen fit`
  options `--group-by`, `--bootstrap`, `--weights`, `-o`.
* Templates: `homogenization_spheres`, `homogenization_gyroid`,
  `homogenization_voronoi`, `rve_poisson_fibers_fe`.
* `rve.py` became the package `rve` (`rve.scaling` holds the phase-1 functions; the
  imports from `morphogen.rve` are unchanged).

## 0.1.0 — 2026-10-21 (phase 1)

First version: specification, architecture and the foundations of the package.

* Licence: GNU General Public License v3 or later (`LICENSE`).
* Models:
  * Boolean spheres;
  * Boolean cylinders and spherocylinders;
  * Poisson fibres;
  * hard-core spheres (RSA);
  * Voronoi and Laguerre tessellations, with phases and crystal orientations;
  * Gaussian excursions;
  * TPMS (gyroid, Schwarz P, Schwarz D, Neovius);
  * strut lattices (cubic, BCC, FCC, octet).
* Periodic and non-periodic domains; reproducible seeds independent of how a study is
  split into jobs.
* Representations: explicit geometry (JSON, CSV), level sets, voxel images with
  optional partial volumes.
* Theory: closed-form characteristics of each model.
* Measurements: volume fraction, specific surface, covariance, connected components and
  percolation, cell statistics.
* RVE statistics: ensemble and window methods, scaling-law fit, integral range, RVE
  size, fit of external tables.
* Exports:
  * images: VTI, legacy VTK, MetaImage, NPZ/NPY, TIFF;
  * voxel meshes: INP, MSH 2.2, VTU, meshio;
  * STL surfaces;
  * Gmsh OpenCASCADE scripts.
* Interfaces: Python API, YAML/JSON/TOML configurations, command line (`run`, `fit`,
  `analyze`, `models`, `template`, `info`), cluster task splitting.
* Test suite and continuous integration on Linux, Windows and macOS.
