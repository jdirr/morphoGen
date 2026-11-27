# The graphical interface

The interface runs the functions of morphoGen from forms, in the web browser
(`morphogen gui`) or as a stand-alone application in a window of its own
(`morphogen desktop`, `morphogen-desktop`; a bundle that needs no Python from
`packaging/desktop`), with the dialogs of the system for its folders and files. The
[tour of the interface](tutorial_interface.md) follows a study in it, screen by screen.

## Starting it

`morphogen gui` starts a local web server (standard library only, address
`127.0.0.1`) and opens the interface in the browser; `morphogen desktop` (or the program
`morphogen-desktop`, `pip install ".[desktop]"`) opens the same interface as a
stand-alone application, in a window of its own whose server is private to it (a free
port of the local machine): its menu holds the workspace (Open…, Save…), the folder of
the session and the help; the folders of the workspaces and the paths that the forms
take (**Browse…**) are chosen in the dialogs of the system, the files of the results are
saved where one chooses or opened by their applications, the folder of a result is
shown in the file manager (**Show its folder**), and closing the window asks first when
the workspace holds items not saved.

## Sections, tabs and tools

The forms of every model open filled with its example configuration (its template: parameters and domain), so that
any model runs as it opens. Its menus follow the work in four sections, each with its
tabs: **Create** (Generate, Topology optimization, Inverse design, Tile assemblies),
**Characterize** (Statistics), **Simulate** (Linear homogenization, Mechanics, Learning,
Fracture, Contact, Nonlocal damage, Stability, Physics, Couplings, Explicit
dynamics) and **Export & compare**
(Export, Solvers & formats, Comparisons, Ashby charts). A tab lists its tools in a menu
on its left, under the headings of what they do: in Mechanics, *Loading paths* (of a
morphology, its macroscopic law, the NTFA), *Behaviours of materials* (material point,
identification) and *Linear elasticity* (tensors, frames of beams); in Statistics,
*Morphology*, *Representative volume* and *Reconstruction*; in Fracture, *Cracks*, *Cohesive
zones* and *Phase field*; in Solvers & formats, *AMITEX_FFTP*, *Abaqus and meshes*,
*Code_Aster and OpenFOAM* and *Behaviours of materials*. Generate lists its models by family, with a filter.

The **search** of the header (`/`) finds a tool, a model or an option of their forms (a
behaviour, a strain energy, a file format) and opens it with the option set: "arruda"
opens the loading path of a morphology with the energy of Arruda and Boyce chosen,
"finite strains" its kinematics, "stl" the surfaces of Export. The address of the page
names what is shown (`#mechanics/path`, `#generate/voronoi`): a link to keep or to
give. **?** opens the map of the interface (every tab and every tool, as links),
`Ctrl + Enter` runs the form shown (its button stays in sight under the long forms), and
the colours follow the system or are set light or dark.

## The workspace

Each morphology, simulation and optimized design is
kept in the **workspace** of the session (a strip of its items under the menus), from
which any tab takes its input, before or after another as needed: an item is sent on by
the buttons of its usual next steps or the menu of the other tools (by section), which
set the forms to take it (e.g. a morphology as the specimen of a crack), and the item
stays when another tool of the tab is chosen; a click on an item shows it, sends it on,
renames or removes it, and its cross (or a middle click) closes it at once, with an
**Undo** for a few seconds. Each run of a tool joins the workspace too (R), so that a run with other
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

The workspace is saved to a folder and opened again in another session (**Save…** and
**Open…** beside it): the morphologies, simulations and results with their files.

## The tabs

* **Generate**: any model, chosen in a menu by family (random sets and particles, fibres
   and layers, tessellations and foams, lattices and TPMS, bio-inspired, growth and
   patterns, assemblies), with a form built from its parameters (each size monodisperse,
   or polydisperse by the law chosen beside it: uniform, log-normal, normal, discrete);
   sections of its
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
  or finite elements; strains, stresses or both over time) with their curves and their
  fields drawn in the page in 3D (any field on the deformed mesh or the voxels, any step
  of the path by a slider, played on one colour scale), the videos on demand (here or in
  Export, from the steps); the NTFA reduced
  model trained on paths and compared with the full field; a material point under a
  path, with the points of a test, its behaviour kept in the workspace for the other
  tools; the identification of a von Mises, Hill or porous behaviour (yield stress,
  hardenings, viscosity, elastic constants, Hill's coefficients; Gurson's q1, q2, the
  initial porosity...) on the curves of
  runs followed together (the macroscopic law of a morphology, a tension and a cycle in
  shear to tell its isotropic hardening from its kinematic one, the Gurson model of a
  porous morphology under several triaxialities) or on the points of a test, as Zopt
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
* **Learning**: a behaviour learned from data: the responses of a behaviour at a
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
  the phases, its frames (a video on demand); the toughness of each phase and of their
  interfaces, the
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
  width set by the internal length, the fields of the steps in 3D (their videos on
  demand);
* **Physics**: the other physics of a morphology of the workspace, on its voxels: its
  transport properties (thermal and electrical conductivities, permittivity, permeability,
  diffusivity) and their tensor between the bounds of Wiener and of Hashin and Shtrikman;
  the flow through its pores (Darcy's permeability), the acoustic parameters of the rigid
  frame and the absorption curve of a layer of it on a rigid wall; the storage and loss
  moduli and the loss factor of a composite with a viscoelastic phase over a range of
  frequencies;
* **Couplings**: transient diffusion into a morphology (uptake curve, the fields of the
  times in 3D, the effective diffusivity), the thermal oxidation of an epoxy plate (Ernault et al.) and
  of a morphology in its volume (the oxidized layer around impermeable phases), and
  hygromorph multilayers bending and twisting as moisture enters them, their layers
  homogenized from morphologies of the workspace;
* **Explicit dynamics**: the crushing, tension, shear or impact of a morphology run by
  OpenRadioss (its voxels or a conforming mesh; its phases elastic, von Mises
  elastoplastic or Johnson–Cook, with failure; the self contact of its outer surface;
  engineering units: MPa, kg/m³, m/s, 1/s, ms), its nominal stress–strain curves at both
  faces, energies, peak and plateau stresses, densification strain and absorbed energy,
  its fields over time in 3D step by step (a video on demand); without OpenRadioss here,
  its inputs and the
  scripts to run them elsewhere; the results of a run made elsewhere read back; the
  installation found, or a release installed from GitHub. Its curves go to Comparisons
  with the loading paths (E33 and S33 for a compression along z);
* **Export**: the files of an item of the workspace with an archive of them all: of a
   morphology, its geometry, sections, voxel images of its level set and phases (up to
   2048 voxels per edge, memory permitting), its surface (STL) and conforming mesh (for
   a morphology of several phases: the implicit function and the surface of each phase,
   `level_set_phase1.vti`, `surface_phase1.stl`..., and the mesh conforming to all their
   interfaces), the particles of one phase without its matrix for the discrete element
   method (surfaces, Gmsh meshes of triangles or tetrahedra for LMGC90, tables, spheres
   for LAMMPS, LIGGGHTS and YADE) and a
   video (sections of its level set sweeping it, orthoslices, a turntable); of a
   simulation or of the run of a tool, a view of its fields (field, colormap,
   magnification, cut, range) and a video (the field on the deformed mesh along the
   steps of a loading path or of a run, one frame per step on one colour range, or under
   a growing or oscillating load, the colours following the load — a pattern of a
   bifurcation oscillating; a turntable, a sweep; the frames that a tool kept instead of
   writing its video), its tables, fields (those of the steps with their ParaView
   collection, `steps.pvd`), mesh and configuration (to rerun it with `morphogen run`),
   drawn without solving again. No tool writes a video unless asked;
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
  a morphology, the study of Code_Aster of its apparent properties and the case of
  OpenFOAM of the flow through its pores or the diffusion through a phase; its meshes and images in the formats of other codes (Gmsh, Medit,
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

## Views in 3D

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

## Tools of one's own

The tabs of tools are built from a registry (`morphogen.gui.tools`): each tool declares
its form (typed fields, fields shown on conditions, items of the workspace, the
materials of their phases) and runs as a job whose progress the page follows; its
values, tables, figures, videos and files come back with an archive, and the
morphologies it makes join the workspace. A module of tools added to the package
appears in the interface by itself.
