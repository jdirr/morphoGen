# Configuration files and commands

The keys of the configuration files, the commands of `morphogen` and the files that a
run writes. [Configuration files and the command line](running.md) shows them at work,
and `morphogen template` gives a configuration to start from for every example of the
[gallery](gallery.md).

## Configuration files

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
  curved: true         # 10-node elements (6-node triangles of the plane): their nodes on
                       # the interfaces of the phases (curved edges); false: the flat
                       # faces (straight edges) of the mesher
  phases: auto         # all | solid (pores not meshed); auto: solid if a phase is a void
  hausdorff: null      # interface accuracy (default size / 20)
  min_size: null       # default size / 10
  gradation: 2.0       # size ratio of neighbouring elements
  background: null     # level-set sampling (default size / 2): must resolve thin features
  grains: null         # tessellations: one subdomain per grain (default), false: phases
  min_feature: null    # grains: features below this length removed (default size / 4; the
                       # cells in space, the polygons of the plane)
  cells: whole         # periodic tessellations: whole grains (default) or box (cut cells)
  method: auto         # exact: no background grid, the exact cells, convex pieces or
                       # point-wise level set (theory §94; background unused); grid: the
                       # level set on a grid; auto: exact when possible, grid otherwise
  plane: auto          # true: triangles of the plane (theory §119; the exact columnar
                       # grains of a tessellation of coplanar germs, the section of any
                       # other structure at z) and the finite elements of the plane;
                       # false: tetrahedra (those grains extruded through the box);
                       # auto: triangles for those tessellations and for the orientation
                       # maps (ebsd: two triangles per pixel), tetrahedra otherwise
  z: null              # plane: the height of the sections (default: the middle of the box)
outputs:
  directory: out/run   # relative to the configuration file
  name: sample         # file stem
  geometry: [json, csv, geo]
  image: [vti, vtk, mhd, npz, npy, tif]
  labels: [vti]        # tessellations: cell labels
  level_set: [vti]
  mesh: {formats: [inp, msh, vtu], phases: [1]}   # voxel hexahedra
  conforming_mesh: [vtu, inp, msh, mesh, xdmf, mail]   # tetrahedra (needs the mesh section);
                       # the triangles of the plane: vtu, inp (plane strain elements), msh,
                       # mesh and the formats of meshio
  periodic_pairs: true
  periodic_equations: true   # *EQUATION files for the meshes written (periodic domains;
                       # the plane elements of the meshes of the plane too)
  damask: true         # tessellations with orientations: DAMASK grid and material file
  neper: true          # tessellations: Neper seeds, weights, orientations, command
  amitex: true         # AMITEX_FFTP runs of the image (needs homogenization), or options:
                       # {fields, tolerance, elastic_only, processes, library}; the
                       # nonlinear materials as MFront laws, along the paths
  code_aster: true     # the study of Code_Aster of the conforming mesh (needs the mesh
                       # section and the materials of homogenization), or options:
                       # {physics: [elastic, thermal], conditions: periodic | kinematic,
                       # mesh_format: aster | med, fields, memory, time_limit, threads}
  openfoam: true       # the case of OpenFOAM of the image (or sample: mesh, the conforming
                       # mesh), or options: {kind: flow | diffusion, phases: [0], axis,
                       # lateral: symmetryPlane | cyclic, viscosity, pressure_drop,
                       # diffusivity, difference, iterations, tolerance, connected,
                       # dual: the flow through the mesh on its polyhedral dual}
  surface: [stl]
  database: null       # SQLite file recording the run and its results (results.py)
  tables: [txt, xlsx]  # the tables of the run in these formats too (besides .csv)
analysis: [volume_fraction, specific_surface, connectivity, phase_fractions, cells]
                       # on the voxels; on the implicit geometry itself, without voxels:
                       # exact_fraction, exact_surface (with the deviations of their
                       # probes), exact_chords, exact_minkowski (the four densities of a
                       # union of spheres exactly: exact_V_V, exact_S_V, exact_M_V, exact_chi_V),
                       # exact_grains (the grains of a tessellation on its exact cells; the
                       # grains of seeds against their seeds)
homogenization:        # apparent properties of every realization
  method: auto         # fe (mesh section) | fft (voxels, periodic) | xfem (the level set
                       # of the model cut by the grid of the voxels section, periodic) |
                       # auto: fe (a light mesh of edge/10 when none), fft only for
                       # what only voxels do (config.voxels_only) or without MMG
  materials:           # one per phase: void, {young, poisson, conductivity}, {bulk, shear}
    0: void            # or crystals: {cubic: [C11, C12, C44]}, {hexagonal: [C11, C12,
    1: {young: 210000.0, poisson: 0.3, conductivity: 100.0}   # C13, C33, C44]},
                       # {stiffness: 6 x 6}, conductivity: k | [k_a, k_c] | 3 x 3;
                       # tows: {unidirectional: {fibre, matrix, fibre_fraction}}
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
  plane: generalized   # the meshes of the plane: generalized (generalized plane strain:
                       # the anti-plane fluctuation free, the stiffness of the structure in
                       # space; the default) | strain (plane strain); kubc, periodic, subc;
                       # their paths on their layer in space, periodic (generalized)
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

## Command line

| Command | Purpose |
|---|---|
| `morphogen run CONFIG [-o DIR] [--seed S] [--task-id I\|auto --n-tasks N] [-j J] [--threads T] [--resume] [--mpi]` | batch or RVE study, in J processes, one task of a split run, or one task per MPI rank |
| `morphogen merge DIR [--no-report]` | gather the records of the tasks of a split run into its results and report |
| `morphogen status DIR [--json]` | realizations done and missing in a run directory |
| `morphogen submit CONFIG --scheduler slurm\|pbs\|torque\|sge\|oar\|lsf --tasks N [-j J] [--threads T] [--time T] [--memory M] [--queue Q] [--account A] [--setup LINE...] [--submit]` | job array and merging job scripts for a cluster, submitted with `--submit` (the merging job waits for the array) |
| `morphogen fit RESULTS... [--property P] [--group-by C] [--volume-column C] [--eps E...] [--n N...] [--gamma G] [--bootstrap B] [--weights n] [-o DIR]` | statistical RVE analysis of CSV tables (morphoGen results directories, or external FE/FFT results) |
| `morphogen homogenize SAMPLE --material P=... [--bc ...] [--property ...] [--size H --order O --min-feature F] [--voxels N] [--device cpu\|gpu\|auto] [--fields DIR] [--save-mesh FILE] [--json]` | apparent properties of a geometry (`.json`: meshed, by default a light mesh of the shortest edge of its domain over 10, or voxelized with `--voxels` for the FFT solver), a tetrahedral mesh (`.vtu`, `.mesh`: FE) or an image (FFT, on the CPU or an NVIDIA GPU); crystals as `P=cubic:C11/C12/C44` or `hexagonal:...`; a stiffness printed with its compliance S = C^-1, their notation and its engineering constants (in the JSON: `tensor`, `compliance`, `notation`) |
| `morphogen homogenize ... [--path [NAME:]PATH ...] [--paths FILE] [--finite] [--contact [MU]] [--interfaces KEY:VALUE,...] [--curves DIR]` | loading paths followed with the nonlinear behaviours of the materials (their curves as `.csv`); in finite strains; with the contact of the walls of the pores of a mesh (friction `MU`); with the interfaces of the phases of a mesh that debond (`normal_strength:30,mode_I:0.05,friction:0.3,viscosity:0.02`; `law:exponential,strength:..,energy:..`; `law:contact`; `faces:phases\|grains\|0-1+0-2`) |
| `morphogen analyze IMAGE [--array A] [--measure M...] [--json]` | measurements on an image file |
| `morphogen analyze GEOMETRY.json [--measure exact_...] [--seed S] [--json]` | the measures of the implicit geometry of a realization (exact_fraction, exact_surface, exact_chords, exact_minkowski, exact_grains), beside the theory of its model (`theory_*`) |
| `morphogen texture MAP [--threshold T] [--min-size N] [--axes map\|file] [--setting 1\|2\|3\|4] [--poles HKL...] [--direction x\|y\|z\|a,b,c] [--halfwidth H] [-o DIR] [--format png\|svg\|pdf] [--convert FILE [--clean]] [--json]` | the grains and the texture of an orientation map (`.ang`, `.ctf`): their numbers phase by phase; the tables of the grains and of their boundaries, the map in the colours of its orientations, the pole figures, the inverse pole figure and the disorientations (`-o`); the map written in the other format, as it is or cleaned |
| `morphogen texture FIELDS [--lattice cubic\|hexagonal] [--poles HKL...] [--direction D] [--halfwidth H] [-o DIR] [--json]` | the texture of the lattices of the fields of a loading path (`.vti`, `.vtu`: crystals in finite strains, or turned by their twins): the largest densities of its pole figures and of its inverse pole figure, and their figures |
| `morphogen compress SAMPLE --material P=... --strain E [--axis 0\|1\|2] [--friction MU] [--no-self-contact] [--points gauss\|nodes] [--steps N] [--small] [--size H --order O] [--curve F] [--fields F] [-o F] [--json]` | compression of a sample (a geometry, meshed, or a tetrahedral mesh; its solid phases) between two rigid platens, with contact and friction on the platens and of its walls on themselves: nominal stress against strain, areas in contact |
| `morphogen indent SAMPLE --material P=... --depth D [--indenter sphere\|cone] [--radius R] [--half-angle A] [--point U V] [--unload] [--base fixed\|smooth] [--poisson NU] [--steps N] [--small] [--curve F] [--fields F] [-o F] [--json]` | indentation of the upper face of a sample by a rigid sphere or cone: force against depth, area of the contact, the reduced modulus read (Hertz, Sneddon, the layer on its base, the slope of the unloading) |
| `morphogen bands SAMPLE --material P=... [--waves elastic\|acoustic\|shear\|tm\|te\|electromagnetic] [--path G X M G] [--bands N] [--points N] [--order O] [--coarsen N] [-o bands.csv] [--figure F] [--json]` | band structure of the Bloch waves of a periodic cell (an image or a tetrahedral mesh): elastic waves, the scalar waves of sonic and planar photonic crystals, or light in a three-dimensional crystal (plane waves on the voxels of an image, `--coarsen` times coarser with averaged interfaces); the band gaps along the path |
| `morphogen percolation CONFIG [--parameter P] [--lower L] [--upper U] [--phase N] [--axis x\|y\|z\|any\|all] [--connectivity 4\|6\|8\|18\|26] [--periodic\|--no-periodic] [--method labels\|fire] [--directions axes\|lattice\|N] [--sizes L1,L2,...] [--nu NU] [--fit-nu] [--rigidity] [--cut C] [--exponent thermal\|elastic] [--threshold P] [--threshold-stderr S] [--fractions F1,F2,...] [--contrast C] [--device cpu\|gpu\|auto] [--solver fft\|fe] [--realizations N] [--tol T] [--seed S] [--json]` | percolation threshold of the model of a configuration (its domain and voxel grid) against a parameter, by default the volume fraction: the threshold, its bracket and the probabilities tried (a 2-D configuration, one voxel along z: along x or y, through 4 or 8 neighbours); with `--directions`, its anisotropy: the thresholds across the cores of the box along the axes, the axes and diagonals of the lattice or N directions sampled, their paired deviations and ratio, the threshold tensor; with `--sizes`, the thresholds of cubes of these sizes at its voxel size extrapolated to p_c + a L^(-1/nu); with `--rigidity`, those of its elastic moduli by the solver of the voxels (where half the realizations bear a uniaxial strain above `--cut`, 1e-8 of the phase's modulus, the others void), extrapolated alike, with the solutions counted; with `--exponent`, the critical exponent of the conductivity (t) or of the elastic moduli (f) by the solver of the voxels: the means of the realizations of the `--sizes` at the threshold (given, or extrapolated first: through 26 neighbours wrapping around the box for t, by the moduli as with `--rigidity` for f) fitted to L^(-t/nu), the error of the threshold propagated, a sweep over `--fractions`, the other phases at `--contrast` (1e-8); `--solver fe`: the voxels assembled and factorized instead of the FFT solver (on the CPU; a few iterations near the rigidity threshold, void around) |
| `morphogen homogenize ... [--cells whole\|box] [--loading [NAME:]COMPONENTS ...]` | whole periodic grains; macroscopic loadings such as `tension: S11=100`, `E11=0.001 S22=0 dT=10` (strain `E`, stress `S`, gradient `G`, flux `Q`; components not given are free) |
| `morphogen frame-stiffness FILE [--seed S] [--realization R] [--beams euler-bernoulli\|timoshenko] [--shear-coefficient K] [--rigid-joints C] [--young E] [--poisson NU] [--softening axial\|bending\|modulus\|area ...] [--axis x\|y\|z] [--replicas N] [--cv CV ...] [--distribution normal\|lognormal\|bimodal] [--replica-seed S] [--table CSV] [--figure FILE] [--json]` | the stiffness of the frame of beams of a realization of `delaunay_network`, `open_foam` or `random_lattice` (a configuration, realized as the realization R of its batch run, or a geometry `.json`; its framework kept in its metadata): Euler–Bernoulli or Timoshenko beams (Cowper's coefficient, or K), rigid over C radii at the joints, fibres of modulus E and Poisson's ratio NU; the Voigt stiffness (per unit area for a planar network) and its compliance S = C^-1 with their notation (JSON: `stiffness`, `compliance`, `notation`), its engineering constants, along each axis the Young's modulus, Poisson's ratio, ratio to the affine estimate, energy split (stretching, bending, torsion, shear) and non-affinity of a uniaxial stress, their means (the summary of the models), and Ban et al.'s softening coefficients s and v of the modes given along an axis; with `--replicas`, N replicas per CV of the stiffness of the beams of each mode (the law of `--distribution`, seed S) against them: the plain and corrected means of E/E0 with their standard errors, 1 - s CV^2, the scatter against v CV^2 (`--table`: CSV; `--figure`: the means against CV and the scatter against v CV^2) |
| `morphogen amitex export SAMPLE --material P=...\|--materials FILE [--voxels N] [--loading ...] [--path ...] [--stiffness] [--fields] [--tol T] [--elastic-only] [--name NAME] [-o DIR]` | AMITEX_FFTP runs of an image (or of a geometry with `--voxels`): the six unit strains of the stiffness by default, named loadings and paths; the files and `<name>_run.sh` in DIR |
| `morphogen code-aster export SAMPLE --material P=...\|--materials FILE [--size H] [--order 1\|2] [--periodic] [--physics elastic thermal] [--conditions periodic\|kinematic] [--format aster\|med] [--fields] [--name NAME] [-o DIR]` | the study of Code_Aster of the apparent properties of a conforming mesh (a geometry meshed with `--size`, or a mesh `.vtu`, `.mesh`): the mesh and its groups, the command file of the six unit strains and three unit gradients, the export file of `run_aster`, in DIR |
| `morphogen code-aster read DIR [--name NAME] [--json] [-o CSV]` | the results of a study: the mean stress or flux of each loading, the apparent stiffness and its compliance (JSON: `stiffness`, `compliance`, `notation`), the conductivity and the moduli (a row appended to a CSV table: `C11` ... `C66`, `S11` ... `S66`) |
| `morphogen code-aster run DIR [--name NAME] [--json] [-o CSV]` | the study run by Code_Aster (`run_aster` on the path, or the program that `MORPHOGEN_RUN_ASTER` names; its output in `<name>.log`), then its results as `read` gives them; a job that fails stops with its diagnostic |
| `morphogen openfoam export SAMPLE [--kind flow\|diffusion] [--phases P...] [--axis 0\|1\|2] [--lateral symmetryPlane\|cyclic] [--viscosity NU] [--pressure-drop DP] [--diffusivity D] [--difference DT] [--iterations N] [--all-cells] [--dual auto\|yes\|no] [--voxels N\|--size H] [--periodic] [-o DIR]` | the case of OpenFOAM of an image, a geometry (voxelized, or meshed) or a tetrahedral mesh: its polyMesh, the flow through the pores (`simpleFoam`; through tetrahedra on their polyhedral dual, `--dual`) or the diffusion through the phases (`laplacianFoam`), `Allrun` |
| `morphogen openfoam read DIR [--json] [-o CSV]` | the results of a case run by OpenFOAM: the flow rate, the mean velocity and Darcy's permeability, or the effective diffusivity and its ratio to that of the phase; the verdict of `checkMesh` |
| `morphogen openfoam run DIR [--json] [-o CSV]` | the case run by OpenFOAM (`Allrun`, with its solvers on the path, or after the file that `MORPHOGEN_OPENFOAM` names, its `etc/bashrc`), then its results as `read` gives them; a program that fails stops with the end of its log |
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
| `morphogen alloy ALLOY [--config DIR] [--homogenize] [--shape N] [--seed S] [--json FILE]` | the scales of an alloy described by a tool of alloy design (`.json`, `.yaml`: its phases, fractions, radii and elastic constants; `morphogen.interop.alloys`): the configuration of each scale for `morphogen run`, and their effective elasticity by FFT beside the modulus predicted |
| `morphogen learn CONFIG [-o MODEL] [--data FILE] [--model feedforward\|potential\|recurrent\|standard] [--epochs N] [--paths N] [--seed S] [--json]` | a behaviour learned from data: random paths of a material point (`material`) or of a voxel image (`sample`, `materials`), a network trained and saved (`.npz`), its errors on the validation paths and on named `paths` that the learned law runs at a material point |
| `morphogen fit-rule IMAGE [--neighbourhood N] [--steps S] [--lattice L] [--tiling T] [--population P] [--generations G] [-o CONFIG] [--json]` | automaton rule whose structures match an image, on the grid or the cells of a tiling `T`; writes the configuration that generates them |
| `morphogen view FILE [--shape N] [--labels] [--tile T] [--surface] [--history] [--kinetics] [--animate GIF --frames F] [-o PNG]` | figures of a geometry, configuration or image; animation of the formation of a structure |
| `morphogen export INPUT OUTPUT [--kind K] [--field F] [--displacement U] [--magnification M] [--wave ramp\|sine] [--section AXIS FRACTION] [--frames N] [--fps F] [--mode once\|loop\|bounce] [--repeat R] [--cmap C] [--clim LO HI] [--size W [H]] [--shape N] [--format frames]` | an image, a mesh with its arrays (`.vtu`), a geometry or a table to another format (by the suffix), a figure or a video |
| `morphogen export INPUT OUTPUT --particles PHASE [--format yade] [--tetrahedra] [--element-size H] [--shape N]` | the particles of a phase of a geometry or an image without its matrix, for the discrete element method: a closed surface per particle (`.stl`, `.obj`, `.vtu`; `.msh`, Gmsh 2.2 as LMGC90 reads it, its tetrahedra with `--tetrahedra`), whole across periodic faces, remeshed by MMG at `--element-size`; a table (`.csv`, `.txt`, `.xlsx`); the spheres of a union of spheres and segments as a LAMMPS / LIGGGHTS data file (`.data`) or YADE's columns (`--format yade`) |
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

## Output files

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
