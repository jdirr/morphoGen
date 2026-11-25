# Other codes and file formats

What morphoGen generates and computes goes to the codes of the field, and comes back:
images and meshes for ParaView, Abaqus, CalculiX, Gmsh, FEniCSx, DAMASK and Neper, runs of
AMITEX_FFTP prepared and read back, behaviours written as MFront sources and Z-set
material files, and Z-set files run as they are; the particles of a phase go to the
codes of discrete elements (LMGC90, LAMMPS, LIGGGHTS, YADE), microstructures to the arrays
of JAX, PyTorch and CuPy, to Microgen and to Blender. `morphogen.export(obj, path)` picks the
writer from the object and the suffix of the file.

## File formats

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
  strains), algorithm file and the MPI script; their results read back (averages of the
  cell, materials and zones, apparent stiffness and moduli) for RVE studies
  (`morphogen amitex export|read`, `outputs.amitex`).
* MFront sources of the nonlinear materials (von Mises or Hill plasticity, porous
  plasticity, crystal plasticity and the learned laws, in small and finite strains,
  with their thermal expansion from 293.15 K,
  where the Abaqus inputs then start their nodes), for AMITEX_FFTP, Cast3M, Abaqus or
  Code_Aster (`morphogen mfront`), checked against morphoGen's own kernels; the user's
  own MFront laws go to AMITEX and Abaqus inputs too, in small or finite strains
  (every phase an MFront law then, Hencky's elasticity for the elastic ones).
* Watertight STL surfaces for additive manufacturing.
* Particles for the discrete element method (`morphogen.io.dem`, `export(...,
  particles=k)`, `morphogen export --particles k`): one phase without its matrix, a
  closed surface per particle (STL solids or files, OBJ objects, VTU), whole across
  the faces of periodic boxes, cut by those of windows; Gmsh files (`.msh`) of their
  triangles or of their tetrahedra (`elements="tetrahedra"`, `--tetrahedra`: gmsh's
  mesher fills each surface), a particle per entity and physical group, as LMGC90 reads
  them (`readMesh`, `separateMeshes`, `surfacicMeshToRigid3D` or
  `volumicMeshToRigid3D`); the surfaces remeshed by MMG at a triangle size (`size`,
  `--element-size`: fewer, regular faces for the contacts of polyhedra); the exact
  spheres of the models of spheres and fibres (a fibre a clump of spheres along its
  axis, a capsule among the surfaces) for LAMMPS and LIGGGHTS (`.data`, the clumps as
  molecules) and YADE (`ymport.text`, `textClumps`); tables of the particles (volume,
  area, centroid, equivalent radius).
* Gmsh OpenCASCADE scripts.
* Raw data and figures: tables as `.csv`, `.txt` and `.xlsx` (no dependency),
  figures as `.png`, `.tif`, `.svg`, `.pdf` and TikZ (pgfplots code for LaTeX).

The meshes of a polycrystal keep their grains and orientations for Abaqus, FEniCSx and
DAMASK ([Polycrystals](properties.md#polycrystals)); the [recipes](howto.md) show the
usual exchanges in a few lines.

## Libraries and programs

morphoGen hands its microstructures to other libraries and programs, and takes theirs
back ([optional](installation.md#optional-components); morphoGen never requires them):

* **Arrays of other libraries** (`morphogen.interop.arrays`). Voxel images handed over by
  the DLPack protocol without copies: JAX (on buffers aligned on 64 bytes, which XLA
  shares), PyTorch, CuPy; the diffuse interface of a phase of a realization, the profile
  of the plane interface at equilibrium of Allen–Cahn's and Cahn–Hilliard's equations,
  with the grid and the state that spectral solvers take (Exponax's steppers), and
  ensembles of realizations for `jax.vmap` rollouts: microstructures as initial
  conditions of phase-field coarsening and of differentiable pipelines, their fields
  back as images to measure and homogenize.
* **Microgen** ([3MAH/microgen](https://github.com/3MAH/microgen), developed by the 3MAH
  team; GPL-3.0, installed apart; `morphogen.interop.microgen`, the model `microgen`).
  morphoGen calls Microgen and copies none of it. The shapes of Microgen that have a field
  — TPMS sheets and skeletons of its many surfaces (Schoen's I-WP, lidinoid, Fischer–Koch
  S...), spinodoids, spheres, boxes, cylinders, capsules, ellipsoids — taken by morphoGen
  through its model `microgen`, described in a configuration and saved; the phases of any realization of
  morphoGen as shapes of Microgen (its field the implicit function of the phase), for its
  surface and volume meshes, infills and smooth booleans; the tetrahedral meshes of both
  through PyVista. Cite Microgen
  ([10.5281/zenodo.6793573](https://doi.org/10.5281/zenodo.6793573)) when you use them
  ([Citing](citing.md)).
* **Alloys** (`morphogen.interop.alloys`, `morphogen alloy`). The phases of an alloy as
  a tool of alloy design gives them (CALPHAD equilibria and Scheil solidification, the
  KWN kinetics of the precipitates, the grain size, the moduli it predicts: Leil-Mat,
  Thermo-Calc, pycalphad), in a description of JSON or YAML: its matrix, its other
  grains, its precipitates and particles, their fractions, radii and elastic constants
  (isotropic, or the cubic constants of crystals turned at random); its scales as
  configurations of morphoGen — the precipitates in a box of the matrix, the grains with
  their particles on the matrix homogenized —, their effective elasticity by FFT within
  the Hashin–Shtrikman bounds of their phases, and its departure from the modulus
  predicted. Leil-Mat gives its programming interface on request: the description is
  filled from its canvas meanwhile (`examples/alloy_design.py`).
* **Blender** (`morphogen.interop.blender`). A microstructure as a scene of Blender: the
  closed surface of each phase from its conforming mesh (STL, for OpenFOAM's
  snappyHexMesh too), and the fluid of Mantaflow around them (its domain, the phases as
  obstacles, an inflow and an outflow along an axis) to bake in Blender, the flow read
  back as voxel images (`write_blender_scene`, `read_blender_flow`; the box *Blender
  (CFD)* of the tab Export). The particles of a phase as the rigid bodies
  of Blender's physics (Bullet): its exact spheres and spherocylinders (Bullet's own
  shapes) or the convex hulls of its particles, dropped under gravity in an open box;
  the scene and its script written for Blender (`blender --background`, or a Python with
  the module `bpy`), run, and the packing read back as a microstructure (the spheres and
  segments moved, or the particles of their surfaces, `MeshParticles`, voxelized by the
  parity of the crossings of their columns) to measure and homogenize; the `.blend` scene
  to look at, render or go on with in Blender.

## AMITEX_FFTP and MFront

```bash
# an image for AMITEX_FFTP: its inputs and the MPI script, then its results
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

Behaviours written in MFront also take the place of morphoGen's own along loading paths
([MFront behaviours](behaviours.md#mfront-behaviours)).

## Z-set

The material files of Z-mat run as they are, and the behaviours of morphoGen, among them
those identified on the responses of a morphology, are written as such files
([Z-mat material files](behaviours.md#z-mat-material-files); what of Z-mat's library is
covered: [](zmat.md)).

## FEniCSx

`solver: fenicsx` assembles and solves the finite element problems of
tetrahedral meshes with FEniCSx (dolfinx, dolfinx_mpc, PETSc; Linux or WSL), with the
same results as morphoGen's own solver.
