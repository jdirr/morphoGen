# Other codes and file formats

What morphoGen generates and computes goes to the codes of the field, and comes back:
images and meshes for ParaView, Abaqus, CalculiX, Gmsh, FEniCSx, DAMASK and Neper, runs of
AMITEX_FFTP and studies of Code_Aster prepared and read back, cases of OpenFOAM (the flow
through the pores, the diffusion through a phase) and their results, behaviours written
as MFront sources and Z-set material files, and Z-set files run as they are; the particles of a phase go to the
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
* Code_Aster's meshes (`.mail`, its native format, with the groups of the phases, of
  the grains, of the faces of the box and of their nodes; MED with h5py) and its studies
  of the apparent stiffness and conductivity of a conforming mesh, read back
  (`morphogen code-aster export|read`, `outputs.code_aster`;
  [Code_Aster](#code-aster)).
* OpenFOAM's meshes (`constant/polyMesh` of the voxels of some phases or of the
  tetrahedra of a conforming mesh, with the patches of the faces of the box, cyclic
  pairs and the walls) and its cases of the flow through the pores and of the diffusion
  through a phase, read back (`morphogen openfoam export|read`, `outputs.openfoam`;
  [OpenFOAM](#openfoam)).
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

## Code_Aster

`morphogen.io.write_code_aster` writes the study of Code_Aster (EDF's finite element
code, [code-aster.org](https://code-aster.org)) of the apparent properties of a
conforming mesh: the mesh in Code_Aster's own format (`.mail`, read by
`LIRE_MAILLAGE(FORMAT='ASTER')`; or MED), with the groups of elements of its phases
(`PHASE_<p>`), of the grains of its crystals (`GRAIN_<c>`) and of the triangles of the
faces of the box (`XMIN`... `ZMAX`), and the groups of nodes that the conditions need;
the command file (`.comm`): the materials of the phases (`ELAS` and `THER`; the crystals
`ELAS_ORTH` and `THER_ORTH` in the frames of their grains, `AFFE_CARA_ELEM` with their
Bunge angles), the six unit strains and three unit gradients of the apparent stiffness
and conductivity, solved by `MECA_STATIQUE` and `THER_LINEAIRE`, the integrals of the
stresses and fluxes over the volume (`POST_ELEM`, `INTEGRALE`) and their tables; the export
file of `run_aster`. The periodic conditions tie each node of an upper face to its image
on the lower faces (`LIAISON_GROUP`, the nodes grouped by the translation that brings
them there, a corner fixed): without a redundant equation at the edges and corners, they
are those of morphoGen's own finite elements on the same mesh; kinematic conditions
impose the affine displacement on the faces (functions of the coordinates). The
voids are left out of the mesh, the nonlinear materials written by their elasticity
([theory notes](theory.md), §111).

`morphogen.io.run_code_aster` (`morphogen code-aster run`) runs a study with Code_Aster,
`run_aster` on the PATH or the program that `MORPHOGEN_RUN_ASTER` names (Code_Aster 18 is
on conda-forge, `code-aster`), and reads its results; a job that fails raises its
diagnostic. Run by Code_Aster 18.1.11, the studies give the apparent stiffness and
conductivity of morphoGen's finite elements on the same meshes within $10^{-13}$, in
Code_Aster's format and in MED, of linear and quadratic tetrahedra, under both conditions,
crystals turned by their grains included (§111).

```bash
# a geometry meshed (or a .vtu mesh) with its materials: the study, run by Code_Aster
morphogen code-aster export composite.json --size 0.1 --materials composite.yaml -o aster/
morphogen code-aster run aster/               # run_aster, then the results
morphogen code-aster read aster/              # the apparent stiffness and conductivity
morphogen code-aster read aster/ -o properties.csv   # a row of a table of properties
```

In a configuration, `outputs.code_aster: true` (or its options `physics`,
`conditions`, `mesh_format`, `fields`) writes the study of the conforming mesh of each
realization (`<name>_<index>_aster/`; the `mesh` section and the materials of the
`homogenization` section); `outputs.conforming_mesh: [mail]` the mesh alone. The tab
Solvers & formats of the interface writes it too (*Code_Aster study*).

## OpenFOAM

`morphogen.io.write_openfoam` writes a case of OpenFOAM (the finite volume code of the
OpenFOAM Foundation, [openfoam.org](https://openfoam.org), and of OpenCFD,
[openfoam.com](https://www.openfoam.com)) of a voxel image or a conforming mesh: its
`constant/polyMesh` (a hexahedral cell per voxel of the phases chosen, or a cell per
tetrahedron; the internal faces in the order OpenFOAM requires, the patches of the six
faces of the box, cyclic in pairs along the periodic axes, and the walls of the phase,
a cell zone per phase), the dictionaries of `system/`, the properties of the fluid in
the files of both lines of OpenFOAM, the fields of `0/` and `Allrun`. The flow
(`kind="flow"`, `simpleFoam`; `foamRun` with the solver `incompressibleFluid` in the
versions of the Foundation that have it) runs through the pores from the face
`XMIN` to `XMAX` (along `axis`) under a pressure drop, without slip on the walls, the
other faces symmetry planes or cyclic pairs; the flow rate through the outlet gives
Darcy's permeability (`read_openfoam`). The diffusion (`kind="diffusion"`,
`laplacianFoam`) gives the effective diffusivity of a phase (the conductivity of a phase
in an insulating matrix, the inverse of its formation factor). The pores that join the
inlet to no outlet are left out (`connected`), where the pressure would be undetermined
([theory notes](theory.md), §111).

The flow through a tetrahedral mesh is solved on its polyhedral dual, which `Allrun`
makes first (`polyDualMesh`; `dual`, `--dual`): on tetrahedra themselves, the finite
volumes of OpenFOAM give a permeability that converges to a value about a fifth too low,
on their dual to the exact one. The dual of a mesh with slivers can be invalid
(`checkMesh`, which `Allrun` runs before the solver), and the conforming meshes of curved
interfaces hold slivers, the more so as they are refined: for the flow through such
interfaces the voxels are the robust case, and `write_openfoam` warns of a flow through a
conforming mesh (§111).
`morphogen.io.run_openfoam` (`morphogen openfoam run`) runs `Allrun` in the environment of
OpenFOAM (its solvers on the PATH, or the file that `MORPHOGEN_OPENFOAM` names, its
`etc/bashrc`, sourced first) and reads the results with the verdict of `checkMesh`. Run by
OpenFOAM v2412, the cases of voxels give the permeability and the diffusivity of finite
volumes written directly on the voxels within $10^{-10}$ (§111).

```bash
# the pores of an image (phase 0) along x: the case, run by OpenFOAM, read back
morphogen openfoam export foam.vti --periodic --viscosity 1e-6 --pressure-drop 1e-3 -o foam/
morphogen openfoam run foam/     # ./Allrun in the environment of OpenFOAM, then the results
morphogen openfoam read foam/    # the flow rate, the mean velocity, the permeability
# the diffusion through the pores of a geometry voxelized
morphogen openfoam export spheres.json --voxels 96 --kind diffusion -o diffusion/
```

In a configuration, `outputs.openfoam: true` (or its options `kind`, `phases`, `axis`,
`lateral`, `viscosity`, `pressure_drop`, `diffusivity`, `difference`, `iterations`,
`connected`, `dual`, and `sample: mesh` for the conforming mesh rather than the image) writes the
case of each realization (`<name>_<index>_openfoam/`); the tab Solvers & formats of the
interface writes it too (*OpenFOAM case*). ParaView opens a case by its empty
`<case>.foam` file. morphoGen's own Stokes solver gives the permeability tensor of
periodic pores on the voxels ([Flow through the pores](physics.md#flow-through-the-pores-and-the-sound-absorbed-by-a-porous-layer)).

## Z-set

The material files of Z-mat run as they are, and the behaviours of morphoGen, among them
those identified on the responses of a morphology, are written as such files
([Z-mat material files](behaviours.md#z-mat-material-files); what of Z-mat's library is
covered: [](zmat.md)).

## FEniCSx

`solver: fenicsx` assembles and solves the finite element problems of
tetrahedral meshes with FEniCSx (dolfinx, dolfinx_mpc, PETSc; Linux or WSL), with the
same results as morphoGen's own solver.

## VIMSEO: the credibility of the simulations

VIMSEO (IRT Saint Exupéry, LGPL-3.0) demonstrates the credibility of simulations with the
tools of GEMSEO: the verification of a code against reference values, the verification of
a solution against its discretization (Richardson's extrapolation, the observed order,
the grid convergence index), the validation against measurements with uncertain inputs
(the area metric), plans of experiments, sensitivities and calibration.
`morphogen.interop.vimseo` gives it the computations of morphoGen as its models
([theory notes](theory.md) §112): a configuration of `morphogen run` whose entries are the
inputs and whose results of a realization are the outputs, and the exact Minkowski
densities of the realizations of a model of spheres with the closed forms of the model as
their references. VIMSEO runs on Python 3.10 to 3.12: morphoGen and VIMSEO together on
Python 3.12 (`pip install "morphogen[all,vimseo]"`).

```python
import numpy as np
from morphogen.interop import vimseo as mv

config = {
    "model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.3},
    "domain": {"size": 6.0, "periodic": True},
    "seed": 1,
    "voxels": {"voxel_size": 0.25},
    "homogenization": {
        "method": "fft",
        "materials": {
            0: {"young": 1.0, "poisson": 0.3, "conductivity": 1.0},
            1: {"young": 10.0, "poisson": 0.3, "conductivity": 10.0},
        },
        "properties": ["conductivity"],
    },
}
simulation = mv.configuration(config, {"h": "voxels.voxel_size"}, ["conductivity.periodic"])
model = mv.model(simulation)  # a model of VIMSEO, its runs archived
tool = mv.solution_verification(model, "h", [0.4, 0.2, 0.1, 0.05], "conductivity.periodic")
print(tool.result.extrapolation["q_extrap"], tool.result.extrapolation["beta"])  # 2.278, 0.90

spheres = {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.3}
exact = mv.model(mv.exact_morphology(spheres, 12.0))  # the exact densities of realizations
points = {
    "model_volume_fraction": np.array([0.2, 0.3, 0.4]),
    "model_radius": np.ones(3),
    "seed": np.arange(3.0),
}
references = mv.dataset(points, mv.theory_of(spheres, points))  # Miles' formulae
tool = mv.code_verification(exact, references, output_names=["volume_fraction"])
```

