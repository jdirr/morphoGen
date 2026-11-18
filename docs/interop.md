# Other codes and file formats

What morphoGen generates and computes goes to the codes of the field, and comes back:
images and meshes for ParaView, Abaqus, CalculiX, Gmsh, FEniCSx, DAMASK and Neper, runs of
AMITEX_FFTP prepared and read back, behaviours written as MFront sources and Z-set
material files, and Z-set files run as they are. `morphogen.export(obj, path)` picks the
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
* Gmsh OpenCASCADE scripts.
* Raw data and figures: tables as `.csv`, `.txt` and `.xlsx` (no dependency),
  figures as `.png`, `.tif`, `.svg`, `.pdf` and TikZ (pgfplots code for LaTeX).

The meshes of a polycrystal keep their grains and orientations for Abaqus, FEniCSx and
DAMASK ([Polycrystals](properties.md#polycrystals)); the [recipes](howto.md) show the
usual exchanges in a few lines.

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
