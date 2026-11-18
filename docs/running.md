# Configuration files and the command line

morphoGen runs the same computations, and writes the same files, from Python, from
configuration files run by the command line, and from a
[graphical interface](interface.md). This page covers the configuration files and the
command line, the figures and videos of a run, and runs in parallel on a workstation or
a cluster; every key and every command are listed in the
[configuration reference](configuration.md).

## Configuration files

A configuration file (YAML, JSON or TOML) describes the model, the domain, the voxels,
the outputs and the measurements:

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

## The command line

Each command has its `--help`; the main ones:

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
    --property stiffness --loading "tension: S11=1" --loading "E11=0.01 S22=0"
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

## Figures and videos

morphoGen draws sections, periodic tilings, 3D surfaces, histories, Avrami
kinetics and animated GIFs of the formation of structures (`morphogen view`).
Videos (`.mp4` through ffmpeg, `.gif`), played once, in a loop or back and forth:
sections sweeping through a volume and the three orthogonal sections (as in
Avizo), turntables of phases and meshes, cross-sections sweeping through a mesh,
fields and modes animated with a magnified displacement, matplotlib colormaps
(3D views rendered off screen by VTK).

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

`morphogen view` draws the same figures from the command line: sections, `--tile 3`
(periodic tiling), `--surface` (3D view), `--labels` (grains), `--history`,
`--kinetics` (Avrami), and `--animate formation.gif` for the formation of a structure
described by a configuration.

## Parallel and cluster runs

Realizations are independent, and the seed of each one depends only on its place in
the study. They run in parallel processes (`-j`), in job arrays written and submitted
for SLURM, PBS Pro, Torque, SGE, OAR and LSF (`morphogen submit`), or under MPI; each
one leaves a record, so that runs can be resumed, checked (`morphogen status`) and
merged, with results that do not depend on how a run is split.

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
