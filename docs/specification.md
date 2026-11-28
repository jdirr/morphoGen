# morphoGen — Specification (cahier des charges)

| | |
|---|---|
| Document | Software requirements specification |
| Version | 1.197 |
| Date | 2026-11-27 |
| Author | @jdirr |
| Status | Version 1.1 **validated** on 2026-10-21. The later versions, proposed, are listed in the change log (§13); changes go through pull requests that update this file and its change log. |

Companion documents: [architecture.md](architecture.md) (modules, data model, formats,
tests) and [theory.md](theory.md) (formulas implemented and validated).

---

## 1. Context

Determining the size of the representative volume element (RVE) of random or
architected materials needs three things:

* large populations of virtual microstructures with controlled morphology;
* their morphological characterization;
* discretizations suited to full-field simulations (finite elements, FFT).

The statistical method of Kanit et al. (2003) relates the ensemble variance of an
apparent property Z to the volume V of the samples:

    D_Z^2(V) = K V^(-gamma),   eps_rel = 2 D_Z(V) / (Zbar sqrt(n)),
    V_RVE = (4 K / (eps_rel^2 Zbar^2 n))^(1/gamma)

It has been applied to Poisson fibres (Dirrenberger et al., 2014), Voronoi
polycrystals (Yang et al., 2019) and nanolayered films imaged by AFM (Bironeau et al.,
2016). Each study needs many realizations at several sizes.

The previous tool chain was:

* **vtkSim** (Centre de Morphologie Mathématique, Mines ParisTech; C++/VTK): vector
  generation of random structures, surface meshing on a Cartesian grid, then commercial
  remeshing (YAMS) and tetrahedral meshing (TetMesh-GHS3D). It is research software
  that is not openly maintained.
* **PuMA** (NASA; Python/C++; NASA Open Source Agreement 1.3): voxel-based generation
  and analysis of fibrous and porous media, with property solvers.

morphoGen is meant to replace this chain with one open, portable and scriptable program
centred on morphological modelling.

## 2. Objectives

* **O1.** Generate virtual microstructures from stochastic morphological models (Boolean
  models, point processes, tessellations, random fields, fibres...) and from
  deterministic architectures (strut lattices, TPMS), periodic or not, reproducibly.
* **O2.** Provide the associated morphological and statistical characteristics: the
  theoretical values when they are known, and measurements on realizations.
* **O3.** Export microstructures in three representations — vector (explicit
  primitives), implicit (level sets) and voxel — and as meshes for finite-element
  codes and volumes for FFT codes.
* **O4.** Support RVE studies end to end: populations of realizations at several sizes,
  measurements, ensemble statistics, scaling-law fits and RVE sizes. This includes
  properties computed by external FE/FFT codes.
* **O5.** Run on a laptop (Windows, macOS on Apple Silicon, Linux) for moderate volumes,
  and in batch on clusters for very large studies or volumes. Use GPU acceleration when
  available (later phase).
* **O6.** Remain maintainable and extensible by a research group working in Python, and
  publishable as an open distribution.

## 3. Requirements (2026-10-21)

| Topic | Content |
|---|---|
| Purpose | Morphological modelling: the generation of stochastic or deterministic virtual microstructures, by morphoGen's own kernels; the existing tools (VTK, PuMA, gmsh) as assessed in §7.3. |
| Users | Researchers in the mechanics and physics of materials, first the author's research group (O6); an open-source distribution (phase 4, §10). |
| First version | Random microstructures from several morphological schemes, their statistical and morphological characteristics, and meshes for FE (or 3D volumes for FFT), for RVE-size studies as in Dirrenberger et al. (2014) (version 0.1, §10). |
| Inputs | A morphology type (granular, porous, fibrous, tessellation, etc.) and its morphological parameters, the volume fraction first among them (porous media). |
| Outputs | 3D volumes, vector and implicit geometry (§5.2), meshes and volumes for FE and FFT codes (§5.5). |
| Downstream codes | Finite element codes, open source (FEniCSx) or in-house as well as Abaqus/CalculiX, and FFT codes (§5.5); morphoGen also solves these problems itself (§5.8). |
| Usage | Python API, command line, graphical interface, batch processing on clusters (§5.6, §5.7). |
| Performance | An ordinary laptop for moderate volumes, clusters for very large ones (O5, NFR-2); optional GPU acceleration (FR-B3, §7.4). |
| Language | Python with compiled kernels (Numba), §7.1. |
| Dependencies | Open-source libraries, no commercial licence (§7.2, NFR-6; the optional CUDA libraries are free of charge, D3). |
| Platforms | Windows, macOS (Apple Silicon), Linux (NFR-1). |
| Language of the project | English (NFR-8). |
| Licence | GPL-3.0-or-later, decided on 2026-10-21 (decision D1); the author holds the rights, which the institution confirmed on 2026-11-17. |
| Existing code | None reused: the statistical post-analysis is written from Dirrenberger et al. (2014) in phase 2 (§3.1, D6). |
| Reference data | The RVE studies of Dirrenberger et al. (2014), Yang et al. (2019) and Bironeau et al. (2016) (§1, §9). |

### 3.1 Phase 2 (2026-10-21)

| Topic | Content |
|---|---|
| Statistical code | The statistical analysis written from Dirrenberger, Forest & Jeulin (2014), no earlier code existing (FR-M4, D6). |
| Finite element workflow | The cases of the Z-set input files of the earlier workflow (D7): KUBC, mixed (MBC) and periodic conditions; bulk (k) and shear (μ) loadings; outputs: volume averages of strain, stress and energy, volume, second moments. |
| Solvers | Free codes rather than Z-set (D7, §7.3). |
| Discretization | No finite elements on voxels when the morphologies come from implicit functions: their surfaces are meshed and filled with volume meshes that follow the morphology instead of approximating it with voxels (FR-E5, FR-H3). Voxels are reserved for the FFT solver (FR-H4). |

### 3.2 Phase 3 (2026-10-21)

| Topic | Content |
|---|---|
| Graphical interface and visualization | A graphical interface and visualization tools (FR-I4, FR-I6, D2). |
| Cellular automata | A generation module based on cellular automata and what follows from them, notably combinatorial design and topology optimization problems, with the algorithms dedicated to such systems (§5.9–§5.11). |

### 3.3 Phase 2b (2026-10-21)

| Topic | Content |
|---|---|
| Polycrystals (D8) | Finite elements on conforming meshes of polycrystals (D8, FR-H10, FR-H11). |
| Phase 2b | The rest of phase 2b of the roadmap (§10): exports (FR-E6), measures (FR-M3), combinations of models (FR-G12), further models (FR-G9), chunked large volumes (FR-R4). |
| Parallel runs | Parallel computations, notably on clusters, as a priority (FR-B5, D11). |

### 3.4 Phase 2c (2026-10-22)

| Topic | Content |
|---|---|
| Periodic polycrystal meshes | The grains periodized rather than the mesh of the cube, with a volume of grains equal to that of the cube: periodic at the design, not at the mesh, which naturally gives matching edges and nodes for the PBC (FR-H12). |
| Periodic element | A periodic element like `***mesh periodic` in Z-set: a 3D element with extra degrees of freedom in macroscopic strain and stress, applied at the scale of the element, which prescribes a macroscopic strain or stress field over a whole mesh, hence homogenization (FR-H13, D13). Reference: the author's homogenization notice for Z-set (2022). |
| Post-processing | Macroscopic values, homogenized values (relative to the volume fraction for porous media) (FR-H15). |
| Ashby charts and database | Normalized Ashby-type charts that display the values of the generated microstructures, and a database of all the configurations tested, filterable for display (§5.12, D14). |
| Nested models | Nested models, for example a Cox-type model of spheres for nanocomposites, and hierarchical, even fractal architectures (Rayneau-Kirkhope, Mao & Farr, PRL 2012) (FR-G13, D15). |
| Bio-inspiration | Morphological and growth models inspired by the living world (FR-G14, D15). |
| Wood rays (2026-10-22) | The rays of the wood model (FR-G14). |
| Nonlinear behaviour (2026-10-22) | The macroscopic strains as degrees of freedom of the periodic element, for nonlinear behaviours (viscoplasticity in particular) (FR-H16, FR-H17, D16). |
| Local continuous integration (2026-10-22) | The jobs of the continuous integration run locally when GitHub Actions is unavailable (D18). |
| Crystal plasticity, finite strains (2026-10-22) | Méric–Cailletaud crystal plasticity for polycrystals (FR-H18); finite strains for every nonlinear behaviour and loading path (FR-H19, D17). |
| Nonlinear FFT solver (2026-10-22) | The nonlinear FFT solver of voxel images (viscoplasticity and crystal plasticity, macroscopic strains as degrees of freedom): the iterations of D16 on voxels (FR-H20, D19). |
| RVE statistics of nonlinear responses (2026-10-23) | RVE statistics of the viscoplastic responses (the methodology of Yang et al. 2019 with the nonlinear FFT solver) (FR-V7); the GPU backend, which needs large packages, in phase 3b (D3, §3.5). |

### 3.5 Phase 3b (2026-10-23)

| Topic | Content |
|---|---|
| GPU backend, further models (2026-10-23) | A GPU backend for the FFT solvers of voxel images (FR-B3): CuPy (D3), chosen after benchmarks on the development machine (NVIDIA RTX 3000 Ada laptop GPU); the remaining models of FR-G9: hard-core fibres, curved fibres (worm-like chains), rotated and graded lattices. |
| GPU integration (2026-10-23) | The behaviours integrated on the GPU to accelerate the loading paths: they are ported to CUDA (D3) and the integration points stay on the GPU; the crystal plasticity paths gain 1.4 to 1.9 times on the development laptop, bound by the double precision of its GPU (§7.4). |
| AMITEX workflows (2026-10-23) | The AMITEX workflows (exports and reading of the results): the inputs of AMITEX_FFTP runs of an image and the reading of their results (FR-E6); the formats follow AMITEX's documentation and its official Python interface (py4amitex, cpp4amitex); no AMITEX installation on the development machine (Windows), so the runs themselves are not tested. |
| MFront (2026-10-23) | The integration of MFront laws into morphoGen's solvers through MGIS (finite elements and FFT voxels, small and finite strains), validated against morphoGen's laws. MGIS and MFront exist on conda-forge for the three systems; MFront compiles the behaviours with a C++ compiler, which the Windows development machine lacks: they are installed with conda's compiler in WSL (Ubuntu 22.04), where the tests of the MFront behaviours run, on the CPU and the GPU (FR-H21, D20). |
| MFront exports (2026-10-23) | The exports of morphoGen's laws for AMITEX and Abaqus: morphoGen writes its nonlinear materials as MFront sources (FR-E7), which its AMITEX exports compile with the umat interface (FR-E6); the order of the shear components (11, 22, 33, 12, 13, 23 is that of Z-set and TFEL, the usual order 11, 22, 33, 23, 13, 12) is applied at every interface and checked with distinct shear moduli. AMITEX and Abaqus are not installed on the development machine: the laws are checked through MGIS and through the umat interface called as AMITEX calls it. |
| Abaqus materials (2026-10-23) | The MFront exports for Abaqus: the Abaqus inputs of conforming meshes define the materials of their phases, the elastic ones (isotropic and anisotropic in the axes of the crystals) and the nonlinear ones as MFront laws of the abaqus interface (FR-E6, FR-E7). Abaqus is not installed on the development machine: CalculiX, which reads Abaqus inputs, is installed in WSL to run the elastic materials, and the MFront laws are called as Abaqus calls them. |
| Voxel meshes, crystals and user laws in the exports (2026-10-23) | The materials of the Abaqus inputs of voxel meshes, crystal plasticity in finite strains as an MFront source (morphoGen's kernel in it, D20), user-supplied MFront laws in AMITEX and Abaqus inputs (FR-E6, FR-E7). |
| Finite strains in the exports (2026-10-23) | The MFront laws in finite strains in the AMITEX and Abaqus inputs. The AMITEX runs and the Abaqus inputs of finite strains take every phase but the voids as an MFront law of finite strains, Hencky's elasticity for the elastic ones (the native elasticities of AMITEX and Abaqus in finite strains are not morphoGen's), user-supplied laws of finite strains included (FR-E6, FR-E7, D20). AMITEX and Abaqus are not installed on the development machine: the laws are called as the codes call them in finite strains. |
| Dead leaves and Torquato (2026-10-24) | The dead-leaves type models and Torquato's approaches in the module of morphological generation. The dead leaves get families of leaves of their own sizes, spheroidal leaves and a finite coverage over a background, with their exact bivariate laws (Jeulin 1997); after Torquato (2002), dense packings of hard spheres grown by the Lubachevsky–Stillinger algorithm (the dense packings beyond the RSA limit of FR-G9), penetrable spheres with impenetrable cores (the cherry-pit model), the stochastic reconstruction of Yeong & Torquato (FR-G15) and the descriptors of random heterogeneous materials (FR-M3). |
| NTFA (2026-10-24) | An NTFA solver: `homogenization.ntfa` (FR-H24). |
| Level-set foams (2026-10-24) | The level-set tools for the modelling of foams, as in the paper of N. Degdeg and J. Dirrenberger (an anisotropic open-cell copper foam): the model `level_set_foam` (FR-G16). |
| FEniCSx (2026-10-24) | The FEniCSx solver integrated, after the demos of COMET-FEniCSx, as the solver of morphoGen's own problems: `solver: fenicsx` for the finite element problems (FR-H23, D21). |
| Transparent dead leaves and hyperuniformity (2026-10-24) | The random function of transparent leaves of Jeulin and its excursions, with their exact covariance, and Voronoi tessellations of stealthy hyperuniform germs (the collective coordinates of Torquato and coworkers) (FR-G9). |
| Post-bifurcation (2026-10-24) | A module for a post-bifurcation analysis in the manner of Azulay (R. Azulay's thesis and papers with C. Combescure and J. Dirrenberger): `symmetry`, `homogenization.frames` and `homogenization.bifurcation` (FR-H26). |
| Visualization, exports, behaviours, fracture (2026-10-24) | A graphical interface organized by the steps of a study; exports in several forms (domain-specific files, raw data as .csv, .xlsx, .txt, vector files as svg or TikZ, bitmaps as png, tiff); videos (mp4) of morphologies (mesh exploration, as in Avizo) and of simulations (colormaps, magnified fields, back and forth, loops, cross-sections); the behaviour bricks of Z-set reused; cohesive zone elements, crack propagation by remeshing and by XFEM, in 3D and in finite strains; the analysis in the manner of Azulay in 3D; every module connected in a workflow from the morphological model to the fields of the FE mesh, their exports, images and videos (FR-E8, FR-I7, FR-I8, FR-H26 to FR-H30: versions 1.37 to 1.43 of §13). |
| Mixed-mode cracks (2026-10-25) | 3D cracks (modes I+II+III) in finite strains: `fracture.mixed` (FR-H30). Also examined, all in finite strains: the analysis in the manner of Azulay with a nonlinear behaviour, Abouaf's porous plasticity, Forest's microfoam model and 3D micromorphic media (the next row). |
| Finite strains, porous plasticity, microfoam (2026-10-25) | The analysis in the manner of Azulay in finite strains; the porous_plastic behaviours of Z-mat beyond gen_evp; the microfoam model of Forest's article (Dillard, Forest & Ienny 2006); for fracture, finite strains in elasto-viscoplasticity, with crystal plasticity and nonlocal models: `homogenization.path_bifurcation` (FR-H26), `homogenization.porous` (FR-H31), `morphogen.micromorphic` (FR-H32), `fracture.bulk` (FR-H33). |
| Conforming meshes (2026-10-25) | Conforming meshes from level sets rather than voxels: `mesh.mirrored_mesh`, `mesh.ImplicitBody`, matching faces along chosen axes (FR-E5), the post-bifurcation on them (FR-H26). |
| Micromorphic models, remeshed cracks, point groups (2026-10-25) | The micromorphic models in finite strains and micromorphic ductile damage (FR-H32), remeshed crack propagation in inelastic bodies with the transfer of their state (FR-H33), the full point groups of the mirrored meshes and the equibiaxial bifurcations (FR-E5, FR-H26). |
| Buckling of 2 x 2 x 2 cells, polycrystal cracks, triaxiality (2026-10-25) | The buckling of the cubic cell of 2 x 2 x 2, crack growth in a remeshed polycrystal and the triaxiality in the damage: the linear post-bifurcation of cells of several primitive cells split by their translations and the little co-groups of their wave vectors (FR-H26), the grains on the crack meshes of a polycrystal and its state transferred grain by grain (FR-H33), the micromorphic ductile damage driven by the plastic strain weighted by the stress triaxiality (FR-H32); the forms of the graphical interface open with the example configuration of each model. |
| Interface workflow (2026-10-25) | The fields of a simulation shown as they evolve along its video; a workflow from module to module (a morphology generated, then sent to the simulation, then to the export), its items carried from one module to the next rather than videos recorded at each simulation; boxes of Generate larger than 160 voxels per edge (up to 2048); the Optimize module started from a simulation result or a morphology; the Formation and Design space modules documented in the interface; the level-set functions used rather than voxels. Delivered: the videos of a load colour the fields as they grow; the workflow from tab to tab through the workspace of the session (Generate, Simulate, Export, Optimize); the views and measures of the level sets, sections at up to 2048 pixels per edge without a voxel image of the volume, voxel images on demand up to 2048 voxels per edge within the free memory; optimizations from a morphology or a simulation; the formation shown in Generate with the share of the cells changed at each step, and the tile assemblies explained (FR-I4). |
| Every function in the interface (2026-10-25) | Every function reachable from the graphical interface (the NTFA, bifurcation, fracture, nonlocal models and triaxiality were not); a Statistics tab for the RVE and related tools; tabs without numbers: the modules can be called in any order, although the operations may have an intrinsic one. Delivered: tools built from a registry, the tab Statistics, tabs of the mechanics (loading paths, NTFA, material points), fracture, stability, nonlocal damage, couplings and the formats of other solvers (FR-I4). |
| Interface menus, cracks through any morphology (2026-10-26) | The modules of the graphical interface in a sequence by the steps of a study, some gathered in submenus; cracks through an arbitrary morphology, beyond the plates and polycrystals of the fracture tools. Delivered: the menus in four sections with their tabs, the items sent on to the forms that take them, compact forms (FR-I4); cracks through any morphology, in the library and in the tab Fracture (FR-H33). |

### 3.6 Phase 4 (2026-10-28)

| Topic | Content |
|---|---|
| Release preparation (2026-10-28) | A faster local CI and the preparation of version 1.0, publication excepted: the local CI runs the tests in parallel (pytest-xdist, one file per process, the longest first) and the Python versions at once, its environments kept from one run to the next (D18); the documentation site is built from `docs/` (Sphinx, the API reference from the docstrings, NFR-8); the public interface is defined, documented and kept by a test (NFR-10); the citation file (`CITATION.cff`), a draft paper for JOSS (`paper/`), a conda-forge recipe (`packaging/conda-forge/`) and the steps of a release (`docs/release.md`) are ready. Publication (PyPI, conda-forge, the DOI, JOSS) follows, once the points of D1 are settled; JOSS reviews software whose repository has been public for six months with active development. |
| Shorter CI (2026-10-29) | A shorter CI, by two tiers of tests and lighter tests of the interface: the slowest tests (marker `slow`, the numerical checks of more than about 15 s, two thirds of the time) run under one Python version, the others take the fast tier (D18, NFR-7); the tests of the interface run on coarser meshes, fewer steps and smaller images where the tests of the library check the same physics, their assertions unchanged. |
| Hill plasticity (2026-10-29) | Hill's anisotropic plasticity in morphoGen's kernels, on the CPU and the GPU (FR-H16, FR-B3), in Z-set's form (its coefficients by component), identified on the responses of a morphology (FR-H34), read and written in Z-mat files (FR-H27), in the forms of the interface (FR-I4); the macroscopic law of an anisotropic morphology (fibres, laminates, lattices) can then be identified and written back for Z-set; then the macroscopic law of a morphology in one go (its loadings run, a law estimated on them, von Mises, Hill or, of a porous morphology, Gurson identified), in the library and as a tool of the interface; then the kernels of Hill's criterion and of porous plasticity made faster, and the cost of each behaviour of the points measured (NFR-2). |
| Kernels (2026-10-29) | The analytical Jacobian of the porous update, porous plasticity on the GPU, the tangent of finite strains from the six directions of dC: the porous update and its consistent tangent take the analytical Jacobian of their four equations (the second derivatives of the matrix stress, implicit ones for GTN), and the FFT solvers on the GPU integrate porous points with a CUDA port of the update (FR-B3); the tangent of finite strains takes the six directions of dC (3 to 12 % faster, Jacobi's eigen decomposition keeping most of the cost); the local CI clears the Numba caches of its kept environments, which a change of a kernel called from another file left stale (D18). |
| GPU benchmark, crystal kernel, softening points (2026-10-29) | The porous paths in the benchmark of the GPU (stiff spheres in a GTN matrix, the CPU and the GPU), and the crystal kernel 1.4 to 1.7 times faster, the exponentials of its isotropic hardening computed once per family; then, on the stall found in that benchmark, MINRES takes the linear solves over from the conjugate gradients when the tangent of softening points is indefinite; then the softening paths studied (the reference medium of the preconditioner kept, their divided steps a localization of the damage), and the FFT tools of the tab Mechanics on the GPU (a device field). |
| Exports (2026-10-30) | The thermal expansion of the MFront sources in small strains (von Mises and Hill: the brick's Hooke potential; crystals: `@ComputeThermalExpansion`), missing from the exports of 2026-10-23, checked through MGIS against morphoGen's kernels; then in finite strains for von Mises and Hill, MFront's mean coefficient matched to morphoGen's logarithmic one, then the crystals and Hencky's elasticity of finite strains (morphoGen's own sources); then the Abaqus inputs, whose nodes start at 293.15 K when a material expands (MFront's abaqus interface cannot expand the brick's laws of finite strains, refused there); the AMITEX_FFTP inputs (no change of temperature there) still refuse it. |
| Porous exports, percolation threshold (2026-10-30) | Porous plasticity in the MFront sources, its kinematic hardening, and an estimate of the percolation threshold. Porous plasticity in the MFront sources: the materials of Gurson-Tvergaard-Needleman with the brick `StandardElastoViscoPlasticity` and its porous criterion `GursonTvergaardNeedleman1982` (the effective porosity f or Tvergaard & Needleman's coalescence, `f_c` and `f_r`), the porosity in the implicit system of the brick (morphoGen's backward Euler), Plastic and Norton flows, linear and Voce hardening, the constant nucleation from p = 0, small and finite strains, thermal expansion; the porosity f0 until the matrix flows (the natural state of the internal variables zero); checked through MGIS against morphoGen's kernel within 10⁻¹⁰ of the stress scale along paths that grow the porosity, and through the umat (AMITEX) and abaqus (Abaqus) interfaces; refused with their reasons: Rousselier's criterion (MFront's is homogeneous of degree 1), the elliptic one, the Gaussian and exponential nucleations, thresholds of the nucleation, other effective porosities. The kinematic hardening of porous plasticity on the CPU and the GPU, with the analytical Jacobian and the Z-mat import: the Armstrong-Frederick and Prager backstresses of GTN and the elliptic criterion, the potential of σ - X (Besson & Guillemer-Neel 2003), the backstresses with their hydrostatic part, integrated along the invariants of the relative trial stress with the analytical Jacobian and the consistent tangent, on the CPU and the GPU, in small and finite strains; read and written in Z-mat's porous_plastic with the bricks of gen_evp; refused for Rousselier's criterion and in the MFront export (MFront 5.1's brick has no kinematic hardening with a porosity evolution); dense, von Mises's kernel within 10⁻¹²; then a new tool for the percolation threshold, `analysis.percolation_threshold` (the percolation probability of independent realizations, the crossing of 1/2 by bisection on the fraction, common random numbers; joining the faces, or wrapping around a periodic box), checked against the site percolation of the cubic lattice and overlapping spheres. |
| Percolation in the interface (2026-10-30) | The percolation threshold in the interface and on the command line: the threshold of `analysis.percolation_threshold` for the model of a run configuration (`analysis.percolation.family_from_config`: its `model`, `domain` and `voxels` sections, the parameter varied, by default the volume fraction; a model without it refused), the command `morphogen percolation` (the threshold, its bracket, the probabilities tried; JSON) and the tool "Percolation threshold" of the tab Statistics for the model of a morphology of the workspace (the probabilities against the fraction, drawn and tabulated, with its progress and its stops); checked on Boolean spheres through the command, its errors and the harness of the interface. |
| Forest fire (2026-10-30) | A forest-fire algorithm for the percolation, which was computed otherwise (the components labelled by scipy, joined by a union-find on periodic images); `analysis.burn` burns the phase from a face as Herrmann, Hong & Stanley (1984), a compiled breadth-first search over a queue of flat indices that stops when the opposite face is reached, with the chemical distance, the front at each step and the burning times (`Fire`); `percolation_threshold(method="fire")`, `morphogen percolation --method fire` and the field "algorithm" of the tool "Percolation threshold" take it as the criterion (face to face, periodic or open across the axis: as the labels on open boundaries, implied by their wrapping on periodic ones); checked against the labels and scipy's components of the graph on 200 site lattices, on exact chemical distances and a pure-Python breadth-first search, and on the threshold 0.3116 of the simple cubic lattice. |
| Finite-size scaling, porous MFront sources (2026-10-30) | The finite-size scaling of percolation thresholds (the thresholds of several box sizes fitted to p_c + a L^(-1/ν), the error of p_c, ν given or fitted; the sized family of a configuration; `morphogen percolation --sizes` and the field of the tool) and the wrapping of a phase around a periodic box in the measure `connectivity`; then every porous behaviour in the MFront sources: the brick where it matches morphoGen exactly (GTN), elsewhere a source that calls morphoGen's own porous update (the CUDA port compiled as host C++: Rousselier, the elliptic criterion, every nucleation and its threshold, the tables of f*, the kinematic hardening), equal to the kernel to round-off through MGIS, umat and abaqus. |
| Anisotropy, fibres, tilings, learned behaviours (2026-10-30) | The anisotropy of percolation, in 2D and 3D, and anisotropic versions of the morphological models; the fibre models of Altendorf and Jeulin, and other published ones; grids other than square for the cellular automata (e.g. triangular, hexagonal, quasi-periodic in 2D, and in 3D); a module of behaviours learned from data; a gallery of examples and a tutorial. Delivered: the anisotropy of percolation in 2-D and 3-D (planar images in the percolation tools, 4 or 8 neighbours, ν = 4/3; the forest fire across congruent cores of the box along any direction; the thresholds of a family along directions with paired deviations, ratio and threshold tensor; `morphogen percolation --directions`, the field of the tool, the measure `directional_percolation`); cellular automata on triangular, hexagonal and Penrose (pentagrid, quasi-periodic) tilings, their prisms in 3-D and the BCC and FCC Voronoi lattices, with CSR neighbourhoods, generalized rules and cell-by-cell rasterization to images (the square grid unchanged; a 3-D quasi-periodic lattice left for later); Altendorf and Jeulin's fibres (`random_walk_fibers`: random walks of multivariate von Mises-Fisher steps between a main direction, with Schladitz's β law of orientations, and the previous step, packed without overlap to high fractions by the paper's force-biased repulsions and recovery springs); anisotropic versions of the morphological models through one `anisotropy` field (stretches along turned principal axes, preserving volumes), which maps the isotropic realization of point processes, tessellations, leaves, growths and lattices and enters the covariance of the Gaussian field, an exact map of grids in periodic boxes for axes along the box; the other fibre models: the contact packing of Keilmann, Redenbach and Willot (2026) in `random_walk_fibers`, and entangled cross-linked fibres (`cross_linked_fibers`, Chatti et al. 2018 and 2020, with their Gaussian girdle law), checked against Toll's estimate and the exact contact statistics of Poisson lines; a first module of behaviours learned from data (`morphogen.learning`: a perceptron, an input-convex potential and a gated recurrent unit with analytic gradients, trained on the responses of morphoGen's behaviours or microstructures along random paths, integrated as `learned` materials at a material point, by finite elements and on voxels; `morphogen learn`); the exact covariances of oriented fibre systems (Peyrega et al.: Poisson lines, cylinders and spherocylinders for every orientation law, with von Mises-Fisher and mixed laws), the initial placement of least overlap of Altendorf, Jeulin and Willot (2014) and Delaunay fibre networks (Ban et al. 2016); the quasi-periodic Ammann-Beenker tiling and the 3-D icosahedral tiling of golden rhombohedra (de Bruijn's dual method), Gray-Scott reaction-diffusion on any tiling with a finite-volume Laplacian, and rule fitting on any tiling from the CLI (`morphogen fit-rule --tiling`); for the learned behaviours: a Learning tab in the interface, learned laws turned with the orientation of their material (a crystal learned once serves every grain of a polycrystal, the 24 rotations of the cube in the training) and the stress-free thermal strain taken out before the law; thermodynamically consistent learned laws (a learned generalized standard material: a convex free energy and a convex dissipation potential of learned internal variables, updated by the incremental variational principle, with a nonnegative dissipation and a symmetric tangent by construction); the stiffness of random fibre networks and open foams by periodic frames of Euler-Bernoulli beams (Delaunay networks stretching-dominated but below the affine estimate, Voronoi foams bending-dominated) and the softening of networks of non-identical beams of Ban et al. (2016); periodic approximants of the Penrose, Ammann-Beenker and icosahedral tilings (the Fibonacci and Pell approximants of the cut-and-project, the tiles unchanged), so that quasi-periodic lattices fill periodic boxes; Timoshenko beams in the periodic frames (the exact two-node stiffness, Cowper's coefficient, the geometric stiffness tending to Engesser's load), which settle the softening of Ban et al.'s Voronoi networks of thin sections (case 4) and bring their Delaunay case 9 fifteen times closer |
| Learned laws in finite strains, beam frames (2026-10-31) | Consistent learned laws in finite strains, the beam frames against the FFT solver and Ban et al.'s case 9: the learned generalized standard materials in finite strains (a free energy of the logarithmic strain or of the Green-Lagrange strain, objective, with a nonnegative dissipation and a consistent tangent), 11 % on large-strain von Mises paths where the recurrent law and the Green-Lagrange space reach 64 %; the frames of beams against the FFT solver on the voxel images of the same foams and networks (frame/FFT 0.88 to 0.98 for Euler-Bernoulli beams, the joints the cause), rigid joints as an option (Timoshenko beams rigid over a third of their radius within 4 % of the continuum, a fitted length), and Ban et al.'s case 9 examined (a factor 1.45 left, its coordination explained); the stiffness of the frames of beams of network realizations in the interface (Mechanics) and the command line (`morphogen frame-stiffness`); the perceptron, the learned potential and the recurrent network in the logarithmic strain space (the recurrent network 11 % on large-strain von Mises paths instead of 64 % in the Green-Lagrange strain, Hencky's elasticity 0.02 % instead of 4 to 16 %), the earlier models read as before; the thermal strain of the learned laws in finite strains aligned on that of the Hencky points (the strain less the thermal one in the logarithmic space, no factor J_th), so that a learned Hencky elasticity gives the Hencky points' stress to round-off when heated; Ban et al.'s replicas of random beam stiffnesses and a drawing of the frame coloured by its forces or energies, in the interface and the command line; a gallery of the 64 templates and a tutorial of a whole study in the documentation, both kept in step with the code by tests (required on 2026-10-30); the thin networks of the gallery (Delaunay fibres, open foams, strut lattices) seen as surfaces in 3D |
| Learned MFront laws, percolation exponents, tensors, tangents (2026-11-01) | The MFront sources of the learned laws and the critical exponents of percolation: the learned laws written as MFront behaviours (the four models in C++, small and finite strains, heated and turned), equal to morphoGen's integration to round-off through MGIS, umat and abaqus; the critical exponent t of the conductivity at the percolation threshold by finite-size scaling (2-D discs 1.13 +- 0.16 +- 0.21 for 1.30, 3-D spheres 2.37 +- 0.36 +- 0.32 for 2.0), the voxels of the FFT solvers found to conduct through their corners; then the elastic stiffnesses C_ij and compliances S_ij beside the engineering constants E1, E2, E3, mu12, mu23, mu31, nu12, nu21, etc.: the stiffness and the compliance, their notation stated, in every elastic output of the interface, the command line, the JSON files and the results database; then, for phase 4 (version 1.0, publication excepted): the JOSS draft, the conda-forge recipe and the citation file brought up to 0.6, kept in step with the package by a test, and the steps of the release updated (`docs/release.md`); the mean stiffness and compliance of each volume of the RVE studies, their scatter and the RVE size of each component; then the tensors of the window method, the elastic exponent f and the nonsymmetric tangents in morphoGen's solvers: the stiffness, compliance and conductivity tensors of the RVE studies by windows, the tensor of a window the window means of the mutual energy densities (symmetric, positive, their mean over the windows tiling the box its tensor to round-off); the rigidity threshold of the voxels from their moduli by finite-size scaling (sites of the square lattice: 0.4647 +- 0.0056, between the thresholds through 8 and 4 neighbours) and the elastic exponent f there (2.47 +- 0.21 +- 0.34 with nu = 4/3, 2.24 with the 1.21 of central-force rigidity, below the 3.96 of bond bending); the nonsymmetric tangents in morphoGen's solvers: the full tangents of the laws that have no symmetric one (learned perceptrons and recurrent networks, MFront laws declaring none, crystals with latent hardening, porous and kinematic laws, finite strains) solved by restarted GMRES on the CPU and the GPU, on voxels and finite elements, the symmetric cases unchanged to the bit (a learned recurrent network from 148 to 47 Newton iterations on voxels; the learned MFront laws converging through MGIS); then a solver fast near the rigidity threshold and f in 3D, shifted windows, the edges of planar windows and the restarts of GMRES: the shifted tilings of the window method on periodic images (no new solve; the variance pooled within the tilings, the count of the uncertainties that of one tiling) and the edges of planar windows as the square roots of their areas; the voxels' own trilinear system on the bearing phase, solved by conjugate gradients on a sparse LU factorization in 0.04 to 0.6 s near the rigidity threshold where the FFT solver takes thousands of iterations or stops (an option, not the default: dense phases favour the FFT), and f measured with it, 2.88 +- 0.12 +- 0.23 on 16² to 256² sites of the square lattice (p_r = 0.4627 +- 0.0016) and 3.23 +- 0.30 +- 0.25 on 8³ to 32³ sites of the cubic lattice (p_r = 0.1307 +- 0.0009, between the thresholds through 26 and 18 neighbours); the restarts of GMRES deflated (GMRES-DR(50, 20): Rousselier's softening points on 8³ voxels from 41,068 to 11,064 operator products, without stalled solves) and the tangent chosen by its measured asymmetry (the symmetric solvers below 2e-2 over all the points, where morphoGen's own laws stay, up to 1.6e-2; the full tangents of the learned laws, 0.10 to 0.21), the localizing GTN composite found to bifurcate (the symmetric path the stable solution, the full one an unstable equilibrium); then the rigidity decided by the energies of the solutions, checked against the round-off of the element matrices (1,919 of 1,920 decisions on the planar realizations unchanged), the cut of the factorized voxels lowered to 1e-12, the voxels that cannot bear pruned exactly; the 3-D sites again on 8³ to 48³ (p_r = 0.12904 +- 0.00029, f = 4.58 +- 0.19 +- 0.07, the slope still steepening with the size), Boolean discs (p_r = 0.608 +- 0.006, f = 3.20 +- 0.19 +- 0.24) and spheres (p_r = 0.202 +- 0.004, f = 4.65 +- 0.59 +- 0.05), their exponents those of the sites within the errors; then the 2-D sites redone and the 3-D sites from 64³, the recycling of GMRES and the stalls of 16³, the shifts in the interface: the shifts of the window method in the RVE tool of the interface (refused on non-periodic morphologies with the messages of the API) and the areas of planar windows named on the axes of the figures; the recycling of GMRES (GCRO-DR from the first restart of a path: Rousselier on 16³ voxels from 55,869 to 14,947 products and 819 to 106 s, below CG with MINRES at 35,244), the solves stopped where their residual stagnates and the steps divided after three such corrections (the stalls of 16³ in one softening step whose preconditioned tangent has dozens of eigenvalues of both signs near zero), and auto decided by the largest asymmetry of a phase (a learned phase of 0.6 % of the volume keeps its full tangent); the rigidity confirmed at the floor of the residual (the factorized voxels solved to 1e-10, the energies under 1000 cuts solved again at 1e-11: the old 3-D threshold low by floppy images left bearing at 1e-8), the 2-D sites on 16² to 256² (p_r = 0.4617 +- 0.0004, nu = 1.14 +- 0.02 from the widths, f/nu = 2.26 +- 0.07 +- 0.05 rising with the size) and the 3-D sites on 8³ to 80³ (p_r = 0.1304 +- 0.0002, f/nu = 4.48 +- 0.09 +- 0.09, 4.85 +- 0.21 +- 0.18 with a correction to scaling: bond-bending-like in 3-D, not central-force, the 2-D class not told) |

### 3.7 Laser-architectured sheets (2026-11-06)

| Topic | Content |
|---|---|
| Laser-architectured sheets (2026-11-06) | A module dedicated to the simulation and optimization of sheets architectured by localized laser treatment, based on the thesis of Zhige Wang, on morphoGen and on the scientific literature, with the six choices of a study: the loading case, which defines the objective function; the base material; the treatment (thermal, carburizing, nitriding, diffusion through a coating...); the shape of the sheet or specimen; the laser configuration (single or symmetric); and the architecturing pattern, i.e. the path of the laser treatment. |
| Delivered | `morphogen.laser` (FR-L1–L6, §5.13) and `docs/laser.md`: the six choices stated with their reasons and the literature (uniaxial tension, `R_m A_g` at a minimum yield strength; annealed MS1500; transformation hardening, with carburizing, nitriding and alloying as options; the dog-bone of Wang's thesis; two beams on both faces; corrugated tracks optimized), the chain of models and its checks against closed forms and the thesis. |
| Cellular design of laser-treated sheets (2026-11-07) | The results of the paper of Wang et al. (2023) and of chapter 5 of the thesis reproduced; design and optimization with cellular automata and topology optimization, as proposed in SCOLASTIC; a survey of the scientific literature on such approaches, other materials included. |
| Delivered | FR-L7–L9 (`morphogen.laser.cells`): the phases of the paper with a nonlocal fracture of the martensite; the cellular design loop of SCOLASTIC (cells, phases, sensitivities by replays of the deformation history, BESO and neighbour exchanges tested by finite elements, periodic orbits, treatability by a spot, multigrid; HCA and density-based topology optimization with MMA); the designs turned into laser tracks and treated by the process model. |
| Other loading cases and materials (2026-11-07) | Other loading cases: the blunting of the crack of a CT specimen, proposed in SCOLASTIC, and the strength of an L-shaped plate; the control of the plastic anisotropy by the treatment; the application to alternative materials. |
| Delivered | FR-L10–L12 (`morphogen.laser.structures`): the compact tension specimen and the L-shaped plate, the anisotropy of treated sheets (Lankford's coefficients, Hill's coefficients), the boron steel 22MnB5 and the aluminium alloy AA6082-T6 through the whole chain. |
| Compensation of the plastic anisotropy (2026-11-07) | A textured sheet, typically with a rolling texture, treated so as to compensate its plastic anisotropy. |
| Delivered | FR-L13 (`morphogen.laser.periodic`): Hill's 1948 criterion in plane stress for the rolled sheet (the Lankford coefficients of a γ-fibre steel), the tracks isotropic or keeping a share of the texture; the periodic cell of the sheet treated all over (the anisotropy of the sheet, not of a narrow specimen). |

### 3.8 Multiphysics, three dimensions, implicit geometry, documentation (2026-11-12 to 2026-11-14)

| Topic | Content |
|---|---|
| Multiphysics homogenization (2026-11-12) | A homogenization module, documented for researchers in the mechanics and physics of materials, that treats mechanical, thermal, electrical, electromagnetic, chemical, diffusive, optical and acoustic problems, static and dynamic, and the weak and strong couplings between these physics; the statistical analysis of RVE sizes for every physical property; nonlinear mechanical behaviour as a foundation: every module compatible with plasticity, viscoelasticity, hyperelasticity, ductile and brittle fracture mechanics, and damage models; an XFEM solver for the problems of mesh dependence. |
| Delivered | One homogenization module for every physics, with its guide for researchers ([physics.md](physics.md); [theory.md](theory.md) §59–§66): the transport of a scalar potential (thermal, electrical, dielectric, magnetic, diffusive), the strong couplings on voxel images (piezoelectric, piezomagnetic, magneto-electro-elastic, thermoelectric), the weak couplings (swelling, the poroelasticity of Biot), the harmonic regimes (complex moduli and permittivities, the optical index), the scalar Bloch waves of sonic and photonic crystals; viscoelasticity, hyperelasticity and damage in every solver, beside the plasticity; the extended finite elements of level sets; the phase field of brittle fracture on voxel images; the RVE statistics of every result. |
| Porous flow and acoustics (2026-11-13) | The multiphysics module extended to the flow of a fluid through the pores and to the acoustics of porous media. |
| Delivered | The flow of a fluid through the pores (lattice Boltzmann) and the acoustics of porous media ([theory.md](theory.md) §67–§68), with their templates; a tab Physics and the new behaviours in the graphical interface; a faster phase field and its tool; the band structures from the command line (`morphogen bands`); two examples to run and change (`examples/multiphysics.py`, `examples/acoustic_foam.py`). |
| Three dimensions (2026-11-13) | Every module of morphoGen available in 3D, not only in 2D. |
| Delivered | Every module checked, and what worked only in the plane extended ([theory.md](theory.md) §69–§74): light in three-dimensional photonic crystals (the vector waves of Maxwell's equations), the particles of a phase separated in a volume, diamond and pyrochlore lattices of struts, the thermal oxidation in the voxels of a microstructure, cracks of any surface and front, the laser-treated sheet in its volume, sound at any incidence and in a diffuse field, a cell of negative Poisson's ratio along its three axes, the vertex stars of the icosahedral tiling. |
| Implicit geometry, finite strains, hyperelasticity (2026-11-13) | Every model described in implicit geometry, so that exact geometries can be exported; finite strains in every case; the hyperelastic models completed (Mooney–Rivlin, Arruda–Boyce and the others of §61); the menus of the graphical interface reorganized. |
| Delivered | An implicit function for every phase of the 38 models, and conforming meshes of structures of several phases ([theory.md](theory.md) §75); finite strains for the extended finite elements, for the phase field of fracture (§64) and for the volume of the laser-treated sheet; the strain energies of Rivlin's polynomial, Yeoh, Arruda–Boyce and the eight chains, Gent, van der Waals, Fung, Ogden, Holzapfel–Gasser–Ogden, Blatz–Ko and Ogden–Hill, the softening of Mullins, and their fit to tests (§61); the interface in four sections, the tools of a tab under headings, a search of the tools and of their options, an address for each tool, a map, the exports of every phase. |
| Documentation (2026-11-13) | Tutorials, a gallery of examples, the documentation and a website (NFR-8). |
| Delivered | NFR-8. Five tutorials whose snippets the tests run ([tutorials.md](tutorials.md): a whole study, the size of the representative volume, a rubber from its tests to its tearing, exact geometry from implicit functions to meshes, a tour of the graphical interface with screenshots taken by a script); recipes ([howto.md](howto.md)), questions and troubles ([faq.md](faq.md)) and a glossary ([glossary.md](glossary.md)); in the gallery, four templates more (a rubber, two cells that break, three phases meshed; a new family, Fracture), the curves that the runs compute on the cards of six templates, and planar sections of a volume in the configurations (`voxels.section`); for the site, a landing page, a logo, the pages in four groups, a button that copies each code block, a filter of the gallery, and the site kept by the continuous integration. |
| Quadratic elements (2026-11-14) | Quadratic elements for the finite element analyses. |
| Delivered | Before: 10-node tetrahedra existed, with straight edges, and were the default of two tools only; six of the nine templates that analyse a mesh used 4-node elements, one with a plastic matrix (locking). Now ([theory.md](theory.md) §76): isoparametric 10-node tetrahedra in the linear problems, the loading paths and the finite strains (the Jacobian of each integration point; straight elements unchanged); the nodes of the interfaces put on the implicit surfaces of the phases (`mesh.curved_mesh`, the default of `conforming_mesh(order=2)`), exactly for the unions of spheres and segments and their coatings (`Microstructure.implicit_at`); 10-node elements by default for every mesh that the finite elements analyse (configurations, command line, interface, templates). Against Sangani & Acrivos (1983) the conductivity of a cubic array of spheres is within 0.01 % with elements of three quarters of the radius (1.1 % off with 4-node elements, 3.2 % with straight 10-node ones). In the fracture modules ([theory.md](theory.md) §46, §47): interface elements of six nodes a side between 10-node tetrahedra, on curved faces (the area of the interface of a sphere within 0.03 %, 6 % less on flat faces), and 10-node elements with quarter points at the fronts of cracks on conforming meshes (the energy release rate of an edge crack at 0.99 of the handbook value with elements of 1, 0.76 with 4-node ones); the XFEM, the mixed-mode analyses and the inelastic bulk of crack problems keep 4-node elements. Also corrected: the opposite faces of periodic meshes of three phases were not split alike; the reference volume of the coatings of coated spheres ignored their overlaps. Known limits ([theory.md](theory.md) §76): the sharp edges of a phase (the grooves between overlapping spheres) and the gaps thinner than the grid of the mesher are rounded or bridged by the mesher, with any element. |
| Isogeometric analysis, reduced integration (2026-11-14) | Isogeometric analysis (IGA) and its application to shape optimization; quadratic elements with reduced integration for problems such as the near incompressibility of elastomers (Poisson's ratio close to 0.5). |
| Delivered (reduced integration) | `integration: bbar` for the 10-node tetrahedra ([theory.md](theory.md) §77): the volume change of each element taken as its mean (selective reduced integration: `B`-bar in small strains, `F`-bar in finite strains, with the conjugate stresses and the consistent symmetric tangent), in the elastic problems and the loading paths of a mesh, from `homogenize`, the configurations, the command line and the interface; the default stays `full`, with a warning for phases of Poisson's ratio 0.49 or more. A uniform reduced integration does not exist for tetrahedra (a single point leaves 18 deformations without energy). Stiff spheres in a matrix of Poisson's ratio 0.4999: shear moduli converged within 0.1 % with 4,800 nodes, where the full integration is 2.4 to 6 % too stiff and still 0.5 to 1 % with 40,000; Lamé's cylinder, a rubber against its material point, the tangents against central differences. |
| Delivered (isogeometric analysis) | A package `morphogen.iga` ([theory.md](theory.md) §78, §79): B-splines and NURBS patches assembled along their sides, with exact circles and ellipses and refinements that keep the geometry; linear elasticity (plane strain, plane stress, solids) and conduction with the functions of the geometry, at their optimal orders on Lamé's cylinder and Kirsch's plate; the effective tensors of periodic cells (within 0.05 % of the curved finite elements of the solid with 180 times fewer unknowns); shape optimization whose variables are control points of a coarse design, with analytical sensitivities (the energy-momentum tensor of Eshelby, checked against central differences) and SLSQP or the moving asymptotes. Validated on two shapes known in closed form: the hole of largest bulk modulus of a periodic cell, which reaches the bound of Hashin and Shtrikman (Vigdergauz 1994), and the hole of a plate under biaxial tension, an ellipse. |
| Delivered (isogeometric shape optimization applied) | Periodic cells of any lattice — parallelograms (hexagonal arrays), or whole pieces of a pattern cut along no straight line —, validated by other cuts of the same cell (10⁻¹³) and by the series of Perrins et al. for arrays of holes (0.002 %); a sheet with alternating elongated holes (the rotating squares), auxetic down to a Poisson's ratio of -0.975, confirmed by the FFT solver; design variables linked by the symmetries of a pattern and linear constraints on sizes; the lowest Poisson's ratio at a given stiffness (-0.814 from -0.530) and an inverse design with a known answer; the image of an assembly on a grid, by which an optimized shape goes to the voxel solvers; a tutorial ([tutorial_iga.md](tutorial_iga.md)) whose snippets the tests run. Limit stated: for this pattern the free shape of the holes brings nothing beyond their two sizes — the ligaments decide. |
| Sharp edges (2026-11-14) | The sharp edges of the phases followed by the mesher: the limit that the quadratic elements had left, sharp edges rounded by the mesher. |
| Delivered | The conforming mesher follows the sharp edges of the phases ([theory.md](theory.md) §80): the structures made of spheres and segments, their coatings and hollow members give the smooth pieces of their phases (primitives that do not overlap, by a colouring of their overlaps); the box is cut by the surface of each piece in turn and the lines where two cuts meet are kept as ridges through the remeshing (`mesh.sharp`); the curved elements put the nodes of these edges on the two surfaces that meet there. Two overlapping spheres: the volume of the union within 0.001 % for 0.05 %, every node of the interface but one on its surface; coated spheres: the interphase within 0.1 % for 0.9 %, the bulk modulus at its converged value. Stated: the meshes take 40 to 80 % longer where primitives overlap; 4-node meshes lose more volume at the same sizes (the cuts are smoother, MMG uses the whole Hausdorff distance); flat caps, shallow overlaps and the structures whose functions are those of a grid are rounded as before. |
| Contact and damage (2026-11-14) | Contact, self-contact and friction; then damage models. |
| Delivered (first lot) | Coulomb's friction of the interfaces whose nodes match ([theory.md](theory.md) §81): the cohesive laws and the contact of the lips of cracks take a coefficient of friction (the broken share of a point sticks and slides: Alfano & Sacco 2006), in small and finite strains; loadings in stages; the stress intensity factors and the energy release rate of cracks whose lips rub. Two blocks slide at T = μ N, a closed crack as under the shear less Coulomb's; K_II within 2 % of the shear alone, G within 1 %. |
| Delivered (second lot) | Contact between surfaces that slide far, self-contact and rigid obstacles, with friction ([theory.md](theory.md) §82): the package `morphogen.contact` (surfaces of a mesh, pairs, a surface against itself, planes, spheres, cylinders and cones that move; the integration points of the slave faces against the nearest point of the master, penalty or augmented Lagrangian, Coulomb's radial return on a frame-indifferent slip) in the quasi-static solver of the cohesive zones, in small and finite strains, with 4- and 10-node elements. A pressure transmitted between meshes that do not match to 1e-4, T = μ N along a slide longer than the slider, Hertz's line contact within 2 %, his sphere within 5 %, a slot that closes, arms that come to rest on each other. Stated: corners and edges carry no point; a first contact by an edge may fail in small strains; a direct solver. |
| Delivered (third lot) | Self-contact in the periodic cells of the nonlinear homogenization ([theory.md](theory.md) §82): `contact.ContactCell`, the periodic element of the loading paths with the contact and friction of the walls of the pores, between themselves and with their images across the faces of the cell (the stress that the contacts carry through the faces is in the macroscopic stress and in its equilibrium). Layers that close on their images: the stress of the solid to 1e-6; a cell cut through its slot or beside it: the same response. To follow: configurations, command line and interface; then the damage models. |
| Delivered (fourth lot) | The contact of the walls of the pores in configurations (`homogenization.nonlinear.contact`), in `homogenize` and on the command line (`morphogen homogenize --contact [MU]`); its curves (`contact_area`, `contact_pressure`, `contact_penetration`, `contact_dissipation`: `PathResult.series`) in the files of the paths and in the statistics of volume elements; the template `homogenization_contact` (plates parted by gaps and held by pillars: five times stiffer once the gaps close). Three corrections found on real morphologies ([theory.md](theory.md) §82): the two sides of a thin ligament were taken for surfaces gone through each other; the walls cut by the faces of a periodic cell had a rim there; the walls of a notch that closes were not seen until their normals were nearly opposite. Limit found: the creases of a honeycomb crushed beyond half its height. To follow: compression between platens and indentation as ready-made tests, with their command and tool of the interface; then the damage models. |
| Delivered (fifth lot) | Ready-made tests of contact ([theory.md](theory.md) §82): `contact.compression` (a sample between two rigid platens, which rub, its walls free to close on themselves: nominal stress against strain, areas in contact) and `contact.indentation` (a sphere or a cone: force against depth, the reduced modulus read by Hertz's and Sneddon's laws, by the slope of the unloading and, for a sphere, with the rigid base under the sample: `contact.layer_correction`, Dimitriadis et al. 2002); the commands `morphogen compress` and `morphogen indent`; the tab Contact of the interface. A block between lubricated platens: the uniaxial stress to 2e-4; a sphere on a block: its modulus within 5 to 7 % with its base. This ends the contact; the damage models follow. |
| Delivered (damage models) | The second part of the requirement above ([theory.md](theory.md) §83). What existed: the isotropic damage of quasi-brittle solids (crack band, delayed damage), the micromorphic damages (brittle and ductile), porous plasticity (Gurson, Rousselier), the phase field, cohesive zones, the softening of Mullins. Added, at the material point of every solver of the loading paths, in small and finite strains: damage coupled to the von Mises plasticity of a material (`damage: {law: lemaitre}`: ductile damage by the energy release rate, the triaxiality and the closure of the microcracks; `{law: kachanov}`: the creep damage of Kachanov and Rabotnov with Hayhurst's stress), Mazars' law of concrete (`{law: mazars}`), and the unilateral effect of the quasi-brittle law (`split: spectral`); a delayed damage keeps the softening of a cell well posed. Checked against closed forms (a perfectly plastic bar to 1e-12, the time to rupture and the tertiary creep, concrete in tension and compression to 1e-9); the forms of the interface; the template `homogenization_ductile_damage`. Not offered: anisotropic damage; Hill's criterion, MFront sources, Z-set files and the NTFA with a damaged plastic phase. |
| Continuous integration, debonding, fatigue (2026-11-15) | The tests of the continuous integration passing on every system; interfaces that debond in the periodic cells of the loading paths; multiaxial high-cycle fatigue; the damage coupled to plasticity with Hill's criterion. |
| Delivered (continuous integration) | The two tests that failed on macOS in the last run of the workflow (the factorized solver of the rigidity percolation took 4 iterations where they asked for 3 at most) ask for fewer iterations than the FFT solver, and 10 at most; the numbers of tests of the README updated. |
| Delivered (interfaces that debond) | Cohesive interfaces in the periodic cells of the loading paths ([theory.md](theory.md) §84): `fracture.CohesiveCell` (the bilinear and exponential laws, lips without adhesion, the contact and Coulomb's friction of the broken lips; phases, pairs of phases or grains; periodic and kinematic conditions, small and finite strains, 4- and 10-node elements), the interfaces inserted on the torus of the cell (`fracture.open_cell`: those that cross its faces, those that lie on them), the option `interfaces` of `homogenize`, of the configurations and of the command line, the curves `debonded_area`, `interface_damage`, `interface_opening` and `interface_dissipation`, the template `homogenization_debonding`. For the iterations: a predictor of the lips that touch, a nonmonotone line search, an elastic slip of the friction, a viscosity of the damage. A defect corrected: a path that brought an elastic cell back to zero stress did not converge (finite elements and FFT). Limits: the tangent is factorized (cells of 10⁴ to 10⁵ unknowns), one law for all the interfaces of a cell, slips small against the elements, no arc-length control (the viscosity passes the snap-backs). |
| Delivered (high-cycle fatigue) | Criteria of multiaxial fatigue on the stresses of the points of a cell along a cycle of a loading path ([theory.md](theory.md) §85): `homogenization.fatigue` — the amplitude of a multiaxial cycle (the smallest hypersphere that holds the path of the deviator), the criteria of Crossland, Sines and Dang Van and the critical planes of Findley and Matake, each calibrated on two fatigue limits of a phase; the utilization of each point, a criterion per phase, the largest utilization, the factor of safety and the share of the volume beyond its limit (`fatigue`, `path_fatigue`). In configurations, `homogenization.fatigue`: the map written beside the curve of the path, the three numbers among the results and the quantities of the statistics of volume elements; the template `homogenization_fatigue`. Checked against closed forms, and against Goodier's stresses at a spherical pore. Limits: local criteria (no gradient or size effect), judged on the cycle given (an elastic shakedown); the critical planes cost a few hundred planes per point. |
| Delivered (a limit lifted) | The damage coupled to plasticity (Lemaitre, Kachanov-Rabotnov: [theory.md](theory.md) §83) with Hill's criterion, which the damage laws refused: the cumulated plastic strain is that of the norm dual to the criterion. |
| Interface laws, figures, tools, tutorial (2026-11-15) | A law per pair of phases and the maps of the interfaces, the fatigue in the figures, the tools of the interface, a tutorial. |
| Delivered (interfaces: laws, fields, grains) | A law for each pair of phases of a cell whose interfaces debond (`laws`), the damage and the opening of the interfaces as fields of the elements (files and figures), and the grain boundaries of real polycrystals ([theory.md](theory.md) §84): checked by the work that breaks a polycrystal, the fracture energy times the area of the crack, on whole grains and on grains cut by the box; the template `homogenization_intergranular`. |
| Delivered (interface: debonding, fatigue) | The tool *Loading path of a morphology* of the graphical interface has the interfaces that debond (finite elements: those of the phases or of the grains, their strength, fracture energy, friction and viscosity; the debonded area, the damage and the opening in the values, the curves and the fields) and the fatigue of the cycle (a criterion and its two limits: utilization, factor of safety, share of the volume beyond, the fields and a figure); the configuration written beside the simulation reproduces both. |
| Twinning and cleavage (2026-11-15) | Twinning, cleavage, and intragranular and intergranular cracking in crystal plasticity. |
| Delivered (twinning) | Twins in the crystal plasticity of the grains ([theory.md](theory.md) §86): systems that shear in one sense until the crystal has twinned (the pseudo-slip of Kalidindi, a saturation), the twins of the cubic lattices and four modes of the hexagonal one with their shears, and the slip families of hexagonal crystals (basal, prismatic, pyramidal); a fibre texture of the grains of tessellations; the twinned fraction as a field and a curve; checked against Schmid's law in one sense, the shears of the modes, the saturation; the template `homogenization_twinning` (the asymmetry of a textured magnesium). |
| Delivered (cleavage, cracks in and between the grains) | Cleavage planes in the crystals ([theory.md](theory.md) §87): opening and accommodation systems that soften (Marchal et al., Aslan & Forest), the fracture energy of the planes dissipated in a band of one element, a bound on the size of the elements; with the cohesive boundaries of the grains, cracks between the grains or through them, whichever is weaker; the opening and the damage of the planes as fields and curves; checked against the closed forms of a plane (strength, softening, work), the fracture energy of a crack band and the energy of an intergranular crack; the templates `homogenization_cleavage` and `homogenization_grain_fracture`. |
| Delivered (the local approach of cleavage) | The criterion of Beremin on the fields of a loading path ([theory.md](theory.md) §88): `homogenization.beremin` — the Weibull stress of the points of the plastic zone, driven by their largest principal stress or by the normal stress of the cleavage planes of their grain, and the probability that the weakest link has failed; in the configurations (`homogenization.beremin`: curves, field, quantities of the RVE studies); checked against closed forms; the template `homogenization_beremin`. |
| Delivered (tutorial: how a microstructure breaks) | A tutorial whose snippets the tests run ([tutorial_fracture.md](tutorial_fracture.md)): interfaces that debond, the fatigue of a pore, the twins of a crystal and of a textured polycrystal of magnesium, the cleavage of a crystal and of a polycrystal of iron, the probability of cleavage of Beremin, each against a number known in advance; and a correction: the fatigue of a cycle on a voxel image with pores, whose voids are now left out. |
| Delivered (interface: cleavage; compiled kernels) | The box *Cleavage* of the tool of the loading paths of the graphical interface (the criterion of Beremin: its parameters, the stress that drives the cleavage, the plastic zone; the curves, a figure, the field, the configuration); and a revision of the compiled functions that hold the kernels of the crystals, so that the caches of an earlier version are not run in place of the new kernels. |
| Delivered (contact: corners and edges) | Contact points at the nodes of the faces ([theory.md](theory.md) §82), the limit that the lots of the contact had left: `points="nodes"` of the pairs, the obstacles, the periodic cells and the compression test (`contact: {points: nodes}`, `morphogen compress --points nodes`, the tool of the interface). The weight of a point is the share of its face that the other surface sees, so that the sides of a block carry nothing of what its base carries; a point of the rim of a surface faces the rim of the other. Checked: the weights of the rules; two blocks of meshes that do not match (the uniaxial stress to 0.5 %); a block whose whole skin is slave to a plane (the uniaxial stress to 1e-6, as with the points inside the faces); a cube that stands on a corner against a rigid plane and against a block (nothing with the points inside the faces, carried at the nodes; the penetration taken off by the multipliers). |
| Crack closure and twin reorientation (2026-11-15) | The closure of cleavage cracks and the reorientation of twins. |
| Delivered (cracks that close) | The unilateral condition of the cleavage planes ([theory.md](theory.md) §89): the opening of a plane follows the sign of its normal stress, back to nothing (Aslan & Forest 2009), and the closed plane carries the pressure as the sound crystal; the lips of a plane under pressure rub with Coulomb's `friction` — in place of the accommodation systems stopped on a closed plane, whose discontinuity kept a polycrystal with cohesive boundaries from converging. `cleavage: {closure, friction}`; the default is the closure. Checked: the flow rule there and back (10⁻⁷), the stress of the sound crystal once closed (10⁻⁹), the work of a cycle (1 %), sliding at `μ p`, the consistent tangents (2·10⁻⁶), cells against the point, a cracked polycrystal that carries the pressure within 1 % (the template `homogenization_crack_closure`). |
| Delivered (twins that shear back and turn the lattice) | Detwinning: the second variable of a twin is its net shear, which goes back to nothing under a reversed stress (`twin_detwinning`, the default). The lattice of a twin (`twin_reorientation`): at a threshold of its twinned fraction the lattice of a point becomes that of its predominant twin (Van Houtte 1978; Tomé, Lebensohn & Kocks 1991), turned by half a turn about the normal of the twin plane with its systems, elasticity and thermal strain; it twins back by the same system, the other twins of the family held (the conjugate variant of a twin of small shear would otherwise take the detwinning). The field `twin_misorientation` (86.3° for {10‑12} in magnesium, 60° for the cubic twins). Checked: the fraction there and back (10⁻⁹), the disorientations against closed forms, the Schmid law of the turned lattice, no strain left after a twin and back, tangents and warm starts, cells against the point (small and finite strains), a textured polycrystal (the template `homogenization_detwinning`). In the interface: the fields of the crystal form. Also: the tolerance of the material point at no stress in finite strains. |
| Delivered (twins: grain by grain; a resistance on the way back) | What the reversal had left as limits ([theory.md](theory.md) §89). `twin_reorientation: {threshold, growth}`: the lattices turned grain by grain, the predominant twin reorientation of Tomé, Lebensohn & Kocks (1991) with the integration points of a grain in place of the grains of an aggregate — the most twinned points first, past `threshold + growth × turned volume / twinned volume`, while the turned volume of the grain is below its twinned volume; back in the same way when the twins shear back. `twin_detwinning` and `cleavage.closure` take the share of the resistance at which the twins shear back and the planes close (Lou et al. 2007; Hama et al. 2016). The twinned fraction is that of the shears of the twins, unchanged by a turn (it was 1 for a turned point), and what is left of a twin goes on shearing back once the lattice of the grain is back; the field `twin_reoriented` is the turned volume. The fields have the `orientation` of the turned lattices in small strains too, and the criterion of Beremin takes the cleavage planes of the lattice as it is. Checked: the flow rules at a share of the resistance (10⁻⁹) and their tangents; the rule of the grains on points of given fractions (exact), a material point (the constant threshold), a grain of 512 points (the turned volume within a point of the twinned one) and a polycrystal (grain by grain); the basal plane of a twin under the criterion of Beremin (10⁻⁶). The template `homogenization_detwinning` turns its lattices grain by grain; the crystal form of the interface has the three settings. |
| Delivered (cleavage that the slip nucleates) | The coupling of plasticity and damage of Aslan, Cordero, Gaubert & Forest (2011) for the cleavage planes of the crystals ([theory.md](theory.md) §90), the limit that the cleavage had left (a constant strength: slip shielded cleavage for ever). `cleavage: {coupling: Θ}`: the strength of the planes falls as `exp(−Θ γ_cum)` of the cumulated slip of the crystal, and their damage takes `Θ exp(−Θ γ_cum) Σ_p (Y₀/w)(1 − exp(−w d_p))` off the resistance of its slip systems — the two thresholds of one energy. Checked: the onset of the cleavage of a crystal that slips (`γ_cum = ln(Y₀/σ)/Θ`), the flow rules of its slips and of its plane in tension and in compression (10⁻⁶), the Jacobian of the local residual and the tangents (small and finite strains), cells against the point, a polycrystal that breaks sooner. A coupling that would leave the slip without resistance is refused. In the interface: the coupling of the crystal form. |
| Regularized cleavage (2026-11-16) | The cleavage regularized by a microdamage (Aslan & Forest 2009), as an option. |
| Delivered (cleavage planes regularized by their microdamage) | The micromorphic regularization of Aslan & Forest (2009) for the cleavage planes of the crystals ([theory.md](theory.md) §91), the limit that the coupling had left (a crack band of one element). `cleavage: {energy, regularization: {width, ratio}}`, or `{modulus, length}` with the linear softening of the planes: one threshold for the planes of a point, `Y₀ (e^{−Θ γ} − w d) + χH (d − χd)`, and the equation of the microdamage `χd − ℓ² Δχd = d`, solved by the loading paths between the equilibria of a step (alternate scheme) — in Fourier space on voxel images, by finite elements on meshes. The planes that have come to nothing are broken through (no strength whatever they open; the microdamage keeps the damage that they had), so that the band does not widen as the crack opens. Opt-in: the crack band of one element stays the default. Checked against the closed form of a bar that breaks across its band (derived in [theory.md](theory.md) §91: width, profiles, opening, fracture energy): softening branch and fields within 0.5 %, work within 0.3 % of the fracture energy on 64 voxels and at the second order of their size (−1.2 %, −0.28 %, −0.04 % on 32, 64, 128); above it on meshes of tetrahedra (+2.4 % with 64 slabs), by less as they are refined. Template `homogenization_cleavage_band`, a field of the crystal form of the interface, fields `cleavage_microdamage` and `cleavage_damage` (the share of its strength that a plane has lost). Limits: one modulus and one length per crystal; a point broken on a plane is broken on the others of its family; on voxels the microdamage crosses pores and other phases; the rules that break the planes through are those of this implementation; a crack that runs across the cell takes some equilibria per element crossed; the extended finite elements refuse these planes. |
| Alternate scheme of the microdamage (2026-11-16) | The tolerance and the passes of the microdamage in the configuration, and a faster alternate scheme where a crack runs. |
| Delivered (the alternate scheme of the microdamage, accelerated) | The options of the scheme in the configurations, `nonlinear: {microdamage: {tolerance, passes, acceleration}}` (`microdamage_options` of the solvers of the paths), and its acceleration ([theory.md](theory.md) §91), after the accelerated staggered scheme of Storvik et al. (2021) for the phase field: the updates of the microdamage extrapolated where their residual falls (Anderson's acceleration of depth one, within ten residuals of the plain update) and relaxed where it grows; the equilibria between two updates solved to a tenth of the last change of the microdamage, the last one of a step to the tolerance of Newton's iteration; the condensation of the macroscopic unknowns kept from a pass to the next. No plane is broken through where the microdamage had been extrapolated ahead of the damage. Measured: the bar that breaks across its band in 346 equilibria for 588, with a work 0.10 % above its settled value for 0.36 % (the plain scheme lags); the template of the columnar polycrystal in 25 s for 49 s, a third of the linear iterations, the same peak and the same work to 0.2 %. `acceleration: false` gives the plain scheme. Limit: a crack still crosses the cell at a row of elements a pass — the acceleration shortens what comes before and after, and makes each pass cheaper; on a grid of 48 x 48 voxels the gain is a factor 1.6. |
| Crystal kernel (2026-11-16) | A faster crystal kernel, which took more than half of the time of the polycrystals. |
| Delivered (the integration of the points, faster with the same results) | The integration of the points of the loading paths ([theory.md](theory.md) §32, *The cost of a point*), two to three times faster for the crystals with the same stresses, tangents and internal variables to the bit (372 arrays of short histories of 12 laws compared with those of the earlier kernels; the curves of five templates identical to the last digit). Two causes, found by replaying evaluations of the points taken from runs of the templates and by counting the cycles of each part of the kernels. (1) The work arrays of the laws were allocated in the parallel loop over the chunks of points: the rows that the kernels take of them at each call were objects whose references the compiled code counted, atomically — a third to a half of the time of a crystal point. They are allocated per chunk before the loop (`work_blocks`, `work_block` of `homogenization.crystal`; `point_chunks` of `homogenization.behaviours`: 64 points or more per chunk, 128 chunks at most), for the crystals and for the von Mises, Hill and porous points in small and finite strains. (2) The crystals computed what a point does not need: a point that stays elastic is now evaluated once, without derivatives; the derivatives of the resolved stresses are those of the active systems (and of an inactive one where it starts), or the constants kept with the Schmid tensors of the grain in small strains (`cp_small(..., key)` keeps them from a point to the next); the correction that the line search accepts is evaluated with its derivatives and is the next iterate; the resolved stresses are summed over the systems that slip, the hardening over those that have slipped, and a plane that has not opened has no exponential. On one thread, per point: 7.4 → 2.6 µs for an iron polycrystal that cleaves (21 systems, 88 % of elastic points), 26.4 → 10.8 µs for the first iterate of a step where 90 % of its points flow, 15.8 → 8.7 µs for the twins of magnesium, 21.5 → 7.7 µs for a nickel polycrystal in finite strains; von Mises 0.5 → 0.3 µs, Hill 3.2 → 2.4 µs, GTN in finite strains 8.0 → 6.4 µs. On four threads: `homogenization_crystal_plasticity_fft` 148 s for 210 s (its points 64 s for 123 s), `homogenization_cleavage` 36 s for 49 s, `homogenization_slip_cleavage` 35 s for 60 s, `homogenization_twinning` 46 s for 58 s, `homogenization_cleavage_band` 12 s for 18.5 s. A test integrates points together and alone (slip, cleavage, twins that turn the lattice; small and finite strains): the same to the bit. Limit: the points are now a third to two thirds of the time of these runs; what remains of a crystal point is the evaluations of its local iteration (three to four per point that flows). |
| Linear iterations (2026-11-16) | Fewer linear iterations for the polycrystals, or a more compact tangent, the results allowed to change within the tolerances. |
| Delivered (loading paths in fewer iterations) | Fewer Newton and linear iterations along the loading paths ([theory.md](theory.md) §31, *The prediction of a step, the secant of the condensed block*), by two changes of the path solver (`homogenization.nonlinear`; finite elements and voxels). (1) A step starts from a second-order prediction of its unknowns — the rate of the last step and the change of the rates of the last two, exact for a parabola with unequal steps — where that would have predicted the last step clearly better than the first order (half its error, itself below half of the increments); a response that turns keeps the first order (`prediction = 1` on a solver keeps it everywhere). (2) The condensed block of the macroscopic unknowns that the iterative solvers keep from a correction to the next is updated by the secant of each correction that converges (the merit below a hundredth of what it was): BFGS with Powell's damping for a symmetric block, Broyden's update otherwise. In a steady plastic flow the first residual of a step is three to four times smaller and two corrections reach the tolerance where three were needed. Newton and linear iterations of the templates: `homogenization_crystal_plasticity` 200 → 170 and 5,510 → 3,905 (42 s for 54 s), `homogenization_crystal_plasticity_fft` 149 → 129 and 3,535 → 2,678 (115 s for 132 s), `homogenization_ductile_damage` 133 → 120 and 12,059 → 9,352 (64 s for 73 s), `homogenization_viscoplastic` 79 → 69 and 2,225 → 1,899 (176 s for 187 s), `homogenization_slip_cleavage` 479 → 436 with 26 divided steps for 34; the same curves to the tolerance of the equilibria where the steps are the same (differences below 3·10⁻⁶ of the stresses). The paths where cracks or twins start keep their iterations and their divided steps (cleavage, crack closure, twinning, grain fracture, the regularized band): the first versions of the two changes divided more steps there (18 for 13 on the cleavage of a polycrystal with the second order taken whenever it had been the closer; a step divided where twins start, the block updated by a secant taken far from the solution), hence the two safeguards. What the requirement named was measured first and not kept: a reference medium of the Fourier preconditioner that follows the mean tangent, isotropic or not (3,816 and 3,572 linear iterations for 3,535: the 21 iterations per solve of the plastic flow, for 6 in the elastic steps, come from the contrast between grains that flow along different directions, which no homogeneous medium follows); full tangents with GMRES, the full integration of the voxels, the condensation computed again at every correction (the same residuals: neither the symmetrization, nor F-bar, nor the kept condensation limits Newton's iteration in the flow — the law does); a copy of the tangents in single precision for the Krylov iterations (the same iterations, no faster: the application of the tangent is not bound by the memory on four threads). Limit: a path made of events (cracks, twins) gains nothing; the time of `homogenization_viscoplastic` follows its iterations only in part (the next lot). |
| Multithreaded solvers (2026-11-17) | The iterative solvers of the finite element paths on several threads: the sweeps of the preconditioner and the products of the Krylov iterations ran on one thread, so that the time of the paths no longer followed their iterations. |
| Delivered (the finite element solvers on several threads) | The iterative solvers of the finite element problems ([theory.md](theory.md) §11, *Linear solvers*) on all the threads, with results that do not depend on their number. (1) Gauss–Seidel sweeps in the order of the colours of the unknowns (`homogenization.amg.Smoother`; matrices of 200,000 stored values or more): a greedy colouring of the graph of the couplings (both sides of the diagonal: the products of sparse matrices leave a few values on one side only) so that no stored value couples two unknowns of a colour, which are relaxed together; the backward sweep is the adjoint of the forward one, the preconditioner stays symmetric positive definite. (2) The sweeps read a copy of the matrix made for them: rows and columns sorted by colour, values in single precision, 32-bit column indices, the parts below and above the diagonal apart (the forward sweep from zero reads the first only, the residual that follows it the second only); the cycle of the two-level preconditioner of 10-node elements and that of each level of the multigrid run in the order of the colours. (3) The products of a matrix and a vector of the Krylov iterations share the rows between the threads (`homogenization.amg.product`), the same product to the bit as SciPy's. (4) The reduced tangent `TᵀKT` of a loading path is filled by the places of the stored values of `K`, found once (`homogenization.fe.Reduction`; a reduction that interpolates keeps the products). Measured on four threads, back to back: `homogenization_spheres` (368,577 unknowns) 192 s for 342 s (solves 106 s for 248 s; 1,034 iterations for 1,021), `homogenization_gyroid` 124 s for 200 s, `homogenization_polycrystal` 24 s for 34 s, `homogenization_viscoplastic` 86 s for 129 s (the reduction of its tangents 1.3 s for 17.5 s), `homogenization_crystal_plasticity` 21 s for 32 s; one application of the preconditioner on 76,398 unknowns, 10.8 ms for 32 ms. Verified in `tests/test_solvers.py`: the product against SciPy's to the bit, the colouring (one-sided values included), the sweeps against Gauss–Seidel on the sorted matrix, the preconditioner (symmetric positive definite, the iterations of the sweeps in the order of the unknowns), the same results on one thread and on several, the reduction against the sparse products. Measured and not kept: the rows of a colour read where they are in the matrix (7.7 ms a sweep for 11 ms on one thread: bound by the memory, 3.9 ms once sorted). What remains of the 86 s of `homogenization_viscoplastic`: the sweeps (22 s) and the products (some 13 s), which the memory bounds; the assembly of the tangent, on one thread (12 s); the preconditioner built at each step (6 s). |
| Delivered (the sweeps by nodes, the assembly by colours, the threads of BLAS) | From the profile that the delivery above left ([theory.md](theory.md) §11, *By nodes*, *The threads of BLAS*, *The assembly by colours*); the iterations and the results are those of before. (1) Sweeps by nodes: where the unknowns are components of the displacements of the nodes, the nodes are coloured and the copy of the matrix that the sweeps read holds the 3 × 3 blocks of their couplings (4.4 bytes a value for 8); the three components of a node are relaxed one after the other. The sweep is the Gauss–Seidel sweep of the unknowns coloured one by one (the greedy colours of the components of a node follow each other), so the preconditioner is the same operator; the solvers take the degree of freedom of each unknown (`dofs`), and a component that a condition holds leaves its place empty, so that the four kinds of boundary conditions work by nodes; the rows of the part above the diagonal are shared between the threads by equal numbers of stored values. One application of the two-level preconditioner on 76,398 unknowns: 4.9 ms for 9.0 ms, 49 iterations in both cases. (2) The threads of BLAS: a product of BLAS between two compiled kernels wakes the threads of OpenBLAS, which stay busy for a tenth of a second and take the cores of the kernels that follow (a product of a sparse matrix 11.7 ms for 3.7 ms; some 80 ms of work lost per product of BLAS). The coarsest level of a multigrid cycle, the norms of the solves, the merit and the condensation of a path are summed without BLAS (`amg.dense_product`, `solvers.dot`, `norm`, `times`), and the loading paths and the Krylov solvers run with the BLAS on one thread, restored afterwards (`settings.blas_on_one_thread`, `without_blas_threads`; no new dependency: the functions of the OpenBLAS libraries of the process are called directly, and another BLAS is left as it is). (3) The assembly by colours: the elements are coloured so that two elements of a colour have no node in common (`fe.element_colours`), and the tangent, the nodal forces and the couplings of the finite element paths (small and finite strains, B-bar and F-bar; the bulk of the fracture module) are assembled colour by colour, the elements of a colour in parallel by chunks, the sums over all the elements chunk by chunk: the same results for any number of threads, those of the elements taken one by one to round-off. The tangent of 19,025 10-node tetrahedra: 42 ms for 158 ms. Measured on four threads, back to back: `homogenization_viscoplastic` 49 s for 93 s, `homogenization_ductile_damage` 30 s for 59 s, `homogenization_crystal_plasticity` 16 s for 22 s, `homogenization_crystal_plasticity_fft` 74 s for 86 s, `homogenization_rubber` 58 s for 67 s, `homogenization_polycrystal` 15 s for 26 s, `homogenization_gyroid` 103 s for 128 s, `homogenization_spheres` 160 s for 198 s; curves the same to 1e-12. Verified in `tests/test_solvers.py` (the sweeps by nodes against Gauss–Seidel in the order of the colours of the nodes, components held by a condition, the products without BLAS, the BLAS on one thread within a solve and restored after it) and `tests/test_assembly.py` (the colours of the elements, the assemblies against the elements taken one by one, on one thread and on several). What bounds the solves now: the product of the matrix in double precision, about half of the time of an iteration (stored by blocks of nodes it would read a third less memory: not done); then the preconditioner built again at each step of a path. |
| Product by nodes, preconditioner along a path (2026-11-17) | The product of the matrix stored by blocks of nodes, and the preconditioner kept from a step of a loading path to the next. |
| Delivered (the product by nodes, the preconditioner of a path kept from a step to the next) | ([theory.md](theory.md) §11, *The product by nodes*, *The preconditioner along a path*.) (1) The product by nodes: the products of the Krylov iterations read the matrix by the 3 × 3 blocks of its nodes in double precision (`homogenization.amg.NodeBlocks`; `blocks` of `LinearSolver`), each component of a row summed term by term in the order of the columns: the same product to the bit as that of the stored values, hence the same iterations and results; 1.2 ms for 3.9 ms alone and 2.2 ms for 7.0 ms between the sweeps on 76,398 unknowns, a fifth of the solves of the large linear analyses (bound by the memory read). The blocks are held with the stored values (70 % of the memory of the matrix more), and serve the four kinds of boundary conditions (a component that a condition holds is an empty row and column). (2) The sweeps of the tangent as it is: along a path the pattern of the tangent does not change; the place of each stored value in the blocks, and of each block in the copy of the sweeps, is found once, and at each linearization the blocks take the values of the tangent (`NodeBlocks.refresh`, 10 ms for 6.5 million values) and the sweeps follow (`Smoother.refresh`, 3 ms), where the sweeps were those of the first Newton iteration of the step; a matrix with other stored values has its blocks made again, and a matrix that is not symmetric leaves the sweeps as they are. (3) The coarse levels kept while they serve: the interpolation, the stiffness of the corner nodes and its multigrid hierarchy are kept from a step to the next and built again at the start of a step once the solves of the last one took `solvers.WORN` = 1.3 times the iterations per decade of residual of the step for which they were built (`LinearSolver.worn`; a factorization and a diagonal are built with their solver, as before). The preconditioner stays symmetric positive definite whatever the age of its coarse levels (the symmetric Gauss–Seidel preconditioner of the current tangent plus a positive semi-definite term). (4) The reduction of the tangent sums the values of each place of the reduced matrix on several threads (16 ms for 23 ms). Measured: `homogenization_viscoplastic` builds 2 preconditioners for 25 steps, `homogenization_crystal_plasticity` 4 for 67; linear iterations 1,867 for 1,917 and 3,937 for 4,071, the same Newton iterations, curves the same to 1e-8 (the tolerance of the equilibria). On four threads, back to back: `homogenization_viscoplastic` 41 s for 61 s, `homogenization_crystal_plasticity` 14 s for 20 s, `homogenization_fatigue` 21 s for 25 s, `homogenization_polycrystal` 14 s for 18 s, the solves of `homogenization_gyroid` 33 s for 42 s and of `homogenization_spheres` 68 s for 85 s; the paths solved by factorization keep their times. Measured before the choice: keeping the first preconditioner of a path for all its steps without giving its sweeps the tangents costs 6 % (viscoplastic) to 21 % (crystals) more linear iterations; with the sweeps of each tangent and coarse levels never built again, 6 % fewer and 10 % more: hence the rule. Verified in `tests/test_solvers.py`: the product against that of the stored values to the bit, the values of another matrix at their places, the sweeps that follow the blocks against those built for the new matrix to the bit, an updated solver (products, sweeps, solution, a matrix that is not symmetric), the wear rule, a plastic path with its preconditioner kept against the direct solver. `LinearSolver.updated` changes the solver that it is called on, which is not to be used afterwards. |
| Textures and orientation maps (2026-11-17) | Crystallographic textures and the import of EBSD maps, the first of four features considered alongside the release of version 1.0; the three others are on hold: the effects of the grain size in crystal plasticity (hardening by the gradients of slip), the second-gradient homogenization of architectured cells, real volumes (the segmentation of tomographs, out of scope in §4). The rights of the licence confirmed (decision D1, §11). A complete test point (both tiers of the tests and the timed templates on 20 threads) precedes version 1.0. |
| Delivered (textures and orientation maps: the core) | ([theory.md](theory.md) §92.) (1) `morphogen.texture`: the proper rotations of the eleven Laue classes; the disorientation of two crystals, exact to rounding for crystals that are close, and Mackenzie's density for random cubic crystals; the mean orientation and the spread of a grain; poles of the families of planes and directions of the cube and of the hexagonal lattice, stereographic and equal-area projections, pole figures in multiples of a random texture (an axial von Mises–Fisher kernel of a given half-width); inverse pole figures and their colours; the disorientations of the boundaries of an image of grains; orientations drawn from a measured texture. (2) `morphogen.io.ebsd`: orientation maps read and written as `.ang` (EDAX/TSL, square and hexagonal grids) and `.ctf` (Oxford Instruments/HKL) files. The orientations are brought into the axes of the image of the map (the first to the right, the second upwards, the third towards the viewer): those of a `.ctf` file; those of an `.ang` file turned about the normal according to the setting of the acquisition (1 to 4, numbered as by MTEX's reader, whose rotations were checked against its sources); the hexagonal and trigonal crystals of a `.ctf` file are turned by 30° about `c` to morphoGen's axes (the first along `a`, as for its slip and twin systems). The grains of a map: connected sets of neighbouring points less than a threshold apart (5° by default), the small ones dropped, the points without a grain given to the nearest one; mean orientations, spreads, areas; their polycrystal. The map that an acquisition would give of a layer of a computed polycrystal. (3) The model `ebsd` (the 39th): a window of a map of the size of the domain, its first rows and columns or drawn at random for each realization, as a polycrystal of one cell through the thickness for every solver; the files of the models of a configuration are taken from its folder. A demonstration map comes with morphoGen, with the template `ebsd`, its card in the gallery, a recipe and the example `examples/ebsd_map.py`. Checked (`tests/test_texture.py`, 28 tests): the disorientations of known rotations (10⁻⁶ degrees), Mackenzie's density (integral, mean, 200,000 random pairs), the poles of the cube orientation, the pole figures of a random texture and of one crystal, the files of the two systems and their axes, the grains of a noisy map of known grains, the stiffness of a map of one crystal (10⁻⁹), the bulk modulus `(C₁₁ + 2C₁₂)/3` of windows of a map (10⁻⁶). Limits: a map is a section, its grains are taken as columns through the thickness; a window is not periodic and is taken as a period by the solvers; one orientation per grain (the spread inside the grains is measured, not modelled); a hexagonal grid is read row by row; the Laue classes of the trigonal and monoclinic phases are taken in their usual settings. To follow (the second lot): the textures along a loading path (the lattices that slip and twins turn), pole figures among the outputs and the figures, a command, a tool of the interface and a tutorial. |
| Delivered (textures: figures, command, the texture of a map in a computed polycrystal) | The second lot of the requirement above ([theory.md](theory.md) §92). (1) `morphogen texture MAP`: the numbers of a map and of its grains phase by phase (grains, mean diameters by number and by area, spread inside the grains, the mean disorientation of the boundaries by length, the largest density of the pole figures); with `-o` the tables of the grains and of their boundaries, the map in the colours of its orientations and the figures; with `--convert` the map in the other format, as it is or cleaned. (2) Figures: pole figures on one colour range, the inverse pole figure with the direction of each grain, the map with its grain boundaries and its colour key, the disorientations against Mackenzie's density. (3) The density of an inverse pole figure in multiples of a random texture. (4) The texture of a map for the cells of a tessellation of space (`texture: {type: measured, file, spread}` of `voronoi` and `laguerre`): each cell takes the orientation of a grain of the map, drawn with its area (in a section the share of the area of the crystals of an orientation is that of their volume), then scattered; a polycrystal of space with the measured texture, where the columns of the model `ebsd` do not do. Checked: the inverse pole figure of one crystal against the kernel over the number of equivalent axes (2 %, the grid), the boundaries against the sides counted one by one, the orientations of a Voronoi polycrystal against the grains of the map (the cube component of the demonstration map: 39 % of the cells for 40 % of the area). Limits: the orientations are drawn cell by cell, so that the texture is that of the map but not the disorientations of its boundaries; the figures need matplotlib (the tables and the coloured map do not). To follow: the textures along a loading path, a tool of the interface and a tutorial. |
| Delivered (textures: the lattices along a loading path) | The third lot of the requirement above ([theory.md](theory.md) §92). The lattices of the crystals turn in finite strains (the rotation of the polar decomposition of the elastic part of the gradient; that of their twins): the material point gives the orientation of its lattice at each step (`PathResult.orientation`), a volume element the lattices of its crystals with their volumes at a time of the path (`PathResult.texture`), read also from the files of its fields (`morphogen texture FIELDS`: pole figures and inverse pole figure). Checked against closed forms: in single slip (basal slip of a hexagonal crystal) the tensile axis turns towards the slip direction with `l/l₀ = sin λ₀/sin λ = sin χ₀/sin χ` and the axis of a compression towards the normal of the slip plane with `h/h₀ = sin φ/sin φ₀ = cos λ/cos λ₀` (Schmid & Boas), within the elastic strains (2 × 10⁻³; 29.63° and 28.82° to 0.05°); a polycrystal of Taylor (50 face-centred cubic crystals in the same isochoric extension) takes the ⟨111⟩ and ⟨100⟩ fibres in tension (3.8 and 1.2 times random, ⟨110⟩ at 0.4) and the ⟨110⟩ fibre in compression (2.2, ⟨111⟩ at 0.2). Limits: in small strains the lattices do not turn (but for their twins), so that the textures of large strains are those of the paths in finite strains; no pole figures among the outputs of a configuration yet (the command draws them from the fields). |
| Delivered (textures: a tutorial and a tool of the interface) | The fourth lot of the requirement above. A tutorial whose snippets the tests run ([tutorial_texture.md](tutorial_texture.md), *From an orientation map to a textured polycrystal*): the map that comes with morphoGen read with its axes, its grains and its texture, the stiffness of its polycrystal (the exact bulk modulus; the soft cube axes: 188 GPa for 227 GPa with orientations at random), a tension in finite strains in which the lattices turn (0.7° on average at 2 %), the fibres of a Taylor polycrystal at a stretch of 1.5, a polycrystal of space with the texture of the map; with its three figures. The tool *Orientation map (EBSD)* of the tab Statistics of the interface: the numbers of the grains and of the texture of a file, the tables of the grains and of their boundaries, the map in the colours of its crystals, pole figures, the inverse pole figure and the disorientations of the boundaries; its polycrystal joins the workspace. The first feature (textures and the import of EBSD maps) is delivered with these four lots; the three others stay on hold. |

### 3.9 Scale-rich networks and exact meshes (2026-11-11 to 2026-11-17)

| Topic | Content |
|---|---|
| Scale-rich networks (2026-11-11) | A model of the scale-rich networks of architectured materials, whose scales are nested: Both, Chen, Gao, Mooij, Charara, Wadhwa, Portela & Barabási, *Scale-rich network-based metamaterials*, Nature 659, 345, 2026. |
| Delivered | FR-G17 (`morphogen.models.scale_rich`, `docs/theory.md` §93): the growth of the paper as convex voids split by strips (its reference implementation's scheme, exact partition of the box), its laws checked (thicknesses exact, tails of the lengths and degrees near 3), the relative density and the jamming threshold in closed form (mean field, a measured length constant: the paper's first-order formula is 19 % short on the lengths), the density solved on each realization, the variants of its Fig. 6 (aligned, polarized, graded), a frame for periodic cells; `analysis.power_law_tail` and `homogenization.postprocess.directional_modulus`; `examples/scale_rich_networks.py` (the elastic anisotropy against the geometric one; the anisotropy of the frame and of kinematic conditions on the edges alike, an open difference with the paper at small `alpha`). |
| Scale-rich networks in 3D (2026-11-12) | Three-dimensional microstructures of the model. |
| Delivered | FR-G18 (`morphogen.models.scale_rich_3d`, `docs/theory.md` §93.1): `dimension: 3` of `scale_rich_network`, the plates and beams of Fig. 6a of the paper: plates splitting convex polyhedral voids (closed cells, exact partition, rims and neighbours), beams grown until they touch (open cells); orientation laws and poles, gradients, frame; the first plate area in closed form (the mean inverse distance in the box), a measured area constant, the density and jamming in closed form (`1 - (1 - rho_1/3)^3`), tails 5/2 checked; a mean field of the beams, whose density is the volume of their union; the elastic anisotropy growing with `alpha` under kinematic conditions as in the paper (an independent code; `homogenization.postprocess.spatial_modulus`); `examples/scale_rich_networks_3d.py`, the template `scale_rich_plates`. |
| Exact meshes (2026-11-12) | Every morphology meshed from its exact geometry, without voxelization (clean geometry, then clean mesh: the approach of vtkSim, §7.3); exact meshes for all, without a background grid (exact polyhedra, adaptive refinement on the exact function of implicit surfaces). |
| Delivered | FR-E9 (`morphogen.mesh.exact`, `morphogen.mesh.octree`, `docs/theory.md` §94): `method: exact` of the conforming meshes: tessellations from their exact cells; faceted morphologies from their exact convex pieces (scale-rich plates and planar networks, fractal sponges), their faces overlaid into a face-to-face complex; the others from their exact point-wise level sets (`level_set_at`, given by every two-phase model with a level set; one per phase for the partitions of dead leaves, osteons and inclusions, `phase_level_sets_at`) on a balanced adaptive octree, cut by MMG (in turn for several phases), the interface nodes (and, for 10-node elements, the middles of the interface edges) moved onto the exact surface; periodic faces cut exactly and matching. Transparent dead leaves, tiling fields and oriented cells but tessellations stay on their images. |
| Exact meshes by default (2026-11-13) | `exact` as the default method instead of the grid. |
| Delivered | FR-E9: `method: auto`, the default of the configurations, of `conforming_mesh` and of `--mesh-method`: the exact mesh when the microstructure has an exact route (`exact_route`), the grid otherwise (transparent dead leaves, tiling fields, oriented cells but tessellations, objects with a level set on grids only) or should the exact mesh fail (MMG) or its periodic faces not match; tessellations from their exact cells (`min_feature` 0; a positive one regularizes them as before). |
| Exact meshes and the meshes of §3.8 (2026-11-17) | The exact meshes of this section and the meshes of §3.8 (10-node elements, several phases, sharp edges) behind one choice of method |
| Delivered | `method: auto` keeps the exact mesh as the default of two-phase structures and of the grains of tessellations, its 10-node elements those of its 4-node mesh curved as on the grid; what the grid alone met in §3.8 stays on it: the phases kept by a list, several phases, the sharp edges of smooth pieces ([theory.md](theory.md) §94; `method: exact` forces the exact route) |
| Exact route everywhere (2026-11-17) | The decision of 2026-11-13 (the exact mesh as the default) applied in full: the two interfaces of the implicit functions, on grids and at points, made one, and the exact route extended to what stayed on the grid: partitions of several phases, the sharp edges of smooth pieces, the phases kept by a list; one curvature of the 10-node elements |
| Delivered | FR-E9 ([theory.md](theory.md) §75, §76, §80, §94): one interface of the functions of the phases, `phase_level_sets_at(points, reach)` at any point and `phase_level_set` on a grid, the same values at the centres of the voxels (`level_set_at` and `level_set`: phase 1; `implicit_at`: one phase of `phase_level_sets_at`), given by every structure but transparent dead leaves and the coatings of structures that are not unions of spheres and segments: coatings and shells of unions, clustered spheres, inclusions and nested structures of any phases, tessellations with grain boundaries or without their grains, tilings, affine images; `method: auto` exact for all of them, the phases kept by a list (the whole partition meshed, then the phases taken), the sharp edges followed on the octree (the cuts of the pieces, their nodes on both surfaces even with 4-node elements); a cut made at once, then the grid, should MMG fail; `curved_mesh` the one curvature of the 10-node elements of both routes, with the exact functions wherever they are given. On the grid stay transparent dead leaves, the coatings of other structures (the distance of their voxels), oriented cells but tessellations, objects with level sets on grids only |

### 3.10 Version 0.7 (phase 3c) (2026-11-24 to 2026-11-25)

| Topic | Content |
|---|---|
| Arrays without copies (2026-11-24) | Voxelized realizations shared without copies with other array libraries (DLPack, JAX arrays), as inputs of JAX spectral solvers of partial differential equations (the ETDRK steppers of Exponax for Cahn–Hilliard and Allen–Cahn) and of differentiable pipelines, and as the initial conditions of the ensemble rollouts of stochastic Allen–Cahn equations (exponax-spde: Q-Wiener noise). |
| Delivered | FR-E11 (`morphogen.interop.arrays`, [theory.md](theory.md) §99): images handed over by DLPack, buffers aligned on 64 bytes that JAX shares on the CPU, fields of JAX back as images; the diffuse-interface phase field of a phase (the equilibrium profile of Allen–Cahn's and Cahn–Hilliard's equations), the grids and states of the steppers, ensembles of realizations for `jax.vmap`; checked by the motion by curvature of a disc in Exponax's Allen–Cahn equation within 1 % |
| Alloy design (2026-11-24) | Microstructures and their elasticity from the description of an alloy given by a tool of alloy design, such as Leil-Mat (mat.leil.app). |
| Delivered | FR-E14 (`morphogen.interop.alloys`, `morphogen alloy`, §103): an alloy described by the quantities such tools give (composition, phases and their fractions, precipitates and the law of their radii, grain size, moduli), its scales as configurations (precipitates in the matrix, grains with their phases and particles), their effective elasticity by FFT against the moduli predicted. Leil-Mat gives its programming interface on request only: the description is filled from its canvas until a client reads it |
| Microgen (2026-11-24) | Interoperability with Microgen (3MAH): its shapes in morphoGen, morphoGen's phases in Microgen. |
| Delivered | FR-E12 (the model `microgen`, `morphogen.interop.microgen`, §101): the shapes of Microgen that have a field as a model, the phases of any realization as shapes of Microgen, the tetrahedral meshes of both through PyVista |
| Particles for the discrete element method (2026-11-24) | Meshes of one phase without its matrix, as an export. |
| Delivered | FR-E10 (`morphogen.io.dem`, `export(..., particles=k)`, `morphogen export --particles`, the box *Particles (DEM)* of the tab Export, §100): a closed surface per particle, whole across the periodic faces; the exact spheres and clumps of unions of spheres and segments; STL, OBJ, VTU, tables, LAMMPS/LIGGGHTS and YADE files |
| LMGC90 (2026-11-25) | The particles exported as the `.msh` meshes that LMGC90 takes as input. |
| Delivered | FR-E10 (`morphogen.io.dem`, §100): Gmsh 2.2 files of the particles, their triangles or their tetrahedra (gmsh's mesher, or the fans of star-shaped particles), a particle per elementary entity and physical group, as LMGC90's pre-processor reads them (`readMesh`, `separateMeshes`, `surfacicMeshToRigid3D` or `volumicMeshToRigid3D`; checked against its reader of the version 2026.rc1, and by gmsh and meshio); the surfaces remeshed by MMG at a triangle size; the segments as capsules; `--tetrahedra`, `--element-size`, the formats of the box *Particles (DEM)* |
| Closing items of the workspace (2026-11-24) | An item of the workspace closed directly, without its menu. |
| Delivered | FR-I9: a cross on each chip of the strip (and the middle click), an Undo for a few seconds, the last sixteen items closed kept by the session |
| Fields of the mechanics in the page (2026-11-24) | The fields (stresses, strains...) of the tab Mechanics shown in the interface without ParaView; no video by default, one on demand in the export. |
| Delivered | FR-I9: the 3D view of the fields of a loading path opened with its result, a slider and Play through its steps; the video off by default, made by the export from the steps, which writes their files and their ParaView collection |
| Videos of every tool on demand (2026-11-25) | The tools of every tab write no video unless asked; their fields are shown in the page and their videos made by the export. |
| Delivered | FR-I9: Couplings, Explicit dynamics, Fracture, Nonlocal damage and Stability with their video off by default; the fields of their steps in the 3D view (`<name>_steps.json`), the frames of the others kept (`<name>.frames.npz`); the export of the runs of the tools (tables, fields and steps, the videos of a field along the steps, of a pattern oscillating, of the frames kept) |
| Finite elements by default (2026-11-25) | Conforming finite element meshes as the default of the homogenizations of every model, the voxels as the last resort. |
| Delivered | FR-H35: `homogenization.method: auto` takes the finite elements on the conforming mesh (exact wherever the model knows its geometry; a light mesh of the shortest edge over 10 when none is described), the FFT solver only for what only it does (`config.voxels_only`) or without MMG; `morphogen homogenize` and the tools of Mechanics mesh by default ([theory.md](theory.md) §94) |
| Delivered | FR-G21, FR-H35 (`models.nested_grains`, [theory.md](theory.md) §97): the subgrains of nested grains meshed from their exact convex pieces, each element with its subgrain and crystal (the finite elements turn them); grains filled with another structure by the functions of their phases at any point (the route of the level sets; slow where the inner phase meets the faces of the grains at sharp edges) |
| Blender (2026-11-24) | Interoperability with Blender and its physics engine. |
| Delivered | FR-E13 (`morphogen.interop.blender`, §102): the particles of a phase as Blender's rigid bodies, settled under gravity in a box (headless, by `blender --background` or the module `bpy`), the packing read back as a microstructure; particles given by closed surfaces (`MeshParticles`) |
| Microstructures for the fluids of Blender (2026-11-25) | Microstructures exported to Blender, notably for its fluid simulations. |
| Delivered | FR-E13 (`morphogen.interop.blender`, §102): `write_blender_scene`, the closed surface of each phase from its conforming mesh (STL, for OpenFOAM's snappyHexMesh too) and the fluid of Mantaflow around them (a domain along an axis, the phases as obstacles, an inflow and an outflow), baked by Blender, the flow read back as voxel images (`read_blender_flow`); the box *Blender (CFD)* of the tab Export |
| Long fibres (2026-11-24) | The classic model of long fibres, parallel or misaligned by a parameter. |
| Delivered | FR-G19 (`long_fibers`, §95): fibres along an axis, closed across the period, their sections a random packing to 0.8 or an array, misaligned by an RMS angle (a random field correlated along and across the fibres), the overlaps removed in space |
| Monodisperse or polydisperse (2026-11-24) | The objects of every model of one size or of a distribution of sizes. |
| Delivered | FR-G22 (§2, §98): every size of the objects of a model a number or a law, one size per object; the closed forms with the moments of the laws; the scale of a law solved for a volume fraction; the editor of the laws in the form of Generate. The sizes that stay single (the lengths of curved and cross-linked fibres, the thicknesses of coatings, honeycomb walls and nacre layers) are listed in §98 with the reason |
| Woven textiles (2026-11-24) | A model of 3D woven textile composites. |
| Delivered | FR-G20 (`woven_textile`, §96): 2D weaves in plies and 3D orthogonal and angle-interlock weaves, tows of elliptic, lenticular or power-ellipse sections along crimped paths, a partition of their union, the orientation of their material along their paths for the FFT solvers and the exports; meshed from their exact geometry (no background grid), a label and an orientation per element, the finite elements of the template on that mesh |
| Nested crystalline structures (2026-11-24) | Grains with subgrains, or another model within the grains. |
| Delivered | FR-G21 (`nested_grains`, §97): the subgrains of an inner tessellation in each grain, turned by small angles from its crystal; or the grains filled with another model turned with their crystals |

### 3.11 Version 0.8 (phase 3d) (2026-11-27)

| Topic | Content |
|---|---|
| Implicit geometry first (2026-11-27) | The realizations generated in implicit geometry, their measures exact with respect to the theory of mathematical morphology and of random closed sets; the exports as meshes or voxels after that. |
| Delivered | FR-M5 (`morphogen.morphology`, [theory.md](theory.md) §104, §105): the probes of mathematical morphology answered by the implicit functions of the phases at any point — points, pairs, segments (sphere tracing), balls (the distance), lines (crossings and chords) — for the fraction, the covariance, the Choquet capacities, the contact distributions, the chords and the specific surface (Crofton), with the standard deviations of their estimators; the four Minkowski functionals of unions of spheres (volume, surface, integral of mean curvature, Euler characteristic) and the three of their plane sections exactly, on the torus of a periodic box (power diagram and dual complex); the exact dilation of unions of spheres and segments; the measures of a realization beside the closed forms of its model; the named measures of the implicit geometry in configurations (`analysis`, without voxels), on the command line (`morphogen analyze` of a geometry) and in the interface (*Morphological measures*) |
| Mathematical morphology and its teaching (2026-11-27) | The theory of mathematical morphology reinforced in the package, and a tutorial to learn the theory and its applications with morphoGen. |
| Delivered | FR-M5 and the tutorial `docs/tutorial_morphology.md` (random closed sets, the Choquet capacity and Matheron's tests of a Boolean model, dilation, erosion, opening, closing and granulometry, the Minkowski functionals and Miles' formulae, stereology, the model of an image), run by the tests; the openings and closings of images of voxels not of unit size taking the voxels for both their steps |
| The model of an image (2026-11-27) | A morphological model identified for an arbitrary microstructure from an image (stereology) or a volume (voxels, a stack of sections) by mathematical morphology. |
| Delivered | FR-M6 (`morphology.identification`, §106): the measures of an image (fraction, specific surface or `L_A`, covariance, capacities of balls or discs and segments, Euler density), candidate models (Boolean spheres of one or a lognormal law of radii, hard spheres, Gaussian excursions, Poisson fibres, Voronoi mosaics) fitted by their closed forms, simulated on voxels of the size of the image, calibrated and ranked by the shapes of the covariance and the contact distributions; from a volume or from one section; `morphogen identify`, the best model written as a configuration |
| Models from segmentation (2026-11-27) | A model based on the segmentation of images and volumes. |
| Delivered | FR-M7 (`morphology.segmentation`, §107): Otsu's thresholds (one or several), smoothing, equalization, the particles of a phase and the ellipsoids of their moments, the laws of their sizes, the scores of a segmentation (accuracy, precision, recall, intersection over union, Dice); FR-G25 (`image`, §107): the image of a file segmented into its phases as a microstructure of implicit functions at any point (the iso-surfaces of its interpolated levels), in windows at its origin or drawn at random, and a synthetic tomography that comes with morphoGen |
| Thermo-chemo-elastic couplings in time (2026-11-27) | The coupled thermal, chemical and elastic fields of a microstructure in time (4D), and the inverse design (topology optimization) of structures in that context. |
| Delivered | FR-H37 (`coupling.thermochemoelastic`, §108): temperature, activity of a species and displacement solved together in time on voxel images and meshes of linear tetrahedra (heat of a reaction, Arrhenius' laws, stress-assisted diffusion; implicit Euler or TR-BDF2; a staggered fixed point with Aitken's relaxation; faces held, clamped or symmetric, periodic axes with a free or imposed mean strain), the fields at each output time and their XDMF time series; FR-O7 (`optimization.morphing`, §109): the layout of two materials that expand with a stimulus (or of one and voids) for target displacements at equilibrium or at times along a diffusion, SIMP with filter, projection and extrusion, MMA, exact adjoints through the steps of the diffusion, the layout verified by the coupled solver |
| Woven textiles enriched (2026-11-27) | The 3D woven model enriched. |
| Delivered | FR-G23 (`woven_textile`, §110): the variability of the tows along their paths (the deviations of the centrelines and the lognormal widths and thicknesses as stationary periodic Gaussian processes along the tows, correlated between neighbouring tows; their realized statistics, the mean sections and lengths in the theory); FR-G24: 3D interlocks of any binding depth and angle interlocks without stuffer warps; FR-H36 (`homogenization.unidirectional`): the tows from their fibres and matrix by the composite cylinder assemblage, the generalized self-consistent transverse shear or Hashin's bounds, Hill's universal relations, the conductivities and Levin's expansions |
| Credibility of the simulations (2026-11-27) | VIMSEO integrated for the demonstration of the credibility of the numerical simulations. |
| Delivered | FR-E17 (`interop.vimseo`, §112): the computations of morphoGen as models of VIMSEO (a configuration of `morphogen run` with its entries as inputs and its results as outputs; the exact densities of the realizations of a model of spheres with the closed forms of the model as references), their GEMSEO disciplines and VIMSEO models, the reference datasets, the verification of the code and of the solution (Richardson's extrapolation, the observed order, the grid convergence index); the optional extra `vimseo` (Python 3.10 to 3.12) |
| Code_Aster and OpenFOAM (2026-11-27) | Interoperability with Code_Aster and OpenFOAM. |
| Delivered | FR-E15 (`io.code_aster`, `morphogen code-aster`, `outputs.code_aster`, §111): the conforming mesh in Code_Aster's format or MED with its groups, the command and export files of the six unit strains and three unit gradients under periodic (`LIAISON_GROUP`) or kinematic conditions, crystals turned by their grains, run by `run_code_aster`, the tables read back into the apparent tensors, within 10⁻¹³ of morphoGen's finite elements by Code_Aster 18.1.11; FR-E16 (`io.openfoam`, `morphogen openfoam`, `outputs.openfoam`, §111): the polyMesh of voxels or tetrahedra with cyclic pairs, walls and cell zones, the cases of the flow through the pores (Darcy's permeability; tetrahedra on their polyhedral dual) and of the diffusion through a phase for both lines of OpenFOAM, run by `run_openfoam`, their outputs read back, the finite volumes on voxels in closed form within 10⁻¹⁰ by OpenFOAM v2412; the tools *Code_Aster study* and *OpenFOAM case* of the interface |
| Methods of other repositories (2026-11-27) | Repositories of microstructure generation, characterization, segmentation and learning examined (MicroStructPy, MCRpy, Fe-Cr_CNN, dendrite-segmentation, ml-microstructure-property, NPJCOMPUMATS-03747, Wood-Microstructure-Modeling, the segmentation of Ti-6Al-4V, VIL-GAN) for the methods to add to morphoGen, re-implemented from their papers. |
| Delivered | The segmentation front-end and its scores (FR-M7), the fit of morphological models to images (FR-M6), the statistics of particles for their models (FR-M7); FR-G26 (`spinodal`, §113): spinodal decomposition by the Cahn–Hilliard equation (the microstructures of the phase-field studies of those repositories) at an age of its coarsening. The deep-learning parts (networks, trained weights) and the data of the repositories without a licence are not taken; FR-M8 (`morphology.reconstruction`, `reconstruction` with `method: gradient`, §114): the descriptor-based reconstruction of MCRpy, by gradient descent on the relaxed cells (the two-point probability at every lag, the lineal paths), from the closed forms of a model or the measures of an image, a volume from a planar section; FR-G27 (`seeded_laguerre`, `models.seeded`, §115): the grains of MicroStructPy grown from ellipsoidal seeds — the seeds broken down into the spheres inscribed about their medial surfaces, placed the largest first within the overlap tolerance of Hart and Rimoli, the grain of a seed the union of the Laguerre cells of its spheres; the crowded seeds placed where they overlap least (every seed its grain) or left out as MicroStructPy does; the meshes and surfaces of the polycrystals by grain; the seeds of each phase of their own laws, amorphous phases merged into one region (a void: an amorphous phase that the meshes leave out), the verification of the grains against their seeds on their exact cells (§116: the ellipsoids of their second moments, the errors and the Kolmogorov–Smirnov distances to the laws of the seeds, `exact_grains`); the seeds of the plane, ellipses in the middle plane of the box whose grains are columnar, the figures of the verification and the tool *Grains and their seeds* (§117); seeds of boxes and of rectangles, the seeds of each phase filling its fraction of the volume (§118) |
| Meshes of the plane (2026-11-27) | Two-dimensional triangular meshes of the structures of the plane and their finite elements. |
| Delivered | FR-E18 (`mesh.triangles`, §119): the meshes of the plane (three- and six-node triangles, phases, grains and orientations, sides, the layer of tetrahedra extruded from them), the exact columnar grains of the tessellations of coplanar germs (the power diagram of the plane, the edges graded from their small features, matching periodic sides, `mmg2d`) and the sections of any structure by the level sets of its phases, their files (the plane elements of Abaqus and CalculiX, Gmsh, VTK, Medit) read back; FR-H38 (`homogenization.plane`, §119): their finite elements in generalized plane strain or plane strain and in conduction, periodic, kinematic and static conditions, the loadings of the solvers in space, verified against the uniform fields, the laminates, the checkerboard of Dykhne and the extruded layer in space; the meshes of the plane in the runs (`mesh.plane`, `mesh.z`, `homogenization.plane`, their files and figures) and in `morphogen homogenize` (`--plane`, `--z`), the columnar grains meshed in space by their polygons extruded, the exact sections of tessellations in space, the small features of the polygons removed; the loading paths of the meshes of the plane on their periodic layer in space; the orientation maps (EBSD) meshed by their pixels; the curved six-node triangles of the sections, their nodes on the curves of the functions of the phases, verified against the areas of discs, Perrins et al., Keller and the neutral coated cylinder of Hashin; the Stokes flow through the pores of the plane by the elements of Taylor and Hood and its flow along z (`homogenization.plane_flow`, `hydraulic_permeability` of the meshes of the plane), verified against Poiseuille, Sangani & Acrivos, Hagen–Poiseuille and the lattice Boltzmann method; the exact discs of the sections of straight fibres and of spheres, graded from their gaps, the rings of coated and hollow ones and the union of plain discs that cross each other (the Boolean models of discs), graded toward its corners; the exact contours of the sections of oblique and wavy fibres, traced body by body on their own distance functions (`mesh.contours`), their corners exact where capsules cross and found on their own, coated and hollow ones by the contours of their layers, verified against the ellipses of oblique fibres, crossing discs, counts on grids and Keller; the Stokes flow through the pores of the conforming meshes of space by the elements of Taylor and Hood (`homogenization.mesh_flow`), MINRES preconditioned beyond small meshes, verified against Poiseuille, the flows of the plane and Zick & Homsy; the axial stiffness and conductivity of misaligned plies, the tensors of their sections turned along the segments of their fibres (`homogenization.unidirectional.misaligned_stiffness`, `misaligned_conductivity`: the bounds of Voigt and Reuss and the layers of Backus), verified against the laminates, the fibres turned in a plane and the FFT solution of their voxels, and recorded by the runs (`homogenization.misaligned`); the tutorial *Fibre composites in their plane* ([tutorial_fibres.md](tutorial_fibres.md)); the tutorial *Polycrystals of the plane, meshed and solved in their plane* ([tutorial_plane.md](tutorial_plane.md)), whose snippets the tests run |

## 4. Scope

In scope:

* the generation of microstructures, their description and measurement;
* RVE statistics;
* exports for FE and FFT codes and for additive manufacturing;
* the Python API, the command line and configuration files;
* later, a graphical interface and GPU acceleration.

* computational homogenization (phase 2): finite elements on conforming tetrahedral
  meshes and an FFT solver on voxel images, for the apparent elastic and thermal
  properties used in RVE studies; polycrystals of anisotropic crystals (phase 2b, §3.3).
* parallel runs on one machine and on clusters (phase 2b).
* generation by cellular automata and lattice growth processes, combinatorial design
  of assemblies and frameworks, topology optimization of periodic cells, visualization
  and a graphical interface (phase 3a, §3.2).
* the periodic element (macroscopic strains and stresses as unknowns), thermo-elasticity,
  post-processing of macroscopic and normalized values, a database of the
  configurations tested with Ashby charts, nested and bio-inspired models (phase 2c,
  §3.4).
* nonlinear behaviours along loading paths, crystal plasticity and finite strains on
  cells and RVEs (phase 2c, §3.4); behaviours written in MFront, integrated through MGIS
  (phase 3b, §3.5).

Out of scope for now:

* **Non-linear analyses of structures:** morphoGen's nonlinear behaviours (phase 2c)
  and the MFront behaviours (phase 3b) follow loading paths on cells and RVEs; for
  structures, morphoGen exports meshes, volumes and material files for the codes that
  handle them (Abaqus/CalculiX, FEniCSx, DAMASK, AMITEX_FFTP), and fits their results
  (FR-V5).
* **Segmentation of real images.** morphoGen measures binary and label images, and
  finds the grains of orientation maps (EBSD: their points come with the orientation of
  the crystal, §3.8), but does not segment micrographs or tomographs (on hold: §3.8).
* **CAD editing, and features tied to commercial software.**

## 5. Functional requirements

Priority: **M** must, **S** should, **C** could. Phase: see §10 (1 = this first
version, 0.1).

### 5.1 Morphological models

| Id | Requirement | Prio. | Phase |
|---|---|---|---|
| FR-G1 | Boolean model of spheres (Poisson germs, constant or random radii), given by its volume fraction or its intensity | M | 1 |
| FR-G2 | Poisson fibres: Boolean model on isotropic Poisson lines (infinite cylinders), non-periodic (Dirrenberger et al. 2014) | M | 1 |
| FR-G3 | Boolean model of finite fibres: cylinders or spherocylinders with radius and length distributions and an orientation distribution (isotropic, aligned, planar) | M | 1 |
| FR-G4 | Hard-core spheres by random sequential addition: target fraction, minimal gap, polydispersity | M | 1 |
| FR-G5 | Voronoi and Laguerre tessellations (Poisson or fixed number of germs), with a phase per cell (multiphase mosaics) and random crystal orientations | M | 1 |
| FR-G6 | Excursion sets of Gaussian random fields, given by their correlation length and volume fraction | S | 1 |
| FR-G7 | TPMS (gyroid, Schwarz P and D, Neovius), sheet or network, given by a level or a target fraction | S | 1 |
| FR-G8 | Strut lattices (cubic, BCC, FCC, octet), given by a strut radius or a relative density | S | 1 |
| FR-G9 | Further models: random laminates with layer breakup (Bironeau 2016), Laguerre germs from packings, dead leaves, Kelvin cells (2b); Johnson–Mehl tessellations (3a); multiscale and clustered (Cox-like) inclusions as combinations (FR-G12); hard-core fibres, dense packings beyond the RSA limit (Lubachevsky–Stillinger), curved fibres (worm-like chains, arcs of distributed radii of curvature as in PuMA, helices; overlapping or not; hollow), graded or rotated lattices, dead leaves of families of their own sizes, of spheroids and of finite coverage, penetrable spheres with impenetrable cores (the cherry-pit model), transparent dead leaves, Voronoi tessellations of stealthy hyperuniform germs (3b) | S | 2b–3b |
| FR-G10 | Every model works in periodic and non-periodic boxes when this is meaningful; non-periodic samples are unbiased windows of the stationary model | M | 1 |
| FR-G11 | A realization is determined by (model, domain, seed, version); seeds of a study are derived from the study seed and the item indices, independently of how the study is split across jobs | M | 1 |
| FR-G12 | Combination of models: union, intersection, difference, complement (with combined level sets), inclusions in a polycrystal (keeping its grains and orientations), grain-boundary phase of tessellations | S | 2b |
| FR-G13 | Nested models: a model filling the phase 1 of another (Cox spheres in aggregates, nested combinations), coatings and hollow members, hierarchical lattices of frames of tubes (Rayneau-Kirkhope et al. 2012), fractal sponges (iterated function systems), hollow strut lattices | M | 2c |
| FR-G14 | Bio-inspired models: nacre (brick and mortar), Bouligand plywoods, honeycombs and wood (radial files, growth rings, rays), Haversian bone (remodelled osteons), open-cell foams and trabecular bone (anisotropic, relaxed cells), branching networks grown by space colonization with Murray's law; with their closed-form characteristics | M | 2c |
| FR-G15 | Stochastic reconstruction of two-phase microstructures from their descriptors (Yeong & Torquato 1998): the two-point probability and the lineal-path function along the axes, the targets of a Debye medium, of a model or of an image | S | 3b |
| FR-G16 | Foams as level sets of distance fields (3b, after Degdeg & Dirrenberger 2026, Sonon et al. 2012 and Kilingar et al. 2019): seeds of aligned polydisperse ellipsoids densely packed, struts with Plateau-border cross-sections of controlled curvature thickening towards the nodes, open or closed cells, the thickness solved for a solid fraction, a level set for conforming meshes; the scale and shape factors of the seeds calibrated on the mean radii of the sections of the pores | S | 3b |
| FR-G17 | Scale-rich networks (after Both et al. 2026): ligaments of thicknesses `lambda_0 t^-alpha` grown one after another from points of the voids until they meet, extruded prisms; the initial thickness or the relative density, uniform, aligned or polarized orientations, gradients of nucleation, a frame; their dual graph, orientation tensor and power-law tails; the density and the jamming threshold in closed form | S | 3b |
| FR-G18 | Scale-rich networks in 3D: plates splitting convex polyhedral voids into closed cells, or beams grown until they touch; normals or axes uniform, from an orientation law or polarized by a pole; the density and the jamming threshold of plates in closed form, the tails of their areas and degrees; a mean field of the beams | S | 3b |
| FR-G19 | Long fibres along an axis (unidirectional composites): closed across the period, their sections a random packing (to 0.8) or a hexagonal or square array with a disorder, straight or misaligned by an RMS angle with correlation lengths along and across the fibres, without overlap; mono- or polydisperse | S | 3c |
| FR-G20 | Woven textile composites: 2D weaves (plain, twill, satin, basket) in plies and 3D weaves (orthogonal, layer-to-layer and through-thickness angle interlocks), tows of elliptic, lenticular or power-ellipse sections along crimped paths, the orientations of their material along their paths; nominal fractions, crimps and thickness in closed form | S | 3c |
| FR-G21 | Nested crystalline structures: grains of a tessellation with the subgrains of an inner tessellation turned by small angles from their crystals, or filled with another model turned with them | S | 3c |
| FR-G22 | Monodisperse or polydisperse objects in every model where objects have a size: a number (one size) or a distribution (one size per object), the closed forms with the moments of the laws, the scale of a law solved where a volume fraction sets the size | S | 3c |
| FR-G23 | Variability of the tows of woven textiles along their paths: deviations of the centrelines and lognormal widths and thicknesses as stationary periodic Gaussian processes along the tows, correlated between neighbouring tows; their realized statistics and the theory of the mean sections and lengths | S | 3d |
| FR-G24 | 3D woven interlocks of any binding depth (layer-to-layer and orthogonal), and angle interlocks without stuffer warps | S | 3d |
| FR-G25 | The microstructure of an image (a volume, a stack of sections, a section) segmented into its phases, described by implicit functions at any point, in windows at its origin or drawn at random | S | 3d |
| FR-G26 | Spinodal decomposition by the Cahn–Hilliard equation (semi-implicit spectral), at a fraction of the lever rule and an age of its coarsening | S | 3d |
| FR-G27 | Grains grown from seeds (MicroStructPy): ellipsoidal seeds of laws of sizes and shapes, at random or aligned, the seeds of each phase of their own laws, broken down into spheres and placed within an overlap tolerance; the grain of a seed the union of the Laguerre cells of its spheres, meshed as one subdomain; amorphous phases merged into one region, voids left out of the meshes; ellipses in the plane (columnar grains); boxes and rectangles; the grains verified against their seeds on their exact cells, with their figures and a tool of the interface | S | 3d |

### 5.2 Representations

| Id | Requirement | Prio. | Phase |
|---|---|---|---|
| FR-R1 | Vector: explicit primitives (centres, radii, axes), germs and weights, Fourier modes, analytic surfaces; saved as JSON, CSV for tables | M | 1 |
| FR-R2 | Implicit: level-set fields on any grid (exact signed distances for unions of grains) and point membership queries | M | 1 |
| FR-R3 | Voxel: phase or label images at any resolution, with optional partial-volume voxels (supersampling) | M | 1 |
| FR-R4 | Chunked generation of volumes larger than memory: slabs written to memory-mapped raw files with an XDMF or MetaImage header | S | 2b |

### 5.3 Morphological characterization

| Id | Requirement | Prio. | Phase |
|---|---|---|---|
| FR-M1 | Theoretical characteristics of each model when known: volume fraction, specific surface, covariance, integral range, variance exponent, cell statistics | M | 1 |
| FR-M2 | Measurements on images: volume fraction, specific surface (Crofton, 3 or 13 directions), covariance (periodic or not), connected components and percolation, cell volumes and neighbours | M | 1 |
| FR-M3 | Granulometry by openings, chord-length distributions, Minkowski functionals and Euler characteristic, pore sizes (local thickness), geometric tortuosity (2b); the lineal-path, two-point cluster and pore-size functions, the radial distribution function and the structure factor of point patterns (3b); (3b) Choquet capacities T(K) of balls, segments, pairs of points and cubes (dilations, erosions, openings, closings), spherical and linear contact distributions, the morphological covariance of grey-level images, the integral of mean curvature (plane sections) and the Minkowski curves of excursion sets, Betti numbers and connectivity density; (3b) mean intercept lengths and the MIL fabric tensor (principal MILs and axes, degree of anisotropy), the first minima of the covariance along the axes (hole effect); stereology: sections by the grid planes (watershed separation, frame correction), Saltykov's unfolding for spheres and oriented ellipsoids, the shape consistency of the section radii | S | 2b–3b |
| FR-M4 | Statistical analysis of apparent properties after Dirrenberger et al. (2014): intervals, relative errors, fitted and imposed exponents, RVE sizes and object counts, discrepancy between boundary conditions, isotropy (written from the paper, no earlier code existing) | M | 2 |
| FR-M5 | Measures on the implicit geometry of the realizations, without a grid: probes (points, pairs, segments by sphere tracing, balls by the distance, lines and their crossings) for the fraction, the covariance, the Choquet capacities, the contact distributions, the chords and the specific surface, with the standard deviations of their estimators; the exact Minkowski functionals of unions of spheres and of their plane sections (power diagram, dual complex), on the torus of periodic boxes; the exact dilation of unions; the measures beside the closed forms of the model | S | 3d |
| FR-M6 | Identification of a morphological model from an image or a volume: candidates fitted by their closed forms to the fraction and the specific surface, simulated on voxels of the size of the image, calibrated and ranked by the shapes of the covariance and of the contact distributions; from a volume or from one section (stereology) | S | 3d |
| FR-M7 | Segmentation of grey-level images and stacks: Otsu's thresholds (one or several), smoothing and equalization, the particles and the ellipsoids of their moments, the laws of their sizes, the scores of a segmentation against a reference | S | 3d |
| FR-M8 | Reconstruction from descriptors by gradient descent: the cells relaxed, the two-point probability at every lag and the lineal paths with their exact gradients, towards the closed forms of a model of random sets or the measures of an image; a volume from a planar section | S | 3d |

### 5.4 RVE statistics

| Id | Requirement | Prio. | Phase |
|---|---|---|---|
| FR-V1 | Point variance (two-phase and multiphase), ensemble statistics per size, relative error | M | 1 |
| FR-V2 | Fit of D²(V) = K V^-γ with γ free or imposed, on a chosen range of sizes; A₃ or A₃* when D² is known | M | 1 |
| FR-V3 | RVE volume for (ε_rel, n), and number of realizations for (V, ε_rel) | M | 1 |
| FR-V4 | Ensemble method (independent realizations per size) and window method (sub-volumes of a large realization or of any scalar field) | M | 1 |
| FR-V5 | Fit of external tables (CSV), e.g. apparent moduli from FE/FFT codes or image measurements | M | 1 |
| FR-V6 | Bootstrap confidence intervals of γ, K and V_RVE (phase 2); detection of several scaling regimes (Bironeau 2016) | C | 2 |
| FR-V7 | RVE statistics of nonlinear responses: quantities of the curves of loading paths (a component of the strain or stress, a mean of a phase, the cumulated plastic strain, at the end of the path or of one of its steps: the flow stress at given strains) analysed as the apparent properties, in configurations and by `morphogen fit` | S | 2c |

### 5.5 Exports and interoperability

| Id | Requirement | Prio. | Phase |
|---|---|---|---|
| FR-E1 | Voxel volumes: VTK XML (.vti), legacy VTK (.vtk, cell data, for FFT codes), MetaImage (.mhd/.raw), NumPy (.npz/.npy), TIFF stacks (optional dependency) | M | 1 |
| FR-E2 | Hexahedral voxel meshes of selected phases, with phase element sets, face node sets and periodic node pairs: Abaqus/CalculiX .inp, Gmsh .msh 2.2 with boundary groups, VTK .vtu; other formats through meshio (e.g. XDMF for FEniCSx) | M | 1 |
| FR-E3 | Watertight surface meshes (STL) from level sets, closed on the box faces (additive manufacturing) | M | 1 |
| FR-E4 | Gmsh OpenCASCADE scripts (.geo) of unions of spheres and cylinders, for conforming tetrahedral meshes with the external gmsh program | S | 1 |
| FR-E5 | Robust conforming tetrahedral meshes from implicit geometry (MMG level-set discretization), with matching opposite faces when the microstructure is periodic (along all or some of the axes); meshes of cells symmetric about their mid-planes mirrored from their irreducible part (exact symmetries, matching faces, copies side by side), with the diagonal mirrors of square and cubic cells (the groups 4/mmm and m-3m: the chamber of the group meshed and mapped by its operations); implicit bodies of level-set functions; exports to Abaqus/CalculiX, Gmsh, VTK, Medit and meshio formats | M | 2 |
| FR-E6 | XDMF outputs (raw binary data, no HDF5 dependency), FEniCSx meshes with phase, grain and face tags, Neper seed files, DAMASK grids and material files, periodic constraint equations for Abaqus/CalculiX (2b), the materials of the phases in Abaqus inputs (3b); AMITEX_FFTP runs of images (material and zone fields, material, loading and algorithm files, MPI script; small and finite strains) and the reading of their results (3b), the nonlinear materials as MFront laws of the umat interface (3b; in finite strains every phase) | S | 2b–3b |
| FR-E7 | MFront sources of the nonlinear materials (von Mises or Hill plasticity and crystal plasticity, in small and finite strains; Hencky's elasticity in finite strains), for the codes that integrate MFront behaviours (AMITEX_FFTP and Cast3M, Abaqus, Code_Aster, MGIS), and the materials of AMITEX and Abaqus inputs (conforming and voxel meshes, small and finite strains), user-supplied MFront laws included; checked against morphoGen's kernels | S | 3b |
| FR-E8 | Raw data and figures (3b): tables as .csv, tab-separated .txt and .xlsx (no dependency); figures as .png, .tif, .svg, .pdf and TikZ (pgfplots code of matplotlib figures, compiled by LaTeX in the tests); one entry point `export` choosing the writer from the object and the suffix (images, meshes, microstructures, band structures, bifurcation analyses, loading paths, frames, tables, figures, videos) | S | 3b |
| FR-E9 | Exact conforming meshes without a background grid: from the exact cells of tessellations, the exact convex pieces of faceted morphologies, or the exact point-wise level set of the others on an adaptive octree, the nodes of the interface on the exact surface, matching opposite faces when periodic; the default (`method: auto`: exact wherever the model has an exact geometry, several phases, the phases kept by a list and the sharp edges of smooth pieces included; on a grid otherwise, or should the exact mesh fail; `method: exact` or `grid` forces one) | S | 3b |
| FR-E10 | The particles of one phase without its matrix for the discrete element method: a closed surface per particle (whole across periodic faces), the exact spheres and clumps of unions of spheres and segments; STL, OBJ, VTU, Gmsh meshes of triangles or tetrahedra for LMGC90, tables, LAMMPS/LIGGGHTS data files, YADE columns | S | 3c |
| FR-E11 | Voxel images as the arrays of other libraries without copies (DLPack; JAX, PyTorch, CuPy); diffuse-interface phase fields, grids and states of spectral phase-field solvers (Exponax), ensembles of realizations | S | 3c |
| FR-E12 | Microgen (3MAH): its shapes as a model, the phases of morphoGen as its shapes, the tetrahedral meshes of both (optional dependency) | C | 3c |
| FR-E13 | Blender: a microstructure as a scene of Blender (the closed surfaces of its phases from its conforming mesh, the fluid of Mantaflow around them, the flow baked read back), and the particles of a phase as its rigid bodies settled under gravity, headless (optional program) | C | 3c |
| FR-E14 | Alloys from tools of alloy design: a description of their phases, the microstructures of their scales and their effective elasticity | C | 3c |
| FR-E15 | Code_Aster: the study of the apparent stiffness and conductivity of a conforming mesh (mesh `.mail` or MED with the groups of the phases, grains, faces and periodic nodes; materials `ELAS`/`ELAS_ORTH`/`THER`/`THER_ORTH`, crystals by `ANGL_EULER`; periodic `LIAISON_GROUP` or kinematic conditions; `POST_ELEM` integrals; export file of `run_aster`), run by Code_Aster and its tables read back (`morphogen code-aster export|run|read`, `outputs.code_aster`); verified with Code_Aster itself where it is installed | S | 3d |
| FR-E16 | OpenFOAM: the polyMesh of the voxels of some phases or of a conforming mesh (OpenFOAM's face order, the patches of the box, cyclic pairs, walls, cell zones) and the cases of the steady flow through the pores (through tetrahedra on their polyhedral dual) and of the diffusion through a phase, for both lines of OpenFOAM, run by OpenFOAM and their outputs read back into the permeability and the effective diffusivity (`morphogen openfoam export|run|read`, `outputs.openfoam`); verified with OpenFOAM itself where it is installed | S | 3d |
| FR-E17 | VIMSEO: the computations of morphoGen as its models (configurations, exact morphological densities with their closed forms as references), for the verification of the code and of the solutions, the validation, the uncertainties and the calibration (optional dependency, Python 3.10 to 3.12) | C | 3d |
| FR-E18 | Meshes of the plane: triangular meshes (three or six nodes) of the structures invariant across a plane, the exact polygons of the tessellations of coplanar germs (columnar grains) graded from their small features, the sections of any structure in a plane, matching periodic sides; the plane elements of Abaqus and CalculiX, Gmsh, VTK and Medit files read back | S | 3d |

### 5.6 Interfaces

| Id | Requirement | Prio. | Phase |
|---|---|---|---|
| FR-I1 | Documented Python API (library usable from scripts and notebooks) | M | 1 |
| FR-I2 | Command line: run a configuration, fit RVE data, analyse an image, estimate the percolation threshold of a model, homogenize a sample, list and describe models, print example configurations | M | 1–2 |
| FR-I3 | Configuration files in YAML (JSON and TOML accepted), validated with explicit error messages | M | 1 |
| FR-I4 | Graphical interface: forms built from the parameter descriptions, sections, generation, measures, export; animations of the formation of structures; live topology optimization; design-space exploration; the whole chain on one realization (model, mesh or voxels, materials, apparent property, views of the fields on the deformed mesh, videos, every file and the configuration to download; views redrawn without solving again) (`morphogen gui`, decision D2); since 2026-10-25, a workflow from tab to tab through the workspace of the session (Generate, Simulate, Export, Optimize: a morphology sent to the simulation, the export or the optimization, a simulation to the export or the optimization, the optimized design back as a morphology), the views of the level sets (sections at up to 2048 pixels per edge without a voxel image of the volume, measures of the level sets), voxel images on demand up to 2048 voxels per edge within the free memory, optimizations started from a morphology or a simulation, the formation shown in Generate, the videos of a load colouring the fields as they grow; tabs without numbers taking items of the workspace in any order, tools built from a registry (`morphogen.gui.tools`: forms of typed fields, jobs with their progress), the tab Statistics (RVE sizes by the ensemble and window methods, measures, correlation functions, stereology, the RVE fit of pasted tables, reconstructions, percolation thresholds), the tabs Mechanics (loading paths, NTFA, material points, linear tensors), Fracture (stress intensity factors, crack growth, cohesive zones, polycrystals), Nonlocal damage (microfoam, ductile damage with the triaxiality, quasi-brittle damage), Stability (post-bifurcation of frames and morphologies, nonlinear paths, Bloch bands), Couplings (diffusion, oxidation, hygromorph multilayers), Design (automaton rules of morphologies, rigidity of strut lattices) and Solvers & formats (AMITEX_FFTP, Abaqus, mesh and image formats, MFront, Z-set); FFT simulations on the GPU, and the FFT tools of Mechanics there (a device); the workspace saved and opened; since 2026-10-26, menus in four sections (Create, Characterize, Simulate, Export & compare) with their tabs, the items of the workspace sent on to the forms that take them (and kept when another tool of the tab is chosen) or removed, compact forms with their fields in boxes, the runs of the tools kept in the workspace and shown again with their parameters, crack growth through the morphologies of the workspace; the models by family, the forms offering only the items their tools take; the 3D view of the items of the workspace (WebGL: the interfaces of the phases from the level sets, the facets of polycrystals, the fields of simulations on the deformed mesh, a section plane); the long computations stopped from the page at their next step (a tool or a simulation keeping nothing, an optimization its design so far); the behaviours of materials kept in the workspace (a material point, an import of Z-set) and taken by the loading paths of morphologies, the NTFA, the material point, the bulk of a cracked plate or a phase of a cracked morphology, the inputs of AMITEX_FFTP and Abaqus, and the MFront sources; the tab Comparisons (the curves of several runs on one figure, with the points of a test); a tool run on several items one after another; the identification of a behaviour on runs followed together or on the points of a test (FR-H34), the behaviours of the workspace written as Z-set files (FR-H27) | M | 3a |
| FR-I6 | Figures: sections, periodic tilings, 3D surfaces, histories, transformation kinetics, animated GIFs (`morphogen view`, `morphogen.viz`) | S | 3a |
| FR-I7 | Videos (3b): .mp4 (ffmpeg), .gif or PNG frames, played once, in a loop or back and forth; sections sweeping through a voxel image and the three orthogonal sections (Avizo-like), turntables of phases and meshes, cross-sections sweeping through a mesh and fields animated with a magnified displacement, coloured by matplotlib colormaps on a fixed range (3D views rendered off screen by VTK) | S | 3b |
| FR-I8 | The whole chain in one configuration (3b): from the morphological model to its voxel image, conforming mesh and the fields of the homogenization; the `visualization` section draws figures (bitmaps, vector graphics, TikZ) and videos of the first realizations: sections and orthoslices of the image and its labels, views, turntables and sweeps of the mesh coloured by phase, the fields of properties, loadings and loading paths (at every step) on the deformed mesh (magnified, total or fluctuation, cut by a plane) or the voxels, the curves of the paths, colour bars on fixed ranges; `outputs.tables` writes the tables as .txt and .xlsx; missing dependencies (VTK, ffmpeg) are reported before the run; the template `workflow`; `morphogen export` converts images, meshes with their fields, geometries and tables to other formats, figures and videos | S | 3b |
| FR-I9 | The workspace and the results in the page (0.7): a cross and an Undo on each item; the fields of the loading paths in 3D step by step without an external viewer, videos on demand only; an editor of the laws of the sizes in the forms | S | 3c |
| FR-I5 | Parameter metadata (description, kind, unit, default) available programmatically, for the command line, the documentation and the future GUI | S | 1 |

### 5.7 Batch and HPC

| Id | Requirement | Prio. | Phase |
|---|---|---|---|
| FR-B1 | Multithreaded kernels, with a thread-count option | M | 1 |
| FR-B2 | Studies split across cluster jobs (task id and number of tasks, e.g. SLURM job arrays), with per-task result files and a merge step | M | 1 |
| FR-B5 | Parallel realizations on one machine (process pool with a thread budget) and on clusters: job arrays of SLURM, PBS Pro, Torque, SGE, OAR and LSF written and submitted with a dependent merging job, task id read from the scheduler, MPI; per-realization records, resumption of interrupted runs, status; results independent of the splitting | M | 2b |
| FR-B3 | GPU acceleration of the heaviest kernels as an optional backend: the FFT solvers of voxel images, linear and along loading paths (the integration of the behaviours included), on NVIDIA GPUs (CuPy, D3), chosen per problem (`device`: CPU, GPU or automatic) | S | 3b |
| FR-B4 | Distributed generation of a single gigantic volume (slab decomposition) | C | 4 |

### 5.8 Homogenization

| Id | Requirement | Prio. | Phase |
|---|---|---|---|
| FR-H1 | Apparent elastic (bulk and shear moduli, stiffness tensor) and thermal (conductivity, tensor) properties of each realization, with the volume averages, energy estimates and per-phase statistics of the fields; pores carry no stress nor flux | M | 2 |
| FR-H2 | Kinematic (KUBC), mixed (MBC, Dirrenberger et al. 2014), periodic (PBC) and static (SUBC, non-porous media) boundary conditions | M | 2 |
| FR-H3 | Finite elements on the conforming meshes of FR-E5: 4-node and 10-node tetrahedra; no finite elements on voxels | M | 2 |
| FR-H4 | FFT solver of voxel images (periodic conditions), in particular for models without a level set (tessellations) | M | 2 |
| FR-H5 | Solvers for 10⁵–10⁶ unknowns on a laptop, without compiled dependencies beyond binary wheels | M | 2 |
| FR-H6 | Validation: exact solutions (homogeneous media, laminates), an independent finite element code, ordering of the boundary conditions, bounds (Table 4 of the 2014 paper) | M | 2 |
| FR-H7 | Isotropy criteria (eqs. 72–74) and discrepancy between conditions (eq. 69) | M | 2 |
| FR-H8 | Local fields (stress, strain, energy, flux) written for ParaView | S | 2 |
| FR-H9 | Homogenization within configurations and RVE studies (one results row per realization, a long table per property and condition), and from the command line | M | 2 |
| FR-H10 | Conforming meshes of tessellations (polycrystals) for finite elements: one subdomain per grain, from the exact cells, with the features smaller than a length removed, matching periodic faces | M | 2b |
| FR-H11 | Anisotropic phases: cubic, hexagonal or general stiffness and conductivity tensors turned by the orientation of each grain, in the finite element and FFT solvers; bounds of polycrystals | M | 2b |
| FR-H12 | Periodic polycrystals meshed from their whole cells (periodized at the design): the boundary of the union made of translated facets meshed alike, node-to-node periodic links | M | 2c |
| FR-H13 | Macroscopic loadings mixing prescribed strain and stress components (the periodic element of Z-set), in elasticity and conduction, for every boundary condition and both solvers; named loadings in configurations and on the command line | M | 2c |
| FR-H14 | Thermo-elasticity: thermal expansion of the phases, apparent thermal expansion tensor, thermal stresses; Levin's relation and bounds | M | 2c |
| FR-H15 | Post-processing: engineering constants, anisotropy indices, macroscopic values of porous media, relative, normalized and specific values | M | 2c |
| FR-H16 | Nonlinear behaviours of the phases: von Mises elasto-viscoplasticity (Norton flow, linear and nonlinear isotropic and kinematic hardening) and rate-independent plasticity, integrated implicitly with their consistent tangent, and Hill's anisotropic criterion with the same hardenings (2026-10-29); a material point simulator | M | 2c |
| FR-H17 | Loading paths in time (each component a strain or a stress, and a change of temperature, piecewise linear) solved by finite elements with the macroscopic strain components as degrees of freedom of the Newton iterations (the periodic element), under periodic and kinematic conditions; macroscopic curves, phase averages and element fields; configurations and command line | M | 2c |
| FR-H18 | Crystal plasticity of Méric and Cailletaud for polycrystals: slip families of cubic crystals (octahedral, cube, {110} and {112} bcc), Norton flow, nonlinear isotropic hardening with an interaction matrix (Franciosi's six interactions) and nonlinear kinematic hardening per system; slip systems turned by the orientation of each grain; implicit integration with its consistent tangent, robust for large steps | M | 2c |
| FR-H19 | Finite strains for every nonlinear behaviour and loading path: total Lagrangian finite elements, elasticity and von Mises plasticity in the logarithmic strain space, multiplicative crystal plasticity with lattice rotations; nominal, true (Cauchy) stresses and logarithmic strains as outputs; the material point simulator, configurations and command line | M | 2c |
| FR-H20 | Loading paths on voxel images by the FFT solver: the periodic element on trilinear voxels whose volumetric strain is averaged over each voxel (B-bar, F-bar: no locking under plastic flow), Newton–Krylov iterations (conjugate gradients preconditioned in Fourier space, the tangent applied without assembly), small and finite strains, von Mises and crystal plasticity in the grains of a label image; configurations (`method: fft`), command line, fields as images | M | 2c |
| FR-H21 | Behaviours written in MFront (TFEL) for every nonlinear solver: standard strain-based behaviours in small strains, standard finite strain behaviours in finite strains (first Piola–Kirchhoff stress and its derivative), integrated point by point by MGIS with their consistent tangents, by finite elements, on voxels (CPU and GPU) and at a material point; crystals turned by the orientation of each grain; material properties, parameters, temperature and external state variables; `.mfront` sources compiled on first use and cached; validated against MFront twins of morphoGen's behaviours | S | 3b |
| FR-H22 | Couplings of transport, chemistry and mechanics (3b): transient diffusion in voxel images (heterogeneous diffusivities and solubilities, exposed faces, first-order consumption); thermal oxidation of polymers by the closed-loop scheme, homogeneous or diffusion-limited (Ernault et al. 2019); properties as functions of an ageing tracer and graded phases of tracer fields for every solver and export (shrinkage and swelling as eigenstrains); Ekenstam's hydrolysis of cellulose with the critical degree of polymerization of Vibert et al. (2024); hygromorph multilayers: moisture through the thickness and curvatures of the free plate (Mou et al. 2025); transient diffusion in configurations (`diffusion` section); diffusion and free swelling of tetrahedral meshes with FEniCSx (bending of strips, small and finite strains); the stresses of oxidized samples (Ernault et al. 2019); orthotropic hygromorph layers from microstructures | S | 3b |
| FR-H23 | FEniCSx as the solver of the finite element problems of tetrahedral meshes (3b): `solver: fenicsx`, the weak forms in UFL, periodic conditions by dolfinx_mpc, the same constrained degrees of freedom as morphoGen's solver for KUBC, SUBC and MBC, PETSc's direct or multigrid solvers; the same averages and statistics, so that the two solvers are interchangeable and check each other | S | 3b |
| FR-H24 | Nonuniform transformation field analysis (3b; Michel & Suquet 2003, 2004): reduced models of periodic voxel microstructures of von Mises phases with isotropic and kinematic hardening (plastic or viscoplastic), modes by POD of full-field snapshots, elastic influence fields by FFT, reduced laws integrated implicitly along loading paths under strain, stress or mixed control, local fields rebuilt, models stored; in configurations (`homogenization: ntfa`) | S | 3b |
| FR-H25 | Bloch waves in periodic cells (3b): the band structure of the elastic waves of tetrahedral meshes and voxel images along the edges of the Brillouin zone (Bloch-periodic map of the fluctuations, consistent mass matrices of 4- and 10-node elements, shift-invert Lanczos), band gaps and figures | S | 3b |
| FR-H26 | Post-bifurcated patterns of periodic architectured materials by group theory (3b, after Azulay & Combescure 2024): symmetry groups of cells of several primitive cells restricted to the loading, real irreducible representations computed from the regular representation, the axial isotropy subgroups of the equivariant branching lemma, the patterns that minimize the elastic energy in their fixed-point and isotypic subspaces, the first bifurcation of a loading by linear buckling in each isotypic subspace; periodic plane frames of Euler-Bernoulli beams (honeycomb, triangular, square, kagome) and frames of struts in space (cubic, body-centred cubic, octet, Kelvin; 3D Euler-Bernoulli beams of circular section, torsion, the O_h group with the centring translations, their apparent stiffness against the scaling laws of lattice materials), and their Bloch-wave stability, which selects the cell; Schoenflies symbols of the point groups in space; continua (tetrahedral meshes and voxel images): the initial-stress matrix of the periodic prestress, the symmetries detected on the mesh, mechanisms left out; along nonlinear loading paths in finite strains with nonlinear behaviours (Hencky elasticity, von Mises elasto-(visco)plasticity, crystals, MFront laws): the consistent tangent of each converged step block-diagonalized in the isotypic subspaces of the group restricted to the path, the macroscopic components of prescribed stress condensed, the first loss of positive definiteness of a block (time, irrep, multiplicity, softest patterns) and the switch to the bifurcated branch by an equivariant imperfection; checked against the linear buckling of slender columns, the pattern transformation of a square array of holes (alternating ellipses, plateau, negative Poisson's ratio) and an elastoplastic column; conforming mirrored meshes of level sets, whose symmetries are exact, with the diagonal mirrors of square and cubic cells (equibiaxial compression of square arrays of holes, cubic cells and their three-dimensional irreps) (Bertoldi's specimen 1 within the extrapolation of 2.3-2.6 % against 2.25 % in uniaxial strain); cells of several primitive cells split first by their translations (the wave vectors Γ, X, M and R and their stars) and then by the little co-groups of the wave vectors (Clifford's theory), without dense bases of the whole cell, and the wave vectors of each irrep (the patterns that double the period) | S | 3b |
| FR-H27 | The material files of Z-mat (Z-set) read and translated (3b): gen_evp behaviours (elasticity, thermal strain, potentials with their criterion, flow, isotropic and kinematic hardening) on morphoGen's kernels when they are von Mises elasto-(visco)plasticity, otherwise as MFront behaviours of the brick StandardElastoViscoPlasticity (Hill's criterion, user-defined viscoplastic flows, power-law hardening, softening, Burlet-Cailletaud, several potentials); linear elastic behaviours of any symmetry; in configurations (`zmat`) and on the command line (`morphogen zmat`); a table of the coverage of Z-mat's library (docs/zmat.md); checked against MFront and Hill's analytical yield stresses; the elastic, von Mises and porous materials of morphoGen written as material files of Z-mat (gen_evp, porous_plastic), read back as the same (a behaviour identified on a morphology goes back to Z-set; in the interface, the behaviours of the workspace) | S | 3b |
| FR-H28 | Cohesive zone elements (3b: the interfaces of composites and fracture): bilinear mixed-mode (Benzeggagh-Kenane) and exponential (Ortiz-Pandolfi) traction-separation laws, irreversible, with contact in compression; zero-thickness interface elements between linear tetrahedra, in small and finite strains (the frame of the current mid-surface; a neo-Hookean bulk); interfaces inserted between phases or along chosen faces by duplicating nodes (crack fronts kept closed); a quasi-static Newton solver with step divisions and viscous regularization; the length of the cohesive zone, the strength of coarse meshes and the double cantilever beam against the corrected beam theory (Linke & Lammering 2023) | S | 3b |
| FR-H29 | Crack propagation in 3D (3b: by remeshing as Z-cracks, and by XFEM): planar cracks described by two level sets (straight and circular fronts); conforming meshes (MMG through the level set of the plane, or faces on it) with the crack faces opened by duplicated nodes and the body remeshed at each step; XFEM on linear tetrahedra with jump and branch enrichments and the integration of cut elements on sub-tetrahedra; the energy release rate by the compliance of the uniform advance of the front; propagation by Paris' law or Griffith's criterion; checked against the handbook factors of an edge crack (small strains, mode I) | S | 3b |
| FR-H30 | Mixed-mode cracks in 3D, in small and finite strains (3b): the energy release rate along the front by the G-θ method (Eshelby's energy-momentum tensor of the reference configuration, exact for a neo-Hookean body; its mean or its profile at stations along the front); the stress intensity factors of modes I, II and III by interaction integrals (the plane-strain and antiplane fields of Williams; the mean rotation removed; the work of the tractions of lips in contact; the curvature terms of curved fronts), turned back by the polar rotation in finite strains (objective); frictionless contact of the lips; dead loads; kinked cracks (polylines extruded along a straight front) remeshed by MMG with their front on edges, kinking by the largest hoop stress (Erdogan & Sih) and fatigue propagation by Paris' law; checked against the handbook (mode I), the exact antiplane strip (mode III), a penny crack in shear (Kassir & Sih: the three modes along a curved front), the energy of the body (each mode, finite strains, kinked cracks), the superposition of modes, a superposed rotation and the local symmetry of kinked paths | S | 3b |
| FR-H31 | Porous elasto-(visco)plasticity, the `porous_plastic` behaviours of Z-mat beyond `gen_evp` (3b): the matrix stress of a porous potential (Gurson-Tvergaard-Needleman with the effective porosity of a table and Tvergaard and Needleman's coalescence, Rousselier and its modified criterion, the elliptic criterion of Abouaf's compaction with coefficients tabulated in the porosity), the Norton or rate-independent flow of the matrix and its isotropic hardening, normality and the equivalence of the plastic works, the growth of the porosity by the mass of the matrix and Chu and Needleman's nucleation; implicit integration along the invariants (Aravas) with a consistent tangent, in small and finite (logarithmic) strains, at every point where von Mises points work on the CPU (material points, finite elements, FFT; the porosity in the curves and fields); Z-mat files translated natively (tables, `fs = f`, `*initialize_variable`); checked against the hydrostatic yield of GTN, the von Mises limit, the closed form of nucleation, the yield of compaction along the densification, finite differences of the tangents and a superposed rotation | S | 3b |
| FR-H32 | Micromorphic continua (3b: the microfoam model of Forest's article, and nonlocal models to crack with, in finite strains): mixed tetrahedra (quadratic displacements, linear generalized degrees of freedom) with a Newton solver (optionally from the increment that the last tangent predicts); the microfoam model of Dillard, Forest & Ienny (2006: the anisotropic compressible plasticity of nickel foams with the article's parameters, a full micromorphic microdeformation) in small and finite strains (total Lagrangian: the plasticity in the logarithmic strains, the objective measures `χ⁻¹ F` and `χ⁻¹ Grad χ` of Forest & Sievert); a micromorphic damage model (the implicit gradient model of Peerlings et al. as a scalar micromorphic variable, small and finite strains) propagating cracks as bands of damage; micromorphic ductile damage (von Mises elasto-(visco)plasticity whose damage follows the micromorphic cumulated plastic strain, after Engelen, Geers & Baaijens; small and finite strains) cracking as bands of ductile damage; checked against the article's plastic contraction ratios and hole size effect, the one-dimensional characteristic lengths of its Appendix A, softening and ductile bands independent of the mesh, finite differences, a superposed rotation and the small-strain limit; the densification of the nickel foam compressed to half its height; the damage driven by the plastic strain weighted by the stress triaxiality (Rice & Tracey, Johnson & Cook, Lemaitre), so that notches crack in mode I | S | 3b |
| FR-H33 | Cracks in inelastic solids, in small and finite strains (3b): the behaviours of morphoGen's points as the bulk of the crack solvers (porous plasticity GTN, Rousselier, elliptic; von Mises elasto-viscoplasticity with Hencky's logarithmic strains in finite strains, crystal plasticity with the orientations of the grains, MFront laws; internal variables committed at the converged steps, consistent tangents; with cohesive zones, the contact of the lips and dead loads); the works of the loads, the bulk and the interfaces; the driving forces of dissipative solids: the J of the deformation theory (the work density), the configurational force of the free energy and the driving force at the tip with the configurational forces of the internal variables; checked against the elastic bulks (points that never yield), the release of the work of the loads (Begley & Landes) and the independence of the tip value from the domain; the R-curve of a cohesive crack in an elasto-viscoplastic plate and the intergranular cracking of a polycrystal in crystal plasticity; remeshed propagation in inelastic bodies: the state of the points transferred between the meshes of successive cracks (the nearest element, or a projection on the nodes; the lips apart; the `Fp⁻¹` of crystals isochoric again; the dissipated part of the work carried, the free energy that of the new state), its equilibrium regained at the load of the transfer, dissipating nothing; ductile growth by the driving force at the tip or the porosity of GTN points near it, the resistance curve; checked by the identity of the transfer onto the same mesh, the dissipation kept in small and finite strains, onto a refined mesh and for crystals, the elastic limit of the propagation in elastic bodies and the resistance curves of an elasto-viscoplastic plate in small and finite strains by both transfers; remeshed polycrystals: crack meshes carrying the grains of a tessellation and their orientations, the state transferred grain by grain, kinking by the isotropic Voigt-Reuss-Hill estimates of a crystal; cracks through any morphology (2026-10-26): crack meshes carrying its phases and their materials, the sides of the crack apart, as staircases or conforming to its level set, its void phases removed; propagation, kinking and ductile growth through it, the fronts running through the pores they reach, the state transferred material by material; checked by equal materials (the homogeneous plate), the kinking towards a pore (Kirsch's estimate) and a soft cylinder and away from a stiff one at three mesh sizes, the transfer of a state set per phase; the toughness of the phases and of their interfaces: each advance the candidate of the largest G(θ)/Γ(θ), the kink of the bulk or a path along an interface (first-order kinked factors of Cotterell & Rice), checked by the deflection of He & Hutchinson (below a quarter of the toughness of the phases), a tough inclusion raising the load to enter it and the staircase paths of layers and brick-and-mortar structures with weak boundaries | S | 3b |
| FR-H34 | Identification of the parameters of a behaviour on a reference response, as Zopt for Zsim (3b): the elastic constants and the parameters of von Mises plasticity (yield stress, linear, isotropic and kinematic hardenings, Norton's flow) or of porous plasticity (its matrix, the initial porosity, Gurson's q1 and q2, the coalescence, Rousselier's and the elliptic coefficients, the nucleation: the Gurson model of a porous morphology under several triaxialities) such that the material point under the path of the reference gives its stresses over time in the least-squares sense (the parameters positive); the reference the macroscopic response of a loading path of a morphology, a material point or the points of a test; several references followed together, each by its weight whatever its number of points (a tension and a cycle in shear tell the isotropic hardening from the kinematic one); in the interface, the behaviour identified kept in the workspace | S | 3b |
| FR-H35 | Finite elements on conforming meshes as the default of the homogenizations (`method: auto`), from the exact geometry of the model wherever it is known; the FFT solver on voxels only for the features that only it has (GPU, coupled, harmonic and flow properties, phase-field fracture, NTFA, window method, planar sections) or without a conforming mesher | M | 3c |
| FR-H36 | Unidirectional fibre composites in closed form: the composite cylinder assemblage, the generalized self-consistent transverse shear, Hill's universal relations, the conductivities and Levin's expansions; the tows of woven textiles from their constituents | S | 3d |
| FR-H37 | Thermo-chemo-elastic couplings in time: temperature (heat capacity, conduction, heat of a reaction), a diffusing species (solubility, consumption, Arrhenius' laws, drift by the hydrostatic stress) and the quasi-static displacement under thermal and chemical expansions, on voxel images and conforming meshes, the fields at each output time | S | 3d |
| FR-H38 | Finite elements of the plane on its triangular meshes: generalized plane strain (the anti-plane fluctuation free, the stiffness of the structure in space) and plane strain, conduction, periodic, kinematic and static conditions on the sides, the macroscopic loadings of the solvers in space; the same tensors as the extruded layer in space | S | 3d |

### 5.9 Cellular automata and lattice growth

| Id | Requirement | Prio. | Phase |
|---|---|---|---|
| FR-A1 | Binary cellular automata on 2D and 3D lattices: von Neumann, Moore, edge and Euclidean neighbourhoods of any radius; periodic or fixed boundaries; Life-like B/S rules with ranges, majority, annealing (Vichniac), majority at conserved volume fraction, probabilistic and activator–inhibitor (Young) rules | M | 3a |
| FR-A2 | Synchronous, asynchronous and sequential updates; histories of the volume fraction, changes and interface density | M | 3a |
| FR-A3 | Reaction–diffusion (Gray–Scott) with pattern presets | S | 3a |
| FR-A4 | Nucleation and growth with Johnson–Mehl–Avrami–Kolmogorov kinetics: site saturation and continuous nucleation, isotropic or faceted oriented growth, partial transformations; Avrami theory (exponent, transformed fraction, grain density) | M | 3a |
| FR-A5 | Normal grain growth (Monte Carlo Potts model), reproducible in parallel | S | 3a |
| FR-A6 | Aggregation: diffusion-limited, Eden and ballistic, clusters and deposits, with sticking probability; fractal dimension (theory and box counting) | S | 3a |
| FR-A7 | Lattice microstructures usable like the other models: any resolution, level set for conforming meshes, grain labels and orientations, JSON storage, unbiased non-periodic windows | M | 3a |
| FR-A8 | Formation histories (frames) of the automaton and growth models | S | 3a |

### 5.10 Combinatorial design

| Id | Requirement | Prio. | Phase |
|---|---|---|---|
| FR-D1 | Tile sets with labelled faces (equal or opposite matching), rotations, voxel geometry of assemblies; polarized blocks of combinatorial metamaterials (after Coulais et al. 2016) and pipe networks with controlled node degrees | M | 3a |
| FR-D2 | Compatible assemblies by wave function collapse with backtracking, with boundary labels and per-cell restrictions; proof of infeasibility when the search is exhausted | M | 3a |
| FR-D3 | Exact counting and exact uniform sampling of compatible assemblies (transfer matrices); exact count of polarized-block assemblies for larger boxes | S | 3a |
| FR-D4 | Strut lattices with defects (random dilution to a mean coordination, node perturbation) and the rigidity of their pin-jointed frameworks (mechanisms, self-stresses, floppy macroscopic strains; Maxwell–Calladine) | M | 3a |
| FR-D5 | Generic search algorithms (simulated annealing, genetic algorithm) and the inverse design of automaton rules from target descriptors or an image (`morphogen fit-rule`) | S | 3a |

### 5.11 Topology optimization

| Id | Requirement | Prio. | Phase |
|---|---|---|---|
| FR-O1 | Homogenized stiffness and conductivity tensors of periodic voxel cells with graded moduli, and their exact sensitivities, with the FFT solver | M | 3a |
| FR-O2 | Objectives: bulk and shear moduli, Poisson's ratio (auxetics), conductivity, target tensor; 3D and planar (prismatic) cells | M | 3a |
| FR-O3 | Density filter, Heaviside projection with continuation, orthotropic and cubic symmetry | M | 3a |
| FR-O4 | Methods: optimality criteria, method of moving asymptotes, BESO, hybrid cellular automaton | M | 3a |
| FR-O5 | Designs exported as lattice microstructures (level set from the densities) for meshing and finite element validation; configuration files and `morphogen optimize` | M | 3a |
| FR-O6 | Validation against exact solutions, finite differences and Hashin–Shtrikman bounds; black-and-white designs checked for hinges (voxels touching only along an edge or at a corner) and re-evaluated on a finer grid | M | 3a |
| FR-O7 | Inverse design of active structures: layouts of two materials (and voids) that expand with a stimulus, for target displacements at equilibrium or along a diffusion in time, with exact adjoint sensitivities verified against finite differences | S | 3d |

### 5.12 Results database and Ashby charts

| Id | Requirement | Prio. | Phase |
|---|---|---|---|
| FR-P1 | A database (one file, no server) of every configuration tested and of its results per realization, filled by runs or afterwards, exported to CSV | M | 2c |
| FR-P2 | Selection of records by conditions on any configuration key or result (comparisons, lists, patterns) | M | 2c |
| FR-P3 | Normalized Ashby charts (relative property against relative density, logarithmic axes, scaling laws of cellular solids), one family per value of a key, accessible to colour-blind readers, light and dark themes; from Python, the command line and the graphical interface | M | 2c |

### 5.13 Laser-architectured sheets (§3.7)

| ID | Requirement | Priority |
|---|---|---|
| FR-L1 | Thermal cycles of laser tracks on a sheet: moving Gaussian or top-hat beams on one face or on both faces, finite thickness, surface losses; closed forms checked; transient finite volumes of a blank | M |
| FR-L2 | Metallurgical response: transformation hardening and tempering of C–Mn steels, phenomenological hardening of ferritic stainless steels, melt-pool carburizing, nitriding and alloying | M |
| FR-L3 | Mechanical properties of each treated point (flow laws fitted to tensile properties, strength against hardness) and presets of the steels of the PIMM studies | M |
| FR-L4 | Laser paths: parametric patterns (straight, corrugated, Fourier, crossed, honeycomb) and polylines, their process measures and G-code | M |
| FR-L5 | Tensile test of the architectured specimen by finite elements (plane-stress membrane with large rotations, layers through the thickness), Considère's necking | M |
| FR-L6 | Optimization of the laser path for an objective under constraints (grid, Nelder–Mead, differential evolution, genetic, Bayesian), configuration files and the command `morphogen laser` | M |
| FR-L7 | Fracture of brittle points (critical plastic strain, local or averaged over a radius) in the tensile test, the failure strain among the measures; the phases and samples of Wang et al. (2023) reproduced | M |
| FR-L8 | Cellular design of the treated sheet: phases on a grid of cells, their sensitivities by replays of the recorded deformation history, cellular automata (BESO, neighbour exchanges, HCA) and density-based topology optimization under a treatment budget, symmetries, periods and a minimum width | M |
| FR-L9 | Designs of cells turned into laser tracks and treated by the process model | S |
| FR-L10 | Structures of a treated sheet beyond the tensile test: a compact tension specimen (the initiation of a crack, its blunting or shielding by the tracks) and an L-shaped plate (the limit load), with cellular designs over any region | M |
| FR-L11 | The plastic anisotropy of a treated sheet: Lankford's coefficients and Hill's 1948 criterion from tests at 0°, 45° and 90°, isotropy objectives | M |
| FR-L12 | Alternative materials: boron steel hardened, age-hardened aluminium alloys softened by the laser | S |
| FR-L13 | The compensation of the rolling texture of a sheet by the treatment: Hill's 1948 criterion of the textured sheet, the anisotropy of the sheet treated all over on its periodic cell | M |
| FR-L14 | Yld2000-2d for a rolled sheet whose yield stresses and Lankford coefficients both vary; the growth of a crack (CT) through the treated sheet, guided or bridged by the tracks; the anisotropy set by an architecture | M |

## 6. Non-functional requirements

* **NFR-1 Portability.** Windows 10/11 x86-64, macOS ≥ 12 on Apple Silicon, Linux
  x86-64 (and aarch64), with Python 3.12–3.14. Installation with pip, uv or conda from
  binary wheels only, without compilers. Intel Macs are not supported by current Numba
  releases.
* **NFR-2 Performance (targets, laptop with 4–8 cores).** 256³ voxels take seconds to a
  few tens of seconds for every phase-1 model at typical densities. 512³ voxels
  (134 M; 0.13 GB per uint8 field) fit in 16 GB of RAM. Measured figures are in the
  README.
* **NFR-3 Memory.** uint8 phases, int32 labels, float32 fields; no per-voxel coordinate
  arrays; large tessellations processed in slabs.
* **NFR-4 Correctness.** Each model is validated against its theory, with statistical
  tests whose tolerances come from that theory. Each export is validated by a round
  trip, and by an independent reader (VTK, Gmsh) when one is installed.
* **NFR-5 Reproducibility.** The same inputs and seed give the same output on a given
  platform and dependency set. Results files record the configuration, seeds and
  versions.
* **NFR-6 Licences.** morphoGen is distributed under the GNU GPL, version 3 or later.
  No commercial dependency. Required dependencies use permissive licences (BSD, MIT),
  compatible with the GPL. Copyleft tools (gmsh, MMG, Neper) remain optional external
  programs; their licences are compatible with GPL-3.0-or-later. MMG is installed from
  the `pymmg` wheel and run as a separate program.
* **NFR-7 Quality.** Tests (pytest) run in continuous integration on the three systems
  for every pull request (locally with `tools/ci_local.sh` when GitHub Actions is not
  available, D18), in parallel (pytest-xdist), the slowest under one Python version only;
  linting and formatting use ruff. Changes reach `main` through pull requests.
* **NFR-8 Documentation.** English throughout: README (installation, usage, examples),
  this specification, architecture and theory notes, docstrings, change log, and a
  documentation site built from them (Sphinx: the pages of `docs/`, the README in parts
  and the API reference from the docstrings), without warnings. The tutorials, the
  recipes, the landing page and the guide to the physics are written in snippets that
  the tests run; the figures of the pages are written by scripts, not by hand (§3.8).
* **NFR-10 Stability.** From version 1.0, semantic versioning of the public interface
  (`docs/stability.md`): the names of the `__all__` of the packages and of the modules
  of the API reference, the commands, the configuration keys and the files written;
  an incompatible change warns for a minor version before a major one. A test keeps
  the list of the public names and parameters (`tests/public_api.txt`).
* **NFR-9 Usability.** Explicit error messages, sensible defaults, an example
  configuration for each model, and ASCII-only console output (Windows consoles).

## 7. Technical choices

### 7.1 Language: Python with compiled kernels (Numba)

Python is slow only for loops executed by the interpreter. NumPy and SciPy run compiled
C/C++/Fortran code, and the remaining loops are compiled to multithreaded machine code
by Numba (LLVM), at speeds comparable to C. The heart of the code stays in a language
that the group reads and extends.

A Rust, C++ or Fortran core would require either building and distributing binaries for
each system, or a compiler on every machine. It would also put the core out of reach of
most group members. This cost is not justified at this stage.

The kernels live in `morphogen.kernels` with array-in, array-out signatures, so a later
port (Rust through PyO3, or GPU) would not change the rest of the code.

### 7.2 Dependencies

| Package | Use | Licence | Status |
|---|---|---|---|
| NumPy | arrays | BSD-3 | required |
| SciPy | k-d trees, FFT, image labelling, special functions, quadrature | BSD-3 | required |
| Numba (llvmlite) | compiled kernels | BSD-2 | required |
| PyYAML | configuration files | MIT | required |
| scikit-image | marching cubes (surfaces) | BSD-3 | optional `[surface]` |
| tifffile | TIFF stacks | BSD-3 | optional `[io]` |
| meshio | other mesh formats | MIT | optional `[io]` |
| matplotlib (with Pillow) | plots of RVE studies, figures and animations (`morphogen view`) | PSF-based, HPND (BSD-compatible) | optional `[plot]` |
| pymmg | the `mmg3d` executable (conforming meshes), wheels for Windows, macOS and Linux x86-64 | LGPL-3.0-or-later | optional `[mesh]` |
| CuPy | GPU backend of the FFT solvers (FR-B3) | MIT | optional `[gpu]` |
| CUDA runtime, NVRTC and cuFFT (NVIDIA `cuda-toolkit` wheels) | libraries loaded by CuPy | NVIDIA licence (proprietary, free of charge; D3) | optional `[gpu]` |
| dolfinx, dolfinx_mpc (FEniCSx), petsc4py | the FEniCSx solver of finite element problems (FR-H23) | LGPL-3.0-or-later (dolfinx), MIT (dolfinx_mpc), BSD-2 (PETSc) | optional, from conda-forge, Linux and WSL (D21) |
| MGIS (MFrontGenericInterfaceSupport) | MFront behaviours in the nonlinear solvers (FR-H21) | LGPL-3.0-or-later and CeCILL-C | optional, from conda-forge (not on PyPI; D20) |
| JAX, Exponax, PyTorch | arrays shared by DLPack, spectral phase-field solvers (`interop.arrays`, FR-E11) | Apache-2.0 (JAX), MIT (Exponax), BSD-3 (PyTorch) | optional, imported when used |
| Microgen (3MAH), PyVista | the shapes of Microgen and their meshes (`models.microgen`, `interop.microgen`, FR-E12) | GPL-3.0 (Microgen), MIT (PyVista) | optional, imported when used |
| bpy (Blender as a module) | Blender's rigid bodies without the program (`interop.blender`, FR-E13) | GPL-3.0 | optional, imported when used |
| pytest, ruff | tests, linting | MIT | development |
| scikit-fem | independent finite element reference in the tests | BSD-3 | development |

The finite element and FFT solvers, the algebraic multigrid preconditioner, the
topology optimization (including the method of moving asymptotes) and the graphical
interface (a local web server of the standard library, PNG images written without
dependencies) are part of morphoGen (NumPy, SciPy, Numba). pyamg was evaluated: it has no wheels for
Python 3.14, and the internal multigrid is as fast on these problems.

External programs, never imported or linked: gmsh (GPL-2.0-or-later), MMG
(LGPL-3.0-or-later, run as a program), Neper (GPL-3.0), ParaView (BSD-3), FEniCSx
(LGPL-3.0), CalculiX (GPL-2.0), FFT solvers (AMITEX_FFTP, DAMASK, in-house codes),
MFront (TFEL, GPL or CeCILL-A: run to compile behaviours, whose libraries MGIS loads),
Blender (GPL, run as `blender --background` to settle particles), the codes of the
discrete element method (LAMMPS, LIGGGHTS, YADE: they read the files of the particles).

### 7.3 Building on existing tools

| Tool | Assessment | Decision |
|---|---|---|
| VTK / vtkSim | VTK is excellent for visualization and file formats, and vtkSim proved the "vector framework then discretization" approach. Building on VTK would make the core heavy, though, and its implicit-function machinery is slow for 10⁵ primitives. | Same approach, own kernels; VTK file formats written natively (no dependency); VTK/ParaView for visualization. |
| PuMA (NASA) | Strong on property solvers for porous media. Its generators are voxel-based and not built on stochastic-geometry models with known theory. Licence NOSA 1.3; installation through conda. | Do not build on it; interoperability through TIFF/NumPy volumes. |
| PoreSpy | MIT; voxel generators and metrics for porous media. | Possible complementary analysis of morphoGen images. |
| gmsh | Conforming meshing; OpenCASCADE booleans struggle with dense random fibre networks, and the classification of discrete surfaces took minutes for fibre networks in phase-2 trials; GPL. | `.geo` export (phase 1); conforming meshes with MMG (D5). |
| MMG | Level-set discretization and remeshing in one program, adaptive to the curvature of the interface, robust on dense fibre networks, wheels for the three systems (pymmg); LGPL. | Conforming meshes (phase 2). |
| Z-set | The FE code of the earlier workflow; not free software. | Not used; morphoGen solves the same problems (§3.1) and exports meshes in open formats. |
| Neper | Reference for polycrystal generation and meshing (GPL). | Interoperability through germs and weights (phase 2). |

### 7.4 GPU (phase 3b)

Decided (D3): CuPy, for NVIDIA GPUs. With `device: gpu` (or `auto`), the FFT solvers of
voxel images, linear and along loading paths, run their conjugate gradients on the GPU in
double precision: the stiffness and tangent operators of the voxels are CUDA kernels
applied without assembly, and the Fourier preconditioner uses cuFFT. Along loading paths
the integration points stay on the GPU: their measures, the integration of their
behaviours (CUDA ports of the Numba kernels), the conjugate stresses of B-bar and F-bar
and the nodal forces. Measured on a laptop (Intel i9-13900H; NVIDIA RTX 3000
Ada, 8 GB): linear solves 5 to 7 times faster from 64³ to 192³ voxels, 3.5 times at
256³; crystal plasticity paths 1.6 to 2.8 times faster from 24³ to 48³ voxels (1.4 to 1.9
before the faster crystal kernel of 2026-10-29), the integration of the crystals being
bound by the double precision of this GPU (1/64 of its single precision; theory notes,
§35); porous paths 2.4 to 3.5 times faster from 32³ to 64³ voxels. Finite elements, Macs and machines without an
NVIDIA GPU keep the CPU kernels.

### 7.5 Conventions

* Coordinates are in a unit chosen by the user (stored as metadata).
* Arrays are indexed `[i, j, k]` along (x, y, z), and voxel (i, j, k) is centred at
  `origin + (i + ½, j + ½, k + ½) h`.
* Files are written with x varying fastest.
* Phase 0 is the matrix (pores, void) and phase 1 the inclusions, fibres or solid.
* Tessellation labels are the 0-based germ indices.

Details are in [architecture.md](architecture.md).

## 8. Architecture (summary)

```
Model (parameters, theory) --generate(domain, seed)--> Microstructure (vector geometry)
    |                                                    | voxelize / level_set / contains
    v                                                    v
config files, CLI, GUI (web)                 VoxelImage --> analysis (measures)
                                                         --> homogenization: FFT (periodic)
                                                         --> mesh (voxel hexahedra, STL)
                                                         --> io (VTK, MHD, NPZ, TIFF...)
                                             level set --> conforming mesh (MMG) --> TetMesh
                         tessellation --> exact cells --> polycrystal mesh (MMG) --> TetMesh
                                                         --> homogenization: finite elements
                                                         --> io.solvers (equations, DAMASK,
                                                             Neper, FEniCSx)
pipeline: batches and RVE studies (ensemble / windows) --> rve (statistics, fits, sizes)
parallel: realizations in processes, tasks, MPI; job scripts for the schedulers

automata (rules, growth) --> lattice models --> LatticeField (a Microstructure)
design (tiles, frameworks, search) --> tile_assembly, random_lattice models
optimization (graded cells, FFT) --> CellDesign --> LatticeField
viz (figures, PNG), gui (local web server)
```

Packages: `models`, `kernels`, `analysis`, `rve`, `mesh`, `homogenization`, `io`,
`config`, `pipeline`, `cli`, in phase 3a `automata`, `design`, `optimization`, `viz`,
`gui`, and in phase 2b `parallel`. See [architecture.md](architecture.md).

## 9. Validation and tests (summary)

* Unit tests compare the kernels with brute-force references.
* Statistical tests compare each model with its closed-form results. Seeds are fixed and
  tolerances are a few standard deviations derived from the theory: volume fraction,
  covariance, specific surface, integral range, Voronoi cell statistics, Rice formula for
  Gaussian fields, closed-form lattice densities, TPMS symmetries.
* Published values are reproduced: RVE sizes of Dirrenberger et al. (2014, Table 10) and
  Yang et al. (2019, Tables 4–5).
* Every file format is round-tripped, and checked with VTK and Gmsh when they are
  installed.
* Homogenization: exact homogeneous media and laminates (both solvers, every boundary
  condition, 4- and 10-node elements), agreement with scikit-fem, the ordering
  SUBC ≤ PBC ≤ KUBC and MBC between SUBC and KUBC, bounds, iterative against direct
  solvers, Hill's lemma; conforming meshes with matching periodic faces.
* Automata and growth: brute-force neighbour sums, the glider and blinker of Life,
  conservation of the volume fraction, the Avrami law and Johnson–Mehl grain densities,
  exact growth distances, fractal dimensions of DLA (1.71 in 2D, 2.5 in 3D).
* Design: brute force, transfer matrices and the gauge formula of polarized blocks
  agree; uniform sampling (chi-square); rigidity of cubic, bcc and octet lattices and
  Maxwell's transition under dilution.
* Optimization: exact laminates, finite-difference sensitivities, MMA against SLSQP,
  designs within the Hashin–Shtrikman bounds and close to them, without hinges and
  confirmed on finer grids, auxetic cells.
* Polycrystals (2b): exact cells against Voronoi volumes and lattices; regularized
  complexes stay closed and valid; meshes reproduce the cells (without regularization,
  to 10⁻⁶), keep every grain and match on periodic faces; the face meshing succeeds on
  hundreds of tessellations; the bulk modulus of cubic polycrystals, (C11 + 2 C12) / 3,
  is exact for every boundary condition, on meshes and voxels; crystal tensors are
  checked against rotations and the Hashin–Shtrikman bounds.
* Parallel runs (2b): the same results whether serial, in processes, split into tasks
  and merged, or under MPI; job scripts parse in a shell.
* Exports (2b): periodic equations satisfied by every affine displacement; the
  quaternions of DAMASK against its own conversion; Neper weights; XDMF round trips and
  chunked volumes equal to direct voxelizations.
* Measures (2b): chords of Boolean models against theory, Euler characteristics of
  known shapes, granulometries of slabs and balls, tortuosity of straight and diagonal
  paths.
* Nonlinear behaviours (2c): closed forms of the material point (hardening, creep);
  consistent tangents against finite differences; homogeneous elements equal to their
  material point and exact laminates; Schmid's law of single crystals along [001] and
  [111]; the polycrystal flow stress between the Sachs and Taylor estimates; in finite
  strains, Hencky's closed forms in tension and simple shear, objectivity, the small
  strain curve recovered in the logarithmic strain space, a uniaxial Cauchy stress in
  tension.
* Models (2b): set operations against their components and theory; grain-boundary
  distances against brute force; the renewal variance of laminates; the covariance of
  dead leaves; Kelvin cells (24 struts and 12 four-fold nodes per cell).
* Phase 2c: whole periodic grains fill the box and link node to node (sweep of 210
  tessellations); mixed loadings against direct solutions and compliances; thermal
  expansion against Levin's exact relation (10⁻⁹) and its bounds; engineering constants
  and anisotropy indices of known tensors; database filters and Ashby charts (light and
  dark, palette validated for colour-blind readers); Cox spheres against the intensity
  and fraction of the process; coatings and shells exact for unions of spheres; frames of
  regular octahedra; fractal sponges against (N/k³)^level and box counting; nacre and
  honeycomb fractions exact; rays of wood exact (their cells and share of the volume)
  and stiffening the radial direction more than the tangential one (FFT); Bouligand
  fibre families periodic and fractions exact; osteon phases against the dead-leaves
  fractions; foam edges against Meijering's constants; Murray's law at every branching;
  the out-of-plane modulus of prismatic walls, E₃ = φ E_s, by FFT. Nonlinear paths: the
  material point against the closed forms of uniaxial tension, creep and hardening, the
  consistent tangent against finite differences, a homogeneous volume element against
  its material point and a periodic laminate against its exact layer solution along
  cyclic, mixed and thermal paths; linear behaviours against the condensation. On
  voxels (FFT): the tangent of B-bar and F-bar against finite differences and its
  symmetry, homogeneous images against the material point, the linear FFT solution
  with full integration, an exact laminate, the convergence of a plastic composite
  under refinement with B-bar and F-bar (and the locking of full integration) towards
  the quadratic tetrahedra of the same geometry.
* RVE statistics of loading paths: the quantities of the paths equal to their curves at
  the ends of the steps, analysed as the apparent properties in the reports of the runs
  and by `morphogen fit`.
* GPU backend (3b): the linear FFT solutions of the GPU equal those of the CPU (10⁻¹⁰)
  with the same iterations; the tangent operators of the voxels (B-bar, F-bar, full
  integration; small and finite strains) and their coupling columns agree to round-off;
  crystal plasticity paths give the same curves; the behaviours of the points equal those
  of the Numba kernels (every behaviour, small and finite strains, substeps), and the
  paths of a viscoplastic composite their curves, phase averages and fields; the choice
  of the device and its errors in the API, the configurations and the command line.
  These tests are skipped where no GPU is usable.
* Models (3b): hard-core fibres never closer than their gap (brute force over the
  periodic images; the closest points of segments against a fine sampling) and their
  fraction; curved fibres: the tangent correlations and the mean square end-to-end
  distance of the discrete worm-like chain, the fraction of the Boolean model; rotated
  lattices along their turned axes and graded lattices with interpolated radii, their
  mean and local fractions against the theory.
* AMITEX workflows (3b): the formats of the inputs (ten-line VTK headers and big-endian
  integers, read back by VTK; binary zone files; the native laws and the order of their
  constants), the stiffness of every crystal voxel rebuilt from the exported axes of its
  zone against morphoGen's, the loadings and paths; the results (cell, materials, zones;
  small and finite strains; numbers written by Fortran without the letter of their
  exponent) in morphoGen's orders, the apparent stiffness of the unit strains. AMITEX
  itself does not run in the tests (it is not installed on the development machine).
* MFront behaviours (3b): the conventions of MGIS (TFEL's order 12, 13, 23 of the shear
  components, with their factors √2, against morphoGen's usual 23, 31, 12, checked on an
  orthotropic law whose three shear moduli differ; the energy; the order of the
  components in finite strains; tangents and rotations against morphoGen's matrices);
  MFront twins of morphoGen's behaviours (von Mises
  elasto-viscoplasticity in small strains and in the logarithmic strain space,
  Méric–Cailletaud crystal plasticity, isotropic elasticity with material properties)
  against morphoGen's kernels at a material point, by finite elements (periodic and
  kinematic conditions) and on voxels (B-bar, F-bar; a polycrystal), in small and finite
  strains, on the CPU and the GPU: the same curves to 10⁻⁹ (10⁻¹⁰ to 10⁻¹⁶ measured) with
  the same Newton iterations; the errors (kinematics, material properties, parameters,
  sources that do not compile) and the command line. These tests are skipped without
  MGIS and MFront; they run in WSL on the development machine.
* MFront exports (3b): the sources (the brick of von Mises plasticity, the slip systems of
  crystals in TFEL's components, the orthotropic constants in MFront's order), their
  internal variables and errors; compiled by MFront, the exported behaviours against
  morphoGen's kernels through MGIS (von Mises viscous, rate-independent and perfect, in
  small and finite strains; crystals with Franciosi's interactions, two families of
  their own parameters, bcc systems with isotropic elasticity, turned), to 10⁻⁸ (10⁻¹⁰
  to 10⁻¹³ measured); AMITEX exports of nonlinear materials: the coefficients and
  internal variables of the material file against MFront's example of Cast3M, and the
  laws called as AMITEX calls the umat interface against morphoGen's kernels, along a
  path of the six components and for each zone of a polycrystal.
* Abaqus inputs (3b): the 21 anisotropic constants in Abaqus's order (11, 22, 33, 12, 13,
  23); CalculiX under a homogeneous strain gives morphoGen's stresses in an isotropic
  grain and in a turned grain of 21 distinct constants; the MFront laws of an input,
  compiled with the abaqus interface, called as Abaqus calls them in the axes of the
  orientations of the sections, give morphoGen's curves. These tests are skipped without
  CalculiX and MFront; they run in WSL on the development machine.
* Voxel meshes, crystals and user laws in the exports (3b): the voxel meshes in Abaqus
  inputs (CalculiX, 8 integration points, a turned grain of 21 distinct constants);
  crystals in finite strains as MFront sources (morphoGen's kernel in them) against the
  kernel at a material point, with the same Newton iterations; user-supplied MFront laws
  in AMITEX and Abaqus inputs, their coefficients and internal variables against
  MFront's examples (glossary names merged by umat), called as the codes call them
  against morphoGen's integration through MGIS, their parameters read from the directory
  of the runs.
* Finite strains in the exports (3b): AMITEX exports (Small_Perturbations false, the nine
  components of the paths, every phase an MFront law, the functions of the strain
  measures of the umat interface) and Abaqus inputs (no orientation in the sections,
  the axes of the orthotropic laws in their first six state variables from *INITIAL
  CONDITIONS) of finite strains; their laws, compiled, called as AMITEX and Abaqus call
  them in finite strains (the deformation gradients, the Cauchy stress) against
  morphoGen's finite strains along a path whose principal axes turn, to 10⁻⁸ (10⁻¹⁵ to
  3·10⁻¹¹ measured: Hencky's elasticity, isotropic and cubic, von Mises with kinematic
  hardening, crystals); user-supplied laws of finite strains (Hencky's strain measure, the
  deformation gradient, an orthotropic law with three distinct shear moduli) against
  their integration through MGIS.
* Dead leaves and Torquato's approaches (3b): the proportions of the families of leaves,
  the probability of the same leaf (isotropic and aligned spheroids, along each axis),
  the background of a finite coverage and its centred covariance against the bivariate
  laws, their limits; packings grown beyond the RSA limit without overlap (all the
  pairs), jamming near 0.64; the two limits of the cherry-pit model and its bounds; the
  reconstruction of a Debye medium and of a Boolean model (its lineal path with and
  without its weight), from an image; the lineal path and the pore sizes of the Boolean
  model against their closed forms, the two-point cluster function, the radial
  distribution function and the structure factor of Poisson points, hard spheres and a
  lattice; the field of transparent dead leaves (mean, variance, covariance, the fraction
  of its excursion over realizations, opaque leaves as the dead leaves); stealthy
  hyperuniform germs (S(k) zero on the constrained wave vectors) and the regularity of
  their Voronoi cells.
* End-to-end tests cover the command line, split cluster runs and the interface
  server.
* Continuous integration runs on Linux, Windows and macOS (Apple Silicon) with
  Python 3.12 and 3.14.

## 10. Roadmap

| Phase | Version | Content | Acceptance |
|---|---|---|---|
| 1 | 0.1 | Nine models (FR-G1–G8, G10–G11); vector, implicit and voxel representations; measurements; RVE statistics (ensemble and window methods, external tables); exports FR-E1–E4; CLI and configuration files; cluster task splitting; test suite and CI | this specification validated; `pip install` works on the three systems; all tests pass in CI |
| 2a | 0.2 | Homogenization (FR-H1–H9): finite elements on conforming meshes, FFT solver on voxels; conforming tetrahedral meshes (FR-E5); statistical analysis after Dirrenberger et al. 2014 (FR-M4, FR-V6 bootstrap) | the boundary conditions and outputs of the earlier FE workflow (§3.1) reproduced with free tools; validation cases of FR-H6 |
| 3a | 0.3 | Requirements of §3.2: cellular automata and lattice growth (FR-A1–A8), combinatorial design (FR-D1–D5), topology optimization (FR-O1–O6), visualization and graphical interface (FR-I4, FR-I6) | validations of §9; the interface covers every model |
| 2b | 0.4 | Requirements of §3.3: conforming meshes and finite elements of polycrystals (FR-H10, FR-H11); parallel and cluster runs (FR-B5); XDMF, FEniCSx, Neper and DAMASK exports, periodic constraint equations (FR-E6); extra measures (FR-M3); model combinations (FR-G12); extra models (FR-G9); chunked large volumes (FR-R4) | validations of §9; the bulk modulus of cubic polycrystals exact on meshes and voxels; results independent of the parallel splitting |
| 2c | 0.5 | Requirements of §3.4: whole periodic grains (FR-H12), periodic element and mixed loadings (FR-H13), thermo-elasticity (FR-H14), post-processing (FR-H15), nonlinear behaviours along loading paths (FR-H16, FR-H17), crystal plasticity (FR-H18) and finite strains (FR-H19), the nonlinear FFT solver of voxel images (FR-H20) and RVE statistics of loading paths (FR-V7), database and Ashby charts (FR-P1–P3), nested and bio-inspired models (FR-G13, FR-G14) | validations of §9: Levin's relation, Meijering's constants, dead-leaves fractions, E₃ = φ E_s of prismatic cells, closed forms of the material point, exact laminates, Schmid's law, Hencky's closed forms; voxels without locking (B-bar, F-bar) against quadratic tetrahedra; templates run end to end |
| 3b | 0.6 | GPU backend (FR-B3); remaining models of FR-G9; AMITEX workflows; MFront behaviours (FR-H21) and sources (FR-E7); 3D interactive views in the interface | GPU speed-up measured (§7.4); MFront twins of morphoGen's behaviours |
| 3c | 0.7 | Requirements of §3.10: long fibres, woven textiles, nested grains, polydisperse sizes (FR-G19–G22); particles for the discrete element method, arrays and phase fields, Microgen, Blender and alloys (FR-E10–E14); the workspace and the fields in the page (FR-I9) | the validations of [theory.md](theory.md) §95–§103 |
| 3d | 0.8 | Requirements of §3.11: implicit geometry first and the exact measures of mathematical morphology, the identification of models from images and their segmentation, the reconstruction from descriptors, images and spinodal decomposition as models, grains grown from seeds (FR-M5–M8, FR-G25–G27); thermo-chemo-elastic couplings in time and the inverse design of active structures (FR-H37, FR-O7); woven textiles enriched and unidirectional tows (FR-G23–G24, FR-H36); Code_Aster, OpenFOAM and VIMSEO (FR-E15–E17) | the validations of [theory.md](theory.md) §104–§118 |
| 4 | 1.0 | Stable API; documentation site; PyPI/conda-forge release; software paper (e.g. JOSS); DOI (prepared on 2026-10-28, §3.6; the remaining steps in `docs/release.md`) | public release |

## 11. Decisions

* **D1 — Licence. Decided on 2026-10-21: GNU General Public License, version 3 or (at
  your option) any later version** (SPDX `GPL-3.0-or-later`; full text in `LICENSE`).
  CC-BY-NC had first been considered, which Creative Commons recommends against for
  software and which is not open source. The copyleft keeps morphoGen and its
  derivatives open source, and every dependency and external tool of §7.2 is compatible
  with it (the CUDA libraries of the optional GPU backend: see D3). **Confirmed on
  2026-11-17:** the author holds the rights, and the institution confirmed it (in France
  the economic rights to software written by employees in the course of their duties
  belong to the employer — Code de la propriété intellectuelle, art. L113-9, extended to
  people with research missions —, which is why the question had been left open). The
  copyright holder of the licence notices is the author. Points that remain:
  * Documentation and data may use CC-BY 4.0.
  * Until the public release (phase 4) the package keeps the `Private :: Do Not Upload`
    classifier, which makes PyPI refuse any accidental upload.
* **D2 — GUI technology. Decided in phase 3a:** a web interface served locally by the
  Python standard library (`morphogen gui`), with PNG images written without
  dependencies: nothing to install beyond morphoGen, the same on the three systems,
  usable from a cluster front end through an SSH tunnel. Figures and animations use
  matplotlib (`morphogen view`). Interactive 3D views (trame, PyVista) or a napari
  plugin remain possible later (phase 3b).
* **D3 — GPU backend. Decided in phase 3b (§3.5): CuPy**, for NVIDIA GPUs. The solvers
  need double precision, which the Metal backend of PyTorch (MPS) does not offer, so that
  PyTorch would not bring the GPUs of Apple Silicon either. CuPy follows the NumPy and
  SciPy API (FFTs, element-wise and reduction kernels, CUDA C kernels compiled at run
  time): the Numba kernels are ported one by one, and the CPU code stays the reference.
  It needs only the CUDA runtime, NVRTC and cuFFT (about 270 MB of wheels, no CUDA
  toolkit or compiler), where PyTorch brings a deep-learning framework of several
  gigabytes. The backend is optional (`[gpu]`) and chosen per problem (`device: cpu |
  gpu | auto`); the CPU kernels remain the default, and the only backend of finite
  elements, of Macs and of machines without an NVIDIA GPU. The CUDA libraries are
  proprietary (NVIDIA's licence, free of charge): morphoGen neither ships nor requires
  them; they are installed with the extra. Whether this optional use agrees with the
  GPL is to be confirmed before the public release. The behaviours
  of the loading paths are ported to CUDA C (`behaviours.cu`), line by line from the
  Numba kernels, which stay the reference, and tests compare the two point by point:
  numba-cuda, which would compile the Numba kernels themselves for the GPU, was tried,
  but its release 0.30.4 does not run with NumPy 2.5 (it uses `numpy.row_stack`, which
  NumPy removed).
* **D4 — FFT solver. Decided in phase 2:** an FFT solver in morphoGen, for voxel images
  under periodic conditions (elasticity and conduction). It uses the trilinear finite
  element discretization of the voxels with a Fourier preconditioner (Schneider,
  Merkert & Kabel 2017): no ringing artefacts, and pores without an ersatz material.
  External FFT codes remain usable through the voxel exports.
* **D5 — Conforming meshes. Decided in phase 2: MMG** (`mmg3d -ls`, installed by the
  `pymmg` wheel), after trials of gmsh. Opposite faces match for periodic media.
* **D6 — Statistical post-analysis code. Settled in phase 2:** the analysis is written
  from Dirrenberger et al. (2014) (FR-M4), no earlier code existing.
* **D7 — Validation files. Settled in phase 2:** the Z-set input files of the earlier
  workflow (KUBC, MBC and periodic conditions; bulk and shear loadings; averages, volume
  and second moments) give the cases to reproduce. As free codes are preferred (§3.1),
  morphoGen solves these problems itself (FR-H1–H3) rather than writing Z-set files.
* **D8 — Finite elements of tessellations. Decided in phase 2b (§3.3): conforming
  meshes of polycrystals.** morphoGen computes the exact cells of Voronoi and Laguerre
  tessellations (power diagrams clipped to the box, or on the torus), removes the
  features smaller than a chosen length (short edges, thin caps on the faces of the
  box), splits the cells into tetrahedra and lets MMG remesh them as subdomains
  (`mmg3d`, the grain boundaries kept); periodic faces are triangulated in the plane
  and remeshed by `mmg2d`, the same triangles on opposite faces. Neper is not needed
  (its files are written for those who use it). The crystals are anisotropic (FR-H11).
* **D9 — Topology optimization.** Periodic cells are optimized on voxels with the FFT
  solver (inverse homogenization, SIMP); the designs can then be meshed conformingly and
  checked by finite elements. Structural optimization of whole parts (compliance under
  loads and supports) is out of scope; to be confirmed.
* **D10 — Building blocks of combinatorial metamaterials.** The polarized blocks model
  the compatibility rules of Coulais et al. (2016) with a generic block geometry (a
  block pierced along its axis). Users may provide the geometry (or the deformation
  modes) of the blocks of interest; any block set with labelled faces can be used.

* **D11 — Parallel and cluster runs. Decided in phase 2b:** each realization writes its
  own record, so that runs can be split into any number of tasks, resumed and merged;
  processes rather than threads run the realizations of one machine (the kernels are
  multithreaded inside, within a thread budget); job scripts are written for the usual
  schedulers rather than depending on a workflow manager; MPI (mpi4py) is optional.
* **D12 — Large volumes. Decided in phase 2b:** raw binary files with XDMF (ParaView,
  VisIt) or MetaImage headers rather than HDF5 or Zarr, which would add a compiled
  dependency; slabs are written through memory maps.

* **D13 — Periodic element. Decided in phase 2c:** the macroscopic strain and stress
  components are the unknowns of mixed loadings, as in the periodic element of Z-set,
  but solved by condensation (§25 of the theory notes): the linear systems keep their
  form and solvers, the unit solutions are reused by every loading, and the results are
  those of the extra degrees of freedom. For nonlinear behaviours see D16.
* **D16 — Nonlinear periodic element. Decided (§3.4):** for
  nonlinear behaviours the macroscopic strain components are degrees of freedom of the
  Newton iterations, with their residuals ``int sigma dV - V Sigma``; each linearized
  system is condensed on them (a Schur complement of at most six unknowns, §31 of the
  theory notes). Linear problems keep the condensation of D13, which gives the same
  solutions. The behaviours are von Mises elasto-(visco)plasticity and crystal
  plasticity (FR-H18), in small or finite strains (D17); the FFT solver of voxel images
  solves the same iterations (FR-H20, D19).
* **D17 — Finite strains. Decided (§3.4):** a total Lagrangian
  formulation (first Piola–Kirchhoff stress, its tangent); elasticity and von Mises
  plasticity in the logarithmic strain space (Miehe, Apel & Lambrecht 2002), which
  reuses the small strain integration and reduces to it for small strains; crystal
  plasticity multiplicative, with the slip systems fixed in the intermediate
  configuration. The macroscopic gradient is upper triangular (the rotation removed by
  its QR decomposition): its six components keep the Voigt order and the meaning of the
  small strains (shears are simple shears), its conjugates are nominal stresses, and a
  tension with free faces is a uniaxial Cauchy stress whatever the anisotropy (§33 of
  the theory notes).
* **D18 — Continuous integration without GitHub Actions. Decided
  (§3.4):** while the Actions minutes of the account are exhausted, `tools/ci_local.sh`
  runs the jobs of the workflow for the system of the development machine (lint and
  format, the wheel installed in fresh environments for Python 3.12 and 3.14, the tests
  against it, the command-line smoke test) before each push: the Windows job on the
  development machine (Git Bash), the Linux or macOS jobs elsewhere. uv is optional (venv
  and pip otherwise). On the development machine the environments also get VTK and Gmsh
  (`CI_LOCAL_PACKAGES`), whose checks of the exported files the GitHub workflow skips.
  The jobs of the other systems run again when Actions is available. Since 2026-10-28
  (§3.6) the Python versions run at once, each on its share of the cores, their tests in
  parallel (pytest-xdist, one file per process, the longest files first), and the
  documentation site is built with its warnings as errors; an environment is kept while
  `pyproject.toml` and the extra packages do not change, for a week at most, and only
  the wheel is then reinstalled, the disk caches of Numba of the package cleared (they
  are keyed on the contents of each source file: a kernel whose file did not change kept
  the compiled code of the kernels it calls in other files). On the development laptop the tests of one Python take
  about 10 minutes on 10 processes instead of about 50 in one, and the whole CI 20
  minutes instead of 70 to 110. Since 2026-10-29 the slowest tests (marker `slow`, two
  thirds of the time) run under the last Python version only (`CI_LOCAL_SLOW`), which
  gets three times the cores of each other version; the GitHub workflow runs them in
  its job Ubuntu with Python 3.14, the fast tier elsewhere. Since the repository is
  public (2026-11-18), the GitHub workflow runs again at each push, beside the local CI.
* **D19 — Nonlinear FFT solver. Decided (§3.4):** the iterations
  of D16 on the voxels of the linear FFT solver (D4): Newton–Krylov, each correction
  solved by conjugate gradients with the Fourier preconditioner of a homogeneous elastic
  reference medium, the tangent applied matrix-free from the consistent tangents of the
  Gauss points (nothing assembled), rather than the fixed-point scheme of Moulinec and
  Suquet, whose iterations grow with the contrast of the tangents. Fully integrated
  trilinear voxels lock under isochoric plastic flow (the flow stress of elastic cubes
  in a plastic matrix twice too high on 10³ voxels): the volumetric part of the measures
  is averaged over each voxel by default (B-bar in small strains, F-bar in finite
  strains), in variational form so that the tangent stays symmetric for the conjugate
  gradients; `integration: full` keeps the discretization of the linear solver.
* **D21 — FEniCSx. Decided (§3.5):** FEniCSx solves
  morphoGen's own finite element problems (subclasses that replace the solution of the
  fields) rather than a separate set of problems: the constraints and every
  post-processing stay those of morphoGen, which makes the two solvers interchangeable
  and checks FEniCSx against the reference of the project. dolfinx and dolfinx_mpc are
  optional dependencies from conda-forge, on Linux or WSL only (no Windows builds); their
  forms are compiled at run time (a C compiler in the environment).
* **D20 — MFront behaviours. Decided (§3.5):** through MGIS and the
  `generic` interface of MFront, the library that TFEL provides for solvers, rather than
  a port of the generated C++: MGIS integrates the points of a material in batches, with
  its own threads, from states that morphoGen keeps in its own arrays, so that the
  solvers, their Newton iterations and their divided steps are unchanged; on the GPU the
  MFront points are integrated on the CPU. MGIS and MFront are optional dependencies from
  conda-forge (they are not on PyPI): without them a material with an `mfront` key raises
  an error that gives the command, and their tests are skipped. A material keeps its
  elastic constants (the linear problems, the reference medium of the FFT solver). The
  validation compares MFront twins of morphoGen's behaviours with its own kernels.
  The exports of morphoGen's laws (FR-E7) are MFront sources with the values of the
  material as constants, the slip systems written out by morphoGen rather than taken from
  MFront's crystallography, so that the same source serves every interface; for AMITEX,
  the umat interface, whose coefficients carry the axes of each grain. Crystal plasticity
  in finite strains is exported with morphoGen's own kernel in the source (the C port of
  the GPU, compiled for the host) rather than MFront's brick of finite strain single
  crystals, whose update of the elastic deformation differs from morphoGen's at the
  second order of the slips: the source gives morphoGen's integration and tangent.
  User-supplied MFront laws go to AMITEX and Abaqus inputs with their sources, compiled for
  the interface of each code. In finite strains, the exports take every phase but the
  voids as an MFront law (the native elasticities of AMITEX and Abaqus in finite strains
  are not Hencky's), and the interfaces of the codes handle the strain measures of the
  laws: the umat interface by functions of its own (`_malls` for Hencky's logarithmic
  strain, `_frst` for Green-Lagrange's), the abaqus interface with the axes of the
  orthotropic laws in their state variables (its only orthotropy in finite strains),
  which morphoGen's sources declare for Abaqus only.
* **D14 — Results database. Decided in phase 2c:** SQLite from Python's standard
  library, one file per study (or shared), records stored as JSON of flattened keys so
  that any configuration or result can be filtered without a fixed schema.
* **D15 — Nested and bio-inspired models.** Model parameters may be models (a mapping
  with a `type`), so that any model can fill or decorate another; the bio-inspired
  models are morphological (geometry and closed-form characteristics), their mechanics
  coming from the homogenization of their realizations. Other organisms or tissues can
  be added in the same way.

## 12. Glossary and notation

| Symbol | Meaning |
|---|---|
| p, q = 1 − p | volume fraction of phase 1, and of its complement |
| θ | intensity of a Poisson process (germs per unit volume; for Poisson lines in Jeulin's notation θ = L_V / π) |
| L_V | mean fibre length per unit volume |
| C(h) | covariance: P(x ∈ A, x + h ∈ A) |
| A₃, A₃* | integral range, generalized integral range |
| D_Z², D_Z²(V) | point variance of Z, variance of its mean over V |
| K, γ | coefficient and exponent of D_Z²(V) = K V^-γ |
| ε_rel, n | relative error on the mean, number of realizations |
| S_V | specific surface area (interface area per unit volume) |

## 13. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-10-21 | First version, for validation |
| 1.1 | 2026-10-21 | Validated; licence decided (D1): GPL-3.0-or-later |
| 1.2 | 2026-10-21 | Phase 2: requirements (§3.1); homogenization in scope (§4, §5.8, FR-H1–H10); FR-M4, FR-V6, FR-E5 updated; dependencies (pymmg, scikit-fem); decisions D4–D7 settled, D8 added; roadmap: phase 2 split into 2a (0.2) and 2b (0.3) |
| 1.3 | 2026-10-21 | Phase 3a (§3.2): cellular automata and growth (§5.9), combinatorial design (§5.10), topology optimization (§5.11), graphical interface and figures (FR-I4, FR-I6); decision D2 settled, D9 and D10 added; roadmap: 3a (0.3) before 2b (0.4) and 3b (0.5) |
| 1.4 | 2026-10-21 | Phase 2b (§3.3): D8 settled (conforming meshes of polycrystals, FR-H10 raised to M), anisotropic crystals (FR-H11), parallel and cluster runs (FR-B5), FR-E6, FR-M3, FR-R4, FR-G12 and part of FR-G9 delivered; decisions D11 and D12; scope (§4) and roadmap updated |
| 1.5 | 2026-10-22 | Phase 2c (§3.4): whole periodic grains (FR-H12), periodic element and mixed loadings (FR-H13), thermo-elasticity (FR-H14), post-processing (FR-H15), results database and Ashby charts (§5.12), nested and bio-inspired models (FR-G13, FR-G14); decisions D13–D15; roadmap: 2c (0.5), 3b moved to 0.6 |
| 1.6 | 2026-10-22 | Rays of wood (§3.4, FR-G14); nonlinear behaviours (FR-H16) and loading paths solved with the macroscopic strains as degrees of freedom of the periodic element (FR-H17); decision D16 |
| 1.7 | 2026-10-22 | Crystal plasticity of Méric and Cailletaud for polycrystals (§3.4, FR-H18), finite strains for every nonlinear behaviour (FR-H19); decisions D17 (finite strain formulation) and D18 (local continuous integration); D16 updated |
| 1.8 | 2026-10-22 | The nonlinear FFT solver of voxel images (§3.4, FR-H20), decision D19; D16 updated |
| 1.9 | 2026-10-23 | RVE statistics of the curves of loading paths (FR-V7), with the nonlinear FFT solver (§3.4) |
| 1.10 | 2026-10-23 | Phase 3b (§3.5): the GPU backend of the FFT solvers (FR-B3) and its benchmarks (§7.4); decision D3 settled (CuPy); optional dependencies CuPy and the CUDA libraries (§7.2); part of FR-G9 delivered: hard-core fibres, curved fibres, rotated and graded lattices |
| 1.11 | 2026-10-23 | The behaviours of the loading paths integrated on the GPU (§3.5, FR-B3), their benchmarks (§7.4); D3 completed (the CUDA port of the behaviours, numba-cuda tried) |
| 1.12 | 2026-10-23 | The AMITEX_FFTP workflows, exports and results (§3.5, FR-E6) |
| 1.13 | 2026-10-23 | MFront behaviours through MGIS in every nonlinear solver (§3.5, FR-H21), decision D20; MGIS and MFront as optional dependencies (§7.2); the scope (§4) updated for the nonlinear behaviours of phases 2c and 3b |
| 1.14 | 2026-10-23 | MFront sources of morphoGen's behaviours (§3.5, FR-E7), AMITEX runs of nonlinear materials through them (FR-E6); D20 completed |
| 1.15 | 2026-10-23 | The materials of Abaqus inputs, elastic and MFront laws (§3.5, FR-E6, FR-E7) |
| 1.16 | 2026-10-23 | Voxel meshes in Abaqus inputs, crystal plasticity in finite strains as MFront sources, user-supplied MFront laws in AMITEX and Abaqus inputs (§3.5, FR-E6, FR-E7); D20 completed |
| 1.17 | 2026-10-23 | Finite strains in the AMITEX and Abaqus exports, every phase an MFront law (Hencky's elasticity for the elastic ones), user-supplied laws of finite strains (§3.5, FR-E6, FR-E7); D20 completed |
| 1.18 | 2026-10-24 | Dead leaves of families, spheroids and finite coverage, dense packings by the Lubachevsky–Stillinger algorithm, the cherry-pit model (§3.5, FR-G9), the stochastic reconstruction (FR-G15), Torquato's descriptors (FR-M3) |
| 1.19 | 2026-10-24 | Transparent dead leaves, Voronoi tessellations of stealthy hyperuniform germs (§3.5, FR-G9) |
| 1.20 | 2026-10-24 | The measures of mathematical morphology of the appendix of Dirrenberger's thesis (2012) (§3.5): Choquet capacities, dilations, erosions, openings and closings by balls, segments, pairs and cubes, contact distributions, the morphological covariance of grey-level images, the integral of mean curvature and the Minkowski curves of excursion sets, Betti numbers and connectivity density (FR-M3) |
| 1.21 | 2026-10-24 | Curved fibres as in PuMA (§3.5): circular arcs of distributed radii of curvature, helices (crimp), and curved fibres without overlap by random sequential addition (FR-G9) |
| 1.22 | 2026-10-24 | Couplings of diffusion, chemistry and mechanics (§3.5): transient diffusion, thermal oxidation by the closed-loop scheme, ageing laws and graded phases, hydrolysis of cellulose, hygromorph multilayers (FR-H22) |
| 1.23 | 2026-10-24 | FEniCSx as the solver of finite element problems (§3.5, FR-H23, decision D21) |
| 1.24 | 2026-10-24 | Foams as level sets of distance fields, after the paper of N. Degdeg and J. Dirrenberger (§3.5, FR-G16) |
| 1.25 | 2026-10-24 | The nonuniform transformation field analysis (§3.5, FR-H24) |
| 1.26 | 2026-10-24 | The NTFA in configurations and the pipeline, with the mean fields of the phases (FR-H24); hollow fibres as in PuMA (FR-G9) |
| 1.27 | 2026-10-24 | Kinematic hardening in the NTFA (FR-H24); transient diffusion in configurations and the pipeline (FR-H22) |
| 1.28 | 2026-10-24 | Diffusion and free swelling of tetrahedral meshes with FEniCSx, bending of strips (FR-H22, FR-H23) |
| 1.29 | 2026-10-24 | The stresses of oxidized samples with FEniCSx, the method of Ernault et al. (2019) (FR-H22) |
| 1.30 | 2026-10-24 | Orthotropic hygromorph layers, layers from microstructures by homogenization (FR-H22); the mean intercept length and the MIL fabric tensor (FR-M3) |
| 1.31 | 2026-10-24 | The first minima of the covariance (hole effect), the second anisotropy measure of Degdeg & Dirrenberger (FR-M3) |
| 1.32 | 2026-10-24 | Stereology, the sections of the grid planes and the recovery of size distributions of spheres and oriented ellipsoids (Saltykov), as Degdeg & Dirrenberger (FR-M3) |
| 1.33 | 2026-10-24 | The seeds of level-set foams calibrated on the mean radii of the sections of their pores (FR-G16) |
| 1.34 | 2026-10-24 | Bloch waves, the band structures and band gaps of periodic cells (FR-H25) |
| 1.35 | 2026-10-24 | Post-bifurcated patterns by group theory in the manner of R. Azulay, and periodic frames of beams (FR-H26) |
| 1.36 | 2026-10-24 | The post-bifurcation analysis extended to continuum meshes: tetrahedral meshes and voxel images, the initial-stress matrix of the periodic prestress, the symmetries detected on the mesh (FR-H26) |
| 1.37 | 2026-10-25 | Raw data and figure exports, TikZ, videos, one export entry point (FR-E8, FR-I7) |
| 1.38 | 2026-10-25 | Every module connected in one workflow (model, image, mesh, fields, exports, figures and videos; FR-I8), `morphogen export` |
| 1.39 | 2026-10-25 | Post-bifurcation of frames of struts in space, Schoenflies symbols (FR-H26) |
| 1.40 | 2026-10-25 | Z-mat material files (the behaviour bricks of Z-set reused), their translation and the coverage of Z-mat (FR-H27) |
| 1.41 | 2026-10-25 | Cohesive zone elements in 3D, in small and finite strains (FR-H28) |
| 1.42 | 2026-10-25 | Crack propagation on remeshed conforming meshes and by XFEM in 3D (FR-H29) |
| 1.43 | 2026-10-25 | Mixed-mode cracks (I, II, III) in 3D, in finite strains (FR-H30) |
| 1.44 | 2026-10-25 | The post-bifurcation along nonlinear loading paths in finite strains with nonlinear behaviours, the switch of branch (FR-H26) |
| 1.45 | 2026-10-25 | Porous elasto-(visco)plasticity, GTN, Rousselier, Abouaf's elliptic criterion, and the porous_plastic files of Z-mat, beyond gen_evp (FR-H31) |
| 1.46 | 2026-10-25 | The microfoam model of Forest's article on micromorphic continua, and a micromorphic damage model to crack with (FR-H32) |
| 1.47 | 2026-10-25 | Cracks in elasto-viscoplastic and crystalline solids in finite strains (FR-H33) |
| 1.48 | 2026-10-25 | Conforming meshes from level sets rather than voxels: mirrored conforming meshes of symmetric cells, exact symmetries for the analyses by group theory, matching faces along chosen axes, implicit bodies (FR-E5, FR-H26) |
| 1.49 | 2026-10-25 | Finite strains and nonlocal models to crack with: the microfoam model in finite strains, micromorphic ductile damage of von Mises elasto-(visco)plasticity in small and finite strains (FR-H32) |
| 1.50 | 2026-10-25 | Remeshed crack propagation in inelastic bodies, the transfer of the state of the points between meshes, ductile growth and its resistance curve (FR-H33) |
| 1.51 | 2026-10-25 | Diagonal mirrors of square and cubic cells in the mirrored conforming meshes (the groups 4/mmm and m-3m), equibiaxial compression of square arrays of holes and cubic cells in the post-bifurcation analyses (FR-E5, FR-H26) |
| 1.52 | 2026-10-25 | The buckling of a cubic cell of 2 x 2 x 2: the linear post-bifurcation of cells of several primitive cells split by their translations and the little co-groups of their wave vectors, the wave vectors of the irreps (FR-H26) |
| 1.53 | 2026-10-25 | Crack growth in a remeshed polycrystal of crystal plasticity, the grains carried by the crack meshes and the state transferred grain by grain (FR-H33) |
| 1.54 | 2026-10-25 | The micromorphic ductile damage driven by the plastic strain weighted by the stress triaxiality (FR-H32) |
| 1.55 | 2026-10-25 | The interface: its workflow through the workspace of the session, the views and measures of the level sets, voxel images up to 2048 per edge, optimizations from a morphology or a simulation, the videos of a load colouring the fields as they grow (FR-I4) |
| 1.56 | 2026-10-26 | Every function in the interface: tools built from a registry, the tab Statistics, tabs without numbers (FR-I4) |
| 1.57 | 2026-10-26 | The tabs Mechanics, Fracture, Nonlocal damage, Stability, Couplings, Design and Solvers & formats; FFT simulations on the GPU; the workspace saved and opened (FR-I4) |
| 1.58 | 2026-10-27 | The menus of the interface in four sections, the items sent on to the forms that take them; cracks through any morphology (FR-H33, FR-I4) |
| 1.59 | 2026-10-27 | The forms of the interface in boxes, the runs of the tools kept in the workspace, the models by family, the items offered by their traits, the 3D view of the items of the workspace (FR-I4); the toughness of the phases and of their interfaces in crack growth (FR-H33) |
| 1.60 | 2026-10-27 | The long computations of the interface stopped from the page; the behaviours of materials kept in its workspace and taken by its tools; the tab Comparisons; the identification of a behaviour (FR-H34, FR-I4) |
| 1.61 | 2026-10-27 | The identification of a behaviour on several references followed together, each by its weight, of porous behaviours too, from the values of the form or of a behaviour of the workspace, in the library and the interface (FR-H34, FR-I4); the materials written as Z-mat files (FR-H27) |
| 1.62 | 2026-10-28 | Phase 4 prepared (§3.6): the local CI with the tests in parallel and the Python versions at once (D18, NFR-7); the documentation site (NFR-8); the stability of the public interface and its test (NFR-10); the citation file, a draft paper for JOSS, a conda-forge recipe and the steps of a release (§10) |
| 1.63 | 2026-10-29 | Two tiers of tests (§3.6): the slowest under one Python version in the CIs, the fast tier under the others (D18, NFR-7); lighter tests of the interface |
| 1.64 | 2026-10-29 | Hill's anisotropic plasticity (§3.6, FR-H16): kernels, identification (FR-H34), Z-mat files (FR-H27), interface (FR-I4) |
| 1.65 | 2026-10-29 | Hill's criterion on the GPU (FR-B3) and in the MFront sources (FR-E7); the macroscopic law of a morphology in one go (FR-H34, FR-I4) |
| 1.66 | 2026-10-29 | Faster kernels of Hill's criterion and of porous plasticity; the cost of each behaviour of the points in the README (NFR-2) |
| 1.67 | 2026-10-29 | The analytical Jacobian of the porous update, porous plasticity on the GPU (§3.6, FR-B3) |
| 1.68 | 2026-10-29 | The tangent of finite strains from the six directions of dC (§3.6); the Numba caches of the kept environments of the local CI cleared (D18) |
| 1.69 | 2026-10-29 | A faster crystal kernel; the porous paths in the benchmark of the GPU (§3.6, NFR-2) |
| 1.70 | 2026-10-29 | MINRES for the indefinite tangents of softening points (§3.6, FR-B3) |
| 1.71 | 2026-10-29 | The device of the FFT tools of the tab Mechanics (FR-I4, FR-B3) |
| 1.72 | 2026-10-30 | Thermal expansion in the MFront sources of small strains (FR-E7) |
| 1.73 | 2026-10-30 | Thermal expansion in the MFront sources of finite strains (FR-E7) |
| 1.74 | 2026-10-30 | The initial temperature of the Abaqus inputs whose materials expand (FR-E7) |
| 1.75 | 2026-10-30 | Porous plasticity in the MFront sources (FR-E7) |
| 1.76 | 2026-10-30 | Kinematic hardening of porous plasticity (FR-H31) |
| 1.77 | 2026-10-30 | The percolation threshold of a family of microstructures (FR-M2) |
| 1.78 | 2026-10-30 | The percolation threshold in the interface and the CLI (FR-M2, FR-I2, FR-I4) |
| 1.79 | 2026-10-30 | The forest-fire algorithm of percolation (FR-M2) |
| 1.80 | 2026-10-30 | The finite-size scaling of percolation thresholds (FR-M2) |
| 1.81 | 2026-10-30 | Every porous behaviour in the MFront sources, morphoGen's kernel where the brick differs (FR-E7, FR-H31) |
| 1.82 | 2026-10-30 | The anisotropy of percolation, in 2-D and 3-D (FR-M2) |
| 1.83 | 2026-10-30 | Cellular automata on other lattices (FR-A1) |
| 1.84 | 2026-10-30 | Bending fibres packed by forces, Altendorf and Jeulin's model (FR-G3) |
| 1.85 | 2026-10-30 | Anisotropic versions of the morphological models (FR-G1 to FR-G9) |
| 1.86 | 2026-10-30 | Fibres drawn into contact and entangled cross-linked fibres (FR-G3) |
| 1.87 | 2026-10-30 | Behaviours learned from data, the module learning (FR-H21, FR-H34) |
| 1.88 | 2026-10-31 | Covariances of oriented fibres, initial placement of least overlap, Delaunay fibre networks (FR-G3) |
| 1.89 | 2026-10-31 | Quasi-periodic tilings in 2-D and 3-D, reaction-diffusion on any tiling (FR-A1) |
| 1.90 | 2026-10-31 | Learned behaviours: a tool of the interface, laws turned with their material, thermal strains (FR-H21, FR-H34) |
| 1.91 | 2026-10-31 | Thermodynamically consistent learned laws, a learned generalized standard material (FR-H21, FR-H34) |
| 1.92 | 2026-10-31 | Stiffness of random fibre networks and open foams by periodic frames of beams, softening of non-identical beams (FR-G3, FR-H26) |
| 1.93 | 2026-10-31 | Periodic approximants of the quasi-periodic tilings (FR-A1) |
| 1.94 | 2026-10-31 | Timoshenko beams in the periodic frames, Ban et al.'s cases of thin sections (FR-G3, FR-H26) |
| 1.95 | 2026-10-31 | Thermodynamically consistent learned laws in finite strains, in the logarithmic or Green-Lagrange space (FR-H21, FR-H34) |
| 1.96 | 2026-10-31 | The beam frames against the FFT solver, rigid joints, Ban et al.'s case 9 examined (FR-G3, FR-H26) |
| 1.97 | 2026-10-31 | The stiffness of the frames of beams in the interface and the command line (FR-G3, FR-H26) |
| 1.98 | 2026-10-31 | The perceptron, the potential and the recurrent network in the logarithmic strain (FR-H21, FR-H34) |
| 1.99 | 2026-10-31 | The thermal strain of the learned laws aligned on the Hencky points (FR-H21, FR-H34) |
| 1.100 | 2026-10-31 | Ban et al.'s replicas and a drawing of the frame in the interface and the command line (FR-G3, FR-H26) |
| 1.101 | 2026-10-31 | A gallery of the templates and a tutorial of a whole study in the documentation |
| 1.102 | 2026-10-31 | The thin networks of the gallery seen in 3D |
| 1.103 | 2026-11-01 | The learned laws as MFront behaviours (FR-H21, FR-H34, FR-E7) |
| 1.104 | 2026-11-01 | The critical exponent of the conductivity at the percolation threshold (FR-M2) |
| 1.105 | 2026-11-01 | The stiffness C_ij and the compliance S_ij in every elastic output (FR-H15) |
| 1.106 | 2026-11-01 | Phase 4: the JOSS draft, the conda-forge recipe and the citation brought up to 0.6 |
| 1.107 | 2026-11-01 | The mean stiffness C_ij and compliance S_ij of each volume in RVE studies (FR-V6, FR-H15) |
| 1.108 | 2026-11-01 | The stiffness, compliance and conductivity tensors of RVE studies by windows (FR-V4, FR-H15) |
| 1.109 | 2026-11-01 | The rigidity threshold from the moduli and the elastic exponent f at percolation (FR-M2) |
| 1.110 | 2026-11-01 | Nonsymmetric consistent tangents: the full tangents and GMRES in the nonlinear solvers (FR-H17, FR-H20, FR-H21) |
| 1.111 | 2026-11-01 | Shifted tilings of the window method and the edges of planar windows (FR-V4) |
| 1.112 | 2026-11-01 | The voxels factorized near the rigidity threshold, and the elastic exponent f in 2D and 3D (FR-M2) |
| 1.113 | 2026-11-01 | Deflated restarts of GMRES and an automatic choice of the tangent by its measured asymmetry (FR-H17, FR-H20) |
| 1.114 | 2026-11-01 | The rigidity decided by the energies, the voxels that bear, the 3D threshold again, discs and spheres (FR-M2) |
| 1.115 | 2026-11-01 | The shifts of the window method in the interface, and areas on the axes of planar windows (FR-V4) |
| 1.116 | 2026-11-02 | GMRES recycled from its first restart, stagnating solves stopped, and auto by phase (FR-H17, FR-H20) |
| 1.117 | 2026-11-02 | The rigidity confirmed at the floor of the residual, and the 2D and 3D sites again (FR-M2) |
| 1.118 | 2026-11-07 | Laser-architectured sheets (§3.7, §5.13: FR-L1–L6) |
| 1.119 | 2026-11-07 | The fracture of the martensite, the samples of Wang et al. 2023, cellular automata and topology optimization of the treated sheet, designs to laser tracks (§3.7, §5.13: FR-L7–L9) |
| 1.120 | 2026-11-07 | Compact tension, L-shaped plate, plastic anisotropy and alternative materials of laser-architectured sheets (§3.7, §5.13: FR-L10–L12) |
| 1.121 | 2026-11-07 | The rolling texture of a laser-architectured sheet compensated by the treatment, on the periodic cell of the sheet (FR-L13) |
| 1.122 | 2026-11-07 | Yld2000-2d, crack growth along spirals of tracks, the anisotropy of sine architectures (FR-L14) |
| 1.123 | 2026-11-13 | Multiphysics homogenization, viscoelasticity, hyperelasticity and damage in every solver, extended finite elements, the phase field of fracture, RVE statistics of every result (§3.8) |
| 1.124 | 2026-11-13 | The flow through the pores and the acoustics of porous media; the new physics in the interface and on the command line (§3.8) |
| 1.125 | 2026-11-13 | Three dimensions wherever a module worked only in the plane (§3.8) |
| 1.126 | 2026-11-13 | Implicit geometry of every model, finite strains in every case, more hyperelastic laws, the menus of the interface (§3.8) |
| 1.127 | 2026-11-14 | Tutorials, recipes, questions, glossary, gallery and site (§3.8, NFR-8) |
| 1.128 | 2026-11-14 | Quadratic elements for the finite element analyses: isoparametric 10-node tetrahedra, curved interfaces, the default of the analyses (§3.8) |
| 1.129 | 2026-11-14 | Selective reduced integration of the 10-node tetrahedra (mean dilatation: B-bar, F-bar) for nearly incompressible phases (§3.8) |
| 1.130 | 2026-11-14 | Isogeometric analysis (NURBS patches, elasticity, conduction, periodic cells) and isogeometric shape optimization (§3.8) |
| 1.131 | 2026-11-14 | Isogeometric cells of any lattice, the auxetic sheet with alternating holes, variables linked by symmetry, a tutorial (§3.8) |
| 1.132 | 2026-11-14 | The conforming mesher follows the sharp edges of unions of spheres and segments and of their coatings (§3.8) |
| 1.133 | 2026-11-14 | Coulomb's friction of cohesive interfaces and of the lips of cracks (§3.8) |
| 1.134 | 2026-11-14 | Contact, self-contact and friction of surfaces, rigid obstacles: `morphogen.contact` (§3.8) |
| 1.135 | 2026-11-14 | Self-contact in periodic cells along loading paths: `contact.ContactCell` (§3.8) |
| 1.136 | 2026-11-15 | The contact of the walls of the pores in configurations and on the command line; the template `homogenization_contact`; three corrections of the search of the faces of a surface that touches itself (§3.8) |
| 1.137 | 2026-11-15 | Ready-made tests of contact (compression between platens, indentation): functions, the commands `compress` and `indent`, the tab Contact of the interface (§3.8) |
| 1.138 | 2026-11-15 | Damage laws: coupled to plasticity (Lemaitre, Kachanov-Rabotnov), of concrete (Mazars), the unilateral effect; a delayed damage; the template `homogenization_ductile_damage` (§3.8) |
| 1.139 | 2026-11-15 | A tutorial of contact and damage, whose snippets the tests run (§3.8) |
| 1.140 | 2026-11-15 | Interfaces that debond in periodic cells along loading paths: `fracture.CohesiveCell`, the option `interfaces`, the template `homogenization_debonding`; paths back to zero stress; tests that do not depend on the machine (§3.8) |
| 1.141 | 2026-11-15 | Multiaxial high-cycle fatigue of the points of a cell along a cycle: `homogenization.fatigue` (Crossland, Sines, Dang Van, Findley, Matake), the entry `fatigue` of the configurations, the template `homogenization_fatigue` (§3.8) |
| 1.142 | 2026-11-15 | The damage coupled to plasticity with Hill's criterion (§3.8) |
| 1.143 | 2026-11-15 | Interfaces that debond: a law per pair of phases, the fields of their damage and opening, the grain boundaries of polycrystals; the template `homogenization_intergranular`; the fields of the interfaces and of the fatigue in the figures (§3.8) |
| 1.144 | 2026-11-15 | The graphical interface: interfaces that debond and the fatigue of a cycle in the tool of the loading paths (§3.8) |
| 1.145 | 2026-11-15 | Twinning and cleavage in the crystal plasticity of the grains, hexagonal crystals, fibre textures; cracks between the grains and through them; the templates `homogenization_twinning`, `homogenization_cleavage` and `homogenization_grain_fracture` (§3.8) |
| 1.146 | 2026-11-15 | The criterion of cleavage of Beremin on the fields of a loading path: `homogenization.beremin`, the template `homogenization_beremin` (§3.8) |
| 1.147 | 2026-11-15 | The tutorial *How a microstructure breaks*; the fatigue of a cycle on voxel images with pores (§3.8) |
| 1.148 | 2026-11-15 | The criterion of Beremin in the graphical interface; the revision of the compiled kernels of the crystals (§3.8) |
| 1.149 | 2026-11-15 | Contact points at the nodes of the faces: the corners and the edges of a body that touch (§3.8) |
| 1.150 | 2026-11-15 | Cleavage cracks that close (the unilateral condition, friction), twins that shear back and turn the lattice of the crystal (§3.8) |
| 1.151 | 2026-11-15 | Lattices turned grain by grain (the turned volume follows the twinned volume), twins and planes that go back at a share of their resistance, the cleavage planes of the turned lattices (§3.8) |
| 1.152 | 2026-11-15 | Cleavage that the slip nucleates: the coupling of the cleavage planes with the slip of the crystal (§3.8) |
| 1.153 | 2026-11-16 | Cleavage planes regularized by their microdamage: a damage band of a given width, the equation of the microdamage in the loading paths (§3.8) |
| 1.154 | 2026-11-16 | The alternate scheme of the microdamage: its options in the configurations, its acceleration (§3.8) |
| 1.155 | 2026-11-16 | The integration of the points two to three times faster for the crystals, with the same results to the bit (§3.8) |
| 1.156 | 2026-11-17 | Loading paths in fewer Newton and linear iterations: the second-order prediction of a step, the secant of the condensed block of the macroscopic unknowns (§3.8) |
| 1.157 | 2026-11-17 | The finite element solvers on several threads: Gauss–Seidel sweeps in the order of the colours of the unknowns, the products of the Krylov iterations, the reduced tangent of a path by the places of its stored values (§3.8) |
| 1.158 | 2026-11-17 | The sweeps by nodes (3 × 3 blocks), the assembly of the paths by colours of elements, the BLAS on one thread while the compiled kernels run (§3.8) |
| 1.159 | 2026-11-17 | The products of the Krylov iterations by blocks of nodes; along a path, the sweeps of each tangent and the coarse levels of the preconditioner kept while they serve (§3.8) |
| 1.160 | 2026-11-17 | Crystallographic textures and orientation maps: symmetries, disorientations, pole figures; EBSD maps read and written (`.ang`, `.ctf`), their grains, the model `ebsd` (§3.8). The licence: the author holds the rights, confirmed by the institution (D1). Three features on hold (§3.8) |
| 1.161 | 2026-11-17 | The command `morphogen texture` and the figures of the textures; the density of inverse pole figures; the texture of a map for the cells of a tessellation of space (§3.8) |
| 1.162 | 2026-11-17 | The lattices along a loading path: the orientation of a material point at each step, the texture of a volume element at a time, from its fields and from their files; single slip against Schmid and Boas, the fibres of Taylor's polycrystal (§3.8) |
| 1.163 | 2026-11-17 | A tutorial (from an orientation map to a textured polycrystal) and the tool *Orientation map (EBSD)* of the interface (§3.8) |
| 1.164 | 2026-11-17 | Scale-rich networks in two and three dimensions (FR-G17, FR-G18) and exact meshes without a background grid (FR-E9), with the meshes of §3.8: `method: auto`, exact for two-phase structures (§3.9) |
| 1.165 | 2026-11-18 | One interface of the implicit functions on grids and at points; `method: auto` exact for partitions of several phases, the phases kept by a list and the sharp edges of smooth pieces; one curvature of the 10-node elements (FR-E9, §3.9) |
| 1.166 | 2026-11-25 | Version 0.7 (§3.10): long fibres (FR-G19), woven textiles (FR-G20), nested crystalline structures (FR-G21), monodisperse or polydisperse sizes (FR-G22), particles for the discrete element method (FR-E10), arrays without copies and phase fields for spectral solvers (FR-E11), Microgen (FR-E12), Blender (FR-E13), alloys (FR-E14), the workspace and the fields in the page (FR-I9); roadmap 0.7 |
| 1.167 | 2026-11-25 | The particles in Gmsh files for LMGC90 (triangles or tetrahedra, an entity per particle), their surfaces remeshed by MMG, the segments as capsules (FR-E10, §3.10) |
| 1.168 | 2026-11-25 | The requirements stated in the voice of a specification throughout (§3, §5, §9–§11, §13); the videos of every tool on demand (FR-I9) and the finite elements by default, the voxels as the last resort (FR-H35) (§3.10) |
| 1.169 | 2026-11-27 | Version 0.8 (§3.11): the measures of the implicit geometry and the exact Minkowski functionals (FR-M5), the identification of models from images (FR-M6), segmentation (FR-M7), woven variability and architectures (FR-G23, FR-G24), images and spinodal decomposition as models (FR-G25, FR-G26), Code_Aster, OpenFOAM and VIMSEO (FR-E15–E17), unidirectional tows (FR-H36), thermo-chemo-elastic couplings in time (FR-H37), the inverse design of active structures (FR-O7); roadmap 0.8 |
| 1.170 | 2026-11-27 | The runs of Code_Aster and OpenFOAM, and their verification with the programs themselves (FR-E15, FR-E16, §3.11): the flow through tetrahedra on their polyhedral dual |
| 1.171 | 2026-11-27 | The reconstruction from descriptors by gradient descent (FR-M8, §3.11), the method of MCRpy, from the closed forms of the models |
| 1.172 | 2026-11-27 | The named measures of the implicit geometry in configurations, on the command line and in the interface (FR-M5, §3.11) |
| 1.173 | 2026-11-27 | Grains grown from seeds, as MicroStructPy (FR-G27, §3.11): the model `seeded_laguerre`, its meshes by grain |
| 1.174 | 2026-11-27 | Seeds of several phases, amorphous phases and the verification of the grains against their seeds (FR-G27, §3.11) |
| 1.175 | 2026-11-27 | Seeds of the plane, the figures of the verification of the grains and its tool of the interface (FR-G27, §3.11) |
| 1.176 | 2026-11-27 | Seeds of boxes and of rectangles, the phases filled by their seeds (FR-G27, §3.11) |
| 1.177 | 2026-11-27 | Meshes of the plane and their finite elements (FR-E18, FR-H38, §3.11) |
| 1.178 | 2026-11-27 | The meshes of the plane in the configurations, the runs and the command line; the columnar grains meshed in space (FR-E18, FR-H38, §3.11) |
| 1.179 | 2026-11-28 | The loading paths of the meshes of the plane on their periodic layer in space (FR-H38, §3.11) |
| 1.180 | 2026-11-28 | The orientation maps (EBSD) meshed in their plane by their pixels, for the finite elements of the plane (FR-E18, §3.11) |
| 1.181 | 2026-11-28 | The tutorial of the polycrystals of the plane (FR-E18, FR-H38, §3.11) |
| 1.182 | 2026-11-28 | The curved six-node triangles of the sections of the plane (FR-E18, FR-H38, §3.11) |
| 1.183 | 2026-11-28 | The flow through the pores of the plane, the permeability of beds of fibres (FR-H38, §3.11) |
| 1.184 | 2026-11-28 | The exact discs of the sections of fibres and spheres (FR-E18, §3.11) |
| 1.185 | 2026-11-28 | The tutorial of fibre composites in their plane (FR-E18, FR-H38, §3.11) |
| 1.186 | 2026-11-28 | The rings of coated and hollow fibres and spheres in the sections (FR-E18, §3.11) |
| 1.187 | 2026-11-28 | The flow through the pores of the conforming meshes of space (FR-H38, §3.11) |
| 1.188 | 2026-11-28 | The exact union of discs that cross each other in the sections (FR-E18, §3.11) |
| 1.189 | 2026-11-28 | The sizes graded toward the corners of the unions of discs (FR-E18, §3.11) |
| 1.190 | 2026-11-28 | The exact contours of the sections of oblique and wavy fibres (FR-E18, §3.11) |
| 1.191 | 2026-11-28 | The exact corners of the contours where capsules cross, their edges checked (FR-E18, §3.11) |
| 1.192 | 2026-11-28 | The contours of coated and hollow fibres, and of coatings that meet (FR-E18, §3.11) |
| 1.193 | 2026-11-28 | Every corner of a body found on its own and checked against its contours (FR-E18, §3.11) |
| 1.194 | 2026-11-28 | The template of a misaligned ply in its plane (FR-E18, FR-H38, §3.11) |
| 1.195 | 2026-11-28 | The axial stiffness of misaligned plies by the orientations of their fibres (FR-H38, §3.11) |
| 1.196 | 2026-11-28 | The stiffness of misaligned plies recorded by the runs of their sections (FR-H38, §3.11) |
| 1.197 | 2026-11-28 | The conductivities of misaligned plies, as their stiffness (FR-H38, §3.11) |

## 14. References

The references of the phase-3 methods (automata, growth, aggregation, frameworks,
topology optimization) are listed in [theory.md](theory.md#references).

* Bironeau, A., Dirrenberger, J., Sollogoub, C., Miquelard-Garnier, G., Roland, S.
  (2016). Evaluation of morphological representative sample sizes for nanolayered
  polymer blends. *J. Microscopy* 264, 48–58.
* Quey, R., Dawson, P. R., Barbe, F. (2011). Large-scale 3D random polycrystals for the
  finite element method: generation, meshing and remeshing. *Comput. Methods Appl.
  Mech. Eng.* 200, 1729–1745.
* Roters, F. et al. (2019). DAMASK — The Düsseldorf Advanced Material Simulation Kit
  for modeling multi-physics crystal plasticity, thermal, and damage phenomena from the
  single crystal up to the component scale. *Comput. Mater. Sci.* 158, 420–478.
* Chiu, S. N., Stoyan, D., Kendall, W. S., Mecke, J. (2013). *Stochastic Geometry and
  its Applications*, 3rd ed. Wiley.
* Coulais, C., Teomy, E., de Reus, K., Shokef, Y., van Hecke, M. (2016). Combinatorial
  design of textured mechanical metamaterials. *Nature* 535, 529–532.
* Dapogny, C., Dobrzynski, C., Frey, P. (2014). Three-dimensional adaptive domain
  remeshing, implicit domain meshing, and applications to free and moving boundary
  problems. *J. Comput. Phys.* 262, 358–378.
* Dirrenberger, J., Forest, S., Jeulin, D. (2014). Towards gigantic RVE sizes for 3D
  stochastic fibrous networks. *Int. J. Solids Struct.* 51, 359–376.
* Gilbert, E. N. (1962). Random subdivisions of space into crystals. *Ann. Math. Stat.*
  33, 958–972.
* Helfer, T., Michel, B., Proix, J.-M., Salvo, M., Sercombe, J., Casella, M. (2015).
  Introducing the open-source mfront code generator: application to mechanical behaviours
  and material knowledge management within the PLEIADES fuel element modelling platform.
  *Comput. Math. Appl.* 70, 994–1023.
* Helfer, T., Bleyer, J., Frondelius, T., Yashchuk, I., Nagel, T., Naumov, D. (2020). The
  MFrontGenericInterfaceSupport project. *J. Open Source Softw.* 5(48), 2003.
* Jeulin, D. (2011). Variance scaling of Boolean random varieties. Technical report,
  Centre de Morphologie Mathématique, hal-00618967.
* Kanit, T., Forest, S., Galliet, I., Mounoury, V., Jeulin, D. (2003). Determination of
  the size of the representative volume element for random composites. *Int. J. Solids
  Struct.* 40, 3647–3679.
* Lantuéjoul, C. (1991). Ergodicity and integral range. *J. Microscopy* 161, 387–403.
* Matheron, G. (1971). *The Theory of Regionalized Variables and its Applications*.
  École des Mines de Paris.
* Ohser, J., Schladitz, K. (2009). *3D Images of Materials Structures*. Wiley-VCH.
* Schneider, M., Merkert, D., Kabel, M. (2017). FFT-based homogenization for
  microstructures discretized by linear hexahedral elements. *Int. J. Numer. Methods
  Eng.* 109, 1461–1489.
* Sigmund, O. (1994). Materials with prescribed constitutive parameters: an inverse
  homogenization problem. *Int. J. Solids Struct.* 31, 2313–2329.
* Torquato, S. (2002). *Random Heterogeneous Materials*. Springer.
* Lubachevsky, B. D., Stillinger, F. H. (1990). Geometric properties of random disk
  packings. *J. Stat. Phys.* 60, 561–583.
* Yeong, C. L. Y., Torquato, S. (1998). Reconstructing random media. *Phys. Rev. E*
  57, 495–506.
* Tovar, A., Patel, N. M., Niebur, G. L., Sen, M., Renaud, J. E. (2006). Topology
  optimization using a hybrid cellular automaton method with local control rules.
  *J. Mech. Des.* 128, 1205–1216.
* Vaněk, P., Mandel, J., Brezina, M. (1996). Algebraic multigrid by smoothed
  aggregation for second and fourth order elliptic problems. *Computing* 56, 179–196.
* Yang, S., Dirrenberger, J., Monteiro, E., Ranc, N. (2019). Representative volume
  element size determination for viscoplastic properties in polycrystalline materials.
  *Int. J. Solids Struct.* 158, 210–219.
