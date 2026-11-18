---
title: "morphoGen: random and architected microstructures, their representative volume elements and their apparent properties in Python"
tags:
  - Python
  - microstructure
  - mathematical morphology
  - random media
  - representative volume element
  - homogenization
  - finite elements
  - FFT
  - percolation
  - constitutive modelling
authors:
  - name: Justin Dirrenberger
    orcid: 0000-0002-3964-305X
    affiliation: "1, 2"
affiliations:
  - name: PIMM, Arts et Métiers, Cnam, CNRS UMR 8006, Paris, France
    index: 1
  - name: Institut universitaire de France (IUF), Paris, France
    index: 2
date: 18 November 2026
bibliography: paper.bib
---

<!-- Draft for JOSS (docs/release.md). Before submission: the research impact
statement with the uses of morphoGen, the acknowledgements and funding,
the AI usage disclosure reviewed by the authors; at least six months after the
repository is made public. -->

# Summary

The effective properties of a heterogeneous material follow from its microstructure
through a representative volume element (RVE): a volume large enough for its apparent
properties to stand for the material, and small enough to be computed. morphoGen is a
Python package that covers the whole chain of such a study. It generates virtual
microstructures from 40 models: those of mathematical morphology and stochastic geometry
(Boolean models, straight, curved and cross-linked fibres, dead leaves, tessellations,
Gaussian fields, packings), most of them anisotropic as well; architected and
bio-inspired geometries (minimal surfaces, lattices, foams, scale-rich networks, nacre,
bone); cellular automata on periodic and quasi-periodic tilings, growth processes and
polycrystals, textured after measured orientation maps. It measures them, percolation
included, against the closed forms of their models where they exist, meshes them from
the exact functions of their phases, and computes their apparent elastic, transport,
coupled and nonlinear responses, fracture and contact included, by finite elements on
conforming meshes or by FFT on voxel images, on the CPU or an NVIDIA GPU, and their
response to impacts by explicit dynamics with OpenRadioss. It estimates the size of their RVE with the statistical method of @Kanit2003 and
@Dirrenberger2014, identifies their macroscopic law (von Mises, Hill [@Hill1948] or
Gurson [@Tvergaard1981]) or learns one from their responses, and writes microstructures,
meshes and behaviours for other codes (Z-set, MFront, AMITEX_FFTP, Abaqus, FEniCSx,
Neper, DAMASK, OpenRadioss). It runs from Python, configuration files or a graphical
interface, in the web browser or as a stand-alone application.

# Statement of need

The statistical determination of the RVE [@Kanit2003] needs many realizations of a
random microstructure at several sizes, their apparent properties under several boundary
conditions, and the fit of the scaling of their variance with the volume, which gives
the size of the RVE for a given precision. Each step usually takes a different tool,
linked by scripts that the study does not publish: generator, mesher, solver,
statistics, then the fit of a law for structural codes. The generators seldom give the
theoretical characteristics of their models, against which a realization can be checked,
and the meshes seldom follow the interfaces. morphoGen puts these steps in one package,
in which each model carries its theory, every realization is reproducible from its seed,
and the same configuration runs one realization on a laptop or thousands as a job array
on a cluster. It is meant for researchers in the mechanics of materials who study how
microstructures set properties, design architected materials, or need microstructures of
known statistics, or their homogenized behaviours, for their own solvers.

# State of the field

Several open tools cover parts of this chain. Neper [@Quey2011] generates and meshes
polycrystals from tessellations; DREAM.3D [@Groeber2014] builds synthetic polycrystals
from statistics; PoreSpy [@Gostick2019] and PuMA [@Ferguson2018] analyse images of porous
media; microgen [@microgen] builds and meshes lattices and TPMS cells. DAMASK
[@Roters2019] and the FFT solvers in the line of @Moulinec1998, such as AMITEX_FFTP,
compute the response of voxel images; MFront [@Helfer2015] generates the behaviours of
mechanical solvers, and Z-set [@Besson1997] computes the RVE of the finite element
studies that morphoGen follows. None of them gives the models of mathematical morphology
[@Matheron1975; @Jeulin2021; @Torquato2002] with their closed forms, nor ties the
generation, the apparent properties, the statistical RVE study and the macroscopic law
together. Rather than reimplement what these tools do well, morphoGen calls them or
writes for them: MMG [@Dapogny2014] meshes its level sets, MFront behaviours run in its
solvers through MGIS [@Helfer2020], Z-mat material files run as they are, and its
microstructures and behaviours become inputs of the codes above, so that a study can
move to the solver its authors trust.

# Software design

A model is a set of parameters that `generate(domain, seed)` turns into a realization
kept in its explicit form (spheres, fibres, cells), from which a level set and voxel
images of any resolution follow. Seeds derive from a NumPy `SeedSequence` by
realization, so that the results do not depend on how processes or cluster tasks share
the realizations. The kernels are compiled with Numba [@Lam2015], on top of NumPy
[@Harris2020] and SciPy [@Virtanen2020], so that pip installs the package without a
compiler; the FFT solvers also run on NVIDIA GPUs through CuPy [@Okuta2017], with CUDA
ports of the behaviours checked point by point against the CPU.

The finite element solver works on conforming tetrahedra and voxel meshes, with
kinematic, static, mixed and periodic boundary conditions and the macroscopic strain as
degrees of freedom (the periodic element of Z-set); the FFT solver discretizes voxels as
trilinear hexahedra [@Schneider2017] and follows loading paths by Newton–Krylov
iterations [@Gelebart2013]. Both integrate von Mises and Hill elasto-viscoplasticity,
crystal plasticity [@Meric1991] and porous plasticity by the same code, in small and
finite strains. Percolation is measured by burning a phase from a face [@Herrmann1984];
finite-size scaling [@Stauffer1994] gives the thresholds of a family of microstructures
along any direction, and the critical exponents of their conductivity and elastic
moduli. The learned behaviours range from a perceptron to a generalized standard
material [@Halphen1975] whose free energy is an input-convex network [@Amos2017] and
whose dissipation potential is convex: its dissipation is nonnegative and its tangent
symmetric by construction. Trained with their gradients written out, without a
machine-learning framework, they serve as materials of the solvers and are exported as
MFront behaviours.

Every model gives the implicit function of each of its phases at any point, from which
MMG meshes the interfaces, the junctions of three phases and the sharp edges included,
without a background grid. The phase field of brittle fracture [@Bourdin2000] runs on
voxel images in small and finite strains, beside cohesive zones and unilateral contact;
the transport, coupled (piezoelectric, magneto-electro-elastic, thermoelectric) and
harmonic problems of physics are solved on the same voxel images and meshes.

The validation comes first: about 2300 tests compare the realizations with the theory of
their models within a few standard deviations, the solvers with exact solutions
(laminates, Hashin–Shtrikman bounds, the bulk modulus of cubic polycrystals) and with
independent codes (scikit-fem, MFront, CalculiX), and the exported files with
independent readers (VTK, Gmsh).

# Research impact statement

morphoGen implements the methods of studies of the RVE of random media
[@Kanit2003; @Dirrenberger2014], polycrystals [@Yang2019] and nanolayered films
[@Bironeau2016], and of the post-bifurcation of architected materials [@Azulay2023], so
that they can be repeated and extended on other microstructures. <!--
To complete by the authors: the publications, theses and projects that use morphoGen,
and the external users or integrations, as JOSS asks for evidence rather than
expectations. -->

# AI usage disclosure

The code, the tests and the documentation of morphoGen were written with Claude Code
(Anthropic), with the model Claude Opus 5.5, from October 2026, at the request of the
author, who set the scope, the requirements and the choices of methods and tools
recorded in the specification of the repository (`docs/specification.md`), and who
reviewed the results. Every commit of the repository says so. The correctness of the
code rests on its tests, which compare its results with closed forms, exact solutions
and independent codes rather than with its own earlier outputs, and run at each change.
This paper was drafted with the same tool and edited by the author.
<!-- To be confirmed and completed by the authors: what they reviewed, edited and
validated themselves, as JOSS requires that human authors validate all AI-assisted
outputs and make the core design decisions. -->

# Acknowledgements

<!-- To complete by the authors: funding and support. -->

# References
