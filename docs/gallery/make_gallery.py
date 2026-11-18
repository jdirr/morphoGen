"""The gallery of examples of the documentation (docs/gallery.md): a card per template of
morphoGen, by family, with the image of the template, a line of description, the commands
that reproduce it and its configuration.

    python docs/gallery/make_gallery.py                 # the page and every image
    python docs/gallery/make_gallery.py tpms voronoi    # the page, and these images again
    python docs/gallery/make_gallery.py --page          # the page only
    python docs/gallery/make_gallery.py --results       # the page, and the results (below)

An image (docs/_static/gallery/<template>.png) shows the first realization of its template,
the one that ``morphogen run`` makes first: the realization 0 of its seed (seed 1 for a
random model whose template has none, the seed that the command of its card then gives),
in its domain (the largest size of an ensemble RVE study, the size of a window study), or
the design of an optimization template. Three faces of its box are seen in axonometry
(the top, front and right faces, shaded), each the section of the microstructure computed
in its plane as the interface draws them (:mod:`morphogen.viz.sections`: anti-aliased
interfaces of the level set, the colours of the phases, a colour per grain of a
polycrystal, a colour per cell of a mosaic of one phase); a planar structure shows its x-y
section, a template that studies a section of its volume (``voxels.section``) that
section, and the templates of :data:`SURFACE` the surface of their phase 1 in 3D (VTK, the
extra ``video``). The images are deterministic, a few tens of kilobytes each (palettes of
at most 128 colours, by Pillow: the extra ``plot``); all of them take about five minutes
on a laptop, most of it the few slow templates (the optimizations, the window study of
the automaton, the reconstruction, fibres packed by forces). The page is written from
:data:`FAMILIES` and the templates; tests/test_gallery.py checks that it is up to date,
that every template has its card and its image, and that every card has its template.

The cards of the templates of :data:`RESULTS` also show what their run computes
(docs/_static/gallery/<template>_result.png): the template is run as ``morphogen run``
does, and the figure drawn from its files (the curve of a loading path, the damage of a
cell that breaks). ``--results`` writes them (all, or those named): a few minutes each.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "gallery.md"
IMAGES = ROOT / "docs" / "_static" / "gallery"
TEMPLATES = ROOT / "src" / "morphogen" / "templates"

#: The seed of the random templates that have none (given to their command)
SEED = 1
#: The templates without a seed whose model is random (their command gives ``--seed``);
#: the others without one are deterministic, which :func:`realization` checks
RANDOM_WITHOUT_SEED = {"bouligand", "homogenization_mixed_loading", "nacre", "osteons", "wood"}
#: Pixels of the images (width, height), and the factor of their supersampling
CANVAS = (560, 476)
SUPERSAMPLE = 2
#: Largest number of colours of an image
COLOURS = 128
#: The templates seen as the surface of their phase 1 in 3D rather than by the faces of
#: their box: thin structures inside the box (a tree, aggregates, networks of struts or
#: fibres, open foams), which its faces barely cut; the voxels of their surfaces along
#: the longest edge
SURFACE = {
    "aggregation", "vascular_tree", "delaunay_network", "trabecular_bone", "level_set_foam",
    "strut_lattice", "graded_lattice", "random_lattice", "hierarchical_lattice",
}  # fmt: skip
SURFACE_VOXELS = 256
#: The templates whose card also shows a result of their run, and how it is drawn:
#: ``path`` (the stress of a loading path against its strain; ``stretch``: against the
#: stretch, the nominal stress; ``time``: against the time, on a logarithmic axis;
#: ``labels``: the names of its segments, those of the same name drawn alike),
#: ``fracture`` (the damage at the end of the path on the phases, beside the curve)
RESULTS = {
    "homogenization_viscoplastic": {
        "kind": "path",
        "labels": ["tension", "relaxation", "unloading"],
    },
    "homogenization_viscoelastic": {
        "kind": "path",
        "time": True,
        "labels": ["stretching", "strain held", "strain held"],
    },
    "homogenization_crystal_plasticity_fft": {"kind": "path"},
    "homogenization_rubber": {
        "kind": "path",
        "stretch": True,
        "labels": ["first loading", "unloading", "second loading"],
    },
    "homogenization_contact": {"kind": "path", "labels": ["compression", "unloading"]},
    "homogenization_ductile_damage": {"kind": "path", "labels": ["tension", "unloading"]},
    "homogenization_debonding": {
        "kind": "path",
        "labels": ["tension", "unloading", "compression"],
    },
    "homogenization_intergranular": {"kind": "path"},
    "homogenization_fatigue": {"kind": "fatigue"},
    "homogenization_twinning": {"kind": "twinning"},
    "homogenization_detwinning": {
        "kind": "cycle",
        "quantity": "twin_fraction",
        "label": "volume fraction",
        "also": "twin_reoriented",
        "legend": ("twinned", "turned lattices"),
    },
    "homogenization_crack_closure": {
        "kind": "cycle",
        "quantity": "cleavage_opening",
        "label": "opening of the planes",
    },
    "homogenization_cleavage": {
        "kind": "field",
        "field": "cleavage_damage",
        "title": "cleavage damage",
    },
    "homogenization_slip_cleavage": {"kind": "path"},
    "homogenization_cleavage_band": {
        "kind": "field",
        "field": "cleavage_damage",
        "title": "cleavage damage",
        "normal": 2,
    },
    "homogenization_grain_fracture": {"kind": "path"},
    "homogenization_beremin": {"kind": "beremin"},
    "fracture_composite": {"kind": "fracture"},
    "fracture_rubber": {"kind": "fracture", "stretch": True},
}
#: Pixels of the figures of the results, their ink (legible on light and dark pages) and
#: the colours of their curves
RESULT_CANVAS = (560, 330)
INK = "#8a9099"
CURVES = ("#2a78d6", "#9aa1ab", "#eb6834", "#1baf7a")

#: The families of the gallery: their title, a sentence, and their templates with a line
#: of description each
FAMILIES = [
    (
        "Point processes and Boolean models",
        "Grains placed at random points: overlapping, penetrable or hard, clustered in "
        "aggregates, coated.",
        [
            ("boolean_spheres", "Boolean model of spheres: overlapping grains of a porous or "
             "granular medium, with its exact covariance and integral range."),
            ("penetrable_spheres", "Penetrable spheres with impenetrable cores (Torquato's "
             "cherry-pit model), between the Boolean model and hard spheres."),
            ("hardcore_spheres", "Non-overlapping spheres by random sequential addition "
             "(jamming near 0.38)."),
            ("dense_packing", "A dense random packing of hard spheres grown to 0.6 by the "
             "Lubachevsky–Stillinger algorithm."),
            ("cox_nanocomposite", "Particles clustered in the aggregates of a Boolean model "
             "(a Cox model), with a bound-rubber phase: carbon black in an elastomer."),
            ("coated_spheres", "Hard spheres with a coating of constant thickness: three "
             "phases, homogenized by FFT."),
        ],
    ),
    (
        "Dead leaves, Gaussian fields and reconstructions",
        "Random sets defined through random functions, and a random set rebuilt from its "
        "correlation functions.",
        [
            ("dead_leaves", "Dead leaves of spheres: a two-phase mosaic of non-convex grains "
             "with random crystal orientations (Jeulin 1997)."),
            ("sequential_dead_leaves", "Two families of spheroidal leaves, prolate and flat, "
             "falling on a background until it is covered 1.5 times."),
            ("transparent_dead_leaves", "The random function of transparent leaves of random "
             "grey levels, thresholded to a volume fraction of 0.35."),
            ("gaussian_excursion", "Excursion set of a Gaussian random field: a bicontinuous "
             "porous medium."),
            ("reconstruction", "A Boolean model of spheres rebuilt from its two-point and "
             "lineal-path functions by simulated annealing (Yeong & Torquato 1998)."),
        ],
    ),
    (
        "Fibres and layers",
        "Straight, curved and bending fibres, with or without overlap, networks of fibres, "
        "and laminates.",
        [
            ("boolean_cylinders", "Boolean model of finite fibres (cylinders or "
             "spherocylinders) of any law of orientations."),
            ("poisson_fibers", "Infinite fibres on isotropic Poisson lines (Dirrenberger et "
             "al. 2014), in a non-periodic domain."),
            ("hardcore_fibers", "Non-overlapping straight fibres by random sequential "
             "addition, at a minimal gap: a fibre-reinforced composite."),
            ("curved_fibers", "Boolean model of curved fibres: worm-like chains whose "
             "tangents decorrelate along the contour."),
            ("curved_arcs", "Circular arcs of uniform radii of curvature without overlap, "
             "lying close to a plane, as the fibrous preforms of PuMA."),
            ("random_walk_fibers", "Bending fibres packed without overlap by forces to 0.35 "
             "(Altendorf & Jeulin 2011)."),
            ("contact_fibers", "The same bending fibres drawn into contact by contact forces "
             "(Keilmann, Redenbach & Willot 2026)."),
            ("cross_linked_fibers", "Entangled carbon fibres moulded about a plane, half "
             "their contacts bonded by epoxy cross-links (Chatti et al.)."),
            ("delaunay_network", "A fibre network on the edges of the Delaunay "
             "triangulation of random points, cross-linked at every node (Ban et al. "
             "2016)."),
            ("random_laminate", "A nanolayered film: layers of lognormal thicknesses, the "
             "thinner ones broken up by holes (Bironeau et al. 2016)."),
        ],
    ),
    (
        "Tessellations",
        "Mosaics of cells: polycrystals, computed or measured, multiphase mosaics and "
        "closed cells.",
        [
            ("voronoi", "A Poisson–Voronoi polycrystal of 343 grains with random crystal "
             "orientations."),
            ("laguerre", "A Laguerre (power) tessellation as a two-phase mosaic (Kanit et al. "
             "2003)."),
            ("hyperuniform_cells", "The walls of the Voronoi cells of stealthy hyperuniform "
             "germs: disordered but regular closed cells."),
            ("ebsd", "A measured polycrystal: windows of an orientation map (EBSD, an .ang "
             "or a .ctf file), its grains found from the disorientations of its points, "
             "each with its mean orientation; their stiffness by FFT."),
        ],
    ),
    (
        "Foams and lattices",
        "Architected materials: strut lattices, regular or not, hierarchical lattices, "
        "fractal sponges and foams.",
        [
            ("strut_lattice", "An octet-truss strut lattice at 15 % relative density."),
            ("graded_lattice", "An octet lattice graded along x and turned by 45° about z, "
             "with its STL surface for additive manufacturing."),
            ("random_lattice", "An octet lattice with missing struts and displaced nodes, "
             "and the rigidity of its framework (Maxwell–Calladine)."),
            ("hierarchical_lattice", "A hierarchical lattice of hollow tubes: frames of "
             "octahedra of generation 1 (Rayneau-Kirkhope, Mao & Farr 2012)."),
            ("scale_rich_network", "A scale-rich network: 500 ligaments of power-law "
             "thicknesses grown until they meet, at 40 % relative density (Both et al. "
             "2026)."),
            ("scale_rich_plates", "A scale-rich network in space: 400 plates of power-law "
             "thicknesses, each splitting a closed cell in two, at 30 % relative density."),
            ("menger_sponge", "The Menger sponge of level 3, of fractal dimension "
             "ln 20 / ln 3."),
            ("level_set_foam", "An open-cell copper foam as a level set of distance fields "
             "of packed ellipsoids (Degdeg & Dirrenberger 2026)."),
        ],
    ),
    (
        "TPMS",
        "Triply periodic minimal surfaces, as sheets or networks (the stiffness of the "
        "gyroid is under Linear homogenization).",
        [
            ("tpms", "A sheet gyroid at 20 % relative density, with its watertight STL "
             "surface."),
        ],
    ),
    (
        "Automata and tilings",
        "Structures grown on lattices by local rules, and assemblies of building blocks.",
        [
            ("cellular_automaton", "A bicontinuous structure coarsened from noise by "
             "Vichniac's annealing rule."),
            ("activator_inhibitor", "Turing patterns of Young's activator–inhibitor "
             "automaton."),
            ("reaction_diffusion", "Turing patterns of the Gray–Scott reaction–diffusion "
             "system (Pearson 1993)."),
            ("tile_assembly", "A compatible assembly of pipe tiles drawn by wave function "
             "collapse: a network without dead ends."),
        ],
    ),
    (
        "Growth",
        "Structures grown by a process in time: grains, aggregates and trees.",
        [
            ("nucleation_growth", "A Johnson–Mehl polycrystal: grains nucleated at a "
             "constant rate grow until they impinge (JMAK kinetics)."),
            ("aggregation", "Diffusion-limited aggregation: fractal clusters grown by "
             "random walkers (Witten & Sander 1981)."),
            ("vascular_tree", "A vascular tree grown by space colonization, its radii by "
             "Murray's law."),
        ],
    ),
    (
        "Bio-inspired",
        "Microstructures drawn from biology.",
        [
            ("nacre", "Nacre: staggered aragonite tablets bonded by a soft organic mortar "
             "(brick and mortar)."),
            ("bouligand", "A Bouligand twisted plywood: layers of fibres turned by 15° from "
             "one to the next (the dactyl club of the mantis shrimp)."),
            ("wood", "Softwood: radial files of tracheids through a growth ring, with "
             "rays."),
            ("osteons", "Cortical bone: osteons with their Haversian canals and cement "
             "lines, remodelled one after another."),
            ("trabecular_bone", "Trabecular bone or an open-cell foam: struts on the edges "
             "of relaxed, elongated Voronoi cells."),
        ],
    ),
    (
        "Linear homogenization",
        "Apparent elastic and thermal properties, by finite elements on conforming meshes "
        "or by the FFT solver on voxels.",
        [
            ("homogenization_spheres", "The apparent moduli of stiff spheres in a soft "
             "matrix under the four boundary conditions (finite elements: 10-node "
             "tetrahedra whose edges follow the spheres)."),
            ("homogenization_gyroid", "The cubic stiffness tensor and the conductivity of a "
             "sheet gyroid cell (finite elements, quadratic tetrahedra)."),
            ("homogenization_voronoi", "The apparent moduli and stiffness tensor of a "
             "two-phase Voronoi mosaic (FFT)."),
            ("homogenization_polycrystal", "The apparent moduli of a copper polycrystal "
             "meshed grain by grain (finite elements)."),
            ("homogenization_polycrystal_inclusions", "A copper polycrystal with an "
             "amorphous grain-boundary phase and alumina inclusions (FFT)."),
            ("homogenization_mixed_loading", "Thermo-elastic loadings mixing strains and "
             "stresses on a unidirectional glass-epoxy composite."),
            ("homogenization_crystallization", "A partially crystallized material: grains "
             "of the Avrami law meshed conformingly (finite elements)."),
            ("homogenization_interphase", "Spheres, their interphase and the matrix meshed "
             "conformingly from the implicit function of each phase: three phases by finite "
             "elements whose nodes lie on the interfaces, with the level sets and the "
             "surfaces of the phases."),
        ],
    ),
    (
        "Nonlinear behaviours",
        "Loading paths of elasto-viscoplastic composites and polycrystals, a rubber in "
        "finite strains, a reduced model, a matrix that plasticity damages, and a porous "
        "cell whose walls come into contact.",
        [
            ("homogenization_viscoplastic", "SiC particles in a viscoplastic aluminium "
             "alloy: tension, relaxation and unloading (finite elements: 10-node "
             "tetrahedra, which do not lock when the matrix flows)."),
            ("homogenization_crystal_plasticity", "A nickel polycrystal pulled to 10 % in "
             "finite strains, Méric–Cailletaud crystal plasticity (finite elements)."),
            ("homogenization_crystal_plasticity_fft", "The same on 24³ voxels of 20 grains, "
             "to 5 % (the nonlinear FFT solver)."),
            ("homogenization_rubber", "A filled rubber stretched by half its length, "
             "unloaded and stretched again: the eight chains of Arruda and Boyce and the "
             "softening of Mullins, in finite strains (the nonlinear FFT solver)."),
            ("homogenization_ntfa", "A reduced model (NTFA) of an elasto-plastic composite "
             "trained on full-field paths."),
            ("homogenization_ductile_damage", "SiC particles in an aluminium matrix "
             "whose plasticity is coupled to Lemaitre's damage: the composite reaches its "
             "strength, softens as the matrix is damaged and unloads along its damaged "
             "modulus (the nonlinear FFT solver, a delayed damage)."),
            ("homogenization_contact", "Plates parted by gaps and held by pillars, "
             "compressed until the gaps close: the walls of the pores press on each other "
             "(self-contact with Coulomb's friction, in finite strains) and the cell "
             "stiffens fivefold."),
            ("homogenization_intergranular", "A polycrystal that breaks along its grain "
             "boundaries: cohesive interfaces between the grains of a Voronoi "
             "tessellation of cubic crystals. The stress passes a peak and falls to zero "
             "as a crack crosses the cell from boundary to boundary; the work that broke "
             "it is the fracture energy times the area of the crack."),
            ("homogenization_grain_fracture", "Cracks between the grains and through "
             "them: a polycrystal in crystal plasticity whose grain boundaries are "
             "cohesive and whose {100} planes cleave. With boundaries weaker than the "
             "planes the crack follows them (intergranular fracture); with strong ones it "
             "cuts through the grains along their cleavage planes (transgranular)."),
            ("homogenization_twinning", "The twins of a textured magnesium: a polycrystal "
             "whose axes c are near the load. A twin shears in one sense only: pulled, "
             "the grains twin and flow at a low stress; compressed, they do not twin and "
             "are twice as hard (crystal plasticity with twins, the nonlinear FFT "
             "solver)."),
            ("homogenization_detwinning", "The twins of a textured magnesium, there and "
             "back: pulled along the axes c of its grains, the polycrystal twins, and the "
             "lattices of its points become those of their predominant twins, grain by "
             "grain (86 degrees off: the polycrystal hardens), the turned volume of each "
             "grain following its twinned volume; brought back, the twins shear back at "
             "about the stress of the tension (detwinning), and the lattices turn back to "
             "those of their grains."),
            ("homogenization_cleavage", "The cleavage of a polycrystal of iron at a low "
             "temperature: the {100} planes of its grains open at their strength and "
             "soften (a crack band of one voxel), and a crack crosses the grains from "
             "plane to plane. With a softer slip the grains flow first: slip shields "
             "cleavage."),
            ("homogenization_slip_cleavage", "Cleavage that the slip nucleates: a "
             "polycrystal of iron whose grains slip far below the strength of their {100} "
             "planes. The slip of a crystal lowers that strength, and the damage of the "
             "planes eases the slip in return: the polycrystal yields, hardens, and breaks "
             "across a section once its grains have slipped enough, under a lower stress "
             "and at a smaller strain than with planes of a constant strength."),
            ("homogenization_cleavage_band", "Cleavage across a band of a given width: "
             "the {100} planes of a columnar polycrystal of iron regularized by their "
             "microdamage, a field that follows their damage over a length. A plane "
             "breaks across a band five voxels wide, whatever the grid, where the "
             "fracture energy of the planes is dissipated; the crack crosses the grains in "
             "a band that winds a little from one to the next, and the planes of its "
             "middle are broken through."),
            ("homogenization_crack_closure", "A cleavage crack that closes: a polycrystal "
             "of iron pulled until a crack has crossed its grains, then pushed. The "
             "opening of the cleavage planes goes back under a low pressure, and once "
             "the crack is closed the polycrystal carries the pressure as the sound one, "
             "its lips rubbing on each other (the unilateral condition of the planes, "
             "Coulomb's friction)."),
            ("homogenization_beremin", "The probability of cleavage of a ferritic steel "
             "(the local approach of Beremin): the Weibull stress of the voxels of a "
             "polycrystal that have yielded, each driven by the normal stress of the "
             "{100} planes of its grain, gives the probability that the weakest link of "
             "a tensile bar has failed, from nothing at the yield stress to a certainty "
             "as the steel hardens."),
            ("homogenization_fatigue", "Pores in a cast alloy under a tension that "
             "alternates: Dang Van's criterion of multiaxial fatigue on the stresses of "
             "a cycle maps the utilization of each point, the largest at the equators of "
             "the pores, and gives the factor of safety of the cell."),
            ("homogenization_debonding", "Glass beads that debond from their matrix: "
             "cohesive interfaces between the phases (a strength, a fracture energy, "
             "Coulomb's friction once broken). The composite softens to a matrix with "
             "holes in tension and finds most of its stiffness again in compression."),
        ],
    ),
    (
        "Fracture",
        "Cells that break along a loading path, by the phase-field method: cracks start, "
        "run and merge in the phases without being meshed or tracked; a planar section of "
        "the volume (`voxels.section`) makes them quick.",
        [
            ("fracture_composite", "Glass beads in a brittle epoxy pulled until a crack "
             "crosses the cell: the strength and the work of fracture of the composite."),
            ("fracture_rubber", "A perforated rubber sheet that tears at a stretch of 1.45, "
             "in finite strains: the ligaments between the holes break one after another."),
        ],
    ),
    (
        "Multiphysics: transport, couplings, harmonic regimes, flow and sound",
        "The other physics of the homogenization module on the same microstructures: "
        "transport of a scalar potential, weak and strong couplings, harmonic regimes, the "
        "flow of a fluid through the pores and the acoustics of porous media, and the "
        "extended finite elements that cut a regular grid by the interfaces "
        "([guide to the physics](physics.md)).",
        [
            ("homogenization_transport", "Thermal and electrical conductivities, "
             "permittivity and diffusivity of conducting, impermeable spheres in a polymer "
             "(FFT), with their bounds."),
            ("homogenization_piezoelectric", "A piezoelectric composite of barium titanate "
             "particles in a polymer: the displacement and the electric potential solved "
             "together, the constants of the effective law."),
            ("homogenization_viscoelastic", "Glass beads in a viscoelastic resin: storage "
             "and loss moduli over five decades of frequency, and the relaxation in time."),
            ("homogenization_poroelastic", "The poroelastic law of a porous solid: drained "
             "stiffness, tensor and modulus of Biot, undrained response with water in the "
             "pores."),
            ("homogenization_xfem", "Extended finite elements: a regular grid cut by the "
             "level set of the spheres, elastic and thermal properties without a mesher."),
            ("homogenization_acoustic_foam", "The flow and the sound in an open-cell foam: "
             "Darcy's permeability by a Stokes flow through the pores, and the parameters of "
             "its equivalent fluid (tortuosity, characteristic lengths)."),
        ],
    ),
    (
        "RVE studies",
        "The size of the representative volume element by the ensemble and window "
        "methods.",
        [
            ("rve_poisson_fibers", "The variance of the volume fraction and specific "
             "surface of Poisson fibres at five sizes (ensemble method)."),
            ("rve_poisson_fibers_fe", "The RVE sizes of the elastic and thermal properties "
             "of Poisson fibres, KUBC and MBC (finite elements)."),
            ("rve_voronoi_windows", "The window method on one large two-phase Voronoi "
             "mosaic."),
            ("rve_cellular_automaton", "The window method on a structure grown by a "
             "cellular automaton."),
            ("rve_crystal_plasticity_fft", "The RVE sizes of the elastic and viscoplastic "
             "responses of a polycrystal (Yang et al. 2019; FFT)."),
            ("rve_multiphysics", "The RVE sizes of several physics at once: bulk modulus, "
             "conductivity, swelling, loss modulus and the stress of a loading path of the "
             "same realizations (FFT)."),
        ],
    ),
    (
        "Topology optimization",
        "Periodic cells optimized for their homogenized properties; the images show their "
        "designs.",
        [
            ("optimize_bulk", "A 3D cell of maximal bulk modulus at 30 % of solid "
             "(method of moving asymptotes)."),
            ("optimize_auxetic", "A planar cell of negative Poisson's ratio, near -0.87 "
             "(method of moving asymptotes)."),
            ("optimize_auxetic_3d", "A 3D cell of negative Poisson's ratio along its three "
             "axes, near -0.14 on 24^3 voxels (method of moving asymptotes)."),
            ("optimize_conductivity", "A 3D cell of maximal conductivity at 30 % of solid "
             "(hybrid cellular automaton)."),
        ],
    ),
    (
        "Laser-architectured sheets",
        "Steel sheets architectured by localized laser treatment, their tensile test and the "
        "optimization of the laser path (`morphogen laser`); the images show the hardness "
        "of the parallel part of the specimen through its thickness (the mean, the upper "
        "and the lower layers) and the axes of the tracks.",
        [
            ("laser_corrugated", "Corrugated tracks of martensite in an annealed 0.23 C "
             "steel by a double-sided laser (Wang 2022), and their amplitude and period "
             "optimized for the strength-ductility balance (Bayesian optimization)."),
            ("laser_bending", "A strip hardened on one face by longitudinal tracks in "
             "three-point bending: the number of tracks and the power optimized for the "
             "load at a deflection of 1 mm within a budget of laser energy."),
            ("laser_cells", "The corrugated track of Wang et al. (2023) drawn by a 1 mm spot "
             "on cells of 0.5 mm and changed by a cellular automaton (neighbour exchanges "
             "tested by finite elements) for the failure strain at a tensile strength of "
             "700 MPa, the martensite brittle; its tracks treated by the laser."),
        ],
    ),
    (
        "Couplings and the whole chain",
        "Transient diffusion in a microstructure, and every step from the model to the "
        "figures and videos.",
        [
            ("diffusion", "Moisture uptake of a composite of hollow flax-like fibres "
             "exposed on two faces."),
            ("workflow", "The whole chain: model, voxels, conforming mesh, homogenization, "
             "a loading path, tables, figures and videos."),
        ],
    ),
]  # fmt: skip


def templates() -> list[str]:
    """The names of the templates of morphoGen."""
    return sorted(p.stem for p in TEMPLATES.glob("*.yaml"))


def config(name: str) -> dict:
    """The mapping of a template."""
    return yaml.safe_load((TEMPLATES / f"{name}.yaml").read_text(encoding="utf-8"))


def is_laser(spec: dict) -> bool:
    return "laser_sheet" in spec


def is_optimization(spec: dict) -> bool:
    return "optimization" in spec and "model" not in spec and not is_laser(spec)


def commands(name: str, spec: dict) -> list[str]:
    """The commands that reproduce a template: its file, then its run."""
    out = [f"morphogen template {name} -o {name}.yaml"]
    if is_laser(spec):
        out.append(f"morphogen laser {name}.yaml")
    elif is_optimization(spec):
        out.append(f"morphogen optimize {name}.yaml")
    elif spec.get("seed") is None and name in RANDOM_WITHOUT_SEED:
        out.append(f"morphogen run {name}.yaml --seed {SEED}")
    else:
        out.append(f"morphogen run {name}.yaml")
    return out


def needs(spec: dict) -> list[str]:
    """The optional parts that a template needs besides the core."""
    outputs = spec.get("outputs") or {}
    out = []
    homogenization = spec.get("homogenization") or {}
    if "mesh" in spec or "conforming_mesh" in outputs:
        if homogenization.get("solver") == "fenicsx":
            out.append("FEniCSx")
        out.append("the mesher MMG (extra `mesh`)")
    if "surface" in outputs:
        out.append("scikit-image (extra `surface`)")
    if "visualization" in spec:
        out.append("VTK and ffmpeg (extra `video`)")
    return out


# -- the page ---------------------------------------------------------------------------------

HEADER = """\
# Gallery of examples

Each card below is one of the example configurations of morphoGen, its *templates*: the
image of its first realization, a line of description, the commands that reproduce it and
the configuration itself, to change at will. `morphogen template` lists them, `morphogen
template NAME -o NAME.yaml` writes one, `morphogen run NAME.yaml` runs it (`morphogen
optimize` for the topology optimizations, `morphogen laser` for the laser-architectured
sheets), and the graphical interface (`morphogen gui`)
opens the form of each model filled with its template. The [tutorial](tutorial.md) walks
through a whole study; the [guide](guide.md) describes the configurations.

The images show the realization that `morphogen run` makes first (the realization 0 of the
seed of the template), in its domain: three faces of its box, each the section of the
microstructure computed in its plane, as the interface draws them (anti-aliased
interfaces, the colours of the phases, a colour per grain of a polycrystal). A planar
structure shows its x-y section, a template that studies a planar section of its volume
(`voxels.section`) that section, an optimization its design, and the thin structures that
the faces barely cut (the vascular tree, the aggregates, the networks of struts or fibres
and the open foams) the surface of their phase 1 in 3D. Some cards show a second
figure, what the run of the template computes: the curve of its loading path, or the
damage of a cell that broke, drawn from the files of the run. They are written by
`python docs/gallery/make_gallery.py`, which writes this page too.
"""


def card(name: str, text: str) -> str:
    """The card of a template (MyST Markdown)."""
    spec = config(name)
    lines = "\n".join(commands(name, spec))
    extra = needs(spec)
    note = f"\n\n*Needs {', '.join(extra)}.*" if extra else ""
    result = ""
    if name in RESULTS:
        result = f"""
```{{image}} _static/gallery/{name}_result.png
:alt: the result of {name}
:loading: lazy
```
"""
    return f"""\
:::{{container}} gallery-card
:name: gallery-{name.replace("_", "-")}

```{{image}} _static/gallery/{name}.png
:alt: {name}
:loading: lazy
```
{result}
**`{name}`** — {text}{note}

```bash
{lines}
```

<details><summary>Configuration</summary>

```{{literalinclude}} ../src/morphogen/templates/{name}.yaml
:language: yaml
```

</details>
:::
"""


def page() -> str:
    """The text of docs/gallery.md."""
    parts = [HEADER]
    for title, text, cards in FAMILIES:
        parts.append(f"## {title}\n\n{text}\n\n::::{{container}} gallery\n")
        parts.extend(card(name, line) for name, line in cards)
        parts.append("::::\n")
    return "\n".join(parts)


def gallery_names() -> list[str]:
    """The templates of the cards, in their order."""
    return [name for _, _, cards in FAMILIES for name, _ in cards]


# -- the images -------------------------------------------------------------------------------


def _domain_of(cfg):
    """The domain and the key of the seed of the first realization that ``morphogen run``
    makes (the largest size of an ensemble study)."""
    from morphogen.domain import Domain

    rve = cfg.rve
    if rve is None or rve.method != "ensemble":
        return cfg.domain, (0,)
    size = float(rve.sizes[-1])
    domain = Domain(
        size=(size,) * 3, origin=cfg.domain.origin, periodic=cfg.domain.periodic,
        unit=cfg.domain.unit,
    )  # fmt: skip
    return domain, (len(rve.sizes) - 1, 0)


def realization(name: str):
    """The microstructure of the image of a template: its first realization, or the
    design of an optimization."""
    from morphogen.config import load_config
    from morphogen.seeding import derive_seed

    path = TEMPLATES / f"{name}.yaml"
    spec = config(name)
    if is_optimization(spec):
        from morphogen.optimization.run import load_optimization_config

        cfg = load_optimization_config(path)
        design = cfg.optimization.run(np.random.default_rng(cfg.seed))
        return design.to_microstructure()
    cfg = load_config(path)
    domain, key = _domain_of(cfg)
    seed = cfg.seed if cfg.seed is not None else SEED
    micro = cfg.model.generate(domain, derive_seed(seed, *key))
    if cfg.seed is None and name not in RANDOM_WITHOUT_SEED:  # deterministic, as stated
        other = cfg.model.generate(domain, derive_seed(seed + 1, *key))
        # on 64^3 voxels: 24^3 missed the random height of the layers of nacre
        if not np.array_equal(micro.voxelize(shape=64).data, other.voxelize(shape=64).data):
            raise RuntimeError(f"{name}: a random model without a seed: add it to "
                               "RANDOM_WITHOUT_SEED")  # fmt: skip
    return micro


def _colour_kind(micro, cuts) -> str:
    """How the sections are coloured: ``grains`` (a colour per grain in phase 1, the
    polycrystals), ``labels`` (a colour per cell: mosaics of one phase) or ``phase``."""
    if all(s.labels is None for s in cuts):
        return "phase"
    if getattr(micro, "orientations", None) is not None:
        return "grains"
    if all(np.all(s.phase == s.phase.flat[0]) for s in cuts):
        return "labels"
    return "phase"


def _rgb(sec, kind: str) -> np.ndarray:
    """The colours of a section (float, 0 to 255)."""
    from morphogen.viz import sections as vs

    phase = vs.render(sec, "phase").astype(np.float32)
    if kind == "phase" or sec.labels is None:
        return phase
    cells = vs.render(sec, "labels", contour=True).astype(np.float32)
    if kind == "labels":
        return cells
    return np.where((sec.phase == 1)[..., None], cells, phase)


#: The faces seen: the normal axis, the position of the plane (0: lower face, 1: upper
#: face), the corner of the face and its two edges along the axes of the plane, as
#: multiples of the lengths of the box, and the shading of the face
FACES = (
    (2, 1.0, (0, 0, 1), (1, 0, 0), (0, 1, 0), 1.0),  # top: x, y
    (1, 0.0, (0, 0, 0), (1, 0, 0), (0, 0, 1), 0.86),  # front: x, z
    (0, 1.0, (1, 0, 0), (0, 1, 0), (0, 0, 1), 0.72),  # right: y, z
)
#: Colour of the edges of the box
EDGE = np.array([40.0, 40.0, 40.0])


def _projection(azimuth: float = 38.0, elevation: float = 28.0) -> np.ndarray:
    """The rows of the orthographic projection to the screen (right, up) of a view from
    the side x+, y-, z+."""
    a, e = np.radians(azimuth), np.radians(elevation)
    eye = np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])
    right = np.cross([0.0, 0.0, 1.0], eye)
    right /= np.linalg.norm(right)
    up = np.cross(eye, right)
    return np.array([right, up])


def _axonometry(micro, canvas, resolution: int) -> np.ndarray:
    """RGBA (uint8) of three faces of the box of a microstructure in axonometry."""
    from morphogen.viz import sections as vs

    lengths = np.asarray(micro.domain.lengths, dtype=float)
    cuts = [vs.section(micro, axis, fraction, resolution) for axis, fraction, *_ in FACES]
    kind = _colour_kind(micro, cuts)
    proj = _projection()
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) * lengths
    screen = corners @ proj.T
    lo, hi = screen.min(axis=0), screen.max(axis=0)
    w, h = canvas[0] * SUPERSAMPLE, canvas[1] * SUPERSAMPLE
    margin = 0.03
    scale = min(w * (1 - 2 * margin) / (hi[0] - lo[0]), h * (1 - 2 * margin) / (hi[1] - lo[1]))
    centre = 0.5 * (lo + hi)
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    pu = centre[0] + (cols - 0.5 * w) / scale  # screen coordinates of the pixel centres
    pv = centre[1] - (rows - 0.5 * h) / scale
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    alpha = np.zeros((h, w), dtype=np.float32)
    line = 1.1 * SUPERSAMPLE  # pixels of the edges
    for sec, (_, _, corner, ea, eb, shade) in zip(cuts, FACES, strict=True):
        o = proj @ (np.asarray(corner) * lengths)
        a = proj @ (np.asarray(ea) * lengths)
        b = proj @ (np.asarray(eb) * lengths)
        inverse = np.linalg.inv(np.column_stack([a, b]))
        s = inverse[0, 0] * (pu - o[0]) + inverse[0, 1] * (pv - o[1])
        t = inverse[1, 0] * (pu - o[0]) + inverse[1, 1] * (pv - o[1])
        inside = (s >= 0) & (s < 1) & (t >= 0) & (t < 1)
        if not inside.any():
            continue
        colours = _rgb(sec, kind)
        n_rows, n_cols = colours.shape[:2]
        col = np.clip((s[inside] * n_cols).astype(int), 0, n_cols - 1)
        row = np.clip(((1.0 - t[inside]) * n_rows).astype(int), 0, n_rows - 1)
        face = colours[row, col] * shade
        area = abs(a[0] * b[1] - a[1] * b[0]) * scale**2
        across = np.minimum(  # distance to the edges of the face, in pixels
            np.minimum(s[inside], 1 - s[inside]) * area / (np.linalg.norm(b) * scale),
            np.minimum(t[inside], 1 - t[inside]) * area / (np.linalg.norm(a) * scale),
        )
        edge = np.clip(line - across, 0.0, 1.0)[:, None]
        rgb[inside] = face + edge * (EDGE - face)
        alpha[inside] = 1.0
    return _downsampled(rgb, alpha)


def _downsampled(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """RGBA (uint8) of a supersampled image, averaged by blocks (colours weighted by their
    opacity)."""
    f = SUPERSAMPLE
    h, w = alpha.shape[0] // f, alpha.shape[1] // f
    a = alpha[: h * f, : w * f].reshape(h, f, w, f).mean(axis=(1, 3))
    weighted = (rgb * alpha[..., None])[: h * f, : w * f].reshape(h, f, w, f, 3).sum(axis=(1, 3))
    count = alpha[: h * f, : w * f].reshape(h, f, w, f).sum(axis=(1, 3))
    colour = weighted / np.maximum(count, 1e-9)[..., None]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[..., :3] = np.clip(np.rint(colour), 0, 255)
    out[..., 3] = np.clip(np.rint(255 * a), 0, 255)
    return out


def _flat(micro, canvas, axis: int = 2, fraction: float = 0.5) -> np.ndarray:
    """RGBA (uint8) of a section of a structure fitted in the canvas: the x-y section of a
    planar structure, or the section normal to ``axis`` at ``fraction`` of the box."""
    from morphogen.models.planar import plane_axes
    from morphogen.viz import sections as vs

    u, v = plane_axes(axis)
    lengths = np.asarray(micro.domain.lengths, dtype=float)[[u, v]]
    width, height = canvas
    fit = min(width / lengths[0], height / lengths[1])
    n = round(max(lengths[0], lengths[1]) * fit)
    sec = vs.section(micro, axis, fraction, n)
    colours = _rgb(sec, _colour_kind(micro, [sec]))
    rows, cols = colours.shape[:2]
    out = np.zeros((height, width, 4), dtype=np.uint8)
    top, left = (height - rows) // 2, (width - cols) // 2
    out[top : top + rows, left : left + cols, :3] = np.clip(np.rint(colours), 0, 255)
    out[top : top + rows, left : left + cols, 3] = 255
    return out


def _surface(micro, canvas) -> np.ndarray:
    """RGBA (uint8) of the surface of the phase 1 of a microstructure seen in 3D, on a
    transparent background (:func:`morphogen.viz.video.turntable_frames`: VTK off screen,
    on :data:`SURFACE_VOXELS` voxels along the longest edge)."""
    from morphogen.image import VoxelImage
    from morphogen.viz.video import turntable_frames

    voxels = micro.voxelize(shape=SURFACE_VOXELS)
    # the axes (x, z, -y), so that the camera of the turntable, which looks along -z with
    # y up before turning, sees the box from the side x+, y-, z+ with z up, as the faces
    data = np.flip(np.transpose(voxels.data, (0, 2, 1)), axis=2)
    sx, sy, sz = voxels.spacing
    turned = VoxelImage(np.ascontiguousarray(data), spacing=(sx, sz, sy))
    size = (canvas[0] * SUPERSAMPLE, canvas[1] * SUPERSAMPLE)
    frames = turntable_frames(turned, phase=1, frames=1, size=size, elevation=28.0, azimuth=38.0)
    rgb = frames[0].astype(np.float32)
    alpha = (rgb.min(axis=2) < 255).astype(np.float32)  # the white background
    return _downsampled(rgb, alpha)


def image(micro, canvas=CANVAS, *, surface: bool = False, section=None) -> np.ndarray:
    """RGBA (uint8) of the image of a microstructure: three faces of its box, the x-y
    section of a planar structure (thinner than a twentieth of its width, as the
    interface takes them), with ``surface`` the surface of its phase 1 in 3D, or with
    ``section`` (the axis and the position of a plane, as a fraction of the box) that
    section: the one that a template with ``voxels.section`` studies."""
    lengths = np.asarray(micro.domain.lengths, dtype=float)
    if surface:
        return _surface(micro, canvas)
    if section is not None:
        return _flat(micro, canvas, *section)
    if lengths[2] < 0.05 * lengths[:2].max():
        return _flat(micro, canvas)
    resolution = int(1.6 * max(canvas))
    return _axonometry(micro, canvas, resolution)


def save_png(rgba: np.ndarray, path: Path, colours: int = COLOURS) -> int:
    """Write an RGBA image as a PNG of at most ``colours`` colours (Pillow); its size in
    bytes."""
    from PIL import Image

    picture = Image.fromarray(rgba, "RGBA").quantize(
        colours, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    picture.save(path, optimize=True)
    return path.stat().st_size


def laser_image(name: str, canvas=CANVAS) -> np.ndarray:
    """RGBA of a laser template: the hardness of the parallel part of its specimen under
    the pattern of the template (the mean through the thickness, the upper and the lower
    layers, stacked), the axes of the tracks in cyan."""
    from matplotlib import colormaps

    from morphogen.laser.run import hardness_map, load_laser_config

    cfg = load_laser_config(TEMPLATES / f"{name}.yaml")
    case = cfg.case
    path = case.path()
    maps = hardness_map(case, path, resolution=0.05)
    h = maps["hardness"]
    views = [h.mean(axis=2), h[:, :, 0], h[:, :, -1]]
    lo, hi = float(h.min()), float(h.max())
    width, height = canvas
    nx, ny = views[0].shape
    strip = round(width * ny / nx)
    gap = max(4, (height - 3 * strip) // 4)
    out = np.zeros((height, width, 4), dtype=np.uint8)
    cmap = colormaps["inferno"]
    xi = np.minimum((np.arange(width) + 0.5) * nx / width, nx - 1).astype(int)
    yi = np.minimum((np.arange(strip) + 0.5) * ny / strip, ny - 1).astype(int)
    x0, x1, y0, y1 = case.specimen.band()
    top = gap
    for view in views:
        img = (view[np.ix_(xi, yi[::-1])].T - lo) / max(hi - lo, 1e-9)
        rgba = (255 * cmap(img)).astype(np.uint8)
        for t in path:  # the axes of the tracks
            pts = t.points
            u = np.linspace(0.0, 1.0, 4 * width)
            seg = (
                np.interp(u * (len(pts) - 1), np.arange(len(pts)), pts[:, 0]),
                np.interp(u * (len(pts) - 1), np.arange(len(pts)), pts[:, 1]),
            )
            col = ((seg[0] - x0) / (x1 - x0) * width).astype(int)
            row = ((y1 - seg[1]) / (y1 - y0) * strip).astype(int)
            ok = (col >= 0) & (col < width) & (row >= 0) & (row < strip)
            rgba[row[ok], col[ok]] = (64, 224, 255, 255)
        out[top : top + strip] = rgba
        top += strip + gap
    return out


# -- the results of the runs -------------------------------------------------------------------


def run_template(name: str, folder: Path) -> Path:
    """A template run as ``morphogen run`` does, in ``folder``: its output directory."""
    from morphogen import pipeline
    from morphogen.config import load_config

    spec = config(name)
    spec["outputs"]["directory"] = str(folder / name)
    if spec.get("seed") is None and name in RANDOM_WITHOUT_SEED:
        spec["seed"] = SEED
    return pipeline.run(load_config(spec))


def _rows(path: Path) -> dict[str, np.ndarray]:
    """The columns of a table of curves (.csv) as arrays."""
    import csv

    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if rows[0][k] != ""}


def _driven(spec: dict) -> tuple[str, dict, str]:
    """The name of the first loading path of a template, its mapping and the component
    of the strain that it drives the furthest."""
    name, path = next(iter(spec["homogenization"]["paths"].items()))
    component = max(path["strain"], key=lambda c: np.abs(np.atleast_1d(path["strain"][c])).max())
    return name, path, component


def _axes(ax, xlabel: str, ylabel: str) -> None:
    """Axes drawn in an ink legible on light and dark pages, on a transparent ground."""
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK, labelsize=12)
    ax.set_xlabel(xlabel, color=INK, fontsize=13)
    ax.set_ylabel(ylabel, color=INK, fontsize=13)
    ax.grid(True, color=INK, alpha=0.25, lw=0.6)


def _curve(ax, name: str, outdir: Path, options: dict) -> None:
    """The stress of the first path of a template against its strain (its stretch, or
    the time), a colour per segment of the path."""
    from matplotlib.ticker import MaxNLocator

    spec = config(name)
    path_name, path, component = _driven(spec)
    stem = spec["outputs"]["name"]
    table = _rows(next(outdir.glob(f"{stem}_*_{path_name}_periodic.csv")))
    strain, stress, times = table[f"E{component}"], table[f"S{component}"], table["time"]
    against_time = bool(options.get("time"))
    x = times if against_time else 1.0 + strain if options.get("stretch") else strain
    ends = [float(t) for t in path["times"]]
    labels = options.get("labels") or [None] * (len(ends) - 1)
    colours: dict = {}  # of the segments drawn so far, by their label
    for k, (t0, t1) in enumerate(itertools.pairwise(ends)):
        sel = (times >= t0 - 1e-12) & (times <= t1 + 1e-12)
        if against_time:
            sel &= times > 0.0  # a logarithmic axis
        label = labels[k]
        new = label is None or label not in colours
        colour = colours.setdefault(label, CURVES[len(colours) % len(CURVES)])
        ax.plot(x[sel], stress[sel], "-o", ms=3.5, lw=2.2, color=colour,
                label=label if new else None)  # fmt: skip
    if any(labels):
        legend = ax.legend(frameon=False, fontsize=12, loc="best" if against_time else "upper left")
        for text in legend.get_texts():
            text.set_color(INK)
    if against_time:
        ax.set_xscale("log")
        _axes(ax, "time", f"stress {component}")
        return
    what = "stretch" if options.get("stretch") else "strain"
    kind = "nominal stress" if options.get("stretch") else "stress"
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))  # small strains: long tick labels
    _axes(ax, f"{what} {component}", f"{kind} {component}")


def _damage(ax, name: str, outdir: Path) -> None:
    """The damage at the end of the path of a template that breaks, on its phases: the
    colours of the phases, and the broken voxels from red to yellow."""
    from matplotlib import colormaps

    from morphogen import io
    from morphogen.config import load_config
    from morphogen.viz.png import PALETTE

    spec = config(name)
    stem = spec["outputs"]["name"]
    last = sorted(outdir.glob(f"{stem}_*_periodic_*.vti"))[-1]
    fields = {im.name: im for im in io.read_vti(last, names=["damage"])}
    damage = np.asarray(fields["damage"].data, dtype=float)
    phases = np.asarray(load_config(spec).voxels.image(realization(name)).data)
    axis = int(np.argmin(damage.shape))  # the normal of the section
    d = np.take(damage, damage.shape[axis] // 2, axis=axis)
    ph = np.take(phases, phases.shape[axis] // 2, axis=axis)
    rgb = PALETTE[np.minimum(ph, len(PALETTE) - 1)].astype(float) / 255.0
    hot = colormaps["autumn"](np.clip(d, 0.0, 1.0))[..., :3]
    alpha = np.clip(1.6 * d, 0.0, 1.0)[..., None]
    picture = rgb + alpha * (hot - rgb)
    ax.imshow(np.transpose(picture, (1, 0, 2)), origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(INK)
    ax.set_title("damage at the end", color=INK, fontsize=13)


def _fatigue(left, right, name: str, outdir: Path) -> None:
    """The fatigue of a cycle of a template: the utilization of the elements of a slice
    through the worst point (on the left), and the share of the volume above each
    utilization (on the right)."""
    from matplotlib import colormaps
    from matplotlib.collections import PolyCollection

    from morphogen import io

    spec = config(name)
    stem = spec["outputs"]["name"]
    data = io.read_vtu(next(outdir.glob(f"{stem}_*_fatigue.vtu")))
    points, cells = data["points"], data["cells"][:, :4]
    value = np.asarray(data["cell_data"]["fatigue_utilization"], dtype=float)
    corners = points[cells]
    centres = corners.mean(axis=1)
    edges = corners[:, 1:] - corners[:, :1]
    volume = np.abs(np.einsum("ei,ei->e", edges[:, 0], np.cross(edges[:, 1], edges[:, 2]))) / 6
    judged = np.isfinite(value)
    worst = int(np.nanargmax(value))
    # a slice across the second axis through the worst element: the load is horizontal
    size = float(np.cbrt(np.median(volume)))
    near = judged & (np.abs(centres[:, 1] - centres[worst, 1]) < 0.75 * size)
    faces = corners[near][:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]][..., [0, 2]]
    colours = colormaps["magma"](np.repeat(value[near] / np.nanmax(value), 4))
    left.add_collection(PolyCollection(faces.reshape(-1, 3, 2), facecolors=colours,
                                       edgecolors="none"))  # fmt: skip
    left.set_xlim(points[:, 0].min(), points[:, 0].max())
    left.set_ylim(points[:, 2].min(), points[:, 2].max())
    left.set_aspect("equal")
    left.set_xticks([])
    left.set_yticks([])
    left.set_facecolor("none")
    for side in left.spines.values():
        side.set_color(INK)
    left.set_title(f"utilization (up to {np.nanmax(value):.2f})", color=INK, fontsize=13)
    order = np.argsort(value[judged])
    sorted_values = value[judged][order]
    above = 1.0 - np.cumsum(volume[judged][order]) / volume[judged].sum()
    keep = above > 0
    right.plot(sorted_values[keep], above[keep], lw=2.2, color=CURVES[0])
    right.set_yscale("log")
    right.set_ylim(1e-6, 2.0)
    _axes(right, "utilization", "share of the volume above")


def _twinning(left, right, name: str, outdir: Path) -> None:
    """The paths of a template whose crystals twin: the magnitude of the stress of each
    against that of its strain (on the left), and the twinned fraction (on the right)."""
    from matplotlib.ticker import MaxNLocator

    spec = config(name)
    stem = spec["outputs"]["name"]
    for k, (path_name, path) in enumerate(spec["homogenization"]["paths"].items()):
        component = max(
            path["strain"], key=lambda c: np.abs(np.atleast_1d(path["strain"][c])).max()
        )
        table = _rows(next(outdir.glob(f"{stem}_*_{path_name}_periodic.csv")))
        strain = np.abs(table[f"E{component}"])
        left.plot(strain, np.abs(table[f"S{component}"]), "-o", ms=3.5, lw=2.2, color=CURVES[k],
                  label=path_name)  # fmt: skip
        right.plot(strain, table["twin_fraction"], "-o", ms=3.5, lw=2.2, color=CURVES[k])
    legend = left.legend(frameon=False, fontsize=12, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK)
    for ax, label in ((left, "|stress|"), (right, "twinned fraction")):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        _axes(ax, "|strain|", label)


def _cycle(left, right, name: str, outdir: Path, options: dict) -> None:
    """The path of a template whose load is reversed: its stress against its strain (on
    the left), and a quantity of its curve against the strain (``options["quantity"]``,
    on the right; dashed, another one: ``options["also"]``, the two named by
    ``options["legend"]``), the way there and the way back in two colours."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    spec = config(name)
    stem = spec["outputs"]["name"]
    path_name, path = next(iter(spec["homogenization"]["paths"].items()))
    component = max(path["strain"], key=lambda c: np.abs(np.atleast_1d(path["strain"][c])).max())
    table = _rows(next(outdir.glob(f"{stem}_*_{path_name}_periodic.csv")))
    strain = table[f"E{component}"]
    turn = int(np.argmax(strain))  # the end of the way there
    parts = ((slice(0, turn + 1), "there"), (slice(turn, None), "back"))
    for k, (part, label) in enumerate(parts):
        left.plot(strain[part], table[f"S{component}"][part], "-o", ms=3.0, lw=2.2,
                  color=CURVES[k], label=label)  # fmt: skip
        right.plot(strain[part], table[options["quantity"]][part], "-o", ms=3.0, lw=2.2,
                   color=CURVES[k])  # fmt: skip
        if options.get("also"):
            right.plot(strain[part], table[options["also"]][part], "--", lw=1.8, color=CURVES[k])
    left.axhline(0.0, color=INK, lw=0.6, alpha=0.5)
    legends = [left.legend(frameon=False, fontsize=12, loc="best")]
    if options.get("also"):
        handles = [Line2D([], [], color=INK, lw=2.2, marker="o", ms=3.0),
                   Line2D([], [], color=INK, lw=1.8, ls="--")]  # fmt: skip
        legends.append(right.legend(handles, options["legend"], frameon=False, fontsize=11,
                                    loc="lower right", handlelength=1.4))  # fmt: skip
    for legend in legends:
        for text in legend.get_texts():
            text.set_color(INK)
    for ax, label in ((left, "stress"), (right, options["label"])):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        _axes(ax, "strain", label)


def _beremin(left, right, name: str, outdir: Path) -> None:
    """The path of a template judged by the criterion of Beremin: its stress and its
    Weibull stress against its strain (on the left), the probability of failure (on the
    right)."""
    from matplotlib.ticker import MaxNLocator

    spec = config(name)
    path_name, _, component = _driven(spec)
    stem = spec["outputs"]["name"]
    table = _rows(next(outdir.glob(f"{stem}_*_{path_name}_periodic.csv")))
    strain = table[f"E{component}"]
    left.plot(strain, table[f"S{component}"], "-o", ms=3.5, lw=2.2, color=CURVES[0],
              label=f"stress {component}")  # fmt: skip
    left.plot(strain, table["beremin_weibull"], "-o", ms=3.5, lw=2.2, color=CURVES[2],
              label="Weibull stress")  # fmt: skip
    legend = left.legend(frameon=False, fontsize=12, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK)
    right.plot(strain, table["beremin_probability"], "-o", ms=3.5, lw=2.2, color=CURVES[2])
    right.set_ylim(-0.03, 1.03)
    for ax, label in ((left, "stress"), (right, "probability of failure")):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        _axes(ax, f"strain {component}", label)


def _field(ax, name: str, outdir: Path, options: dict) -> None:
    """A field of the voxels at the end of the path of a template (``options["field"]``,
    between 0 and 1), on a section through the middle of the cell across the load (normal
    to the second axis, or to ``options["normal"]``)."""
    from matplotlib import colormaps

    from morphogen import io

    spec = config(name)
    stem = spec["outputs"]["name"]
    last = sorted(outdir.glob(f"{stem}_*_periodic_*.vti"))[-1]
    fields = {im.name: im for im in io.read_vti(last, names=[options["field"]])}
    values = np.asarray(fields[options["field"]].data, dtype=float)
    normal = options.get("normal", 1)  # by default, the plane of the axes 1 and 3
    section = np.take(values, values.shape[normal] // 2, axis=normal)
    ax.imshow(colormaps["magma"](np.clip(section.T, 0.0, 1.0)), origin="lower",
              interpolation="nearest")  # fmt: skip
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(INK)
    ax.set_title(options.get("title", options["field"]), color=INK, fontsize=13)


def result_image(name: str, outdir: Path, canvas=RESULT_CANVAS) -> np.ndarray:
    """RGBA of the figure of the result of a template of :data:`RESULTS`, from the files
    of its run in ``outdir``."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    options = RESULTS[name]
    fig = Figure(figsize=(canvas[0] / 100.0, canvas[1] / 100.0), dpi=100, layout="constrained")
    fig.patch.set_alpha(0.0)
    if options["kind"] == "fracture":
        left, right = fig.subplots(1, 2, width_ratios=(1.0, 1.25))
        _damage(left, name, outdir)
        _curve(right, name, outdir, options)
    elif options["kind"] == "fatigue":
        left, right = fig.subplots(1, 2, width_ratios=(1.0, 1.25))
        _fatigue(left, right, name, outdir)
    elif options["kind"] == "twinning":
        left, right = fig.subplots(1, 2)
        _twinning(left, right, name, outdir)
    elif options["kind"] == "cycle":
        left, right = fig.subplots(1, 2)
        _cycle(left, right, name, outdir, options)
    elif options["kind"] == "beremin":
        left, right = fig.subplots(1, 2)
        _beremin(left, right, name, outdir)
    elif options["kind"] == "field":
        left, right = fig.subplots(1, 2, width_ratios=(1.0, 1.25))
        _field(left, name, outdir, options)
        _curve(right, name, outdir, options)
    else:
        _curve(fig.subplots(), name, outdir, options)
    drawn = FigureCanvasAgg(fig)
    drawn.draw()
    return np.asarray(drawn.buffer_rgba()).copy()


def render_result(name: str) -> tuple[int, float]:
    """The figure of the result of a template written (the template run in a temporary
    folder); its size in bytes and the time taken."""
    import tempfile

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as folder:
        outdir = run_template(name, Path(folder))
        size = save_png(result_image(name, outdir), IMAGES / f"{name}_result.png")
    return size, time.perf_counter() - t0


def render(name: str) -> tuple[int, float]:
    """The image of a template written; its size in bytes and the time taken."""
    t0 = time.perf_counter()
    if is_laser(config(name)):
        size = save_png(laser_image(name), IMAGES / f"{name}.png")
        return size, time.perf_counter() - t0
    micro = realization(name)
    rgba = image(micro, surface=name in SURFACE, section=_studied_section(name, micro))
    size = save_png(rgba, IMAGES / f"{name}.png")
    return size, time.perf_counter() - t0


def _studied_section(name: str, micro):
    """The plane that a template with ``voxels.section`` studies (its axis, and the
    position of the centres of its layer of voxels as a fraction of the box), or None."""
    from morphogen.config import load_config

    spec = config(name)
    if is_laser(spec) or is_optimization(spec):
        return None
    voxels = load_config(spec).voxels
    if voxels is None or voxels.section is None:
        return None
    n = micro.voxelize(**voxels.kwargs()).shape[voxels.section]
    return voxels.section, (n // 2 + 0.5) / n


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help="templates whose images to write (all)")
    parser.add_argument("--page", action="store_true", help="write the page only")
    parser.add_argument("--results", action="store_true",
                        help="write the figures of the results (of the templates named, or "
                             "of all those of RESULTS) instead of the images")  # fmt: skip
    args = parser.parse_args(argv)
    known = set(templates())
    listed = gallery_names()
    missing, unknown = known - set(listed), set(listed) - known
    if missing or unknown or len(listed) != len(set(listed)):
        print(f"FAMILIES: templates without a card {sorted(missing)}, cards without a "
              f"template {sorted(unknown)}, or a card twice", file=sys.stderr)  # fmt: skip
        return 1
    PAGE.write_text(page(), encoding="utf-8", newline="\n")
    print(f"written {PAGE.relative_to(ROOT)}")
    if args.page:
        return 0
    names = args.names or (list(RESULTS) if args.results else listed)
    bad = sorted(set(names) - (set(RESULTS) if args.results else known))
    if bad:
        print(f"unknown templates{' with a result' if args.results else ''}: {bad}",
              file=sys.stderr)  # fmt: skip
        return 1
    total = 0
    for name in names:
        size, elapsed = render_result(name) if args.results else render(name)
        total += size
        print(f"{name:42s} {size / 1024:6.1f} kB {elapsed:7.1f} s", flush=True)
    print(f"{len(names)} images, {total / 1024**2:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
