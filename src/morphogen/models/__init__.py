"""Morphological models (see :func:`list_models`).

===========================  ===============================================================
``boolean_spheres``          Boolean model of spheres (Poisson germs, random radii)
``boolean_cylinders``        Boolean model of finite fibres (cylinders / spherocylinders)
``poisson_fibers``           Infinite fibres on isotropic Poisson lines (non-periodic)
``hardcore_spheres``         Non-overlapping spheres: random sequential addition, or dense
                             packings grown by the Lubachevsky-Stillinger algorithm
``penetrable_spheres``       Spheres with impenetrable cores (Torquato's cherry-pit model)
``hardcore_fibers``          Non-overlapping straight fibres by random sequential addition
``curved_fibers``            Boolean model of curved fibres (worm-like chains of segments)
``random_walk_fibers``       Bending fibres packed without overlap by forces (Altendorf &
                             Jeulin's random walks), drawn into contact (Keilmann et al.)
``cross_linked_fibers``      Straight fibres bonded by cross-links at their contacts
                             (entangled cross-linked fibres, Chatti et al.)
``long_fibers``              Long fibres along an axis (unidirectional composites): parallel
                             or misaligned, their sections packed at random or in arrays
``voronoi``                  Poisson-Voronoi tessellation (multiphase mosaic, polycrystal)
``laguerre``                 Laguerre (power) tessellation with random weights
``seeded_laguerre``          Grains grown from ellipsoidal seeds (MicroStructPy): the Laguerre
                             cells of the spheres of each seed gathered into its grain
``gaussian_excursion``       Excursion set of a Gaussian random field
``tpms``                     Gyroid, Schwarz P/D, Neovius sheet or network lattices
``strut_lattice``            Cubic, BCC, FCC and octet strut lattices
``cellular_automaton``       Two-phase structure grown by a cellular automaton from noise
``activator_inhibitor``      Turing-like patterns of Young's activator-inhibitor automaton
``reaction_diffusion``       Turing patterns of the Gray-Scott reaction-diffusion system
``nucleation_growth``        Nucleation and growth (JMAK): Voronoi / Johnson-Mehl grains
``aggregation``              Diffusion-limited, Eden or ballistic clusters and deposits
``tile_assembly``            Compatible assemblies of building blocks (combinatorial design)
``random_lattice``           Strut lattices with missing struts and displaced nodes
``random_laminate``          Layers of random thicknesses with holes (nanolayered films)
``dead_leaves``              Dead leaves of spheres or spheroids, of families of their own
                             sizes, over a background (finite coverage)
``transparent_dead_leaves``  Excursion of the random function of transparent leaves
``reconstruction``           Annealed to the two-point probability and lineal path of a
                             Debye medium, a model or an image (Yeong-Torquato)
``combination``              Union, intersection, difference or complement of models,
                             inclusions in a matrix (a polycrystal...), or nested models
``cox_spheres``              Spheres clustered in the phase 1 of another model (aggregates)
``coating``                  A coating around the phase 1 of a model, or hollow members
``hierarchical_lattice``     Frames of generation G of (hollow) tubes on an fcc lattice
``fractal_sponge``           Menger sponge, Vicsek fractal, Cantor dust (iterated functions)
``nacre``                    Brick-and-mortar stacks of mineral tablets (bio-inspired)
``bouligand``                Twisted plywood of fibres: arthropod cuticles, fish scales
``honeycomb``                Prismatic cells: hexagonal, square or random honeycombs, wood
``osteons``                  Haversian bone: osteons remodelled one after another
``open_foam``                Open-cell foams and trabecular bone (struts on Voronoi edges)
``delaunay_network``         Fibre networks on the edges of Delaunay triangulations (Ban et
                             al.), in space or in a plane
``level_set_foam``           Foams as level sets of distance fields: Plateau borders, nodes
``branching``                Vascular trees grown by space colonization (Murray's law)
``ebsd``                     Polycrystal of a measured orientation map (EBSD): its grains
                             and their orientations, in windows of the map
``image``                    The microstructure of an image (a tomography, a stack of
                             sections), segmented into phases with implicit functions at
                             any point, in windows of the image
``nested_grains``            Grains with subgrains turned by small angles, or filled with
                             another structure turned grain by grain (lamellar colonies)
``scale_rich_network``       Ligaments of power-law thicknesses grown until they meet: scale-
                             rich networks (Both et al. 2026), extruded prisms, or plates
                             (closed cells) or beams (struts) in 3D
``woven_textile``            Woven composites: warp and weft yarns of elliptic, lenticular
                             or power-ellipse sections along crimped paths, 2D weaves
                             (plain, twill, satin, basket) and 3D weaves (orthogonal,
                             layer-to-layer, through-thickness angle interlock)
``spinodal``                 Spinodal decomposition by the Cahn-Hilliard equation, at an age of
                             the coarsening of its pattern
``microgen``                 A shape of Microgen (3MAH): TPMS sheets and skeletons, spinodoids,
                             primitives, by their fields (needs the package microgen)
===========================  ===============================================================

The point processes, tessellations, fields, leaves, growths and lattices take an
``anisotropy``, the same field everywhere: their realizations are mapped by a stretch
along principal axes preserving volumes (:mod:`morphogen.models.affine`).
"""

from morphogen.models.affine import AffineStructure, Anisotropy
from morphogen.models.automata import (
    ActivatorInhibitorModel,
    CellularAutomatonModel,
    ReactionDiffusionModel,
)
from morphogen.models.base import (
    Microstructure,
    Model,
    get_model,
    list_models,
    model_from_config,
    register,
)
from morphogen.models.boolean import BooleanCylinders, BooleanSpheres
from morphogen.models.bouligand import Bouligand, BouligandStructure
from morphogen.models.branching import BranchingNetwork
from morphogen.models.combination import Combination, Combined
from morphogen.models.deadleaves import DeadLeaves, DeadLeavesModel
from morphogen.models.design import RandomLattice, TileAssembly
from morphogen.models.ebsd import EBSD
from morphogen.models.fibers import (
    CrossLinkedFibers,
    CurvedFibers,
    HardcoreFibers,
    PoissonFibers,
    RandomWalkFibers,
)
from morphogen.models.foam import DelaunayNetwork, OpenFoam
from morphogen.models.gaussian import GaussianExcursion, GaussianField
from morphogen.models.growth import AggregationModel, NucleationGrowthModel
from morphogen.models.hardcore import HardcoreSpheres, PenetrableSpheres
from morphogen.models.hierarchical import HierarchicalLattice
from morphogen.models.honeycomb import Honeycomb, PrismaticCells
from morphogen.models.image import ImageModel, ImageStructure
from morphogen.models.laminate import Laminate, RandomLaminate
from morphogen.models.lattice import StrutLattice
from morphogen.models.lattice_field import LatticeField
from morphogen.models.levelset_foam import (
    DistanceFieldFoam,
    LevelSetFoam,
    SeedCalibration,
    calibrate_seeds,
)
from morphogen.models.long_fibers import LongFibers
from morphogen.models.mesh_particles import MeshParticles
from morphogen.models.microgen import Microgen, MicrogenStructure
from morphogen.models.nacre import Nacre, NacreStructure
from morphogen.models.nested import (
    ClusteredSpheres,
    CoatedStructure,
    Coating,
    CoxSpheres,
    FractalSponge,
    FractalStructure,
)
from morphogen.models.nested_grains import NestedGrains, NestedGrainStructure
from morphogen.models.osteons import Osteons, OsteonStructure
from morphogen.models.reconstruction import Reconstruction
from morphogen.models.scale_rich import ScaleRichNetwork, ScaleRichStructure
from morphogen.models.scale_rich_3d import ScaleRichBeams, ScaleRichPlates
from morphogen.models.seeded import SeededLaguerre, SeededTessellation
from morphogen.models.spinodal import Spinodal, SpinodalStructure
from morphogen.models.tessellation import Laguerre, Tessellation, Voronoi
from morphogen.models.textile import TextileStructure, WovenTextile
from morphogen.models.tiling_field import TilingField
from morphogen.models.tpms import TPMS, TPMSStructure
from morphogen.models.transparent import TransparentDeadLeaves, TransparentDeadLeavesModel
from morphogen.models.union import PrimitiveUnion

__all__ = [
    "EBSD",
    "TPMS",
    "ActivatorInhibitorModel",
    "AffineStructure",
    "AggregationModel",
    "Anisotropy",
    "BooleanCylinders",
    "BooleanSpheres",
    "Bouligand",
    "BouligandStructure",
    "BranchingNetwork",
    "CellularAutomatonModel",
    "ClusteredSpheres",
    "CoatedStructure",
    "Coating",
    "Combination",
    "Combined",
    "CoxSpheres",
    "CrossLinkedFibers",
    "CurvedFibers",
    "DeadLeaves",
    "DeadLeavesModel",
    "DelaunayNetwork",
    "DistanceFieldFoam",
    "FractalSponge",
    "FractalStructure",
    "GaussianExcursion",
    "GaussianField",
    "HardcoreFibers",
    "HardcoreSpheres",
    "HierarchicalLattice",
    "Honeycomb",
    "ImageModel",
    "ImageStructure",
    "Laguerre",
    "Laminate",
    "LatticeField",
    "LevelSetFoam",
    "LongFibers",
    "MeshParticles",
    "Microgen",
    "MicrogenStructure",
    "Microstructure",
    "Model",
    "Nacre",
    "NacreStructure",
    "NestedGrainStructure",
    "NestedGrains",
    "NucleationGrowthModel",
    "OpenFoam",
    "OsteonStructure",
    "Osteons",
    "PenetrableSpheres",
    "PoissonFibers",
    "PrimitiveUnion",
    "PrismaticCells",
    "RandomLaminate",
    "RandomLattice",
    "RandomWalkFibers",
    "ReactionDiffusionModel",
    "Reconstruction",
    "ScaleRichBeams",
    "ScaleRichNetwork",
    "ScaleRichPlates",
    "ScaleRichStructure",
    "SeedCalibration",
    "SeededLaguerre",
    "SeededTessellation",
    "Spinodal",
    "SpinodalStructure",
    "StrutLattice",
    "TPMSStructure",
    "Tessellation",
    "TextileStructure",
    "TileAssembly",
    "TilingField",
    "TransparentDeadLeaves",
    "TransparentDeadLeavesModel",
    "Voronoi",
    "WovenTextile",
    "calibrate_seeds",
    "get_model",
    "list_models",
    "model_from_config",
    "register",
]
