"""The tools of the tab Mechanics: the nonlinear homogenization of a morphology along a
macroscopic loading path, by finite elements on the conforming mesh of its level set or by
FFT on its voxels, in small or finite strains (the periodic element of Z-set: the
macroscopic components prescribed as stresses are unknowns); its reduced model by the
nonuniform transformation field analysis (NTFA; Michel & Suquet 2003, 2004) against the
full field; a material point of a behaviour along a loading path, to calibrate its law;
the identification of a behaviour on runs of the workspace or the points of a test; the
linear tensors of a morphology (stiffness, thermal expansion, conductivity), with its
crystals; and the stiffness of a fibre network, an open foam or a random lattice as the
frame of beams of its framework (:mod:`morphogen.homogenization.networks`).

The behaviour of one phase of the form is elastic, von Mises elasto-(visco)plasticity
(Lemaitre & Chaboche: isotropic and kinematic hardening, Norton's viscosity), the crystal
plasticity of Méric & Cailletaud on the slip systems of its grains (turned by their
orientations), porous plasticity (Gurson-Tvergaard-Needleman, Rousselier, the elliptic
criterion of Abouaf), a material file of Z-set, or an MFront behaviour when MGIS is
installed; the other phases are elastic or voids, with the elastic constants of the form,
and the materials of the configurations may replace those of any phase. Stresses are in
the unit of the elastic moduli (the defaults take the modulus of the first phase, 1, as
unit); times in that of the viscosities."""

from __future__ import annotations

import importlib.util
import json
import time
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from morphogen.gui.tools import Field, elastic_tables, tab, tool
from morphogen.homogenization.behaviours import HILL_COMPONENTS

tab(
    "mechanics",
    "Mechanics",
    "The mechanics of a morphology of the workspace: its macroscopic response along a "
    "loading path (strain, stress or mixed components over time; finite elements or FFT, "
    "small or finite strains) with the fields of its steps, the reduced model of the "
    "nonuniform transformation field analysis (NTFA) against the full field, a material "
    "point to calibrate a behaviour, the identification of a behaviour on runs of the "
    "workspace or the points of a test, the macroscopic law of a morphology in one go (its "
    "tensions and shears, a von Mises or Hill law identified on them), the linear tensors "
    "of the morphology, and the stiffness of a fibre network or an open foam as a frame of "
    "beams.",
    order=40,
)

#: The Voigt components of the loading paths (11, 22, 33, 23, 31, 12; engineering shears)
COMPONENTS = ("11", "22", "33", "23", "31", "12")
#: The stem of the files of the realization, as in the runs of configurations
STEM = "sample_0000"


#: The choice of a behaviour of the workspace (a material point, an import of Z-set)
KEPT = ("workspace", "a behaviour of the workspace (B: a material point, a Z-set file)")


def _behaviours() -> tuple:
    """The behaviours offered for a phase (MFront when MGIS is installed), and those of the
    workspace."""
    out = [
        ("elastic", "elastic"),
        ("mises", "von Mises elasto-(visco)plasticity (Lemaitre-Chaboche)"),
        ("crystal", "crystal plasticity of the grains (Méric-Cailletaud)"),
        ("porous", "porous plasticity (Gurson-Tvergaard-Needleman, Rousselier, Abouaf)"),
        ("zmat", "a material file of Z-set (gen_evp, porous_plastic)"),
        ("viscoelastic", "linear viscoelasticity (generalized Maxwell: Prony series)"),
        ("hyperelastic", "hyperelasticity (neo-Hookean to Ogden, foams, fibres; finite strains)"),
        ("damage", "isotropic damage of quasi-brittle solids"),
        ("mazars", "Mazars' damage of concrete (tension and compression)"),
    ]
    if importlib.util.find_spec("mgis") is not None:
        out.append(("mfront", "an MFront behaviour (integrated by MGIS)"))
    return (*out, KEPT)


def _kept_field() -> Field:
    """The behaviour of the workspace chosen (an item: its material under ``"material"``)."""
    return Field(
        "kept",
        "item",
        label="behaviour of the workspace",
        items=("behaviour",),
        required=True,
        when={"behaviour": "workspace"},
        doc="with its own elasticity (the elastic constants of the form unused)",
    )


FAMILIES = (
    ("octahedral", "octahedral {111}<110> (FCC)"),
    ("cube", "cube {100}<110>"),
    ("bcc110", "{110}<111> (BCC)"),
    ("bcc112", "{112}<111> (BCC)"),
    ("basal", "basal (0001)<11-20> (hexagonal)"),
    ("prismatic", "prismatic {10-10}<11-20> (hexagonal)"),
    ("pyramidal_a", "pyramidal {10-11}<11-20> (hexagonal)"),
    ("pyramidal_ca", "pyramidal {10-11}<11-23> (hexagonal, c + a)"),
    ("pyramidal2_ca", "pyramidal {11-22}<11-23> (hexagonal, c + a)"),
)
TWIN_FAMILIES = (
    ("twin111", "{111}<11-2> (FCC)"),
    ("twin112", "{112}<11-1> (BCC)"),
    ("twin1012", "{10-12}<10-1-1> (hexagonal)"),
    ("twin1011", "{10-11}<10-1-2> (hexagonal)"),
    ("twin1122", "{11-22}<11-2-3> (hexagonal)"),
    ("twin1121", "{11-21}<-1-126> (hexagonal)"),
)
CLEAVAGE_PLANES = (
    ("none", "none"),
    ("cube", "{100} (cubic)"),
    ("dodecahedral", "{110} (cubic)"),
    ("octahedral", "{111} (cubic)"),
    ("basal", "basal (0001) (hexagonal)"),
    ("prismatic", "prismatic {10-10} (hexagonal)"),
)
CRITERIA = (
    ("gurson", "Gurson-Tvergaard-Needleman"),
    ("rousselier", "Rousselier"),
    ("elliptic", "elliptic (Abouaf, Green, Shima-Oyane: powders, foams)"),
)
NUCLEATION = (
    ("none", "none"),
    ("gaussian", "Gaussian (Chu-Needleman): fN, εN, sN"),
    ("constant", "constant: A"),
    ("exponential", "exponential: B, ε0"),
)
#: A material file of Z-set in the unit of the defaults (E = 1)
ZMAT_EXAMPLE = """***behavior gen_evp
 **elasticity isotropic
   young 1.
   poisson 0.3
 **potential gen_evp ep
   *criterion mises
   *flow plasticity
   *isotropic nonlinear
     R0 0.002
     Q 0.001
     b 100.
***return"""

#: The hyperelastic energies of the forms and the parameters they take, in their order
HYPERELASTIC = (
    ("neo_hookean", "neo-Hookean: mu (blank: the shear modulus of the elastic constants)"),
    ("mooney_rivlin", "Mooney-Rivlin: c10, c01 (then c20, c11, c02, c30, c21, c12, c03)"),
    ("yeoh", "Yeoh: c10, c20, c30"),
    ("arruda_boyce", "Arruda-Boyce (series): mu, locking stretch"),
    ("eight_chain", "eight chains (inverse Langevin): mu, locking stretch"),
    ("gent", "Gent: mu, Jm"),
    ("van_der_waals", "van der Waals: mu, locking stretch, a, beta"),
    ("fung", "Fung: mu, b"),
    ("ogden", "Ogden: pairs mu, alpha"),
    ("holzapfel", "Holzapfel-Gasser-Ogden: mu, k1, k2, kappa (and the fibres)"),
    ("blatz_ko", "Blatz-Ko (compressible): mu, f"),
    ("hyperfoam", "hyperfoam (compressible): pairs mu, alpha"),
)
#: (the least number of them: the others have defaults)
HYPERELASTIC_PARAMETERS = {
    "neo_hookean": (("mu",), 1),
    "mooney_rivlin": (("c10", "c01", "c20", "c11", "c02", "c30", "c21", "c12", "c03"), 2),
    "yeoh": (("c10", "c20", "c30"), 1),
    "arruda_boyce": (("mu", "lock"), 2),
    "eight_chain": (("mu", "lock"), 2),
    "gent": (("mu", "lock"), 2),
    "van_der_waals": (("mu", "lock", "a", "beta"), 2),
    "fung": (("mu", "b"), 2),
    "holzapfel": (("mu", "k1", "k2", "kappa"), 3),
    "blatz_ko": (("mu", "f"), 1),
}
VISCOELASTIC = {"behaviour": "viscoelastic"}
HYPER = {"behaviour": "hyperelastic"}
DAMAGE = {"behaviour": "damage"}
MAZARS = {"behaviour": "mazars"}
LEMAITRE = {"behaviour": "mises", "coupled_damage": ["lemaitre"]}
KACHANOV = {"behaviour": "mises", "coupled_damage": ["kachanov"]}
COUPLED = {"behaviour": "mises", "coupled_damage": ["lemaitre", "kachanov"]}
PLASTIC = {"behaviour": ["mises", "porous"]}
MISES = {"behaviour": "mises"}
CRYSTAL = {"behaviour": "crystal"}
CLEAVES = {"behaviour": "crystal", "cleavage": [c[0] for c in CLEAVAGE_PLANES[1:]]}
POROUS = {"behaviour": "porous"}


def _device_field(**kwargs) -> Field:
    """Where the FFT solver of the paths runs."""
    return Field(
        "device",
        "choice",
        "cpu",
        choices=(
            ("cpu", "the CPU"),
            ("gpu", "an NVIDIA GPU (CuPy): the points and the conjugate gradients there"),
            ("auto", "the GPU when there is one"),
        ),
        doc="the GPU gains on images of 32 voxels per edge and more",
        **kwargs,
    )


def _plastic_fields(plastic=None, mises=None) -> tuple[Field, ...]:
    """The fields of von Mises elasto-(visco)plasticity (also those of the matrix of porous
    materials, its kinematic hardening included), shown when the fields of ``plastic``
    (``mises``: Hill's criterion) have their values."""
    return (
        Field(
            "yield_stress",
            "float",
            0.002,
            label="yield stress R0",
            unit="stress",
            when=plastic,
            doc="that of the matrix of porous materials",
        ),
        Field("hardening", "float", 0.0, label="linear hardening H", unit="stress", when=plastic),
        Field(
            "isotropic",
            "floats",
            [0.001, 100.0],
            label="isotropic hardening Q, b",
            when=plastic,
            doc="R(p) = R0 + H p + Σ Q (1 - exp(-b p)): pairs Q, b, at most three (Q = 0: none)",
        ),
        Field(
            "kinematic",
            "floats",
            label="kinematic hardening C, D",
            when=plastic,
            doc="Armstrong-Frederick backstresses dX = 2/3 C dεp - D X dp: pairs C, D, at "
            "most four (D = 0: linear, Prager; blank: none); porous materials: the criteria "
            "gurson and elliptic of σ - X, X with its hydrostatic part",
        ),
        Field(
            "norton",
            "floats",
            label="Norton viscosity K, n",
            when=plastic,
            doc="viscoplastic flow dp/dt = <f / K>^n (blank: rate-independent plasticity)",
        ),
        Field(
            "hill",
            "floats",
            label="Hill's criterion h11, h22, h33, h12, h23, h31",
            when=mises,
            doc="anisotropic plasticity: f = sqrt(3/2 s : M : s) - R, M = diag(h) on the "
            "components of the deviator s (Z-set's hill1 ... hill6; yield stress in tension "
            "along 1: R0 sqrt(6 / (4 h11 + h22 + h33)), in shear 12: R0 / sqrt(3 h12)); blank: "
            "von Mises (all 1)",
        ),
    )


def _behaviour_fields(doc: str = "") -> tuple[Field, ...]:
    """The fields of the behaviour of a phase or of a material point (``doc``: that of the
    choice of the behaviour)."""
    choices = _behaviours()
    gurson = {"behaviour": "porous", "criterion": "gurson"}
    nucleating = {"behaviour": "porous", "nucleation": ["gaussian", "constant", "exponential"]}
    fields = [
        Field("behaviour", "choice", "mises", choices=choices, doc=doc),
        _kept_field(),
        *_plastic_fields(PLASTIC, MISES),
        Field(
            "coupled_damage",
            "choice",
            "none",
            label="damage of the plastic phase",
            choices=(
                ("none", "none"),
                ("lemaitre", "Lemaitre: ductile damage, by the plastic strain"),
                ("kachanov", "Kachanov-Rabotnov: creep damage, by the time under stress"),
            ),
            when=MISES,
            doc="the plasticity is that of the effective stress σ / (1 - D), the stress (1 - "
            "D) of it; the variable p of the results is then the damage D",
        ),
        Field(
            "lemaitre",
            "floats",
            [2.0, 1.0],
            label="Lemaitre's S, s",
            when=LEMAITRE,
            doc="dD = (Y / S)^s dp, Y = σ_eq² R_ν / (2 E) the energy release rate of the "
            "effective stress with the triaxiality function R_ν (S: a stress)",
        ),
        Field(
            "damage_threshold",
            "float",
            0.0,
            label="plastic strain at which the damage starts",
            when=LEMAITRE,
        ),
        Field(
            "closure",
            "float",
            1.0,
            label="closure of the microcracks h",
            when=LEMAITRE,
            doc="the share of the compressions in Y (1: as the tensions; 0.2 for metals)",
        ),
        Field(
            "kachanov",
            "floats",
            [500.0, 4.0],
            label="Kachanov's A, r (and k)",
            when=KACHANOV,
            doc="dD/dt = (χ / A)^r (1 - D)^-k (A: a stress; k: r if not given): under a "
            "constant stress the time to rupture is 1 / ((k + 1) (χ / A)^r); a viscoplastic "
            "phase (Norton)",
        ),
        Field(
            "hayhurst",
            "floats",
            label="Hayhurst's weights α, β",
            when=KACHANOV,
            doc="χ = α σ_I + 3 β σ_m + (1 - α - β) σ_eq (blank: von Mises' stress)",
        ),
        Field(
            "critical_damage",
            "float",
            label="damage of a broken point",
            when=COUPLED,
            doc="a point whose damage reaches it keeps a millionth of its stress (blank: "
            "the damage grows up to 1)",
        ),
        Field(
            "coupled_viscosity",
            "float",
            0.0,
            label="time of the delayed damage",
            when=COUPLED,
            doc="a viscous regularization (0: none): the damage of the stress follows that "
            "of the law with this delay, which keeps the softening of a morphology well "
            "posed along its loading path",
        ),
        Field(
            "families",
            "choices",
            ["octahedral"],
            label="slip families",
            choices=FAMILIES,
            when=CRYSTAL,
        ),
        Field(
            "cubic",
            "floats",
            label="cubic elasticity C11, C12, C44",
            unit="stress",
            when=CRYSTAL,
            doc="in the axes of the crystal (blank: the isotropic elasticity of the phase)",
        ),
        Field(
            "tau0",
            "float",
            0.0008,
            label="critical resolved shear stress τ0",
            unit="stress",
            when=CRYSTAL,
        ),
        Field(
            "slip_isotropic",
            "floats",
            [0.0001, 10.0],
            label="slip hardening Q, b",
            when=CRYSTAL,
            doc="r = τ0 + Q Σ_r h_sr (1 - exp(-b v_r)) of the cumulated slips v (Q = 0: none)",
        ),
        Field(
            "slip_kinematic",
            "floats",
            label="slip kinematic hardening C, D",
            when=CRYSTAL,
            doc="x = C α, dα = dγ - D α |dγ| (blank: none)",
        ),
        Field(
            "slip_viscosity",
            "floats",
            [0.0002, 10.0],
            label="slip viscosity K, n",
            when=CRYSTAL,
            doc="dγ/dt = <(|τ - x| - r) / K>^n",
        ),
        Field(
            "interaction",
            "floats",
            label="interaction matrix",
            when=CRYSTAL,
            doc="h_self, h_latent; or the six coefficients of the octahedral systems (self, "
            "coplanar, Hirth, collinear, glissile, Lomer); blank: uniform (Taylor)",
        ),
        Field(
            "hexagonal",
            "floats",
            label="hexagonal elasticity C11, C12, C13, C33, C44",
            unit="stress",
            when=CRYSTAL,
            doc="in the axes of the crystal, c along the third (blank: the cubic or the "
            "isotropic elasticity)",
        ),
        Field(
            "c_over_a",
            "float",
            label="ratio c / a",
            when=CRYSTAL,
            doc="of a hexagonal lattice: for its slip families, its twins and its cleavage "
            "planes (magnesium 1.624, titanium 1.587, zinc 1.856)",
        ),
        Field(
            "twins",
            "choices",
            [],
            label="twin families",
            choices=TWIN_FAMILIES,
            when=CRYSTAL,
            doc="systems that shear in one sense, and back, until the crystal has twinned "
            "(the hardening and the viscosity of the slip)",
        ),
        Field(
            "twin_tau0",
            "float",
            label="critical resolved shear stress of the twins",
            unit="stress",
            when=CRYSTAL,
            doc="blank: that of the slip",
        ),
        Field(
            "twin_saturation",
            "float",
            1.0,
            label="twinned fraction at saturation",
            when=CRYSTAL,
            doc="the fraction of the crystal that may twin",
        ),
        Field(
            "twin_reorientation",
            "float",
            label="twinned fraction that turns the lattice",
            when=CRYSTAL,
            doc="at this fraction the lattice of a point becomes that of its predominant "
            "twin, which slips and twins as the twin does, and twins back (blank: the "
            "lattice does not turn)",
        ),
        Field(
            "twin_reorientation_growth",
            "float",
            label="growth of that fraction, grain by grain",
            when=CRYSTAL,
            doc="given (0 or more), the lattices are turned grain by grain: the most "
            "twinned points first, while the turned volume of the grain is below its "
            "twinned volume, past the fraction above plus this times their ratio (blank: "
            "point by point, at the fraction above)",
        ),
        Field(
            "twin_detwinning",
            "bool",
            True,
            label="the twins shear back",
            when=CRYSTAL,
            doc="when the stress is reversed, down to no twin (detwinning)",
        ),
        Field(
            "twin_detwinning_share",
            "float",
            1.0,
            label="share of their resistance on the way back",
            when=CRYSTAL,
            doc="the twins shear back at this share of the resistance of the twinning (no "
            "twin to nucleate: below 1)",
        ),
        Field(
            "cleavage",
            "choice",
            "none",
            label="cleavage planes",
            choices=CLEAVAGE_PLANES,
            when=CRYSTAL,
            doc="planes that open at a normal stress and soften (the crack band)",
        ),
        Field(
            "cleavage_strength",
            "float",
            label="cleavage strength",
            unit="stress",
            when=CLEAVES,
            doc="the normal stress that opens a plane",
        ),
        Field(
            "cleavage_energy",
            "float",
            label="fracture energy of the planes",
            when=CLEAVES,
            doc="per unit area, dissipated in a band of the size below",
        ),
        Field(
            "cleavage_length",
            "float",
            label="size of the elements of the band",
            unit="length",
            when=CLEAVES,
            doc="that of the voxels or of the elements: below stiffness x energy / strength²",
        ),
        Field(
            "cleavage_band",
            "float",
            label="width of the damage band of the planes",
            unit="length",
            when=CLEAVES,
            doc="the planes regularized by their microdamage: they break across a band of "
            "this width (some elements), where their fracture energy is dissipated, "
            "whatever the size of the elements (blank: across one element, of the size "
            "above)",
        ),
        Field(
            "cleavage_viscosity",
            "floats",
            label="viscosity of the planes K, n",
            when=CLEAVES,
            doc="of their opening (blank: that of the slip); a crack that crosses the cell "
            "at once passes with it",
        ),
        Field(
            "cleavage_closure",
            "bool",
            True,
            label="the planes close under pressure",
            when=CLEAVES,
            doc="their opening goes back, and the closed plane carries the pressure; "
            "otherwise it stays open",
        ),
        Field(
            "cleavage_closure_share",
            "float",
            1.0,
            label="share of their strength at which they close",
            when=CLEAVES,
            doc="a plane that is not broken through closes under this share of the "
            "strength that it has left (1: all of it; near 0: freely)",
        ),
        Field(
            "cleavage_friction",
            "float",
            0.0,
            label="friction of the planes",
            when=CLEAVES,
            doc="Coulomb's coefficient of the lips of a plane under pressure (0: they "
            "slide freely)",
        ),
        Field(
            "cleavage_coupling",
            "float",
            0.0,
            label="coupling of the planes with the slip",
            when=CLEAVES,
            doc="the strength of the planes falls as exp(-coupling x cumulated slip): the "
            "slip nucleates the cleavage, and the damage of the planes eases the slip (0: "
            "a constant strength)",
        ),
        Field("criterion", "choice", "gurson", choices=CRITERIA, when=POROUS),
        Field("porosity", "float", 0.01, label="initial porosity f0", when=POROUS),
        Field(
            "gurson",
            "floats",
            [1.5, 1.0],
            label="q1, q2 (q3)",
            when=gurson,
            doc="(Σeq / σ*)² + 2 q1 f* cosh(3 q2 Σm / 2σ*) - 1 - q3 f*² = 0 (q3: q1² by default)",
        ),
        Field(
            "coalescence",
            "floats",
            label="coalescence fc, fF",
            when=gurson,
            doc="Tvergaard-Needleman: f* grows faster beyond fc, to 1/q1 at fF (blank: f* = f)",
        ),
        Field(
            "rousselier",
            "floats",
            [0.0013, 2.0],
            label="σ1, D (q2)",
            when={"behaviour": "porous", "criterion": "rousselier"},
            doc="σ* = Σeq / (1 - f) + D σ1 f exp(q2 Σm / ((1 - f) σ1)) (q2: 1 by default)",
        ),
        Field(
            "elliptic",
            "floats",
            [1.2, 0.05],
            label="C, F",
            when={"behaviour": "porous", "criterion": "elliptic"},
            doc="σ*² = C Σeq² + F I1² (C = 1, F = 0: von Mises; the configurations take "
            "tables of the porosity)",
        ),
        Field("nucleation", "choice", "none", choices=NUCLEATION, when=POROUS),
        Field(
            "nucleation_coefficients",
            "floats",
            when=nucleating,
            doc="gaussian: fN, εN, sN; constant: A; exponential: B, ε0; then optionally "
            "the cumulated plastic strain below which none nucleates",
        ),
        Field(
            "zmat",
            "text",
            ZMAT_EXAMPLE,
            label="Z-set material file",
            when={"behaviour": "zmat"},
            doc="its text, or the path of the file: gen_evp (one von Mises potential, "
            "else through MFront) or porous_plastic; its elasticity replaces that of the "
            "phase",
        ),
        Field(
            "prony_shear",
            "floats",
            [0.5, 1.0],
            label="Prony series of the shear modulus g, τ",
            when=VISCOELASTIC,
            doc="G(t) = G0 (1 - Σ g (1 - exp(-t / τ))): pairs g, τ, the elastic constants "
            "above the instantaneous ones (the sum of the g below 1)",
        ),
        Field(
            "prony_bulk",
            "floats",
            label="Prony series of the bulk modulus k, τ",
            when=VISCOELASTIC,
            doc="pairs k, τ (blank: an elastic bulk response)",
        ),
        Field(
            "hyper_model",
            "choice",
            "neo_hookean",
            label="strain energy",
            choices=HYPERELASTIC,
            when=HYPER,
        ),
        Field(
            "hyper_parameters",
            "floats",
            label="parameters of the energy",
            unit="stress",
            when=HYPER,
            doc="in the order of the model (neo-Hookean, blank: the elastic constants "
            "above); given, the elasticity is that of the energy (finite strains: the "
            "kinematics of the loading)",
        ),
        Field(
            "hyper_bulk",
            "float",
            label="bulk modulus of the energy",
            unit="stress",
            when=HYPER,
            doc="U(J) = K/2 (J - 1)² (blank: that of the elastic constants above); the "
            "compressible energies take their Poisson's ratio from it",
        ),
        Field(
            "hyper_fibres",
            "floats",
            label="directions of the fibres",
            when={"behaviour": "hyperelastic", "hyper_model": "holzapfel"},
            doc="x, y, z of each family of fibres (Holzapfel-Gasser-Ogden), in the axes of "
            "the morphology: they carry tension only",
        ),
        Field(
            "hyper_mullins",
            "floats",
            label="Mullins softening r, m, β",
            when=HYPER,
            doc="Ogden and Roxburgh: on the way back from a stretch the isochoric stress is "
            "that of the virgin solid times 1 - erf((Wm - W) / (m + β Wm)) / r, Wm the "
            "largest energy reached (blank: none)",
        ),
        Field(
            "strength",
            "float",
            0.002,
            label="tensile strength",
            unit="stress",
            when=DAMAGE,
            doc="the damage starts at the equivalent strain strength / E",
        ),
        Field(
            "ultimate",
            "float",
            0.02,
            label="ultimate strain of the softening",
            when=DAMAGE,
            doc="no stress beyond it (linear softening), or the strain of the exponential "
            "decay; blank with a fracture energy",
        ),
        Field(
            "fracture_energy",
            "float",
            label="fracture energy",
            when=DAMAGE,
            doc="per unit area, dissipated by a band of the size below (the crack band of "
            "Bažant and Oh): the softening then does not depend on the elements",
        ),
        Field(
            "damage_length",
            "float",
            label="size of the elements of the band",
            unit="length",
            when=DAMAGE,
            doc="that of the voxels or of the elements (with the fracture energy)",
        ),
        Field(
            "softening",
            "choice",
            "exponential",
            choices=("exponential", "linear"),
            when=DAMAGE,
        ),
        Field(
            "equivalent",
            "choice",
            "mazars",
            label="equivalent strain",
            choices=(
                ("mazars", "Mazars: the positive principal strains"),
                ("energy", "the energy norm (as much in compression)"),
                ("mises", "modified von Mises (de Vree et al.)"),
            ),
            when=DAMAGE,
        ),
        Field(
            "damage_viscosity",
            "float",
            0.0,
            label="time of the delayed damage",
            when=DAMAGE,
            doc="a viscous regularization (0: none): it keeps the softening of a "
            "morphology well posed along its loading path",
        ),
        Field(
            "damage_split",
            "choice",
            "none",
            label="stiffness of closed cracks",
            choices=(
                ("none", "none: as soft in compression"),
                ("volumetric", "the bulk response in compression (Amor et al.)"),
                ("spectral", "the negative principal strains (Miehe et al.): unilateral"),
            ),
            when=DAMAGE,
        ),
        Field(
            "mazars_threshold",
            "float",
            1e-4,
            label="strain at which the damage starts",
            when=MAZARS,
        ),
        Field(
            "mazars_tension",
            "floats",
            [1.0, 15000.0],
            label="tension A_t, B_t",
            when=MAZARS,
            doc="d_t = 1 - κ0 (1 - A_t) / κ - A_t exp(-B_t (κ - κ0)), κ the largest norm of "
            "the positive principal strains",
        ),
        Field(
            "mazars_compression",
            "floats",
            [1.2, 1500.0],
            label="compression A_c, B_c",
            when=MAZARS,
            doc="the damage is α_t^β d_t + α_c^β d_c, by the shares of the positive strains "
            "that the tensions and the compressions make",
        ),
        Field(
            "mazars_beta",
            "float",
            1.0,
            label="exponent β of the shares",
            when=MAZARS,
            doc="1; 1.06 lessens the damage in shear",
        ),
    ]
    if any(c[0] == "mfront" for c in choices):
        mf = {"behaviour": "mfront"}
        fields += [
            Field(
                "mfront",
                "str",
                label="MFront source or library",
                when=mf,
                doc="a .mfront file (compiled by mfront --obuild) or a library of the "
                "generic interface",
            ),
            Field(
                "mfront_behaviour",
                "str",
                label="behaviour of the library",
                when=mf,
                doc="blank: the stem of the source",
            ),
            Field("mfront_parameters", "text", label="parameters (JSON)", when=mf),
            Field("mfront_properties", "text", label="material properties (JSON)", when=mf),
        ]
    return tuple(fields)


LOADINGS = (
    ("tension", "monotonic: a strain component to the amplitude, the other stresses free"),
    ("cycle", "cycle: the strain to + and - the amplitude, then back to zero"),
    ("relaxation", "relaxation: the strain reached, then held"),
    ("creep", "creep: a stress component reached, then held"),
    ("custom", "custom: strain, stress or mixed components over time"),
)
PRESETS = ["tension", "cycle", "relaxation", "creep"]


def _path_fields(
    default: str = "tension", prefix: str = "", amplitude: float = 0.01
) -> tuple[Field, ...]:
    """The fields of a loading path (their names after ``prefix``): a preset of one
    component, or the components over time in the compact text of
    :meth:`~morphogen.homogenization.paths.LoadingPath.parse`."""
    loading = f"{prefix}loading"
    presets = {loading: PRESETS}
    return (
        Field(loading, "choice", default, label="loading", choices=LOADINGS),
        Field(
            f"{prefix}component",
            "choice",
            "11",
            label="component",
            choices=COMPONENTS,
            when=presets,
        ),
        Field(
            f"{prefix}amplitude",
            "float",
            amplitude,
            label="strain amplitude",
            when={loading: ["tension", "cycle", "relaxation"]},
            doc="engineering shear strains (twice the tensor component)",
        ),
        Field(
            f"{prefix}stress",
            "float",
            0.0025,
            label="stress",
            unit="stress",
            when={loading: "creep"},
        ),
        Field(
            f"{prefix}duration",
            "float",
            1.0,
            label="loading time",
            unit="time",
            when=presets,
            doc="of the ramp (of each quarter of the cycle)",
        ),
        Field(
            f"{prefix}hold",
            "float",
            10.0,
            label="hold time",
            unit="time",
            when={loading: ["relaxation", "creep"]},
        ),
        Field(
            f"{prefix}increments",
            "int",
            10,
            label="increments",
            when=presets,
            doc="of each ramp (a step that does not converge is divided)",
        ),
        Field(
            f"{prefix}path",
            "text",
            "t=0,1,2 n=10 E11=0,0.01,0.01 S12=0,0,0.002",
            label="components over time",
            when={loading: "custom"},
            doc="the times t, increments n of each segment, strain components E11 ... E12 "
            "(engineering shears) and stress components S11 ... S12 at the times (one "
            "value: reached at the end of the first segment and held), the change of "
            "temperature dT; the others free (zero stress). Or a JSON mapping {times, "
            "increments, strain, stress, temperature}",
        ),
    )


def _path(p: dict, prefix: str = "") -> tuple[str, dict]:
    """The name and the specification (:meth:`LoadingPath.from_spec`, checked) of the
    loading path of the form."""
    from morphogen.homogenization.paths import LoadingPath

    kind = p[f"{prefix}loading"]
    if kind == "custom":
        text = (p[f"{prefix}path"] or "").strip()
        if not text:
            raise ValueError("components over time: e.g. t=0,1 n=10 E11=0,0.01")
        spec = _json(text, "components over time") if text.startswith("{") else None
        path = LoadingPath.from_spec(spec) if spec is not None else LoadingPath.parse(text)
        return "path", path.to_dict() if spec is None else spec
    c = p[f"{prefix}component"]
    d, n = p[f"{prefix}duration"], p[f"{prefix}increments"]
    if not d > 0 or n < 1:
        raise ValueError("loading time: positive, and at least one increment")
    a, hold = p[f"{prefix}amplitude"], p[f"{prefix}hold"]
    if kind in ("relaxation", "creep") and not hold > 0:
        raise ValueError("hold time: positive")
    if kind == "tension":
        spec = {"times": [0.0, d], "increments": n, "strain": {c: [0.0, a]}}
    elif kind == "cycle":
        times = [0.0, d, 3 * d, 4 * d]
        spec = {"times": times, "increments": [n, 2 * n, n], "strain": {c: [0.0, a, -a, 0.0]}}
    elif kind == "relaxation":
        spec = {"times": [0.0, d, d + hold], "increments": [n, n], "strain": {c: [0.0, a, a]}}
    else:
        s = p[f"{prefix}stress"]
        spec = {"times": [0.0, d, d + hold], "increments": [n, n], "stress": {c: [0.0, s, s]}}
    LoadingPath.from_spec(spec)  # checked
    return kind, spec


def _json(text, name: str):
    """A JSON value of a text field (None when blank)."""
    if text is None or (isinstance(text, str) and not text.strip()):
        return None
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name}: not valid JSON ({exc.msg} at character {exc.pos})") from None


# -- behaviours ------------------------------------------------------------------------------


def _pairs(values, name: str, first: str, second: str, most: int) -> list[dict]:
    """The pairs of a field of numbers ``a1, b1, a2, b2...`` (``{first: a, second: b}``)."""
    values = list(values or [])
    if len(values) % 2 or len(values) > 2 * most:
        raise ValueError(f"{name}: pairs {first}, {second} (at most {most})")
    return [{first: values[k], second: values[k + 1]} for k in range(0, len(values), 2)]


def _two(values, name: str, first: str, second: str, optional: str = "") -> list[float]:
    """The two numbers of a field (and an optional third one)."""
    values = list(values or [])
    if len(values) not in ((2, 3) if optional else (2,)):
        extra = f" (and {optional})" if optional else ""
        raise ValueError(f"{name}: two numbers {first}, {second}{extra}")
    return values


def _norton(p: dict) -> dict:
    if not p.get("norton"):
        return {}
    k, n = _two(p["norton"], "Norton viscosity", "K", "n")
    return {"norton": {"K": k, "n": n}}


def _plasticity(p: dict) -> dict:
    """The von Mises plasticity of the form (:meth:`Plasticity.from_config`)."""
    out = {"yield": p["yield_stress"], "hardening": p["hardening"] or 0.0}
    iso = [t for t in _pairs(p.get("isotropic"), "isotropic hardening", "Q", "b", 3) if t["Q"]]
    if iso:
        out["isotropic"] = iso
    kin = _pairs(p.get("kinematic"), "kinematic hardening", "C", "D", 4)
    if kin:
        out["kinematic"] = kin
    if p.get("hill"):
        hill = list(p["hill"])
        if len(hill) != 6:
            raise ValueError("Hill's criterion: six coefficients h11, h22, h33, h12, h23, h31")
        out["hill"] = dict(zip(HILL_COMPONENTS, hill, strict=True))
    return {**out, **_norton(p)}


def _crystal(p: dict) -> dict:
    """The crystal plasticity of the form (:meth:`CrystalPlasticity.from_config`)."""
    q, b = _two(p.get("slip_isotropic") or [0.0, 0.0], "slip hardening", "Q", "b")
    c, d = _two(p.get("slip_kinematic") or [0.0, 0.0], "slip kinematic hardening", "C", "D")
    if not p.get("slip_viscosity"):
        raise ValueError("slip viscosity: K, n (crystal plasticity is viscoplastic)")
    k, n = _two(p["slip_viscosity"], "slip viscosity", "K", "n")
    out = {
        "families": p.get("families") or ["octahedral"],
        "tau0": p["tau0"],
        "Q": q,
        "b": b if q else 0.0,
        "C": c,
        "D": d,
        "K": k,
        "n": n,
    }
    if p.get("interaction"):
        out["interaction"] = list(p["interaction"])
    if p.get("c_over_a"):
        out["c_over_a"] = p["c_over_a"]
    twins = list(p.get("twins") or [])
    if twins:
        own = {"tau0": p["twin_tau0"]} if p.get("twin_tau0") else {}
        out["twins"] = {name: dict(own) for name in twins}
        if p.get("twin_saturation") not in (None, 1.0):
            out["twin_saturation"] = p["twin_saturation"]
        growth = p.get("twin_reorientation_growth")
        if p.get("twin_reorientation"):
            turn = p["twin_reorientation"]
            out["twin_reorientation"] = (
                turn if growth is None else {"threshold": turn, "growth": growth}
            )
        elif growth is not None:
            raise ValueError(
                "growth of the fraction that turns the lattice: give that fraction (the "
                "threshold of the grains before any of their lattices is turned)"
            )
        if p.get("twin_detwinning") is False:
            out["twin_detwinning"] = False
        elif p.get("twin_detwinning_share") not in (None, 1.0):
            out["twin_detwinning"] = p["twin_detwinning_share"]
    planes = p.get("cleavage") or "none"
    if planes != "none":
        if not p.get("cleavage_strength"):
            raise ValueError("cleavage strength: the normal stress that opens the planes")
        band = p.get("cleavage_band")
        if not (p.get("cleavage_energy") and (band or p.get("cleavage_length"))):
            raise ValueError(
                "cleavage: the fracture energy of the planes and the size of the elements "
                "that dissipate it (those of the voxels or of the mesh), or the width of "
                "the damage band of the planes"
            )
        spec = {"planes": planes, "strength": p["cleavage_strength"]}
        spec["energy"] = p["cleavage_energy"]
        if band:  # regularized by their microdamage: the energy is that of their band
            spec["regularization"] = {"width": band}
        else:
            spec["length"] = p["cleavage_length"]
        if p.get("cleavage_viscosity"):
            spec["K"], spec["n"] = _two(
                p["cleavage_viscosity"], "viscosity of the planes", "K", "n"
            )
        if p.get("cleavage_closure") is False:
            spec["closure"] = False
        elif p.get("cleavage_closure_share") not in (None, 1.0):
            spec["closure"] = p["cleavage_closure_share"]
        if p.get("cleavage_friction"):
            spec["friction"] = p["cleavage_friction"]
        if p.get("cleavage_coupling"):
            spec["coupling"] = p["cleavage_coupling"]
        out["cleavage"] = spec
    return out


def _porous(p: dict) -> dict:
    """The porous plasticity of the form (:meth:`PorousPlasticity.from_config`)."""
    plastic = _plasticity({**p, "hill": None})
    out = {"criterion": p["criterion"], "porosity": p["porosity"], **plastic}
    if p["criterion"] == "gurson":
        q = _two(p.get("gurson") or [1.5, 1.0], "q1, q2", "q1", "q2", "q3")
        out.update(zip(("q1", "q2", "q3"), q, strict=False))
        if p.get("coalescence"):
            out["fc"], out["fF"] = _two(p["coalescence"], "coalescence", "fc", "fF")
    elif p["criterion"] == "rousselier":
        values = _two(p.get("rousselier"), "Rousselier", "σ1", "D", "q2")
        out.update(zip(("sigma1", "D", "q2"), values, strict=False))
    else:
        out["C"], out["F"] = _two(p.get("elliptic"), "elliptic criterion", "C", "F")
    kind = p.get("nucleation") or "none"
    if kind != "none":
        names = {"gaussian": ("fN", "eN", "sN"), "constant": ("A",), "exponential": ("B", "e0")}
        values = list(p.get("nucleation_coefficients") or [])
        need = names[kind]
        if len(values) not in (len(need), len(need) + 1):
            raise ValueError(f"nucleation coefficients: {', '.join(need)} (and the threshold)")
        out["nucleation"] = {"type": kind, **dict(zip(need, values, strict=False))}
        if len(values) > len(need):
            out["nucleation"]["yield"] = values[-1]
    return out


def _zmat(p: dict, folder: Path) -> str:
    """The material file of Z-set of the form (its text written in ``folder``), by path."""
    text = (p.get("zmat") or "").strip()
    if not text:
        raise ValueError("Z-set material file: its text (***behavior ... ***return) or its path")
    if "\n" not in text and Path(text).is_file():
        return str(Path(text).resolve())
    from morphogen.homogenization.zmat import describe

    how = describe(text)
    if how["translation"] == "mfront" and importlib.util.find_spec("mgis") is None:
        raise ValueError(
            f"Z-set material file: morphoGen's kernels do not integrate it ({how['reason']}); "
            "it needs MFront and MGIS (conda install -c conda-forge mgis mfront cxx-compiler)"
        )
    path = folder / "material.zmat"
    path.write_text(text + "\n", encoding="utf-8")
    return str(path)


def _mfront(p: dict) -> dict:
    source = (p.get("mfront") or "").strip()
    if not source:
        raise ValueError("MFront source or library: its path")
    out = {"source" if source.endswith(".mfront") else "library": source}
    if p.get("mfront_behaviour"):
        out["behaviour"] = p["mfront_behaviour"].strip()
    for key, name in (
        ("mfront_parameters", "parameters"),
        ("mfront_properties", "material_properties"),
    ):
        value = _json(p.get(key), name)
        if value:
            out[name] = value
    return out


def _behaviour(p: dict, elastic: dict, folder: Path) -> dict:
    """The material (a mapping of the configurations) of the behaviour of the form, with the
    elastic constants ``elastic`` (young, poisson), or that of the workspace chosen (with its
    own)."""
    kind = p["behaviour"]
    if kind == "workspace":
        return p["kept"]["material"]
    base = {"young": float(elastic["young"]), "poisson": float(elastic["poisson"])}
    if kind == "elastic":
        return base
    if kind == "mises":
        out = {**base, "plasticity": _plasticity(p)}
        coupled = p.get("coupled_damage") or "none"
        if coupled != "none":
            out["damage"] = _coupled_damage(p, coupled)
        return out
    if kind == "porous":
        return {**base, "porous_plasticity": _porous(p)}
    if kind == "crystal":
        cubic, hexagonal = p.get("cubic"), p.get("hexagonal")
        if cubic and hexagonal:
            raise ValueError("elasticity of the crystal: cubic or hexagonal, not both")
        if cubic:
            if len(cubic) != 3:
                raise ValueError("cubic elasticity: C11, C12, C44")
            base = {"cubic": [float(c) for c in cubic]}
        if hexagonal:
            if len(hexagonal) != 5:
                raise ValueError("hexagonal elasticity: C11, C12, C13, C33, C44")
            base = {"hexagonal": [float(c) for c in hexagonal]}
        return {**base, "crystal_plasticity": _crystal(p)}
    if kind == "zmat":
        return {"zmat": {"file": _zmat(p, folder)}}
    if kind == "viscoelastic":
        series = {
            "shear": _value_pairs(p.get("prony_shear"), "Prony series of the shear modulus"),
            "bulk": _value_pairs(p.get("prony_bulk"), "Prony series of the bulk modulus"),
        }
        series = {k: v for k, v in series.items() if v}
        if not series:
            raise ValueError("Prony series: at least one pair (shear or bulk)")
        return {**base, "viscoelasticity": series}
    if kind == "hyperelastic":
        return _hyperelastic(p, base)
    if kind == "damage":
        law = {
            "strength": p["strength"],
            "softening": p["softening"],
            "equivalent": p["equivalent"],
            "viscosity": p.get("damage_viscosity") or 0.0,
            "split": p.get("damage_split") or "none",
        }
        if p.get("fracture_energy"):
            if not p.get("damage_length"):
                raise ValueError(
                    "size of the elements of the band: that of the voxels or elements over "
                    "which the fracture energy is dissipated"
                )
            law.update(fracture_energy=p["fracture_energy"], length=p["damage_length"])
        elif p.get("ultimate"):
            law["ultimate"] = p["ultimate"]
        else:
            raise ValueError("damage: the ultimate strain of the softening, or a fracture energy")
        return {**base, "damage": law}
    if kind == "mazars":
        return {
            **base,
            "damage": {
                "law": "mazars",
                "threshold": p["mazars_threshold"],
                "tension": list(_two(p["mazars_tension"], "tension", "A_t", "B_t")),
                "compression": list(_two(p["mazars_compression"], "compression", "A_c", "B_c")),
                "beta": p.get("mazars_beta") or 1.0,
            },
        }
    if kind == "mfront":
        return {**base, "mfront": _mfront(p)}
    raise ValueError(f"behaviour: one of {', '.join(c[0] for c in _behaviours())}")


def _coupled_damage(p: dict, law: str) -> dict:
    """The damage coupled to the plasticity of the form
    (:class:`~morphogen.homogenization.laws.DuctileDamage`)."""
    out: dict = {"law": law}
    if law == "lemaitre":
        values = list(p.get("lemaitre") or [])
        if len(values) not in (1, 2):
            raise ValueError("Lemaitre's S, s: the strength S (and the exponent s)")
        out["S"] = values[0]
        if len(values) == 2:
            out["s"] = values[1]
        if p.get("damage_threshold"):
            out["threshold"] = p["damage_threshold"]
        if p.get("closure") is not None:
            out["closure"] = p["closure"]
    else:
        values = list(p.get("kachanov") or [])
        if len(values) not in (2, 3):
            raise ValueError("Kachanov's A, r (and k): the resistance A and the exponents")
        out.update(A=values[0], r=values[1])
        if len(values) == 3:
            out["k"] = values[2]
        weights = list(p.get("hayhurst") or [])
        if weights:
            if len(weights) != 2:
                raise ValueError("Hayhurst's weights: α and β")
            out.update(alpha=weights[0], beta=weights[1])
    if p.get("critical_damage"):
        out["critical"] = p["critical_damage"]
    if p.get("coupled_viscosity"):
        out["viscosity"] = p["coupled_viscosity"]
    return out


def _value_pairs(values, name: str) -> list[list[float]]:
    """The pairs of a list of floats of a form, as lists of two values."""
    values = [float(v) for v in values or []]
    if len(values) % 2:
        raise ValueError(f"{name}: pairs of values")
    return [[values[i], values[i + 1]] for i in range(0, len(values), 2)]


def _hyperelastic(p: dict, base: dict) -> dict:
    """The material of a hyperelastic energy of the form: with the elastic constants
    ``base`` when it has no parameter of its own (neo-Hookean), else with the elasticity
    of its parameters and the bulk modulus of the form (by default that of ``base``)."""
    model = p["hyper_model"]
    values = [float(v) for v in p.get("hyper_parameters") or []]
    extra = {}
    mullins = [float(v) for v in p.get("hyper_mullins") or []]
    if mullins:
        if len(mullins) != 3:
            raise ValueError("Mullins softening: r, m, β")
        extra["mullins"] = mullins
    if not values:
        if model != "neo_hookean":
            raise ValueError(f"parameters of the energy: those of {model.replace('_', ' ')}")
        return {**base, "hyperelasticity": {"model": model, **extra}}
    if model in ("ogden", "hyperfoam"):
        terms = _value_pairs(values, f"parameters of the energy ({model})")
        law = {"model": model, "terms": terms}
    else:
        names, least = HYPERELASTIC_PARAMETERS[model]
        if not least <= len(values) <= len(names):
            raise ValueError(f"parameters of the energy: {', '.join(names)}")
        law = {"model": model, **dict(zip(names, values, strict=False))}
    if model == "holzapfel":
        fibres = [float(v) for v in p.get("hyper_fibres") or []]
        if not fibres or len(fibres) % 3:
            raise ValueError("directions of the fibres: x, y, z of each family")
        law["fibres"] = [fibres[k : k + 3] for k in range(0, len(fibres), 3)]
    bulk = p.get("hyper_bulk")
    if not bulk:  # that of the elastic constants of the form
        bulk = base["young"] / (3.0 * (1.0 - 2.0 * base["poisson"]))
    return {"hyperelasticity": {**law, **extra, "bulk": float(bulk)}}


def _elastic(item: dict, value) -> dict:
    """The elastic constants of the phases of a morphology from the form (phase -> "void" or
    {young, poisson}); without them, those the page proposes (1 for the first phase, 10 for
    the others, voids for the phases named so)."""
    if value:
        return {int(k): v for k, v in value.items()}
    phases = item.get("phases") or {"0": "phase 0", "1": "phase 1"}
    return {
        int(k): "void" if name == "void" else {"young": 1.0 if i == 0 else 10.0, "poisson": 0.3}
        for i, (k, name) in enumerate(phases.items())
    }


def _materials(p: dict, item: dict, folder: Path) -> tuple[dict, int]:
    """The materials of the phases (mappings of the configurations) and the phase of the
    behaviour of the form: the first that is not a void, unless the form names it; the
    materials of the configurations of the form replace those of the phases they name."""
    elastic = _elastic(item, p.get("materials"))
    phase = p.get("phase")
    if phase is None:
        phase = next((k for k, m in elastic.items() if m != "void"), None)
        if phase is None:
            raise ValueError("materials: every phase is a void")
    elif phase not in elastic:
        raise ValueError(f"phase {phase}: not a phase of the morphology ({sorted(elastic)})")
    if elastic[phase] == "void":
        raise ValueError(f"phase {phase} is a void: choose a solid phase for the behaviour")
    out = dict(elastic)
    out[phase] = _behaviour(p, elastic[phase], folder)
    out.update(_extra(p))
    return out, int(phase)


def _extra(p: dict) -> dict:
    """The materials of the configurations of the form (phase -> material)."""
    extra = _json(p.get("behaviours"), "materials of the configurations")
    if extra is None:
        return {}
    if not isinstance(extra, dict):
        raise ValueError("materials of the configurations: a mapping phase -> material")
    out = {}
    for key, spec in extra.items():
        try:
            out[int(key)] = spec
        except ValueError:
            raise ValueError(f"materials of the configurations: phase {key!r}") from None
    return out


BEHAVIOURS_FIELD = Field(
    "behaviours",
    "text",
    label="materials of the configurations (JSON)",
    doc='phase -> material of homogenization.materials, replacing those above: e.g. {"1": '
    '{"cubic": [1.7, 1.2, 0.75], "crystal_plasticity": {"tau0": 0.001, "K": 0.0002, '
    '"n": 10}}} (all their options: tables, nucleation, several nonlinear phases)',
)


# -- figures ---------------------------------------------------------------------------------


def _driven(res) -> tuple[np.ndarray, np.ndarray, str, str, int]:
    """The strain component that varies most along a response and the stresses (true
    measures in finite strains), their names and the index of the component."""
    true = res.finite and res.log_strain is not None and res.cauchy is not None
    strain = np.asarray(res.log_strain if true else res.strain, dtype=float)
    stress = np.asarray(res.cauchy if true else res.stress, dtype=float)
    i = int(np.argmax(np.ptp(strain, axis=0)))
    e_name, s_name = ("LE", "TS") if true else ("E", "S")
    return strain, stress, e_name, s_name, i


def _phase_figure(res, names: dict, title: str = ""):
    """The mean stress of each phase against the macroscopic strain (the component that
    varies most), and its mean cumulated plastic strain against time."""
    from matplotlib.figure import Figure

    strain, stress, e_name, s_name, i = _driven(res)
    j = int(np.argmax(np.ptp(stress, axis=0)))
    key = "cauchy" if s_name == "TS" else "stress"
    fig = Figure(figsize=(9.0, 3.6), layout="constrained")
    ax, ax2 = fig.subplots(1, 2)
    ax.plot(strain[:, i], stress[:, j], "k-", lw=2.0, label="macroscopic")
    for phase, series in sorted(res.phases.items()):
        label = f"{phase}: {names.get(str(phase), phase)}"
        ax.plot(
            strain[:, i],
            np.asarray(series.get(key, series["stress"]))[:, j],
            lw=1.4,
            marker=".",
            ms=3,
            label=label,
        )
        ax2.plot(res.times, series["plastic"], lw=1.4, marker=".", ms=3, label=label)
    ax.set_xlabel(e_name + COMPONENTS[i])
    ax.set_ylabel(f"mean {s_name}{COMPONENTS[j]} of the phases")
    ax2.set_xlabel("time")
    ax2.set_ylabel("mean cumulated plastic strain p")
    for a in (ax, ax2):
        a.grid(alpha=0.3)
        a.legend(fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def _comparison_figure(curves: list[tuple[str, object, object]], title: str = ""):
    """The responses of the NTFA (dashed) against the full field (solid) along paths:
    ``(name, full field, NTFA)``."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.0 * len(curves), 3.8), layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, len(curves)))
    for ax, (name, ref, red) in zip(axes, curves, strict=True):
        strain, stress, e_name, s_name, i = _driven(ref if ref is not None else red)
        j = int(np.argmax(np.ptp(stress, axis=0)))
        if ref is not None:
            ax.plot(strain[:, i], stress[:, j], "k-", lw=1.8, label="full field")
        ax.plot(
            red.strain[:, i],
            red.stress[:, j],
            "--",
            color="#d6604d",
            lw=1.8,
            marker=".",
            ms=4,
            label="NTFA",
        )
        ax.set_xlabel(e_name + COMPONENTS[i])
        ax.set_ylabel(s_name + COMPONENTS[j])
        ax.set_title(name, fontsize=10)
        ax.locator_params(axis="x", nbins=5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def _point_figure(res, reference=None, title: str = ""):
    """The curves of a material point (:func:`morphogen.viz.figures.path_figure`) with a
    reference curve (strain, stress) on the stress-strain axes."""
    from morphogen.viz.figures import path_figure

    fig = path_figure(res, title=title)
    if reference is not None:
        ax = fig.axes[0]
        ax.plot(reference[:, 0], reference[:, 1], "ko", ms=3.5, mfc="none", label="reference")
        ax.legend(fontsize=8)
    return fig


def _at_times(ref, times) -> np.ndarray:
    """The stresses of a response at the ``times`` of another one (the planned ends of the
    steps: a solver that divided steps has more)."""
    rows = [int(np.argmin(np.abs(np.asarray(ref.times) - t))) for t in times]
    return np.asarray(ref.stress)[rows]


def _relative_error(ref, red, k: int) -> tuple[float, float]:
    """The largest and the final gaps of the component ``k`` of the stresses of a response
    (``red``) to a reference at its times, relative to the largest of the reference."""
    stress = _at_times(ref, red.times)[:, k]
    scale = float(np.abs(stress).max()) or 1.0
    gap = np.abs(np.asarray(red.stress)[:, k] - stress)
    return float(gap.max() / scale), float(gap[-1] / scale)


# -- outputs -------------------------------------------------------------------------------


def _steps_written(res, folder: Path, name: str, bc: str, which: str) -> list[str]:
    """The fields of a response (:attr:`PathResult.fields`) as files: those of the steps
    (``which``: ``"steps"``, the ends of the segments ``"segments"``, ``"end"``) as
    ``<path>_step<k>.vtu`` (``.vti`` on voxels), and the end as
    ``sample_0000_<path>_<bc>.vtu``, the fields of the simulations of the interface (von
    Mises stress, components of the stress, strain and plastic strain, cumulated plastic
    strain, porosity, displacement)."""
    from morphogen.pipeline import write_path_fields
    from morphogen.viz.workflow import path_fields

    times = sorted(res.fields)
    if not times:
        return []
    ends = np.asarray(res.path.times[1:], dtype=float)

    def named(t):
        arrays = res.fields[t]
        out = path_fields(arrays)
        for extra in ("porosity", "orientation"):
            if extra in arrays:
                out[extra] = np.asarray(arrays[extra])
        return out

    written, steps = [], []
    strain, _, e_name, _, i = _driven(res)
    all_times = np.asarray(res.times, dtype=float)
    for k, t in enumerate(times, start=1):
        if which == "steps" or (which == "segments" and np.isclose(t, ends).any()):
            target = folder / f"{name}_step{k:03d}.vtu"
            written.append(write_path_fields(target, res.support, named(t)).name)
            j = int(np.argmin(np.abs(all_times - t)))
            label = f"t = {t:.4g}, {e_name}{COMPONENTS[i]} = {float(strain[j, i]):.4g}"
            steps.append({"file": written[-1], "time": float(t), "label": label})
    last = write_path_fields(folder / f"{STEM}_{name}_{bc}.vtu", res.support, named(times[-1]))
    if steps:  # the steps that the 3D view of the page goes through, in their order
        manifest = {"path": name, "steps": steps}
        (folder / f"{name}_steps.json").write_text(json.dumps(manifest), encoding="utf-8")
    return [*written, last.name]


def _register(ctx, pattern: str, kind: str) -> None:
    """The figures or videos written in the folder of the result (``pattern``) for the page."""
    for path in sorted(ctx.folder.glob(pattern)):
        title = path.stem.removeprefix(STEM + "_").replace("_", " ")
        entry = {"title": title, "url": ctx.url(path.name)}
        if kind == "videos":
            entry["format"] = path.suffix[1:]
        getattr(ctx, kind).append(entry)


# -- the loading path of a morphology --------------------------------------------------------

FIELDS_SHOWN = (
    ("von_mises", "von Mises stress"),
    ("p", "cumulated plastic strain p"),
    ("displacement", "displacement (norm)"),
    *((f"stress_{c}", f"stress {c}") for c in COMPONENTS),
    *((f"strain_{c}", f"strain {c}") for c in COMPONENTS),
    *((f"plastic_strain_{c}", f"plastic strain {c}") for c in COMPONENTS),
)
CMAPS = ("viridis", "plasma", "inferno", "magma", "cividis", "turbo", "coolwarm", "RdBu_r", "jet")
#: The fields of the interfaces that debond, for the video of a path
INTERFACE_FIELDS = (
    ("interface_damage", "damage of the interfaces (those that debond)"),
    ("interface_opening", "opening of the interfaces (those that debond)"),
)
#: The fields of the crystals that twin or cleave
CRYSTAL_FIELDS = (
    ("twin_fraction", "twinned fraction (crystals that twin)"),
    ("twin_reoriented", "turned lattices (the share of the points that their twins turned)"),
    ("twin_misorientation", "disorientation of the twins (lattices that their twins turn)"),
    ("cleavage_damage", "damage of the cleavage planes (crystals that cleave)"),
    ("cleavage_opening", "opening of the cleavage planes (crystals that cleave)"),
    ("cleavage_microdamage", "microdamage of the cleavage planes (planes with a damage band)"),
)
FATIGUE = (
    ("none", "none"),
    ("crossland", "Crossland: amplitude of sqrt(J2), largest hydrostatic stress"),
    ("sines", "Sines: amplitude of sqrt(J2), mean hydrostatic stress"),
    ("dang_van", "Dang Van: shear and hydrostatic stress at each time of the cycle"),
    ("findley", "Findley: critical plane (shear amplitude and largest normal stress)"),
    ("matake", "Matake: plane of the largest shear amplitude"),
)
JUDGED = {"fatigue": ["crossland", "sines", "dang_van", "findley", "matake"]}
DEBOND = {"method": "fe", "interfaces": True}


def _interfaces(p: dict, edge: float) -> dict | None:
    """The interfaces that debond along the path of the form (``homogenization.nonlinear.
    interfaces``), or None: a bilinear law of their strength and fracture energy (blank:
    that of an opening at failure of 2 % of the edge of the domain), Coulomb friction of
    the lips that close, a viscous regularization."""
    if not p.get("interfaces"):
        return None
    strength = p["interface_strength"]
    if not strength or not strength > 0:
        raise ValueError("strength of the interfaces: a positive stress")
    energy = p["interface_energy"]
    if energy is None:
        energy = 0.01 * strength * edge
    if not energy > 0:
        raise ValueError("fracture energy of the interfaces: positive")
    out = {"normal_strength": strength, "mode_I": energy}
    for key, name in (("friction", "interface_friction"), ("viscosity", "interface_viscosity")):
        value = p.get(name) or 0.0
        if value < 0:
            raise ValueError(f"{key} of the interfaces: 0 or more")
        if value:
            out[key] = value
    if p["interface_faces"] != "phases":
        out["faces"] = p["interface_faces"]
    return out


def _fatigue(p: dict) -> dict | None:
    """The fatigue of the points along the path of the form (``homogenization.fatigue``),
    or None: a criterion and its two fatigue limits, the times of the cycle."""
    kind = p.get("fatigue") or "none"
    if kind == "none":
        return None
    names = {"bending": "fatigue_bending", "torsion": "fatigue_torsion",
             "repeated": "fatigue_repeated"}  # fmt: skip
    limits = ("torsion", "repeated") if kind == "sines" else ("bending", "torsion")
    out: dict = {"criterion": kind}
    for key in limits:
        value = p.get(names[key])
        if not value or not value > 0:
            raise ValueError(f"fatigue limit ({key}): a positive stress")
        out[key] = value
    if p.get("fatigue_cycle"):
        out["cycle"] = [float(v) for v in p["fatigue_cycle"]]
    return out


BEREMIN = {"beremin": True}
#: The stress that drives the cleavage of a point, in the criterion of Beremin
DRIVING = (
    ("principal", "the largest principal stress"),
    *((c[0], f"the normal stress of the planes {c[1]}") for c in CLEAVAGE_PLANES[1:]),
)


def _beremin(p: dict) -> dict | None:
    """The criterion of cleavage of Beremin along the path of the form
    (``homogenization.beremin``), or None."""
    if not p.get("beremin"):
        return None
    out = {}
    for key, label in (("modulus", "modulus of Weibull"), ("scale", "scale of the stresses"),
                       ("volume", "reference volume")):  # fmt: skip
        value = p.get(f"beremin_{key}")
        if not value or not value > 0:
            raise ValueError(f"cleavage (Beremin): {label}, a positive number")
        out[key] = value
    if (p.get("beremin_stress") or "principal") != "principal":
        out["planes"] = p["beremin_stress"]
        if p.get("c_over_a"):
            out["c_over_a"] = p["c_over_a"]
    if p.get("beremin_plastic") is not None:
        out["plastic"] = p["beremin_plastic"]
    return out


def _beremin_figure(res, found, title: str = ""):
    """The probability of cleavage along a path (the criterion of Beremin) and its Weibull
    stress, against the strain that the path drives."""
    from matplotlib.figure import Figure

    strain, _, e_name, _, i = _driven(res)
    x = strain[-found.times.size :, i]
    fig = Figure(figsize=(5.4, 3.6), layout="constrained")
    ax = fig.add_subplot()
    ax.plot(x, found.probability, "-o", ms=3, color="#b2182b")
    ax.set_xlabel(f"{e_name}{COMPONENTS[i]}")
    ax.set_ylabel("probability of cleavage", color="#b2182b")
    ax.set_ylim(-0.02, 1.02)
    other = ax.twinx()
    other.plot(x, found.weibull, "--", color="#2166ac")
    other.set_ylabel("Weibull stress", color="#2166ac")
    if title:
        ax.set_title(title, fontsize=9)
    return fig


def _fatigue_figure(cycle, title: str = ""):
    """The share of the volume of the points judged above each utilization of a cycle
    (:class:`~morphogen.homogenization.fatigue.FatigueResult`): the points beyond their
    fatigue limit are right of 1."""
    from matplotlib.figure import Figure

    judged = np.isfinite(cycle.utilization)
    values = cycle.utilization[judged]
    weights = np.ones(values.shape) if cycle.weights is None else cycle.weights[judged]
    order = np.argsort(values)
    above = 1.0 - np.cumsum(weights[order]) / weights.sum()
    fig = Figure(figsize=(5.2, 3.6), layout="constrained")
    ax = fig.add_subplot()
    ax.step(values[order], np.clip(above, 0.0, 1.0), where="post", color="#2166ac")
    ax.axvline(1.0, color="#b2182b", lw=1.0, ls="--", label="fatigue limit")
    ax.set_xlabel("utilization (equivalent stress / limit)")
    ax.set_ylabel("share of the volume above")
    ax.set_xlim(0.0, max(1.05, 1.05 * float(values.max())))
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    if title:
        ax.set_title(title, fontsize=9)
    return fig


@tool(
    "path",
    "mechanics",
    "Loading path of a morphology",
    "The macroscopic response of a morphology along a loading path (nonlinear "
    "homogenization): finite elements on the conforming mesh of its level set (periodic or "
    "kinematic conditions) or FFT on its voxels (periodic), in small or finite strains; the "
    "macroscopic components prescribed as stresses are unknowns (the periodic element of "
    "Z-set). On the mesh, the interfaces of the phases (or of the grains) may debond: "
    "cohesive zones that open, break and rub. A fatigue criterion judges the points along "
    "the cycle of the path (utilization, factor of safety), and the criterion of Beremin "
    "gives the probability of cleavage along it. The curves (macroscopic and "
    "mean of each phase), the values at the end, and the fields of the steps, drawn in the "
    "page in 3D (any field, any step; no other program needed) and written as files; a "
    "video of a field along the path on demand (here, or later in Export); the result "
    "joins the workspace as a simulation, with the configuration that runs it again.",
    [
        Field("source", "item", label="morphology", required=True),
        Field(
            "materials",
            "materials",
            of="source",
            doc="elastic constants of the phases (the unit of the stresses), or voids",
        ),
        Field(
            "phase",
            "int",
            label="phase of the behaviour",
            doc="blank: the first phase that is not a void (the matrix, the solid, the grains)",
        ),
        *_behaviour_fields("of that phase; the other phases are elastic, or voids"),
        BEHAVIOURS_FIELD,
        *_path_fields("tension"),
        Field(
            "method",
            "choice",
            "fe",
            choices=(
                ("fe", "finite elements on the conforming mesh of its geometry"),
                ("fft", "FFT on its voxels (periodic; B-bar or full integration)"),
            ),
        ),
        Field("voxels", "int", 16, label="voxels per edge", when={"method": "fft"}),
        Field(
            "integration",
            "choice",
            "bbar",
            when={"method": "fft"},
            choices=(
                ("bbar", "B-bar (F-bar in finite strains): no locking under plastic flow"),
                ("full", "full: 8 Gauss points per voxel (locks under isochoric flow)"),
            ),
        ),
        _device_field(when={"method": "fft"}),
        Field(
            "element_size",
            "float",
            unit="length",
            when={"method": "fe"},
            doc="largest edge of the elements (blank: 1/4 of the edge of the domain)",
        ),
        Field(
            "order",
            "choice",
            "2",
            label="elements",
            when={"method": "fe"},
            choices=(
                ("2", "10-node tetrahedra (accurate plastic flow)"),
                ("1", "4-node tetrahedra (fast; they lock under isochoric plastic flow)"),
            ),
        ),
        Field(
            "fe_integration",
            "choice",
            "full",
            label="integration",
            when={"method": "fe"},
            choices=(
                ("full", "full: every point of the elements"),
                (
                    "bbar",
                    "B-bar (F-bar in finite strains): the volume change of each 10-node "
                    "element as its mean, for nearly incompressible phases (elastomers)",
                ),
            ),
        ),
        Field(
            "bc",
            "choice",
            "periodic",
            label="boundary conditions",
            when={"method": "fe"},
            choices=(("periodic", "periodic"), ("kubc", "kinematic (u = E x on the boundary)")),
        ),
        Field(
            "kinematics",
            "choice",
            "small",
            choices=(
                ("small", "small strains"),
                ("finite", "finite strains (total Lagrangian; nominal and true measures)"),
            ),
        ),
        Field(
            "tolerance",
            "float",
            1e-6,
            label="Newton tolerance",
            doc="relative residual of the equilibrium and of the macroscopic stress",
        ),
        Field(
            "interfaces",
            "bool",
            False,
            label="the interfaces debond",
            when={"method": "fe"},
            doc="cohesive zones between the phases of the mesh (finite elements, every "
            "phase meshed): they open, soften and break, close again and rub along the path",
        ),
        Field(
            "interface_faces",
            "choice",
            "phases",
            label="interfaces",
            when=DEBOND,
            choices=(
                ("phases", "between the phases"),
                ("grains", "between the grains of a tessellation (and the phases)"),
            ),
        ),
        Field(
            "interface_strength",
            "float",
            0.002,
            label="strength of the interfaces",
            unit="stress",
            when=DEBOND,
            doc="the normal traction at which they start to soften (the same in shear)",
        ),
        Field(
            "interface_energy",
            "float",
            label="fracture energy of the interfaces",
            when=DEBOND,
            doc="per unit area (stress x length); blank: that of an opening at failure of "
            "2 % of the edge of the domain",
        ),
        Field(
            "interface_friction",
            "float",
            0.0,
            label="friction of the interfaces",
            when=DEBOND,
            doc="Coulomb coefficient of the lips that close again",
        ),
        Field(
            "interface_viscosity",
            "float",
            0.0,
            label="viscosity of the interfaces",
            unit="time",
            when=DEBOND,
            doc="a viscous regularization (0: none): the interfaces that break at once (a "
            "snap-back of the response) pass with a time of a few percent of the path",
        ),
        Field(
            "fatigue",
            "choice",
            "none",
            label="fatigue criterion",
            choices=FATIGUE,
            doc="high-cycle fatigue of the points of the solid phases along the cycle of "
            "the path: the equivalent stress of the criterion over its limit (the "
            "utilization), the factor of safety and the share of the volume beyond",
        ),
        Field(
            "fatigue_bending",
            "float",
            label="fatigue limit in bending",
            unit="stress",
            when={"fatigue": ["crossland", "dang_van", "findley", "matake"]},
            doc="fully reversed bending or tension-compression (R = -1)",
        ),
        Field(
            "fatigue_torsion",
            "float",
            label="fatigue limit in torsion",
            unit="stress",
            when=JUDGED,
            doc="fully reversed torsion (R = -1)",
        ),
        Field(
            "fatigue_repeated",
            "float",
            label="fatigue limit in repeated bending",
            unit="stress",
            when={"fatigue": "sines"},
            doc="the amplitude at the limit of a bending from zero (R = 0)",
        ),
        Field(
            "fatigue_cycle",
            "floats",
            label="times of the cycle",
            when=JUDGED,
            doc="start, end (blank: the whole path); e.g. the last cycle of a path that "
            "repeats, once the response is stabilized",
        ),
        Field(
            "beremin",
            "bool",
            False,
            label="probability of cleavage (Beremin)",
            doc="the local approach: the Weibull stress of the points that have yielded and "
            "the probability that the weakest link has failed, at each step of the path",
        ),
        Field(
            "beremin_modulus",
            "float",
            22.0,
            label="modulus of Weibull m",
            when=BEREMIN,
        ),
        Field(
            "beremin_scale",
            "float",
            label="scale of the cleavage stresses σu",
            unit="stress",
            when=BEREMIN,
            doc="of the reference volume: 63 % of them fail at it",
        ),
        Field(
            "beremin_volume",
            "float",
            label="reference volume V0",
            when=BEREMIN,
            doc="in the unit of the domain cubed; divided by the number of such cells that "
            "a part holds under the same loading",
        ),
        Field(
            "beremin_stress",
            "choice",
            "principal",
            label="stress that drives the cleavage",
            choices=DRIVING,
            when=BEREMIN,
            doc="the planes of the crystals of the grains, with their orientations",
        ),
        Field(
            "beremin_plastic",
            "float",
            label="plastic strain of the plastic zone",
            when=BEREMIN,
            doc="the points count once their cumulated plastic strain has passed it "
            "(blank: every point counts)",
        ),
        Field(
            "fields",
            "choice",
            "steps",
            label="fields written",
            choices=(
                ("steps", "at every step"),
                ("segments", "at the end of each segment"),
                ("end", "at the end"),
            ),
        ),
        Field(
            "field",
            "choice",
            "von_mises",
            label="field shown",
            doc="the field drawn first in the 3D view of the result (the view offers every "
            "field of the steps), and that of the video",
            choices=(*FIELDS_SHOWN, *INTERFACE_FIELDS, *CRYSTAL_FIELDS),
        ),
        Field(
            "video",
            "bool",
            False,
            label="video of the field along the path",
            doc="3D views of the deformed mesh (VTK), sections of the voxels; written on "
            "demand only (Export makes it later from the fields of the steps)",
        ),
        Field("cmap", "choice", "viridis", label="colormap", choices=CMAPS),
        Field(
            "clim",
            "floats",
            label="colour range",
            doc="low, high (blank: the 0.5 and 99.5 percentiles of the field along the path, "
            "so that a few singular points do not take the whole range)",
        ),
    ],
    order=1,
    groups=(
        ("Morphology and materials", "source"),
        ("Behaviour", "behaviour"),
        ("Loading", "loading"),
        ("Solver", "method"),
        ("Interfaces", "interfaces", True),
        ("Fatigue", "fatigue", True),
        ("Cleavage", "beremin", True),
        ("Output", "fields", True),
    ),
    menu="Loading paths",
)
def loading_path(p: dict, ctx) -> dict:
    from morphogen import pipeline
    from morphogen.config import load_config
    from morphogen.gui.api import _video_format
    from morphogen.homogenization import NonlinearFEProblem, NonlinearFFTProblem
    from morphogen.io.tables import write_table
    from morphogen.viz.figures import path_figure

    source = p["source"]
    micro = source["micro"]
    materials, phase = _materials(p, source, ctx.folder)
    name, spec = _path(p)
    fe = p["method"] == "fe"
    bc = p["bc"] if fe else "periodic"
    edge = float(np.max(micro.domain.lengths))
    key = f"{name}.{bc}"
    fmt = _video_format()
    views: dict = {"figures": [{"of": key, "kind": "curves"}], "cmap": p["cmap"]}
    if p["video"]:
        video = {"of": key, "color": p["field"], "formats": [fmt]}
        video.update({"frames": 60, "fps": 20} if fe else {"fps": 6})
        views["videos"] = [video]
    nonlinear = {"tolerance": p["tolerance"], "kinematics": p["kinematics"]}
    nonlinear["integration"] = p["fe_integration"] if fe else p["integration"]
    debonding = _interfaces(p, edge) if fe else None
    if debonding:
        nonlinear["interfaces"] = debonding
    elif p["video"] and p["field"] in dict(INTERFACE_FIELDS):
        raise ValueError(
            f"field of the video: {p['field']} is a field of the interfaces that debond "
            "(finite elements, 'the interfaces debond')"
        )
    judged = _fatigue(p)
    if p["video"] and p["field"] in dict(CRYSTAL_FIELDS):
        crystal = materials[phase].get("crystal_plasticity") or {}
        needed = "twins" if p["field"].startswith("twin") else "cleavage"
        if needed not in crystal:
            raise ValueError(
                f"field of the video: {p['field']} is a field of the crystals with {needed} "
                "(behaviour: crystal plasticity)"
            )
        banded = "regularization" in (crystal.get("cleavage") or {})
        if p["field"] == "cleavage_microdamage" and not banded:
            raise ValueError(
                "field of the video: cleavage_microdamage is that of planes with a damage "
                "band (give the width of the damage band of the planes)"
            )
        turns = ("twin_reoriented", "twin_misorientation")
        if p["field"] in turns and "twin_reorientation" not in crystal:
            raise ValueError(
                f"field of the video: {p['field']} is that of lattices that their "
                "twins turn (give the twinned fraction that turns the lattice)"
            )
    model = source.get("model") or {"type": "tpms", "volume_fraction": 0.5}  # a stand-in
    data: dict = {
        "model": model,
        "domain": source.get("domain")
        or {"size": [float(v) for v in micro.domain.lengths], "periodic": True},
        "seed": int(source.get("seed") or 1),
        "homogenization": {
            "materials": {k: _relative(v, ctx.folder) for k, v in materials.items()},
            "method": p["method"],
            "boundary_conditions": [bc],
            "properties": [],
            "paths": {name: spec},
            "nonlinear": nonlinear,
            "fields": True,
        },
        "outputs": {"directory": str(ctx.folder), "name": "sample", "tables": ["xlsx"]},
        "visualization": views,
    }
    if judged:
        data["homogenization"]["fatigue"] = judged
    weakest = _beremin(p)
    if weakest:
        data["homogenization"]["beremin"] = weakest
    if fe:
        size = p["element_size"] or edge / 4
        if not edge / 200 <= size <= edge:
            raise ValueError(f"element size: between {edge / 200:g} and {edge:g}")
        data["mesh"] = {"size": size, "order": int(p["order"])}
        data["outputs"]["conforming_mesh"] = ["vtu"]
    else:
        n = p["voxels"]
        if not 4 <= n <= 128:
            raise ValueError("voxels per edge: from 4 to 128 for the loading paths")
        data["voxels"] = {"shape": n}
        data["homogenization"]["device"] = p["device"]
    cfg = load_config(data, base_dir=ctx.folder)  # the errors of the form, before solving
    homog = cfg.homogenization
    path = homog.paths[name]
    options = homog.options()["nonlinear"]
    ctx.progress(0.02, "conforming mesh" if fe else "voxels")
    mesh = image = None
    info: dict = {}
    if fe:
        mesh, info = pipeline._conforming_mesh(micro, cfg)
        from morphogen import io

        io.save_mesh(mesh, ctx.folder / f"{STEM}_tet.vtu")
        interfaces = options.pop("interfaces", None)
        if interfaces:
            from morphogen.fracture.periodic import CohesiveCell
            from morphogen.homogenization.apparent import _interface_options

            law, cell = _interface_options(interfaces, mesh, homog.solver)
            problem = CohesiveCell(
                mesh, homog.materials, law, bc, tol=homog.tolerance, **cell, **options
            )
        else:
            problem = NonlinearFEProblem(
                mesh, homog.materials, bc, solver=homog.solver, tol=homog.tolerance, **options
            )
    else:
        from morphogen.homogenization.materials import any_oriented

        _check_voxels(micro, cfg.voxels.shape, len(path.schedule()), "loading path on voxels")
        image = micro.voxelize(**cfg.voxels.kwargs())
        grains = {}
        if any_oriented(homog.materials) and getattr(micro, "orientations", None) is not None:
            labels = micro.labels(grid=image.grid())
            grains = {"grains": labels, "orientations": micro.orientations}
        info = {"voxels": int(np.prod(image.shape))}
        problem = NonlinearFFTProblem(image, homog.materials, "periodic", tol=homog.tolerance,
                                      device=homog.device, **grains, **options)  # fmt: skip
    t0, t_end = float(path.times[0]), float(path.times[-1])
    started = time.perf_counter()

    def monitor(state):
        fraction = (state.time - t0) / (t_end - t0)
        ctx.progress(0.1 + 0.75 * fraction, f"t = {state.time:.4g} of {t_end:.4g}")

    ctx.progress(0.1, "first step")
    res = problem.run(path, name=name, fields="steps", monitor=monitor)
    solve_s = time.perf_counter() - started
    cycle = None
    if homog.fatigue:
        from morphogen.homogenization.fatigue import path_fatigue

        ctx.progress(0.85, "fatigue of the cycle")
        cycle = path_fatigue(res, homog.fatigue["criteria"], cycle=homog.fatigue["cycle"])
        res.scalars.update({f"fatigue_{n}": v for n, v in cycle.components().items()})
        res.fields[max(res.fields)].update(cycle.fields())  # with the fields of the end
    found = None
    if homog.beremin:
        from morphogen.homogenization.beremin import path_beremin
        from morphogen.pipeline import _point_orientations

        ctx.progress(0.85, "probability of cleavage")
        labelled = (grains["grains"], grains["orientations"]) if not fe and grains else None
        euler = _point_orientations(mesh, labelled)
        found = path_beremin(res, homog.beremin["criterion"], orientations=euler)
        res.series.update(found.series(res.times) or {})
        res.fields[max(res.fields)].update(found.fields())
    ctx.progress(0.86, "fields and figures")
    _steps_written(res, ctx.folder, name, bc, p["fields"])
    rows = res.rows()
    write_table(ctx.folder / f"{STEM}_{name}_{bc}.xlsx", rows)
    values = res.components()
    values.update(
        steps=res.n_steps,
        iterations=int(np.sum(res.iterations)),
        cuts=int(res.info.get("cuts", 0)),
        solve_s=round(solve_s, 2),
        **info,
    )
    record = {"realization": 0, "seed": source.get("seed") or "", **micro.summary()}
    record.update({f"{key}.{k}": v for k, v in res.components().items()}, **info)
    pipeline._write_results(ctx.folder, [record], [res.row()], 0, 1, cfg.outputs.tables)
    title = f"{name} of {source['title']}"
    ctx.figure(path_figure(res, title=title), f"{STEM}_{name}_{bc}_curves", "curves")
    if res.phases:
        names = source.get("phases") or {}
        ctx.figure(_phase_figure(res, names, title), "phases", "mean of each phase")
    if cycle is not None and np.isfinite(cycle.utilization).any():
        ctx.figure(_fatigue_figure(cycle, title), "fatigue", "fatigue of the cycle")
    if found is not None:
        ctx.figure(_beremin_figure(res, found, title), "beremin", "probability of cleavage")
    if p["video"]:
        from dataclasses import replace

        from morphogen.viz.workflow import visualize

        ctx.progress(0.9, "video")
        clim = _colour_range(res, p["field"], p.get("clim"))
        view = replace(cfg.visualization.videos[0], clim=clim)
        spec_video = replace(cfg.visualization, figures=[], videos=[view])
        if clim is not None:
            data["visualization"]["videos"][0]["clim"] = list(clim)
        try:
            if fe:  # VTK aborts the process when it cannot render: ask it first
                from morphogen.viz.video import NO_RENDERING, vtk_can_render

                if not vtk_can_render():
                    raise RuntimeError(NO_RENDERING)
            visualize(spec_video, STEM, ctx.folder, image=image, mesh=mesh, results={key: res})
        except (RuntimeError, ImportError, ValueError) as exc:
            import warnings

            warnings.warn(f"no video of the field: {exc}", stacklevel=1)
        _register(ctx, f"{STEM}_{name}_{bc}_*.{fmt}", "videos")
    ctx.table(rows, f"{STEM}_{name}_{bc}", f"curves of the path {name} ({bc})")
    _configuration(data, source, ctx.folder)
    elastic = {str(k): v for k, v in _elastic(source, p.get("materials")).items()}
    method = "conforming mesh (FE)" if fe else "voxels (FFT)"
    _, stress, _, s_name, i = _driven(res)
    item = ctx.session.add(
        ctx.key,
        "simulation",
        f"{name} of #{source['number']}, {'FE' if fe else 'FFT'}",
        origin="mechanics",
        source=source["key"],
        property=None,
        method=p["method"],
        values=values,
        materials=elastic,
        behaviours={str(k): v for k, v in materials.items()},
        path={"name": name, **path.to_dict()},
        kinematics=p["kinematics"],
        phases=source.get("phases"),
        thumbnail=source.get("thumbnail"),
        info={
            "path": name,
            "method": method,
            "behaviour": f"{p['behaviour']} (phase {phase})",
            f"{s_name}{COMPONENTS[i]} at the end": float(stress[-1, i]),
            "p at the end": float(res.plastic[-1]),
            **({"interfaces": "debonding"} if debonding else {}),
            **(
                {"fatigue": f"{judged['criterion']}, utilization {cycle.largest:.3g}"}
                if cycle is not None
                else {}
            ),
            **(
                {"cleavage (Beremin)": f"probability {float(found.probability[-1]):.3g}"}
                if found is not None
                else {}
            ),
        },
        files={"folder": ctx.url("")},
    )
    ctx.items.append(ctx.session.public(item))
    # the fields drawn in the page at once: those of the steps, in the 3D view
    return {"values": values, "fields3d": {"field": p["field"], "cmap": p["cmap"]}}


def _check_voxels(micro, n: int, steps: int, what: str) -> None:
    """Refuse a nonlinear simulation on ``n`` voxels along the longest edge that would need
    more than half of the free memory: per voxel, eight integration points (their strains,
    stresses, tangents and two sets of internal variables: about 8 kB) and the fields of
    the ``steps`` kept (about 200 B each)."""
    from morphogen.gui.api import _voxels

    _voxels(micro.domain, n, 8.0e3 + 200.0 * steps, what)


def _colour_range(res, field: str, given=None) -> tuple[float, float] | None:
    """The colour range of a field along a path: the one given (low, high), else its 0.5
    and 99.5 percentiles over the steps (zero below for fields that do not change sign);
    None when the field is uniform."""
    if given:
        if len(given) != 2 or not given[0] < given[1]:
            raise ValueError("colour range: low, high (low < high)")
        return float(given[0]), float(given[1])
    from morphogen.viz.workflow import field_values, path_fields

    values = np.concatenate(
        [np.ravel(field_values(path_fields(res.fields[t]), field)) for t in sorted(res.fields)]
    )
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    lo, hi = (float(v) for v in np.percentile(values, [0.5, 99.5]))
    lo = 0.0 if values.min() >= 0.0 else lo
    return (lo, hi) if hi > lo else None


def _relative(spec, folder: Path):
    """A material of the configuration with the files of the folder by their names (the
    configuration is written beside them)."""
    if isinstance(spec, dict) and isinstance(spec.get("zmat"), dict):
        file = Path(str(spec["zmat"].get("file", "")))
        if file.parent == Path(folder):
            return {**spec, "zmat": {**spec["zmat"], "file": file.name}}
    return spec


def _configuration(data: dict, source: dict, folder: Path) -> None:
    """The configuration that runs the simulation again (``morphogen run``), when the
    morphology is the realization of a model."""
    import yaml

    if source.get("model") is None:
        return
    data = json.loads(json.dumps(data))
    data["outputs"] = {**data["outputs"], "directory": "simulation"}
    (folder / "configuration.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )


# -- the reduced model of the NTFA ---------------------------------------------------------

TRAINING = (
    *((f"tension_{c}", f"tension {c}") for c in COMPONENTS[:3]),
    *((f"shear_{c}", f"shear {c}") for c in COMPONENTS[3:]),
)


@tool(
    "ntfa",
    "mechanics",
    "NTFA reduced model",
    "The reduced model of the nonuniform transformation field analysis (NTFA; Michel & "
    "Suquet 2003, 2004) of a morphology on its periodic voxels: the plastic strain of each "
    "von Mises phase sought on a few modes, the proper orthogonal decomposition of the "
    "plastic strain fields of full-field simulations along training paths; the reduced "
    "model then follows any path in milliseconds. Its responses against the full field "
    "along the training paths and along another path, with their gaps.",
    [
        Field("source", "item", label="morphology", required=True),
        Field(
            "materials",
            "materials",
            of="source",
            doc="elastic constants of the phases (the unit of the stresses), or voids",
        ),
        Field(
            "phase", "int", label="plastic phase", doc="blank: the first phase that is not a void"
        ),
        Field(
            "behaviour",
            "choice",
            "mises",
            choices=(("mises", "von Mises elasto-(visco)plasticity (Lemaitre-Chaboche)"), KEPT),
            doc="the NTFA reduces isotropic von Mises phases",
        ),
        _kept_field(),
        *_plastic_fields(MISES, MISES),
        BEHAVIOURS_FIELD,
        Field("voxels", "int", 16, label="voxels per edge", doc="of the periodic image"),
        _device_field(),
        Field("modes", "int", 6, label="modes per plastic phase"),
        Field(
            "training",
            "choices",
            ["tension_11", "shear_12"],
            choices=TRAINING,
            label="training paths",
            doc="a strain component to the amplitude, the other "
            "stresses free; their directions must span those of the paths followed",
        ),
        Field(
            "training_amplitude",
            "float",
            0.004,
            label="equivalent strain of the training",
            doc="the strain of the tensions, the engineering strain of the shears √3 times "
            "it (the same von Mises equivalent strain); the original NTFA stiffens as the "
            "plastic flow spreads: at a few times the yield strain, a few percent of the "
            "full field",
        ),
        Field("training_increments", "int", 8, label="increments of the training paths"),
        *_path_fields("cycle", prefix="test_", amplitude=0.003),
        Field(
            "compare",
            "bool",
            True,
            label="the full field along the path too",
            doc="the gap of the reduced model (one more full-field simulation)",
        ),
        Field("tolerance", "float", 1e-6, label="Newton tolerance of the full field"),
    ],
    order=3,
    groups=(
        ("Morphology and materials", "source"),
        ("Behaviour", "behaviour"),
        ("Training", "voxels"),
        ("Test path", "test_loading"),
    ),
    menu="Loading paths",
)
def ntfa(p: dict, ctx) -> dict:
    from morphogen.homogenization import NonlinearFFTProblem, train_ntfa
    from morphogen.homogenization.materials import materials_from_config
    from morphogen.homogenization.paths import LoadingPath

    source = p["source"]
    micro = source["micro"]
    if not micro.domain.periodic:
        raise ValueError("NTFA: a periodic morphology (the model is trained on periodic voxels)")
    configs, phase = _materials(p, source, ctx.folder)
    if p["behaviour"] == "workspace" and set(configs[phase]) - {"young", "poisson", "plasticity"}:
        raise ValueError(f"the NTFA reduces isotropic von Mises phases: {p['kept']['title']} "
                         "is not one")  # fmt: skip
    materials = materials_from_config(configs)
    n = p["voxels"]
    if not 4 <= n <= 64:
        raise ValueError("voxels per edge: from 4 to 64 for the NTFA")
    if not p["training"]:
        raise ValueError("training paths: at least one")
    a, steps = p["training_amplitude"], p["training_increments"]
    if not a or steps < 1:
        raise ValueError("training paths: a nonzero amplitude and at least one increment")
    training = {}
    for name in p["training"]:
        kind, _, component = name.partition("_")
        strain = a if kind == "tension" else np.sqrt(3.0) * a  # the same equivalent strain
        spec = {"times": [0.0, 1.0], "increments": steps, "strain": {component: [0.0, strain]}}
        training[name] = LoadingPath.from_spec(spec)
    test_name, spec = _path(p, "test_")
    test = LoadingPath.from_spec(spec)
    if any(test.temperature):
        raise ValueError("NTFA: no change of temperature along the path")
    total = sum(len(t.schedule()) for t in training.values())
    _check_voxels(micro, n, total, "NTFA on voxels")  # the snapshots of the training
    total += len(test.schedule()) if p["compare"] else 0
    image = micro.voxelize(shape=n)
    done = [0]

    def monitor(state):
        done[0] += 1
        ctx.progress(0.05 + 0.9 * done[0] / total, f"full field: step {done[0]} of {total}")

    def full(path, name, keep):
        problem = NonlinearFFTProblem(image, materials, "periodic", tol=1e-8,
                                      newton_tol=p["tolerance"], integration="full",
                                      device=p["device"])  # fmt: skip
        times = [float(t) for t in path.schedule()] if keep else False
        return problem.run(path, name=name, fields=times, monitor=monitor)

    ctx.progress(0.02, "training: full-field paths")
    started = time.perf_counter()
    references, snapshots = {}, []
    for name, path in training.items():
        res = full(path, name, keep=True)
        snapshots += [
            np.asarray(res.fields[t]["plastic_strain"]).reshape(-1, 6) for t in sorted(res.fields)
        ]
        res.fields = {}
        references[name] = res
    model = train_ntfa(
        image,
        materials,
        list(training.values()),
        modes=p["modes"],
        newton_tol=p["tolerance"],
        snapshots=snapshots,
    )
    training_s = time.perf_counter() - started
    model.save(ctx.folder / "ntfa_model.npz")  # NTFAModel.load, homogenization.ntfa.save
    with zipfile.ZipFile(ctx.folder / "ntfa_model.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(ctx.folder / "ntfa_model.npz", "ntfa_model.npz")  # for the page
    values: dict = {"modes": model.n_modes, "training_s": round(training_s, 2)}
    values.update({f"energy_{r}": e for r, e in sorted(model.energy.items())})
    curves, tables = [], []
    for name, path in training.items():
        red = model.run(path, name=name)
        ref = references[name]
        _, _, _, _, k = _driven(ref)
        values[f"gap_{name}"], _ = _relative_error(ref, red, k)
        curves.append((f"training: {name}", ref, red))
        tables.append((name, ref, red))
    started = time.perf_counter()
    red = model.run(test, name=test_name)
    values["ntfa_ms"] = round(1e3 * (time.perf_counter() - started), 2)
    ref = None
    if p["compare"]:
        ctx.progress(0.6, "full field along the path")
        started = time.perf_counter()
        ref = full(test, test_name, keep=False)
        values["full_field_s"] = round(time.perf_counter() - started, 2)
        values["speed_up"] = values["full_field_s"] / max(values["ntfa_ms"] * 1e-3, 1e-9)
        _, _, _, _, k = _driven(ref)
        values[f"gap_{test_name}"], values[f"gap_{test_name}_end"] = _relative_error(ref, red, k)
    curves.append((test_name, ref, red))
    tables.append((test_name, ref, red))
    ctx.figure(
        _comparison_figure(curves, f"NTFA of {source['title']}: {model.n_modes} modes"),
        "ntfa",
        "NTFA (dashed) and full field",
    )
    for name, ref, red in tables:
        ctx.table(_comparison_rows(ref, red), f"ntfa_{name}", f"path {name}: NTFA and full field")
    values.update({f"S{c}_ntfa": v for c, v in zip(COMPONENTS, red.stress[-1], strict=True)})
    return {"values": values}


def _comparison_rows(ref, red) -> list[dict]:
    """The curves of the NTFA and of the full field (when given) along a path: the strain
    component that varies most and the stress components that vary."""
    _, stress, _, _, i = _driven(ref if ref is not None else red)
    spans = np.ptp(stress, axis=0)
    shown = [j for j in range(6) if spans[j] > 0.01 * spans.max()] or [i]
    full = _at_times(ref, red.times) if ref is not None else None
    rows = []
    for k, t in enumerate(red.times):
        row = {"time": float(t), f"E{COMPONENTS[i]}": float(red.strain[k, i])}
        for j in shown:
            if full is not None:
                row[f"S{COMPONENTS[j]} full"] = float(full[k, j])
            row[f"S{COMPONENTS[j]} NTFA"] = float(red.stress[k, j])
        row["p NTFA"] = float(red.plastic[k])
        rows.append(row)
    return rows


# -- a material point ------------------------------------------------------------------------


@tool(
    "point",
    "mechanics",
    "Material point",
    "One behaviour under a loading path at a material point (homogeneous strain and stress, "
    "no microstructure; the components prescribed as stresses solved for by Newton's "
    "method, as Zsim does): its curves, with the points of a test to calibrate its law. "
    "The behaviour joins the workspace (B), for the loading paths of morphologies, cracks "
    "and the inputs of other solvers.",
    [
        Field("young", "float", 1.0, label="Young's modulus E", unit="stress"),
        Field("poisson", "float", 0.3, label="Poisson's ratio ν"),
        Field(
            "expansion",
            "float",
            label="thermal expansion α",
            doc="per unit change of temperature (the dT of custom paths; blank: none)",
        ),
        *_behaviour_fields("with the elasticity above (a Z-set file: its own)"),
        Field(
            "orientation",
            "floats",
            [0.0, 0.0, 0.0],
            label="orientation of the crystal",
            unit="degrees",
            when={"behaviour": ["crystal", "workspace"]},
            doc="Bunge angles φ1, Φ, φ2 (of crystal plasticity, or of a learned law: its "
            "material turned)",
        ),
        Field(
            "material",
            "text",
            label="material of the configurations (JSON)",
            doc='replaces the behaviour of the form: e.g. {"young": 1, "poisson": 0.3, '
            '"plasticity": {"yield": 0.002, "kinematic": [{"C": 0.5, "D": 300}]}}',
        ),
        *_path_fields("cycle"),
        Field(
            "kinematics",
            "choice",
            "small",
            choices=(
                ("small", "small strains"),
                ("finite", "finite strains (nominal and true measures)"),
            ),
        ),
        Field(
            "reference",
            "text",
            label="points of a test",
            doc="strain, stress along the component of the path (true measures in finite "
            "strains), one pair per line (comma, space or tab; or semicolons with decimal "
            "commas): drawn with the curves, and their gap",
        ),
    ],
    order=4,
    groups=(
        ("Elasticity", "young"),
        ("Behaviour", "behaviour"),
        ("Loading", "loading"),
        ("Test", "reference", True),
    ),
    menu="Behaviours of materials",
)
def material_point(p: dict, ctx) -> dict:
    from morphogen.gui.api import behaviour_config, behaviour_summary
    from morphogen.homogenization import simulate_point
    from morphogen.homogenization.materials import Material

    spec = _json(p.get("material"), "material of the configurations")
    kept = spec is None and p["behaviour"] == "workspace"
    if spec is None:
        spec = _behaviour(p, {"young": p["young"], "poisson": p["poisson"]}, ctx.folder)
        if p.get("expansion") and "zmat" not in spec and not kept:
            spec["expansion"] = p["expansion"]
    material = Material.from_config(spec)
    name, path_spec = _path(p)
    turned = material.crystal_plasticity is not None or material.learned is not None
    orientation = p.get("orientation") if turned else None
    if orientation is not None and len(orientation) != 3:
        raise ValueError("orientation of the crystal: three Bunge angles φ1, Φ, φ2")
    ctx.progress(0.1, "integration")
    res = simulate_point(
        material, path_spec, name=name, orientation=orientation, kinematics=p["kinematics"]
    )
    values = res.components()
    values.update(
        steps=res.n_steps, iterations=int(np.sum(res.iterations)), cuts=int(res.info.get("cuts", 0))
    )
    reference = _reference(p.get("reference"))
    strain, stress, _, _, i = _driven(res)
    if reference is not None:
        e = strain[:, i]
        if np.all(np.diff(e) >= 0) or np.all(np.diff(e) <= 0):
            order = np.argsort(e)
            inside = (reference[:, 0] >= e.min()) & (reference[:, 0] <= e.max())
            if inside.any():
                model = np.interp(reference[inside, 0], e[order], stress[order, i])
                gap = model - reference[inside, 1]
                scale = float(np.abs(reference[:, 1]).max()) or 1.0
                values["reference_rms_gap"] = float(np.sqrt(np.mean(gap**2)))
                values["reference_relative_gap"] = values["reference_rms_gap"] / scale
        else:
            import warnings

            warnings.warn("points of a test: their gap along monotonic paths only", stacklevel=1)
    kind = "material of the configurations" if p.get("material") else p["behaviour"]
    if kept:
        kind = p["kept"]["title"]
    ctx.figure(_point_figure(res, reference, f"material point: {kind}, {name}"), "point", "curves")
    ctx.table(res.rows(), f"point_{name}", f"curves of the path {name}")
    config = behaviour_config(material.to_config(), ctx.folder)
    (ctx.folder / "material.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    if kept:
        return {"values": values}
    # a behaviour of its own: it joins the workspace (the run with it), sent on from there
    label, info = behaviour_summary(config)
    item = ctx.session.add(ctx.key, "behaviour", f"{label} (material point, {name})",
                           material=config, info=info)  # fmt: skip
    return {"values": values, "items": [ctx.session.public(item)]}


def _reference(text) -> np.ndarray | None:
    """The points (strain, stress) of a test pasted in the form, or None."""
    if not text or not str(text).strip():
        return None
    rows = []
    for line in str(text).splitlines():
        if ";" in line:
            parts = [x.strip().replace(",", ".") for x in line.split(";")]
        else:
            parts = line.replace(",", " ").replace("\t", " ").split()
        try:
            numbers = [float(x) for x in parts if x]
        except ValueError:
            continue  # a header
        if len(numbers) >= 2:
            rows.append(numbers[:2])
    if len(rows) < 2:
        raise ValueError("points of a test: at least two pairs strain, stress")
    return np.array(rows, dtype=float)


# -- the identification of a behaviour on references ----------------------------------------

#: The parameters a form identifies (:mod:`morphogen.homogenization.identification`)
IDENTIFIED = (
    ("yield", "R0, the yield stress (of the matrix)"),
    ("hardening", "H, the linear hardening"),
    ("Q", "Q, the isotropic hardening"),
    ("b", "b, its rate"),
    ("C", "C, the kinematic hardening (elliptic: C)"),
    ("D", "D, its recall (Rousselier: D)"),
    ("K", "K, Norton's viscosity"),
    ("n", "n, Norton's exponent"),
    ("young", "E, Young's modulus"),
    ("poisson", "ν, Poisson's ratio"),
    *((f"hill{c}", f"h{c}, Hill's coefficient of {c}") for c in HILL_COMPONENTS),
    ("porosity", "f0, the initial porosity"),
    ("q1", "q1 of Gurson"),
    ("q2", "q2 of Gurson or Rousselier"),
    ("fc", "fc, the onset of coalescence"),
    ("fF", "fF, the porosity at failure"),
    ("sigma1", "σ1 of Rousselier"),
    ("F", "F, elliptic"),
)
#: The behaviours an identification starts from
STARTS = (
    ("mises", "von Mises elasto-(visco)plasticity (Lemaitre-Chaboche)"),
    ("porous", "porous plasticity (Gurson-Tvergaard-Needleman, Rousselier, elliptic)"),
    KEPT,
)
#: The fields of the behaviours of Mechanics that an identification starts from
STARTING = ("kept", "yield_stress", "hardening", "isotropic", "kinematic", "norton", "hill",
            "criterion", "porosity", "gurson", "coalescence", "rousselier", "elliptic",
            "nucleation", "nucleation_coefficients")  # fmt: skip
#: The workspace items whose curves an identification follows (runs of paths and points)
RUNS = ("simulation", "behaviour", "result")


def _path_of_run(run: dict, session) -> tuple[str, dict, str]:
    """The loading path (its name and specification) and the kinematics of a run of a
    loading path or a material point, read again from the parameters of its form (of an
    identification: those of its first reference)."""
    from morphogen.gui.tools import TOOLS, Tool

    params = run.get("params") or {}
    if run.get("tool") == "identify" and params.get("reference", "runs") == "runs":
        runs = Field("runs", "items", items=RUNS, required=True).parse(params.get("runs"), session)
        return _path_of_run(runs[0], session)
    tool = TOOLS[run["tool"]]
    # the prefix of the fields of its path (that of the validation path of a learned law)
    prefix = next((f.name[: -len("loading")] for f in tool.fields if f.name.endswith("loading")),
                  "")  # fmt: skip
    names = {f.name for f in _path_fields(prefix=prefix)} | {"kinematics", "reference"}
    fields = tuple(f for f in tool.fields if f.name in names)
    p = Tool("path of a run", "", "", "", fields, None).parse(params, session)
    name, spec = _path(p, prefix)
    return name, spec, p.get("kinematics") or "small"


def _varying(stresses) -> list[int]:
    """The stress components a reference compares: those that vary by more than a tenth of
    the one that varies most (the lateral stresses of a uniaxial strain, which tell the
    triaxiality; not the residues of the free components)."""
    ptp = np.ptp(np.asarray(stresses, dtype=float), axis=0)
    return [int(c) for c in np.flatnonzero(ptp > 0.1 * ptp.max())] or [int(np.argmax(ptp))]


#: The colours of the stress components in the figures of an identification
SHADES = ("#d6604d", "#4393c3", "#5aae61", "#9970ab", "#e08214", "#878787")


def _test_reference(p: dict, start: dict):
    """The reference of the points of a test along the monotonic strain of the path of the
    form, placed in time by that strain (the material point of the starting values)."""
    from morphogen.homogenization import Material, simulate_point
    from morphogen.homogenization.identification import Reference

    name, spec = _path(p)
    points = _reference(p["points"])
    first = simulate_point(Material.from_config(start), spec, kinematics=p["kinematics"])
    i = int(np.argmax(np.ptp(np.asarray(first.strain), axis=0)))
    e = np.asarray(first.strain)[:, i]
    if not (np.all(np.diff(e) >= 0) or np.all(np.diff(e) <= 0)) or np.ptp(e) == 0:
        raise ValueError("points of a test: along a monotonic strain (tension, relaxation)")
    order = np.argsort(e, kind="stable")
    inside = (points[:, 0] >= e.min()) & (points[:, 0] <= e.max())
    if inside.sum() < 2:
        raise ValueError("points of a test: at least two within the strains of the path")
    times = np.interp(points[inside, 0], e[order], np.asarray(first.times)[order])
    stresses = np.zeros((int(inside.sum()), 6))
    stresses[:, i] = points[inside, 1]
    return name, Reference(spec, times, stresses)


@tool(
    "identify",
    "mechanics",
    "Identification of a behaviour",
    "The parameters of a von Mises, Hill or porous behaviour (yield stress, hardenings, "
    "viscosity, elastic constants, Hill's coefficients; the initial porosity, Gurson's q1, "
    "q2, the coalescence, "
    "Rousselier's and the elliptic coefficients) identified on references, as Zopt does for "
    "Zsim: the curves of runs of the workspace, each under its own path (the macroscopic "
    "responses of a morphology, to give it a macroscopic law: a porous morphology under "
    "several triaxialities, its Gurson model; material points), followed together (a "
    "tension and a cycle in shear tell the isotropic hardening from the kinematic one), or "
    "the points of a test along the path of the form; the gap of the stresses of a "
    "material point over time in the least-squares sense, each reference by its weight "
    "whatever its number of points, the parameters positive. It starts from the values of "
    "the form or of a behaviour of the workspace; the behaviour identified joins the "
    "workspace (B).",
    [
        Field(
            "reference",
            "choice",
            "runs",
            choices=(
                ("runs", "runs of the workspace: their curves, each under its path"),
                ("points", "the points of a test, along the path below"),
            ),
        ),
        Field(
            "runs",
            "items",
            items=RUNS,
            needs=("curves",),
            required=True,
            when={"reference": "runs"},
            doc="loading paths of a morphology (S), material points or identifications (B, "
            "R), all in small or all in finite strains",
        ),
        Field(
            "weights",
            "floats",
            label="weights of the runs",
            when={"reference": "runs"},
            doc="one per run checked, in the order of the list (by default 1 each)",
        ),
        Field(
            "points",
            "text",
            required=True,
            label="points of a test",
            when={"reference": "points"},
            doc="strain, stress along the component of the path (monotonic), one pair per "
            "line (comma, space or tab; or semicolons with decimal commas)",
        ),
        Field("behaviour", "choice", "mises", choices=STARTS),
        Field("young", "float", 1.0, label="Young's modulus E", unit="stress", when=PLASTIC),
        Field("poisson", "float", 0.3, label="Poisson's ratio ν", when=PLASTIC),
        *(f for f in _behaviour_fields() if f.name in STARTING),
        Field(
            "fit",
            "choices",
            ["yield", "Q", "b"],
            label="identified",
            choices=IDENTIFIED,
            doc="from the values above (positive), the others held",
        ),
        Field(
            "evaluations",
            "int",
            200,
            label="evaluations at most",
            doc="each a material point under the path of each reference",
        ),
        *(
            replace(f, when={**(f.when or {}), "reference": "points"})
            for f in _path_fields("tension")
        ),
        Field(
            "kinematics",
            "choice",
            "small",
            choices=(("small", "small strains"), ("finite", "finite strains")),
            when={"reference": "points"},
        ),
    ],
    order=5,
    groups=(
        ("References", "reference"),
        ("Starting behaviour", "behaviour"),
        ("Identified", "fit"),
        ("Path of the test", "loading", True),
    ),
    menu="Behaviours of materials",
)
def identify_behaviour(p: dict, ctx) -> dict:
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from morphogen.gui.api import behaviour_summary, curves_of
    from morphogen.homogenization.identification import Reference, identify_references
    from morphogen.pipeline import read_rows

    # the values of the form, or those of a behaviour of the workspace (with its elasticity)
    start = _behaviour(p, {"young": p["young"], "poisson": p["poisson"]}, ctx.folder)
    if not p["fit"]:
        raise ValueError("identified: at least one parameter")
    references, labels = [], []
    if p["reference"] == "runs":
        runs = p["runs"]
        weights = p["weights"] or [1.0] * len(runs)
        if len(weights) != len(runs) or not all(w > 0 for w in weights):
            raise ValueError(f"weights of the runs: {len(runs)} positive values, one per run")
        kinematics = set()
        for run, weight in zip(runs, weights, strict=True):
            name, spec, kind = _path_of_run(run, ctx.session)
            rows = read_rows(curves_of(run, ctx.session.root))
            times = np.array([float(r["time"]) for r in rows])
            stresses = np.array([[float(r[f"S{c}"]) for c in COMPONENTS] for r in rows])
            references.append(Reference(spec, times, stresses, _varying(stresses), weight))
            labels.append((f"#{run['number']}", "custom" if name == "path" else name))
            kinematics.add(kind)
        if len(kinematics) > 1:
            raise ValueError("runs: all in small strains or all in finite strains")
        (kinematics,) = kinematics
        numbers = ", ".join(label for label, _ in labels)
        source = runs[0]["title"] if len(runs) == 1 else f"the runs {numbers}"
    else:
        kinematics = p["kinematics"]
        name, reference = _test_reference(p, start)
        references.append(reference)
        labels.append(("the test", "custom" if name == "path" else name))
        source = "the points of a test"
    total = max(int(p["evaluations"]), 1)

    def report(count, error):
        fraction = min(0.02 + 0.93 * count / total, 0.95)
        ctx.progress(fraction, f"evaluation {count}: gap {error:.3g}")

    fit = identify_references(start, references, p["fit"], kinematics=kinematics,
                              max_evaluations=total, callback=report)  # fmt: skip
    columns = min(len(references), 3)
    lines = -(-len(references) // columns)
    fig = Figure(figsize=(1.0 + 4.4 * columns, 3.8 * lines), layout="constrained")
    axes = fig.subplots(lines, columns, squeeze=False).ravel()
    for ax, (label, name), ref, initial, res in zip(
        axes, labels, fit.references, fit.initial_responses, fit.responses, strict=False
    ):
        handles = []
        for k, c in enumerate(ref.components):  # each component its colour
            shade = SHADES[k % len(SHADES)]
            ax.plot(ref.times, ref.stresses[:, c], "o", color=shade, ms=3.5, mfc="none")
            if initial is not None:
                ax.plot(initial.times, np.asarray(initial.stress)[:, c], "--", color=shade,
                        lw=1.0, alpha=0.6)  # fmt: skip
            handles += ax.plot(res.times, np.asarray(res.stress)[:, c], "-", color=shade,
                               lw=1.8, label=f"S{COMPONENTS[c]}")  # fmt: skip
        shown = source if len(references) == 1 else label
        handles += [
            Line2D([], [], color="k", marker="o", mfc="none", ls="", ms=3.5, label=shown),
            Line2D([], [], color="k", ls="--", lw=1.0, alpha=0.6, label="starting values"),
            Line2D([], [], color="k", ls="-", lw=1.8, label="identified"),
        ]
        ax.set_xlabel("time")
        ax.set_ylabel(f"S{COMPONENTS[ref.components[0]]}" if len(ref.components) == 1 else "stress")
        ax.set_title(f"{label}, path {name}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(handles=handles, fontsize=8, frameon=False)
    for ax in axes[len(references) :]:
        ax.set_visible(False)
    ctx.figure(fig, "identification", "the responses identified, under the paths followed")
    ctx.table([{"parameter": k, "start": fit.initial[k], "identified": v}
               for k, v in fit.values.items()], "parameters", "the parameters")  # fmt: skip
    _elasticity_tables(ctx, fit.material, "of the behaviour identified")
    followed = zip(labels, fit.references, fit.initial_gaps, fit.gaps, strict=True)
    ctx.table(
        [
            {
                "reference": label,
                "path": name,
                "components": ", ".join(f"S{COMPONENTS[c]}" for c in r.components),
                "points": len(r.times),
                "weight": r.weight,
                "starting gap": g0,
                "gap": g,
            }
            for (label, name), r, g0, g in followed
        ],
        "references",
        "the references followed",
    )
    for i, ((label, name), res) in enumerate(zip(labels, fit.responses, strict=True), start=1):
        ctx.table(res.rows(), f"identified_{i:02d}_{name}", f"curves of the path {name} ({label})")
    (ctx.folder / "material.json").write_text(json.dumps(fit.material, indent=1), "utf-8")
    label, info = behaviour_summary(fit.material)
    item = ctx.session.add(ctx.key, "behaviour", f"{label} (identified on {source})",
                           material=fit.material, info=info)  # fmt: skip
    values = {"gap": fit.error, "starting gap": fit.initial_error,
              "evaluations": fit.evaluations, **fit.values}  # fmt: skip
    if len(references) > 1:
        values.update({f"gap {label}": g for (label, _), g in zip(labels, fit.gaps, strict=True)})
    return {"values": values, "items": [ctx.session.public(item)]}


# -- the macroscopic law of a morphology ---------------------------------------------------

#: The laws a morphology may be given
LAWS = (
    ("hill", "Hill's criterion: its coefficients of the loadings run (that of the first stays 1)"),
    ("mises", "von Mises"),
    ("gurson", "Gurson's porous plasticity: q1, q2 of the matrix of the phase, its voids"),
)
#: The loadings of a macroscopic law
LOADING_CHOICES = (
    ("11", "tension along 1"),
    ("22", "tension along 2"),
    ("33", "tension along 3"),
    ("12", "shear in the plane 12"),
    ("23", "shear in the plane 23"),
    ("31", "shear in the plane 31"),
    ("confined11", "tension along 1, its lateral strains held (a high stress triaxiality)"),
    ("confined22", "tension along 2, its lateral strains held"),
    ("confined33", "tension along 3, its lateral strains held"),
    ("hydrostatic", "hydrostatic tension"),
)


@tool(
    "macroscopic_law",
    "mechanics",
    "Macroscopic law of a morphology",
    "The elasto-plastic law that gives a morphology its macroscopic behaviour, in one go: "
    "its responses (nonlinear FFT on its voxels) under a tension along each axis and a "
    "shear in each plane checked, the other stresses free (monotonic, or cycles that show "
    "the Bauschinger effect), then a von Mises or Hill law (Z-set's form) identified on "
    "them together, from a law estimated on the responses: Young's modulus, Poisson's "
    "ratio, the yield stress, the linear hardening, with cycles a kinematic hardening and, "
    "for Hill, the coefficients of the loadings run (an anisotropic morphology: aligned "
    "fibres, laminates, lattices); or Gurson's law of a porous morphology, the matrix that "
    "of the phase and the porosity its voids, q1 and q2 identified on loadings of several "
    "stress triaxialities (a tension, a confined or hydrostatic one), as on unit cells "
    "(Tvergaard 1981). The law joins the workspace (B) and is written as a Z-mat file for "
    "Z-set.",
    [
        Field("source", "item", label="morphology", required=True),
        Field(
            "materials",
            "materials",
            of="source",
            doc="elastic constants of the phases (the unit of the stresses), or voids",
        ),
        Field(
            "phase",
            "int",
            label="phase of the behaviour",
            doc="blank: the first phase that is not a void (the matrix, the solid, the grains)",
        ),
        *_behaviour_fields("of that phase; the other phases are elastic, or voids"),
        BEHAVIOURS_FIELD,
        Field(
            "loadings",
            "choices",
            ["11", "22", "33", "12", "23", "31"],
            choices=LOADING_CHOICES,
            doc="each a loading of its component (monotonic, or a cycle below), the other "
            "stresses free",
        ),
        Field("strain", "float", 0.01, label="strain of each loading", doc="the component loaded"),
        Field("increments", "int", 20, label="increments of each loading"),
        Field(
            "cycle",
            "bool",
            False,
            label="cycles",
            doc="each loading then back to -strain (twice the increments): its Bauschinger "
            "effect, and a kinematic hardening (C, D) in the law",
        ),
        Field("law", "choice", "hill", choices=LAWS),
        Field(
            "evaluations",
            "int",
            200,
            label="evaluations at most",
            doc="each a material point under the paths of the loadings",
        ),
        Field(
            "method",
            "choice",
            "fe",
            choices=(
                ("fe", "finite elements on the conforming mesh of its geometry"),
                ("fft", "FFT on its voxels (periodic)"),
            ),
        ),
        Field("voxels", "int", 16, label="voxels per edge", when={"method": "fft"}),
        Field(
            "integration",
            "choice",
            "bbar",
            when={"method": "fft"},
            choices=(
                ("bbar", "B-bar (F-bar in finite strains): no locking under plastic flow"),
                ("full", "full: 8 Gauss points per voxel"),
            ),
        ),
        _device_field(when={"method": "fft"}),
        Field(
            "element_size",
            "float",
            unit="length",
            when={"method": "fe"},
            doc="largest edge of the elements (blank: 1/4 of the edge of the domain)",
        ),
        Field(
            "order",
            "choice",
            "2",
            label="elements",
            when={"method": "fe"},
            choices=(
                ("2", "10-node tetrahedra (accurate plastic flow)"),
                ("1", "4-node tetrahedra (fast; they lock under isochoric plastic flow)"),
            ),
        ),
        Field(
            "fe_integration",
            "choice",
            "full",
            label="integration",
            when={"method": "fe"},
            choices=(
                ("full", "full: every point of the elements"),
                (
                    "bbar",
                    "B-bar (F-bar in finite strains): the volume change of each 10-node "
                    "element as its mean, for nearly incompressible phases (elastomers)",
                ),
            ),
        ),
        Field(
            "kinematics",
            "choice",
            "small",
            choices=(
                ("small", "small strains"),
                ("finite", "finite strains (the law in the space of the logarithmic strain)"),
            ),
        ),
    ],
    order=2,
    groups=(
        ("Morphology and materials", "source"),
        ("Behaviour of the phase", "behaviour"),
        ("Loadings", "loadings"),
        ("Law", "law"),
        ("Solver", "method", True),
    ),
    menu="Loading paths",
)
def macroscopic_law(p: dict, ctx) -> dict:
    from matplotlib.figure import Figure

    from morphogen import pipeline
    from morphogen.config import load_config
    from morphogen.gui.api import behaviour_summary
    from morphogen.homogenization import NonlinearFEProblem, NonlinearFFTProblem
    from morphogen.homogenization.identification import loading_path
    from morphogen.homogenization.identification import macroscopic_law as law_of
    from morphogen.homogenization.materials import any_oriented
    from morphogen.homogenization.zmat import write_zmat

    source = p["source"]
    micro = source["micro"]
    materials, phase = _materials(p, source, ctx.folder)
    loadings = [c for c, _ in LOADING_CHOICES if c in (p["loadings"] or [])]
    voids = [k for k, m in materials.items() if m == "void"]
    if p["law"] == "gurson" and not voids:
        raise ValueError("Gurson's law: a porous morphology (one of its phases a void)")
    if not loadings:
        raise ValueError("loadings: at least one")
    fe = p["method"] == "fe"
    if not fe and not 4 <= p["voxels"] <= 64:
        raise ValueError("voxels per edge: from 4 to 64 for a macroscopic law")
    if not p["strain"] > 0 or p["increments"] < 2:
        raise ValueError("strain: positive, in two increments at least")
    data = {
        "model": source.get("model") or {"type": "tpms", "volume_fraction": 0.5},  # a stand-in
        "domain": source.get("domain")
        or {"size": [float(v) for v in micro.domain.lengths], "periodic": True},
        "seed": int(source.get("seed") or 1),
        "homogenization": {
            "materials": {k: _relative(v, ctx.folder) for k, v in materials.items()},
            "method": p["method"],
            "boundary_conditions": ["periodic"],
            "properties": [],
            "paths": {  # those of the loadings (checked here; run by the identification)
                f"loading{c}": loading_path(c, p["strain"], p["increments"], bool(p["cycle"]))[0]
                for c in loadings
            },
            "nonlinear": {"kinematics": p["kinematics"]},
        },
        "outputs": {"directory": str(ctx.folder), "name": "sample"},
    }
    edge = float(np.max(micro.domain.lengths))
    if fe:
        size = p["element_size"] or edge / 4
        if not edge / 200 <= size <= edge:
            raise ValueError(f"element size: between {edge / 200:g} and {edge:g}")
        data["mesh"] = {"size": size, "order": int(p["order"])}
        data["homogenization"]["nonlinear"]["integration"] = p["fe_integration"]
    else:
        data["voxels"] = {"shape": p["voxels"]}
        data["homogenization"]["nonlinear"]["integration"] = p["integration"]
    cfg = load_config(data, base_dir=ctx.folder)  # the errors of the form, before solving
    homog = cfg.homogenization
    options = homog.options()["nonlinear"]
    if fe:
        ctx.progress(0.01, "conforming mesh")
        mesh, _ = pipeline._conforming_mesh(micro, cfg)

        def problem():
            return NonlinearFEProblem(mesh, homog.materials, "periodic", solver=homog.solver,
                                      tol=homog.tolerance, **options)  # fmt: skip
    else:
        steps = p["increments"] * (3 if p["cycle"] else 1) + 1
        _check_voxels(micro, cfg.voxels.shape, steps, "macroscopic law on voxels")
        image = micro.voxelize(**cfg.voxels.kwargs())
        grains = {}
        if any_oriented(homog.materials) and getattr(micro, "orientations", None) is not None:
            labels = micro.labels(grid=image.grid())
            grains = {"grains": labels, "orientations": micro.orientations}

        def problem():
            return NonlinearFFTProblem(image, homog.materials, "periodic", tol=homog.tolerance,
                                       device=p["device"], **grains, **options)  # fmt: skip

    done = [0]

    def respond(path):
        c = next(iter(path["strain"]))
        ctx.progress(0.02 + 0.6 * done[0] / len(loadings), f"loading {c} of the morphology")
        done[0] += 1
        return problem().run(path)

    total = max(int(p["evaluations"]), 1)

    def report(count, error):
        ctx.progress(min(0.65 + 0.3 * count / total, 0.95), f"evaluation {count}: gap {error:.3g}")

    porous = {}
    if p["law"] == "gurson":  # the matrix of the phase, the voids of the morphology
        sample = image if not fe else micro.voxelize(shape=32)
        porosity = float(np.isin(np.asarray(sample.data), voids).mean())
        porous = {"matrix": materials[phase], "porosity": porosity}
    law = law_of(respond, law=p["law"], loadings=loadings, strain=p["strain"],
                 increments=p["increments"], cycle=bool(p["cycle"]),
                 kinematics=p["kinematics"], max_evaluations=total, callback=report,
                 **porous)  # fmt: skip
    fit = law.identification
    columns = min(len(loadings), 3)
    lines = -(-len(loadings) // columns)
    fig = Figure(figsize=(1.0 + 4.4 * columns, 3.8 * lines), layout="constrained")
    axes = fig.subplots(lines, columns, squeeze=False).ravel()
    for ax, c, ref, initial, res in zip(axes, loadings, law.responses, fit.initial_responses,
                                        fit.responses, strict=False):  # fmt: skip
        # the component loaded (the axis of a confined tension, 11 of a hydrostatic one)
        k = COMPONENTS.index(c if c in COMPONENTS else (c[-2:] if c[-1].isdigit() else "11"))
        strain = np.asarray(ref.strain)[:, k]
        ax.plot(strain, np.asarray(ref.stress)[:, k], "o", ms=3.5, mfc="none", color=SHADES[0],
                label="the morphology")  # fmt: skip
        if initial is not None:
            ax.plot(np.asarray(initial.strain)[:, k], np.asarray(initial.stress)[:, k], "--",
                    lw=1.0, alpha=0.6, color="k", label="estimated law")  # fmt: skip
        ax.plot(np.asarray(res.strain)[:, k], np.asarray(res.stress)[:, k], "-", lw=1.8,
                color=SHADES[1], label="law identified")  # fmt: skip
        ax.set_xlabel(f"E{COMPONENTS[k]}")
        ax.set_ylabel(f"S{COMPONENTS[k]}")
        if c in COMPONENTS:
            title = ("tension " if c[0] == c[1] else "shear ") + c
        else:
            title = "hydrostatic tension" if c == "hydrostatic" else f"confined tension {c[-2:]}"
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes[len(loadings) :]:
        ax.set_visible(False)
    ctx.figure(fig, "macroscopic_law", "the responses of the morphology and of its law")
    ctx.table([{"parameter": k, "estimated": fit.initial[k], "identified": v}
               for k, v in fit.values.items()], "parameters", "the parameters")  # fmt: skip
    ctx.table([{"loading": c, "starting gap": g0, "gap": g} for c, g0, g in
               zip(loadings, fit.initial_gaps, fit.gaps, strict=True)], "loadings",
              "the loadings followed")  # fmt: skip
    _elasticity_tables(ctx, fit.material, "of the law identified")
    for c, res in zip(loadings, law.responses, strict=True):
        ctx.table(res.rows(), f"morphology_{c}", f"curves of the morphology under the loading {c}")
    title = f"the macroscopic law of #{source['number']} {source['title']}"
    write_zmat(fit.material, ctx.folder / "macroscopic_law.mat", title=title, digits=6)
    (ctx.folder / "material.json").write_text(json.dumps(fit.material, indent=1), "utf-8")
    label, info = behaviour_summary(fit.material)
    item = ctx.session.add(ctx.key, "behaviour",
                           f"{label} (macroscopic law of #{source['number']})",
                           material=fit.material, info=info)  # fmt: skip
    values = {"gap": fit.error, "starting gap": fit.initial_error,
              "evaluations": fit.evaluations, **fit.values}  # fmt: skip
    values.update({f"gap {c}": g for c, g in zip(loadings, fit.gaps, strict=True)})
    return {"values": values, "items": [ctx.session.public(item)]}


# -- the linear tensors of a morphology ----------------------------------------------------

TENSORS = (
    ("stiffness", "stiffness tensor (engineering constants, anisotropy)"),
    ("thermal_expansion", "thermal expansion tensor"),
    ("conductivity_tensor", "conductivity tensor"),
)
CRYSTALS = (
    ("none", "isotropic phases"),
    ("cubic", "a cubic crystal: C11, C12, C44"),
    ("hexagonal", "a hexagonal crystal: C11, C12, C13, C33, C44 (axis c along 3)"),
)


@tool(
    "tensors",
    "mechanics",
    "Linear tensors",
    "The apparent stiffness tensor of a morphology (its engineering constants, anisotropy "
    "indices and isotropic projection), its thermal expansion and its conductivity tensor, "
    "by FFT on its voxels, by finite elements on its conforming mesh (periodic, kinematic "
    "or static conditions) or by extended finite elements (XFEM) on a regular grid that its "
    "level set cuts (periodic; no mesh to build, and the interfaces where the model puts "
    "them); crystals turned by the orientations of the grains of polycrystals; against the "
    "bounds of the phase fractions (Voigt, Reuss, Hashin-Shtrikman, Levin) and with the "
    "directional Young's modulus.",
    [
        Field("source", "item", label="morphology", required=True),
        Field(
            "materials",
            "materials",
            of="source",
            doc="isotropic elastic constants of the phases, or voids",
        ),
        Field("tensors", "choices", ["stiffness"], choices=TENSORS),
        Field(
            "expansions",
            "floats",
            label="thermal expansions of the phases",
            doc="one per phase, in the order of the materials (the thermal expansion)",
        ),
        Field(
            "conductivities",
            "floats",
            label="conductivities of the phases",
            doc="one per phase, in the order of the materials (the conductivity tensor)",
        ),
        Field("crystal", "choice", "none", label="crystals", choices=CRYSTALS),
        Field(
            "crystal_phase",
            "int",
            label="phase of the crystals",
            when={"crystal": ["cubic", "hexagonal"]},
            doc="blank: the first phase that is not a void; its grains turned by their "
            "orientations (tessellations with crystal orientations)",
        ),
        Field(
            "constants",
            "floats",
            [1.684, 1.214, 0.754],
            label="elastic constants of the crystal",
            unit="stress",
            when={"crystal": ["cubic", "hexagonal"]},
            doc="in its axes (the default: copper, in units of 100 GPa)",
        ),
        BEHAVIOURS_FIELD,
        Field(
            "method",
            "choice",
            "fe",
            choices=(
                ("fe", "finite elements on the conforming mesh of its geometry"),
                ("fft", "FFT on its voxels (periodic)"),
                (
                    "xfem",
                    "extended finite elements (XFEM) on a grid cut by its level set (periodic, "
                    "two phases)",
                ),
            ),
        ),
        Field("voxels", "int", 32, label="voxels per edge", when={"method": "fft"}),
        Field(
            "cells",
            "int",
            16,
            label="cells per edge",
            when={"method": "xfem"},
            doc="of the grid of the extended finite elements, along the longest edge (4 to 48)",
        ),
        Field(
            "element_size",
            "float",
            unit="length",
            when={"method": "fe"},
            doc="largest edge of the elements (blank: 1/8 of the edge of the domain)",
        ),
        Field(
            "order",
            "choice",
            "2",
            label="elements",
            when={"method": "fe"},
            choices=(
                ("2", "10-node tetrahedra (curved along the interfaces; accurate)"),
                ("1", "4-node tetrahedra (fast; stiff)"),
            ),
        ),
        Field(
            "bc",
            "choice",
            "periodic",
            label="boundary conditions",
            when={"method": "fe"},
            choices=(
                ("periodic", "periodic"),
                ("kubc", "kinematic (KUBC)"),
                ("subc", "static (SUBC)"),
            ),
        ),
    ],
    order=6,
    menu="Linear elasticity",
)
def linear_tensors(p: dict, ctx) -> dict:
    from morphogen.analysis.basic import phase_fractions
    from morphogen.config import MeshSpec
    from morphogen.homogenization import bounds, homogenize
    from morphogen.homogenization.materials import any_oriented, materials_from_config

    source = p["source"]
    micro = source["micro"]
    wanted = list(p["tensors"] or [])
    if not wanted:
        raise ValueError("tensors: at least one")
    elastic = _elastic(source, p.get("materials"))
    configs: dict = {}
    for name, given in (
        ("expansions", p.get("expansions")),
        ("conductivities", p.get("conductivities")),
    ):
        if given and len(given) != len(elastic):
            raise ValueError(f"{name} of the phases: one per phase ({len(elastic)})")
    if "thermal_expansion" in wanted and not p.get("expansions"):
        raise ValueError("thermal expansions of the phases: one per phase (the thermal expansion)")
    if "conductivity_tensor" in wanted and not p.get("conductivities"):
        raise ValueError("conductivities of the phases: one per phase (the conductivity tensor)")
    crystal = p["crystal"]
    phase = p.get("crystal_phase") if crystal != "none" else None
    if crystal != "none" and phase is None:
        phase = next((k for k, m in elastic.items() if m != "void"), None)
    if phase is not None and elastic.get(phase, "void") == "void":
        raise ValueError(f"phase of the crystals: phase {phase} is a void or no phase")
    for index, (k, m) in enumerate(elastic.items()):
        if m == "void":
            configs[k] = "void"
            continue
        spec = dict(m)
        if k == phase:
            constants = [float(c) for c in p.get("constants") or []]
            if len(constants) != (3 if crystal == "cubic" else 5):
                raise ValueError(
                    "elastic constants of the crystal: C11, C12, C44 (cubic) or C11, C12, C13, "
                    "C33, C44 (hexagonal)"
                )
            spec = {crystal: constants}
        if p.get("expansions"):
            spec["expansion"] = p["expansions"][index]
        if p.get("conductivities"):
            spec["conductivity"] = p["conductivities"][index]
        configs[k] = spec
    configs.update(_extra(p))
    materials = materials_from_config(configs)
    stage = {"fe": "conforming mesh", "xfem": "grid cut by the level set"}
    ctx.progress(0.05, stage.get(p["method"], "voxels"))
    extra_args: dict = {}
    if p["method"] == "xfem":
        from morphogen.homogenization import LevelSetSample

        n = p["cells"]
        if not 4 <= n <= 48:
            raise ValueError("cells per edge: from 4 to 48 for the extended finite elements")
        if micro.implicit_phases() != [0, 1]:
            raise ValueError(
                "extended finite elements: a morphology of two phases, cut by its level set; "
                "those of several phases by finite elements on their conforming mesh, or by FFT"
            )
        if not micro.domain.periodic:
            raise ValueError("extended finite elements: a periodic morphology")
        if any_oriented(materials):
            raise ValueError("extended finite elements: isotropic phases (crystals: FFT or FE)")
        sample = LevelSetSample.of(micro, n)
        bc = "periodic"
        fractions = sample.phase_fractions()
        info = {"cells": int(np.prod(sample.shape))}
    elif p["method"] == "fe":
        from morphogen.mesh.conforming import conforming_mesh

        edge = float(np.max(micro.domain.lengths))
        size = p["element_size"] or edge / 8
        if not edge / 200 <= size <= edge:
            raise ValueError(f"element size: between {edge / 200:g} and {edge:g}")
        voids = {int(k) for k, m in materials.items() if m.void}  # the solid phase is meshed
        sample = conforming_mesh(micro, **MeshSpec(size=size, order=int(p["order"])).kwargs(voids))
        bc = p["bc"]
        fractions = sample.phase_fractions()
        absent = [k for k in materials if k not in fractions]
        if len(absent) == 1:  # the pores, left out of the mesh
            fractions[absent[0]] = max(0.0, 1.0 - sum(fractions.values()))
        info = {"elements": sample.n_elements, "nodes": sample.n_nodes}
    else:
        n = p["voxels"]
        if not 4 <= n <= 256:
            raise ValueError("voxels per edge: from 4 to 256")
        sample = micro.voxelize(shape=n)
        bc = "periodic"
        fractions = phase_fractions(sample)
        if any_oriented(materials) and getattr(micro, "orientations", None) is not None:
            extra_args = {
                "grains": micro.labels(grid=sample.grid()),
                "orientations": micro.orientations,
            }
        info = {"voxels": int(np.prod(sample.shape))}
    ctx.progress(0.2, "solving")
    results = homogenize(sample, materials, bcs=[bc], properties=wanted, **extra_args)
    values: dict = dict(info)
    for name in wanted:
        res = results[f"{name}.{bc}"]
        tensor = np.asarray(res["tensor"])
        if name == "stiffness":  # with its compliance S = C⁻¹, in the same notation
            elastic_tables(ctx, tensor, f"({bc})")
        else:
            labels = ("1", "2", "3")
            rows = [
                {"": a, **{b: float(tensor[i, j]) for j, b in enumerate(labels)}}
                for i, a in enumerate(labels)
            ]
            ctx.table(rows, name, f"{name.replace('_', ' ')} ({bc})")
        values.update({f"{name}.{k}": v for k, v in res.items() if isinstance(v, float)})
        if name == "stiffness":
            ctx.figure(_young_figure(tensor), "young_modulus", "directional Young's modulus")
    limits = bounds({k: v for k, v in fractions.items() if k in materials}, materials)
    rows = [{"property": prop, **values_} for prop, values_ in limits.items()]
    if rows:
        ctx.table(rows, "bounds", "bounds of the phase fractions")
    values.update({f"fraction_{k}": float(v) for k, v in sorted(fractions.items())})
    return {"values": values}


def _elasticity_tables(ctx, material: dict, of: str) -> None:
    """The tables of the elasticity of a behaviour (its configuration): its stiffness and
    compliance (``elasticity_stiffness.csv``, ``elasticity_compliance.csv``), none for a
    behaviour without elastic constants of its own (a law of Z-set or MFront)."""
    from morphogen.homogenization.materials import Material

    try:
        m = Material.from_config(material)
    except (ValueError, TypeError, KeyError, OSError):
        return
    if m.elastic and not m.void:
        names = ("elasticity_stiffness", "elasticity_compliance")
        elastic_tables(ctx, m.matrix("elastic"), of, names=names)


def _young_figure(stiffness):
    """The directional Young's modulus ``1 / (n n : S : n n)`` of a stiffness in the planes
    of the axes (polar curves); of a plane stiffness (3 x 3, Voigt 11, 22, 12), in its
    plane."""
    from matplotlib.figure import Figure

    compliance = np.linalg.inv(np.asarray(stiffness, dtype=float))
    plane = compliance.shape == (3, 3)
    angle = np.linspace(0.0, 2.0 * np.pi, 361)
    fig = Figure(figsize=(5.0, 4.4), layout="constrained")
    ax = fig.add_subplot(projection="polar")
    planes = (((0, 1), "x-y plane"), ((1, 2), "y-z plane"), ((2, 0), "z-x plane"))
    for (a, b), name in planes[:1] if plane else planes:
        n = np.zeros((angle.size, 3))
        n[:, a], n[:, b] = np.cos(angle), np.sin(angle)
        if plane:
            v = np.stack([n[:, 0] ** 2, n[:, 1] ** 2, n[:, 0] * n[:, 1]], axis=1)
        else:
            v = np.stack(
                [
                    n[:, 0] ** 2,
                    n[:, 1] ** 2,
                    n[:, 2] ** 2,
                    n[:, 1] * n[:, 2],
                    n[:, 2] * n[:, 0],
                    n[:, 0] * n[:, 1],
                ],
                axis=1,
            )
        ax.plot(angle, 1.0 / np.einsum("ni,ij,nj->n", v, compliance, v), lw=1.6, label=name)
    ax.set_title("directional Young's modulus", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    return fig


# -- the frames of beams of networks -----------------------------------------------------

#: The beams of the frames of networks
BEAMS = (
    ("euler_bernoulli", "Euler–Bernoulli (rigid in transverse shear)"),
    ("timoshenko", "Timoshenko (transverse shear stiffness κGA)"),
)
#: The modes of the beams whose stiffness varies from beam to beam (Ban et al. 2016)
SOFTENING = (
    ("axial", "axial stiffness EA"),
    ("bending", "bending stiffness EI"),
    ("modulus", "Young's modulus (EA, EI, GJ and κGA)"),
    ("area", "area of a general section (EA and κGA)"),
)
#: The laws of the stiffness of the beams of the replicas (mean 1, coefficient of variation
#: CV; :func:`morphogen.homogenization.networks.fibre_scales`)
DISTRIBUTIONS = (
    ("normal", "normal (draws below 10⁻³ raised to it)"),
    ("lognormal", "lognormal"),
    ("bimodal", "bimodal: 1 ± CV, equally likely"),
)
#: The colours of the beams of the drawing of the frame under a uniaxial stress: the value
#: of each beam (``force``: its axial force; the others: its share of the energy of the
#: network), the scale of :func:`morphogen.viz.figures.frame_figure` and the label
DRAWN = {
    "force": ("symmetric", "axial force N under a stress of 1 along {axis} (tension > 0)"),
    "stretching": ("log", "share of the energy in the stretching of the beam"),
    "bending": ("log", "share of the energy in the bending of the beam (torsion, shear)"),
    "beams": ("linear", ""),
}
DRAWINGS = (
    ("force", "axial force (the force chains)"),
    ("stretching", "stretching energy of each beam"),
    ("bending", "bending energy of each beam (with its torsion and shear)"),
    ("beams", "the beams alone"),
)
#: The Voigt components of plane stiffnesses (11, 22, 12; engineering shear)
PLANE_COMPONENTS = ("11", "22", "12")


@tool(
    "frame",
    "mechanics",
    "Stiffness of the frame of beams",
    "The effective stiffness of a fibre network, an open foam or a random lattice of the "
    "workspace as the periodic frame of beams of circular sections along its fibres or "
    "struts. The realizations of delaunay_network, open_foam and random_lattice keep their "
    "framework; Euler–Bernoulli or Timoshenko beams (Cowper's shear coefficient by default), "
    "rigid over a few radii at the joints on request. Its Voigt stiffness and engineering "
    "constants, the split of the energy of a uniaxial stress along each axis into the "
    "stretching, bending, torsion and shear of the beams, its ratio to the affine estimate "
    "and the non-affinity of the deformation, a drawing of the beams coloured by their axial "
    "forces or energies under a uniaxial stress (the force chains, the beams that bend), and "
    "the softening of networks of non-identical beams (the coefficients s and v of Ban, "
    "Barocas, Shephard & Picu 2016, and replicas of the stiffness of the beams against them).",
    [
        Field(
            "source",
            "item",
            label="network",
            required=True,
            needs=("framework",),
            doc="a realization of delaunay_network, open_foam or random_lattice (Generate), "
            "whose framework becomes the frame",
        ),
        Field("beams", "choice", "euler_bernoulli", choices=BEAMS),
        Field(
            "shear_coefficient",
            "float",
            label="shear coefficient κ",
            when={"beams": "timoshenko"},
            doc="of the transverse shear stiffness κGA (blank: Cowper's, 6(1 + ν)/(7 + 6ν) "
            "for circular sections)",
        ),
        Field(
            "rigid_joints",
            "float",
            label="rigid joints",
            unit="radii",
            doc="the beams rigid over that many of their radii from each node, the struts "
            "merged in the joints of a continuum (blank: none; a third brings Timoshenko beams "
            "within a few per cent of the FFT solver on open foams)",
        ),
        Field("young", "float", 1.0, label="Young's modulus of the fibres", unit="stress"),
        Field("poisson", "float", 0.3, label="Poisson's ratio of the fibres"),
        Field(
            "draw",
            "choice",
            "force",
            label="colour of the beams",
            choices=DRAWINGS,
            doc="the beams in the cell (a 3D view; in its plane for a planar network) under "
            "a uniaxial stress: their axial force (red in tension, blue in compression), or "
            "the share of the energy of the network in their stretching or bending (a log "
            "scale); the strongest drawn widest",
        ),
        Field(
            "draw_axis",
            "choice",
            "x",
            label="axis of the stress drawn",
            choices=(("x", "x"), ("y", "y"), ("z", "z")),
            doc="of the uniaxial stress of the drawing (x or y in a plane)",
        ),
        Field(
            "softening",
            "choices",
            [],
            label="modes of the softening",
            choices=SOFTENING,
            doc="Ban et al.: the stiffness of a mode varying from beam to beam with a "
            "coefficient of variation CV, (E₀ − ⟨E⟩)/E₀ = s CV² and σ_E²/E₀² = v CV², exact "
            "to second order (one solve per beam and mode)",
        ),
        Field(
            "axis",
            "choice",
            "x",
            label="axis of the softening",
            choices=(("x", "x"), ("y", "y"), ("z", "z")),
            doc="of the uniaxial stress whose Young's modulus softens (x or y in a plane)",
        ),
        Field(
            "replicas",
            "int",
            0,
            label="replicas per CV",
            doc="replicas of the stiffness of the beams of each mode on the same geometry, "
            "against the exact coefficients (0: the coefficients alone; one solve each; 40 "
            "per CV give the softening within a few per cent at CV = 0.1, the first-order "
            "part of each replica removed as a control variate)",
        ),
        Field(
            "cvs",
            "floats",
            [0.1, 0.2, 0.3],
            label="coefficients of variation",
            doc="CV of the stiffness of the beams of the replicas, σ/⟨k⟩",
        ),
        Field(
            "distribution",
            "choice",
            "normal",
            label="distribution of the stiffness",
            choices=DISTRIBUTIONS,
        ),
        Field("seed", "int", 1, label="seed of the replicas"),
    ],
    order=7,
    groups=(
        ("Beams", "beams"),
        ("Drawing of the frame", "draw"),
        ("Softening of non-identical beams", "softening", True),
    ),
    menu="Linear elasticity",
)
def network_frame_stiffness(p: dict, ctx) -> dict:
    from morphogen._utils import to_jsonable
    from morphogen.homogenization.frames import cowper_coefficient
    from morphogen.homogenization.networks import (
        frame_stiffness,
        network_frame,
        softening,
        softening_coefficients,
    )
    from morphogen.homogenization.postprocess import notation

    timoshenko = p["beams"] == "timoshenko"
    kappa = None
    if timoshenko:
        kappa = "cowper" if p.get("shear_coefficient") is None else p["shear_coefficient"]
    young, poisson = float(p["young"]), float(p["poisson"])
    modes = list(dict.fromkeys(p.get("softening") or []))
    replicas = int(p.get("replicas") or 0)
    cvs = [float(c) for c in (p.get("cvs") or [])]
    if replicas:
        if not modes:
            raise ValueError("replicas: of the modes of the softening (choose one at least)")
        if replicas < 2:
            raise ValueError("replicas per CV: at least 2 (0: the coefficients alone)")
        if not cvs or min(cvs) < 0 or (p["distribution"] == "bimodal" and max(cvs) >= 1):
            raise ValueError("coefficients of variation: non-negative, one at least (below 1 "
                             "for the bimodal law)")  # fmt: skip
    ctx.progress(0.05, "the frame of the framework")
    frame = network_frame(p["source"]["micro"], young=young, poisson=poisson,
                          shear_coefficient=kappa, rigid_joints=p.get("rigid_joints"))  # fmt: skip
    dim = len(frame.lattice)
    axis = "xyz".index(p["axis"])
    if modes and axis >= dim:
        raise ValueError("axis of the softening: x or y for a plane network")
    drawn = p.get("draw") or "force"
    draw_axis = "xyz".index(p.get("draw_axis") or "x")
    if draw_axis >= dim:
        raise ValueError("axis of the stress drawn: x or y for a plane network")
    ctx.progress(0.1, f"the periodic problem of {len(frame.beams)} beams")
    result = frame_stiffness(frame)
    rows = result.by_axis()
    summary = result.summary(young)
    plane = dim == 2
    radius = float(np.sqrt(frame.area[0] / np.pi))
    density = "fibre volume per unit area" if plane else "fibre volume fraction"
    values: dict = {
        "E / E_s": summary["frame_young"],
        "Poisson's ratio": summary["frame_poisson"],
        "E / E affine": summary["frame_affinity"],
        "stretching energy": summary["frame_stretching"],
        "non-affinity": summary["frame_nonaffinity"],
        density: result.density,
        "E / (E_s φ)": summary.get("frame_young_per_density"),
        **result.engineering(),
        "beams": len(frame.beams),
        "nodes": len(frame.nodes),
        "fibre radius": radius,
        "mean beam length": result.mean_length,
    }
    if timoshenko:
        cowper = kappa == "cowper"
        values["shear coefficient κ"] = float(cowper_coefficient(poisson) if cowper else kappa)
    tensor = result.stiffness
    units = ("units of the modulus of the fibres", "their inverse")
    if plane:
        units = ("per unit area: units of the modulus of the fibres times a length",
                 "their inverse")  # fmt: skip
    compliance = elastic_tables(ctx, tensor, "of the frame", units=units)
    parts = [k for k in ("stretching", "bending", "torsion", "shear") if k in rows[0]]
    ctx.table(
        [
            {
                "uniaxial stress along": row["axis"],
                "E": row["young"],
                "E / E_s": row["young"] / young,
                "ν (the next axis)": row["poisson"],
                "E / E affine": row["affinity"],
                **{f"{k} energy": row[k] for k in parts},
                "non-affinity": row["nonaffinity"],
            }
            for row in rows
        ],
        "axes",
        "uniaxial stresses: the energy split, the affine ratio and the non-affinity",
    )
    ctx.figure(_young_figure(tensor), "young_modulus", "directional Young's modulus")
    stretching = summary["frame_stretching"]
    kind = "stretching" if stretching >= 0.5 else "bending"
    joints = ""
    if p.get("rigid_joints") is not None:
        joints = f", rigid over {p['rigid_joints']:g} radii at the joints"
    text = [
        f"A {'plane ' if plane else ''}frame of {len(frame.beams)} "
        f"{'Timoshenko' if timoshenko else 'Euler–Bernoulli'} beams of radius {radius:.4g} "
        f"between {len(frame.nodes)} nodes{joints}: E / E_s = {summary['frame_young']:.4g} on "
        f"average over the axes, {100 * stretching:.0f} % of the energy of a uniaxial stress in "
        f"the stretching of the beams ({kind}-dominated), {summary['frame_affinity']:.3g} of "
        f"the affine estimate, non-affinity {summary['frame_nonaffinity']:.3g}.",
    ]
    if plane:
        text.append("In the plane, the stiffness is per unit area and φ the fibre volume per "
                    "unit area.")  # fmt: skip
    out = {"stiffness": tensor, "compliance": compliance, "notation": notation(tensor),
           "engineering": result.engineering(), "axes": rows, "summary": summary,
           "affine": result.affine}  # fmt: skip
    ctx.progress(0.2, "the forces and energies of the beams, the drawing")
    drawing = _draw_frame(frame, drawn, draw_axis, ctx)
    values["beams storing half the energy"] = drawing["half"]
    text.append(drawing["text"])
    out["drawing"] = {k: v for k, v in drawing.items() if k != "text"}
    if modes:
        softened, studies = [], []
        for k, mode in enumerate(modes):
            start, span = 0.3 + 0.65 * k / len(modes), 0.65 / len(modes)
            ctx.progress(start, f"softening: the {mode} of each beam")
            if replicas:

                def follow(done, total, start=start, span=span, mode=mode):
                    ctx.progress(start + span * (0.1 + 0.9 * done / total),
                                 f"replicas of the {mode}: {done} / {total}")  # fmt: skip

                study = softening(frame, mode, cvs, replicas=replicas, law=p["distribution"],
                                  axis=axis, seed=p["seed"], callback=follow)  # fmt: skip
                studies.append(study)
                c = study.coefficients
            else:
                c = softening_coefficients(frame, mode, axis)
            softened.append({"mode": mode, "s": c["mean"], "v": c["variance"], "E₀": c["young"]})
        ctx.table(
            softened,
            "softening",
            f"softening of non-identical beams along {p['axis']}: (E₀ − ⟨E⟩)/E₀ = s CV², "
            "σ_E²/E₀² = v CV²",
        )
        out["softening"] = {"axis": p["axis"], "modes": softened}
        found = ", ".join(f"s = {r['s']:.3g}, v = {r['v']:.3g} ({r['mode']})" for r in softened)
        text.append(f"Ban et al.'s softening along {p['axis']}: {found}.")
        if studies:
            text.append(_replicas_shown(studies, p, ctx))
            out["replicas"] = {"distribution": p["distribution"], "replicas": replicas,
                               "seed": p["seed"], "cv": cvs,
                               "rows": [r for s in studies for r in s.rows()]}  # fmt: skip
    ctx.path("frame_stiffness.json").write_text(
        json.dumps(to_jsonable(out), indent=2) + "\n", encoding="utf-8"
    )
    return {"values": values, "text": " ".join(text)}


#: The columns of the table of the replicas: their names in the page, from the keys of
#: :meth:`morphogen.homogenization.networks.Softening.rows`
REPLICA_COLUMNS = {
    "mode": "mode",
    "cv": "CV",
    "replicas": "replicas",
    "mean": "⟨E⟩/E₀",
    "mean_error": "± (σ/√n)",
    "corrected": "⟨E⟩/E₀ corrected",
    "corrected_error": "± (corrected)",
    "theory": "1 − s CV²",
    "std": "σ_E/E₀",
    "scatter": "σ_E²/E₀²",
    "scatter_error": "± (σ_E²/E₀²)",
    "theory_scatter": "v CV²",
}


def _replicas_shown(studies, p: dict, ctx) -> str:
    """The table and figure of the replicas of Ban et al. (one study per mode) and their
    sentence."""
    from morphogen.viz.figures import softening_figure

    rows = [{REPLICA_COLUMNS[k]: r[k] for k in REPLICA_COLUMNS} for s in studies for r in s.rows()]
    law, n = p["distribution"], studies[0].replicas
    ctx.table(rows, "replicas", f"replicas of non-identical beams along {p['axis']} ({law} law, "
              f"{n} per CV, seed {p['seed']}) against the second order")  # fmt: skip
    ctx.figure(
        softening_figure(studies, title=f"replicas along {p['axis']}: {law} law, {n} per CV"),
        "replicas",
        "replicas: the mean and scatter of E against the second order",
    )
    parts = []
    for study in studies:
        found = ", ".join(f"{1 - c:.4g} ± {e:.1g}" for c, e in zip(study.corrected, study.error,
                                                                   strict=True))  # fmt: skip
        theory = ", ".join(f"{t:.4g}" for t in study.predicted())
        parts.append(f"the {study.mode} softens by {found} (s CV² = {theory})")
    cv = ", ".join(f"{c:g}" for c in studies[0].cv)
    return (f"Replicas ({law} law, {n} per CV, corrected by the control variate) at CV = {cv}: "
            f"{'; '.join(parts)}.")  # fmt: skip


def _draw_frame(frame, drawn: str, axis: int, ctx) -> dict:
    """The drawing of the beams of a frame under a uniaxial stress along ``axis``, coloured
    by :data:`DRAWN` (the figure ``frame``), the share of the beams that store half its
    energy, the largest tension and compression and its sentence."""
    from morphogen.homogenization.networks import beam_forces
    from morphogen.viz.figures import frame_figure

    loads = beam_forces(frame, axis)
    parts = [k for k in ("stretching", "bending", "torsion", "shear") if k in loads]
    energy = sum(loads[k] for k in parts)
    total = float(energy.sum())
    stored = np.cumsum(np.sort(energy)[::-1]) / total
    half = float((np.searchsorted(stored, 0.5) + 1) / len(energy))
    name = "xyz"[axis]
    scale, label = DRAWN[drawn]
    if drawn == "force":
        shown = loads["force"]
    elif drawn == "stretching":
        shown = loads["stretching"] / total
    elif drawn == "bending":
        shown = sum(loads[k] for k in parts if k != "stretching") / total
    else:
        shown = None
    view = "in its plane" if len(frame.lattice) == 2 else "in the cell"
    title = f"{len(frame.beams)} beams {view}"
    caption = "the frame: its beams"
    if shown is not None:
        title += f", uniaxial stress along {name}"
        caption = f"the frame: {dict(DRAWINGS)[drawn]} under a stress along {name}"
    ctx.figure(frame_figure(frame, shown, scale=scale, label=label.format(axis=name), title=title),
               "frame", caption)  # fmt: skip
    force = loads["force"]
    text = (f"Under a uniaxial stress along {name}, {100 * half:.3g} % of the beams store half "
            f"the energy; axial forces from {force.min():.3g} to {force.max():.3g} for a stress "
            "of 1.")  # fmt: skip
    return {"axis": name, "colour": drawn, "half": half, "tension": float(force.max()),
            "compression": float(force.min()), "text": text}  # fmt: skip
