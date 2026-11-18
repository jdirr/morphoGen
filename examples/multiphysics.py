"""One microstructure, many physics.

A Boolean model of spheres in a periodic cube is homogenized for its elastic, thermal,
electrical, dielectric and diffusive properties, its thermal expansion and its swelling,
its poroelastic law when the spheres are pores, its piezoelectric law when they are a
poled ceramic, its storage and loss moduli when its matrix is viscoelastic, and, cut by
extended finite elements, without the stairs of its voxels. Under a minute on a laptop; the
guide to the physics (docs/physics.md) explains each step.
"""

import dataclasses

import morphogen as mg
from morphogen.homogenization import VOID, LevelSetSample, Material, bounds, homogenize

micro = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.2).generate(
    mg.Domain.cube(6.0, periodic=True), seed=1
)
image = micro.voxelize(shape=24)
fractions = mg.analysis.phase_fractions(image)
print(f"spheres: {fractions[1]:.3f} of the volume, {image.shape} voxels\n")

# -- elasticity and the transport of a scalar potential (SI units) -------------------------
polymer = Material(young=3e9, poisson=0.35, conductivity=0.2, electrical_conductivity=1e-12,
                   permittivity=3.0, diffusivity=1e-12, expansion=60e-6, swelling=0.4)  # fmt: skip
glass = Material(young=70e9, poisson=0.22, conductivity=1.0, electrical_conductivity=1e-12,
                 permittivity=6.0, diffusivity=0.0, expansion=5e-6)  # fmt: skip
materials = {0: polymer, 1: glass}
properties = ["bulk_modulus", "shear_modulus", "conductivity", "permittivity", "diffusivity"]
out = homogenize(image, materials, properties=[*properties, "thermal_expansion", "swelling"])
limits = bounds(fractions, materials)
print(f"{'property':<16}{'apparent':>12}{'HS lower':>12}{'HS upper':>12}")
for name in properties:
    value = out[f"{name}.periodic"].value
    low, high = limits[name]["hs_lower"], limits[name]["hs_upper"]
    print(f"{name:<16}{value:>12.4g}{low:>12.4g}{high:>12.4g}")
alpha, beta = out["thermal_expansion.periodic"]["mean"], out["swelling.periodic"]["mean"]
print(f"thermal expansion {alpha:.3g} /K, swelling {beta:.3g} per unit of moisture\n")

# -- the spheres as pores: the poroelastic law of Biot ---------------------------------------
porous = {0: Material(young=20e9, poisson=0.2), 1: VOID}
rock = homogenize(image, porous, properties=["poroelastic"], fluid_bulk_modulus=2.2e9)
rock = rock["poroelastic.periodic"]
print(f"porous: Biot's coefficient {rock['biot_coefficient']:.3f}, modulus "
      f"{rock['biot_modulus']:.3g} Pa, Skempton's {rock['skempton_coefficient']:.3f}")  # fmt: skip

# -- the spheres as a poled ceramic: a piezoelectric composite -------------------------------
ceramic = dataclasses.replace(
    Material.hexagonal(166e9, 77e9, 78e9, 162e9, 43e9),
    permittivity=(11.2e-9, 11.2e-9, 12.6e-9),
    coupling={"piezoelectric": {"e31": -4.4, "e33": 18.6, "e15": 11.6}},
)
soft = Material(young=4e9, poisson=0.35, permittivity=3.7e-11)
coarse = micro.voxelize(shape=16)
law = homogenize(coarse, {0: soft, 1: ceramic}, properties=["piezoelectric"])
law = law["piezoelectric.periodic"]
print(f"piezoelectric: d33 {law['d33']:.3g} C/N, g33 {law['g33']:.3g} V m/N, "
      f"k_t {law['k_t']:.3f}")  # fmt: skip

# -- a viscoelastic matrix: storage and loss moduli over five decades ------------------------
resin = dataclasses.replace(
    polymer, viscoelasticity={"shear": [[0.5, 1e-2], [0.3, 1.0]], "bulk": [[0.2, 0.1]]}
)
omegas = [0.01, 1.0, 100.0, 1e4]
dma = homogenize(coarse, {0: resin, 1: glass}, properties=["dynamic_stiffness"],
                 frequencies=omegas)["dynamic_stiffness.periodic"]  # fmt: skip
for k, omega in enumerate(omegas, start=1):
    print(f"omega {omega:>8g}: G' {dma[f'shear_storage_{k}']:.3g} Pa, tan(delta) "
          f"{dma[f'shear_tan_delta_{k}']:.3f}")  # fmt: skip

# -- without the stairs of the voxels: extended finite elements on a coarser grid ------------
sample = LevelSetSample.of(micro, 12)
cut = homogenize(sample, materials, properties=["bulk_modulus", "conductivity"])
print(f"\nextended finite elements (12 cells per edge): bulk modulus "
      f"{cut['bulk_modulus.periodic'].value:.4g} Pa, conductivity "
      f"{cut['conductivity.periodic'].value:.4g} W/(m K)")  # fmt: skip
