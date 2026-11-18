"""From an open-cell foam to the sound it absorbs.

A foam is generated as a level set of distance fields, its pores are solved for the
parameters of the equivalent fluid of its rigid frame (a Stokes flow by the lattice
Boltzmann method, the conduction of the pore fluid, a Poisson problem), and the model of
Johnson, Champoux, Allard and Lafarge gives the absorption of layers of it on a rigid
wall. About a minute on a laptop; the figure is written to acoustic_foam.png.

The parameters converge with the voxels: lower ``VOXEL`` until they settle (the struts
are two voxels thick here), and run several realizations for their statistics (the
template homogenization_acoustic_foam with an ``rve`` section).
"""

import numpy as np

import morphogen as mg
from morphogen.homogenization import VOID, Material, absorption, homogenize
from morphogen.homogenization.acoustics import AIR, along_axis, equivalent_fluid

MM = 1e-3  # the model is in millimetres
VOXEL = 0.025  # mm

# 1. The foam: cells of about 0.3 mm, 8 % of solid in struts that thicken at their nodes
model = mg.models.LevelSetFoam(
    seed_radius={"type": "uniform", "low": 0.12, "high": 0.18},
    packing="dense",
    packing_fraction=0.6,
    volume_fraction=0.08,
    curvature=-0.7,
    c1=50.0,
    c2=1.0,
)
micro = model.generate(mg.Domain.cube(1.5, periodic=True), seed=3)
image = micro.voxelize(voxel_size=VOXEL)

# 2. The parameters of its equivalent fluid (the pores are the void phase)
frame = Material(young=2500.0, poisson=0.35)  # rigid for the sound: only its shape counts
result = homogenize(image, {0: VOID, 1: frame}, properties=["acoustic"], tol=1e-4)
parameters = result["acoustic.periodic"]

# 3. In SI units: lengths times the unit, permeabilities times its square
si = along_axis(parameters)  # the means of the three axes: an isotropic foam
for name, power in (("permeability", 2), ("viscous_length", 1), ("thermal_length", 1),
                    ("thermal_permeability", 2)):  # fmt: skip
    si[name] *= MM**power
resistivity = AIR.viscosity / si["permeability"]
print(f"porosity               {si['porosity']:.3f}")
print(f"permeability           {si['permeability']:.3e} m²")
print(f"resistivity            {resistivity:.0f} Pa s/m²")
print(f"tortuosity             {si['tortuosity']:.3f}")
print(f"viscous length         {1e6 * si['viscous_length']:.0f} µm")
print(f"thermal length         {1e6 * si['thermal_length']:.0f} µm")
print(f"thermal permeability   {si['thermal_permeability']:.3e} m²")

# 4. The absorption of layers on a rigid wall, at normal incidence
frequencies = np.geomspace(100.0, 8000.0, 80)
curves = {d: absorption(frequencies, d, **si)["absorption"] for d in (0.02, 0.04, 0.08)}
for thickness, alpha in curves.items():
    half = frequencies[np.argmax(alpha > 0.5)] if np.any(alpha > 0.5) else float("nan")
    print(f"{1e3 * thickness:3.0f} mm: absorption above 0.5 from {half:5.0f} Hz, "
          f"{alpha[np.searchsorted(frequencies, 1000.0)]:.2f} at 1 kHz")  # fmt: skip
medium = equivalent_fluid([1000.0], **si)
print(f"at 1 kHz the sound travels at {2 * np.pi * 1000.0 / medium['wavenumber'][0].real:.0f} "
      "m/s in the foam")  # fmt: skip

try:
    from matplotlib.figure import Figure
except ImportError:  # the extra `plot`
    Figure = None
if Figure is not None:
    fig = Figure(figsize=(5.6, 3.8), layout="constrained")
    ax = fig.subplots()
    for thickness, alpha in curves.items():
        ax.semilogx(frequencies, alpha, label=f"{1e3 * thickness:.0f} mm")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("absorption coefficient")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ax.legend(title="layer", frameon=False)
    fig.savefig("acoustic_foam.png", dpi=150)
    print("figure: acoustic_foam.png")
