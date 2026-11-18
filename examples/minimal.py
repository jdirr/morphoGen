"""Minimal example: generate, measure, compare with theory, export."""

import morphogen as mg

# 1. A model: Boolean spheres of radius 1 at 30 % volume fraction
model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3)

# 2. One realization in a periodic cube of edge 20 (explicit geometry: centres, radii)
micro = model.generate(mg.Domain.cube(20.0, periodic=True), seed=42)
print(micro)

# 3. Voxelize at any resolution and measure
image = micro.voxelize(shape=128)
theory = model.theory()
vf = mg.analysis.volume_fraction(image)
sv = mg.analysis.specific_surface(image)
print(f"volume fraction  {vf:.4f}  theory {theory['volume_fraction']:.4f}")
print(f"specific surface {sv:.4f}  theory {theory['specific_surface']:.4f}")

# 4. Export: image for ParaView / FFT codes, hexahedral mesh of the spheres, geometry
mg.io.save_image(image, "spheres.vti")
mesh = mg.mesh.voxel_mesh(image, phases=[1])
mg.io.save_mesh(mesh, "spheres.inp")
mg.io.save_microstructure(micro, "spheres.json")  # regenerates the realization exactly
