# Tutorial: mathematical morphology, from random sets to the model of an image

Mathematical morphology describes a random medium by what probes of given shapes find in
it (Matheron 1975; Serra 1982; Jeulin 2021): a point falls in the phase or not, a pair of
points falls in it together, a segment or a ball hits it or not. A realization of
morphoGen is a *random closed set* described exactly — centres and radii, cells, fields —
and each of its phases has an implicit function at any point of space: the probes are
answered exactly, so that the measures compare with the theory of the model without the
bias of a grid, and the voxels and meshes come only afterwards, when a solver needs them.
This tutorial follows the theory step by step on a Boolean model of spheres, whose closed
forms are known, then turns to an image: its phases, its measures, and the model that
reproduces them best.

The snippets follow each other in one session and take about half a minute in all.

## 1. A random closed set, described exactly

The Boolean model is the union of grains ``A'`` implanted on the points of a Poisson
process of intensity ``θ``: here spheres of radius 1 that fill 30 % of the space, in a
periodic cube (a torus: the realization tiles space):

```python
import numpy as np
import morphogen as mg
from morphogen import analysis
from morphogen import morphology as mm

model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3)
micro = model.generate(mg.Domain.cube(12.0, periodic=True), seed=1)
print(model.theta, len(micro.spheres))  # 0.0851 germs per unit volume, 147 spheres
```

The realization knows, at any point, the phase it lies in and its signed distance to the
spheres (negative inside them): the implicit function of the phase 1, exact outside it.
A point at a distance 0.15 from the nearest sphere is outside, whatever a voxel image of
the cube would say of it:

```python
points = np.array([[6.0, 6.0, 6.0], [1.0, 2.0, 3.0], [9.5, 0.2, 4.0]])
print(mm.phases_at(micro, points))  # [1 0 1]
print(micro.level_set_at(points, 2.0))  # [-0.626  0.153 -0.043]
```

## 2. Probes: the Choquet capacity

A random closed set ``A`` is known in law by its *Choquet capacity*, the probability that
a compact set ``K`` placed at random hits it, ``T(K) = P(K ∩ A ≠ ∅) = 1 − Q(K)``
(Matheron's theorem). For an ergodic medium it is the volume fraction of the dilation
``A ⊕ Ǩ`` in one realization. Particular probes give the classical measures: a point the
volume fraction ``T({x}) = p``, a pair of points at the distance ``h`` the covariance
``C(h) = 2p − T({0, h})``, a segment the lineal path of the complement, a ball the
spherical contact distribution. morphoGen draws the probes uniformly in the box and asks
the realization: the estimates carry the standard deviations of their samplings.

```python
vf = mm.volume_fraction(micro, points=2**16)
print(vf.value, vf.error)  # 0.3012 ± 0.0018: the fraction of this realization

lags = np.linspace(0.0, 3.0, 7)
c = mm.covariance(micro, lags)  # pairs of points in random directions
print(np.round(c.values, 3))  # 0.302 0.218 0.149 0.109 0.096 0.094 0.093
print(np.round(model.covariance(lags), 3))  # 0.3 0.214 0.148 0.105 0.09 0.09 0.09
```

For the Boolean model the capacity has a closed form for every compact set,
``T(K) = 1 − exp(−θ E|A' ⊕ Ǩ|)``: a ball of radius ``r`` dilates the spheres to the
radius ``R + r``, a segment of length ``l`` sweeps them along it. The probes of balls ask
the distance to the phase at each point (all radii at once); those of segments march
along them by steps that cannot cross the interface (sphere tracing):

```python
radii = np.linspace(0.0, 1.0, 6)
balls = mm.capacity(micro, "ball", radii, points=2**16)
lengths = np.linspace(0.0, 6.0, 7)
segments = mm.capacity(micro, "segment", lengths, points=2**14)
print(np.round(balls.values, 3))  # 0.301 0.459 0.618 0.761 0.871 0.94
print(np.round(model.choquet_capacity("ball", radii), 3))  # 0.3 0.46 0.624 0.768 0.875 0.942
```

These are Matheron's tests of a Boolean model, in straight lines. For spheres of one
radius ``−log Q(B_r) = θ (4π/3) (R + r)^3``: its cube root grows linearly with ``r``, and
the line gives ``R`` and ``θ``; along segments ``−log Q(l) = θ ((4π/3) R^3 + π R^2 l)``
grows linearly with ``l`` (the chords of the pores are exponential), and its line gives
them again:

```python
slope, intercept = np.polyfit(radii, np.cbrt(-np.log(1.0 - balls.values)), 1)
print(intercept / slope, 3 * slope**3 / (4 * np.pi))  # R = 1.012, θ = 0.0821
slope, intercept = np.polyfit(lengths, -np.log(1.0 - segments.values), 1)
radius = 3 * intercept / (4 * slope)
print(radius, slope / (np.pi * radius**2))  # R = 0.984, θ = 0.0883
```

```python
from matplotlib.figure import Figure

fig = Figure(figsize=(7.6, 3.2), layout="constrained")
ax1, ax2 = fig.subplots(1, 2)
fine = np.linspace(0.0, 1.0, 50)
ax1.plot(radii, np.cbrt(-np.log(1.0 - balls.values)), "o", label="measured")
ax1.plot(fine, np.cbrt(-np.log(1.0 - model.choquet_capacity("ball", fine))), label="theory")
ax1.set_xlabel("radius r of the ball")
ax1.set_ylabel("(−log Q(B_r))^(1/3)")
fine = np.linspace(0.0, 6.0, 50)
ax2.plot(lengths, -np.log(1.0 - segments.values), "o", label="measured")
ax2.plot(fine, -np.log(1.0 - model.choquet_capacity("segment", fine)), label="theory")
ax2.set_xlabel("length l of the segment")
ax2.set_ylabel("−log Q(l)")
ax2.legend()
fig.savefig("morphology_capacities.png", dpi=110)
```

```{image} _static/tutorial/morphology_capacities.png
:alt: the capacities of balls and segments of a Boolean model of spheres, measured on its realization and in closed form, as straight lines
:class: tutorial-figure
```

## 3. Dilation, erosion, opening, closing

A structuring element ``K`` placed at every point transforms the set: the dilation
``A ⊕ Ǩ`` (where ``K`` hits ``A``), the erosion ``A ⊖ Ǩ = (A^c ⊕ Ǩ)^c`` (where it lies
inside), the opening ``(A ⊖ Ǩ) ⊕ K`` (the union of the translates of ``K`` inside ``A``)
and the closing, the complement of the opening of the complement. On the exact geometry
the dilation of a union of spheres by a ball is the union of the same spheres made
thicker: its volume fraction is the capacity of that ball, exactly (§105 of the theory
notes):

```python
grown = mm.dilation(micro, 0.5)
print(mm.minkowski_functionals(grown).volume_fraction)  # 0.6934, the exact A ⊕ B_0.5
print(model.choquet_capacity("ball", [0.5])[0])  # 0.6999 in law
```

On a voxel image the four operations take digital balls, segments, pairs or cubes
(`morphogen.analysis`). The opening is idempotent and keeps less than the phase, the
closing more, the erosion of the phase is the complement of the dilation of the pores:

```python
image = micro.voxelize(shape=64)
eroded = analysis.erosion(image, "ball", 0.5)
dilated = analysis.dilation(image, "ball", 0.5)
opened = analysis.opening(image, "ball", 0.5)
closed = analysis.closing(image, "ball", 0.5)
print(eroded.mean(), opened.mean(), closed.mean(), dilated.mean())  # 0.069 0.290 0.343 0.642
assert np.array_equal(analysis.opening(opened.astype(np.uint8), "ball", 0.5), opened)
assert np.array_equal(eroded, ~analysis.dilation(image, "ball", 0.5, phase=0))
```

The openings by balls of growing radii sieve a phase: the fraction that each keeps
decreases with the radius, and its decrease is the *granulometry* of the phase — here
of the pores, whose largest inscribed balls have a mean radius of 1.18:

```python
sizes = analysis.granulometry(image, phase=0)
print(sizes.mean_radius)  # 1.18
```

```python
fig = Figure(figsize=(9.0, 2.2), layout="constrained")
panels = {"A": image.data, "A ⊖ B": eroded, "A ⊕ B": dilated, "A ∘ B": opened, "A • B": closed}
for ax, (name, data) in zip(fig.subplots(1, 5), panels.items(), strict=True):
    ax.imshow(np.flipud(np.asarray(data)[:, :, 32].T), cmap="Greys", interpolation="nearest")
    ax.set_title(name)
    ax.set_axis_off()
fig.savefig("morphology_operations.png", dpi=110)
```

```{image} _static/tutorial/morphology_operations.png
:alt: a section of the spheres, of their erosion, dilation, opening and closing by a ball of radius 0.5
:class: tutorial-figure
```

## 4. The Minkowski functionals, exactly

Of the measures of a set that add up as volumes do (``φ(A ∪ B) = φ(A) + φ(B) − φ(A ∩ B)``)
and that do not depend on where it lies, only four are independent (Hadwiger's theorem):
the volume ``V``, the surface ``S``, the integral of mean curvature ``M = ∫ H dS`` and the
Euler characteristic ``χ`` (components − tunnels + cavities). For a Boolean model of convex
grains their densities are Miles' formulae (§39). The union of the spheres of a
realization has them exactly, from the power diagram of the spheres and its dual complex
(§105): beside them, the estimates of the voxel image of 64³ lose the surface between the
voxels and count spheres closer than a voxel as touching:

```python
exact = mm.minkowski_functionals(micro)
digital = analysis.minkowski(image)
theory = model.theory()
print(exact.specific_surface, digital.specific_surface, theory["specific_surface"])
# 0.7422 (exact), 0.7104 (voxels), 0.7490 (Miles)
print(exact.mean_curvature_density, digital.mean_curvature, theory["mean_curvature_density"])
# 0.4772, 0.4193, 0.5018
print(exact.euler, digital.euler_density * 12.0**3)  # -5 for this union, -25 on the voxels
```

The Euler density is a difference of nearly equal terms, and a realization of this size
gives it within a few thousandths: the mean of eight realizations is within its error of
Miles' formula, which the estimates of the voxels are not (their bias is larger than the
fluctuation):

```python
densities = [
    mm.minkowski_functionals(model.generate(mg.Domain.cube(12.0), seed=s)).euler_density
    for s in range(8)
]
print(np.mean(densities), np.std(densities, ddof=1) / np.sqrt(8), theory["euler_density"])
# 0.00007 ± 0.0022 for 0.0028
```

## 5. Stereology: what a section tells of a volume

The section of a union of spheres by a plane is a union of discs, which morphoGen measures
exactly too: its area fraction, the length of its boundary per unit area and its Euler
characteristic per unit area. For a stationary isotropic medium the sections give the
volume: ``A_A = V_V``, ``L_A = (π/4) S_V`` and ``χ_A = M_V / 2π`` (Crofton). The means
over nine planes of the cube:

```python
sections = [
    mm.section_functionals(micro, axis, position).densities()
    for axis in range(3)
    for position in (2.0, 6.0, 10.0)
]
print(np.mean([s["boundary_density"] for s in sections]), np.pi / 4 * exact.specific_surface)
# 0.5833 for 0.5829
print(np.mean([s["euler_density"] for s in sections]), exact.mean_curvature_density / (2 * np.pi))
# 0.0733 for 0.0759
```

The sizes of the particles themselves come from the sizes of their sections by the
inversion of Wicksell and Saltykov (`analysis.unfold_sections`, §39).

## 6. From an image to a model

morphoGen comes with a volume of grey levels as a tomography of a particulate composite
would give it: 48³ voxels of 0.25, blurred and noisy (`morphogen.models.image`). Its
phases are the classes of its levels split by Otsu's threshold, after a Gaussian
smoothing of a voxel that removes the noise; its particles are the connected parts of the
phase 1, those that touch split by the watershed of their distance map:

```python
from morphogen.models.image import ImageModel
from morphogen.morphology import identification, segmentation

grey = ImageModel().image()
phases = segmentation.segment(grey, sigma=1.0)
print(phases.metadata["thresholds"], phases.data.mean())  # [100.4], 0.2196
particles = segmentation.particle_shapes(phases)
print(particles.count, particles.moments(1))  # 70 particles of equivalent radius 1.23
```

Which model is this? `identification.identify` fits each candidate to the fraction and
the specific surface of the image by its closed forms, simulates it in the box of the
image on voxels of its size, calibrates its length so that its simulated images have the
specific surface of the image, and scores how well they reproduce the rest: the shape of
the covariance and the contact distributions of balls and segments, which were not
fitted:

```python
found = identification.identify(phases)
print(found.table())
# hardcore_spheres  0.0449 ...   boolean_spheres  0.0595 ...   gaussian_excursion  0.1598 ...
best = found.best.model
print(best.to_config())  # hard spheres of radius 1.31 at a fraction of 0.22
```

The image was made of hard spheres of radius 1.2 at 0.2: the blur of the beam and the
smoothing make the particles of the image larger, and the model reproduces the image as
it is. The model then gives realizations of any size, exactly described — and the image
itself is a structure of morphoGen as well, its phases the iso-surfaces of its levels at
the threshold (implicit functions at any point, interpolated between the voxels), ready
for the probes, the meshes and the solvers, in windows drawn at random for the size of
its representative volume:

```python
larger = best.generate(mg.Domain.cube(24.0), seed=0)
window = ImageModel(sigma=1.0).generate(mg.Domain.cube(12.0, periodic=False))
print(mm.volume_fraction(window).value)  # 0.2189, at points between the voxels too
```

```python
model_section = larger.voxelize(shape=96).data[:48, :48, 48]
fig = Figure(figsize=(7.2, 2.6), layout="constrained")
panels = {
    "grey levels": grey.data[:, :, 24],
    "phases": phases.data[:, :, 24],
    "identified model": model_section,
}
for ax, (name, data) in zip(fig.subplots(1, 3), panels.items(), strict=True):
    ax.imshow(np.flipud(np.asarray(data).T), cmap="Greys_r", interpolation="nearest")
    ax.set_title(name)
    ax.set_axis_off()
fig.savefig("morphology_identification.png", dpi=110)
```

```{image} _static/tutorial/morphology_identification.png
:alt: a section of the grey levels of the volume, of its phases, and of a realization of the model identified
:class: tutorial-figure
```

## Where to read on

* The theory notes: random sets probed by structuring elements and Miles' formulae (§39),
  the implicit geometry of every model (§75), the measures of the implicit geometry
  (§104), the exact functionals of unions of balls and of their sections (§105), the
  identification of a model from an image (§106) and the models of images (§107).
* The pages [Measures](measures.md) and [Geometry](geometry.md) of the user guide, and the
  modules `morphogen.morphology` and `morphogen.analysis` of the [API reference](api.md).
