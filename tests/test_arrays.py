"""Voxel images as the arrays of other libraries without copies (DLPack, JAX), and the phase
fields of the realizations for spectral solvers of phase-field equations: the profile of
the interface against its closed form, and Allen–Cahn's motion by curvature of a disc of
morphoGen in Exponax against R² = R0² - 2 nu t."""

from __future__ import annotations

import math

import numpy as np
import pytest

from morphogen.domain import Domain
from morphogen.image import VoxelImage
from morphogen.interop import arrays as ia
from morphogen.models.base import model_from_config
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Segments, Spheres
from morphogen.seeding import derive_seed


def _sphere(r=0.3, centre=(0.5, 0.5, 0.5)):
    return PrimitiveUnion(Domain(size=1.0, periodic=True),
                          spheres=Spheres(np.array([centre]), np.array([r])))  # fmt: skip


def _disc(r=0.3):
    """A cylinder through the box along z: a disc in every plane normal to z."""
    seg = Segments(np.array([[0.5, 0.5, -0.5]]), np.array([[0.5, 0.5, 1.5]]), np.array([r]))
    return PrimitiveUnion(Domain(size=1.0, periodic=True), segments=seg)


def test_aligned_buffers_and_dlpack_without_copies():
    """Aligned buffers; an image hands its array over by DLPack without a copy, and an
    array comes back as an image on the grid of the one it came from."""
    a = ia.aligned_empty((5, 6, 7), np.float32)
    assert ia.is_aligned(a) and a.ctypes.data % 64 == 0 and a.shape == (5, 6, 7)
    b = np.zeros(1000, np.float32)[1:]  # 4 bytes from an aligned start at best
    assert not ia.is_aligned(b)
    c = ia.aligned(b)
    assert ia.is_aligned(c) and np.array_equal(c, b) and ia.aligned(c) is c
    assert ia.aligned(c, np.float64).dtype == np.float64
    image = VoxelImage(ia.aligned_empty((4, 4, 4)), spacing=0.5, origin=(1, 2, 3), periodic=True)
    image.data[...] = np.arange(64).reshape(4, 4, 4)
    shared = np.from_dlpack(image)
    assert np.shares_memory(shared, image.data) and shared[1, 2, 3] == image.data[1, 2, 3]
    back = ia.from_dlpack(shared[None], like=image, channel=0, name="u")
    assert back.spacing == image.spacing and back.origin == image.origin and back.periodic
    assert np.shares_memory(back.data, image.data) and back.name == "u"
    planar = ia.from_dlpack(np.ones((1, 6, 6)), channel=0)
    assert planar.shape == (6, 6, 1) and planar.periodic and planar.spacing == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="channel="):
        ia.from_dlpack(np.ones((2, 3, 3, 3)))


def test_phase_field_of_a_sphere():
    """The diffuse interface tanh(-d / (sqrt 2 epsilon)) at the centres of the voxels, d the
    distance to the sphere; the signed distance of a smooth function that is not one (the
    gyroid) has a gradient of norm one at its interface."""
    n, r = 48, 0.3
    u = ia.phase_field(_sphere(r), shape=n)
    h = 1.0 / n
    assert u.metadata["epsilon"] == pytest.approx(1.5 * h) and u.data.dtype == np.float32
    assert ia.is_aligned(u) and u.name == "phase_field"
    x = (np.arange(n) + 0.5) * h - 0.5
    d = np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2) - r
    exact = np.tanh(-d / (math.sqrt(2.0) * 1.5 * h))
    # exact but beyond 8 epsilon, where the phases are pure: 1 - tanh(8 / sqrt 2) = 2.4e-5
    assert np.abs(u.data - exact).max() < 1.01 * (1.0 - math.tanh(8.0 / math.sqrt(2.0)))
    assert np.abs(u.data - exact)[np.abs(d) < 6 * h].max() < 1e-6
    two = ia.phase_field(_sphere(r), shape=n, epsilon=2 * h, inside=1.0, outside=0.0)
    # 0 far outside; inside, the centre 14 voxels deep: 1/2 + tanh(14.4 / (2 sqrt 2)) / 2
    assert two.data.min() == 0.0 and 0.9999 < two.data.max() < 1.0
    gyroid = model_from_config({"type": "tpms", "surface": "gyroid", "volume_fraction": 0.5})
    dist = ia.signed_distance(gyroid.generate(Domain(size=1.0, periodic=True), 0), shape=40)
    phi = np.clip(np.asarray(dist.data), -0.2, 0.2)  # the far voxels are infinite
    near = np.abs(phi) < 0.5 / 40
    grad = ia._gradient_norm(phi, dist.spacing, True)
    assert near.sum() > 1000 and np.median(grad[near]) == pytest.approx(1.0, abs=0.05)
    with pytest.raises(ValueError, match="epsilon"):
        ia.phase_field(_sphere(r), shape=8, epsilon=0.0)


def test_spectral_grids_states_and_ensembles():
    """The arguments of a spectral solver of a periodic cube (or square), the state of an
    image in its layout (a view), the phase fields of realizations stacked for ensembles."""
    u = ia.phase_field(_sphere(), shape=16)
    assert ia.spectral_grid(u) == {"num_spatial_dims": 3, "domain_extent": 1.0, "num_points": 16}
    s = ia.state(u)
    assert s.shape == (1, 16, 16, 16) and np.shares_memory(s, u.data)
    plane = u.layer(axis=2)
    assert ia.spectral_grid(plane)["num_spatial_dims"] == 2 and ia.state(plane).shape == (1, 16, 16)
    with pytest.raises(ValueError, match="periodic"):
        ia.spectral_grid(VoxelImage(np.zeros((4, 4, 4)), periodic=False))
    with pytest.raises(ValueError, match="as many voxels"):
        ia.spectral_grid(VoxelImage(np.zeros((4, 4, 8)), periodic=True))
    model = model_from_config({"type": "boolean_spheres", "radius": 0.15, "volume_fraction": 0.2})
    domain = Domain(size=1.0, periodic=True)
    seeds = [derive_seed(3, k) for k in range(3)]
    batch = ia.realizations(model, domain, seeds, shape=16)
    assert batch.shape == (3, 1, 16, 16, 16) and ia.is_aligned(batch)
    second = ia.phase_field(model.generate(domain, seeds[1]), shape=16)
    assert np.array_equal(batch[1, 0], second.data)


def test_jax_arrays_share_the_buffers():
    """JAX reads an aligned image where it is: the same buffer, which an image that was not
    aligned takes once; a field of JAX comes back without a copy."""
    jax = pytest.importorskip("jax")
    image = VoxelImage(np.zeros(4097, np.float32)[1:].reshape(16, 16, 16), periodic=True)
    assert not ia.is_aligned(image)
    j = ia.to_jax(image)
    assert ia.is_aligned(image) and j.unsafe_buffer_pointer() == image.data.ctypes.data
    assert jax.numpy.from_dlpack(image).unsafe_buffer_pointer() == image.data.ctypes.data
    doubled = j * 2.0
    back = ia.from_dlpack(doubled, like=image)
    assert np.shares_memory(back.data, np.from_dlpack(doubled)) and back.periodic


def test_allen_cahn_shrinks_a_disc_by_its_curvature():
    """A disc of morphoGen in Exponax's Allen–Cahn equation (epsilon = sqrt(nu), the profile
    of its equilibrium): its area shrinks by motion by curvature, R² = R0² - 2 nu t in the
    plane, within 1 %; Cahn–Hilliard keeps the mass of a sphere; the realizations of a
    model roll out together (jax.vmap)."""
    pytest.importorskip("jax")
    ex = pytest.importorskip("exponax")
    import jax
    import jax.numpy as jnp

    n, r = 64, 0.3
    u = ia.phase_field(_disc(r), shape=n).layer(axis=2)
    nu, h, dt = u.metadata["epsilon"] ** 2, 1.0 / n, 0.05
    steps = round(0.016 / (2 * nu) / dt)  # R² falls by 0.016
    stepper = ex.stepper.reaction.AllenCahn(**ia.spectral_grid(u), dt=dt, diffusivity=nu)
    trajectory = ex.rollout(stepper, steps, include_init=True)(ia.to_jax(ia.state(u)))

    def area(v):
        return float(jnp.sum((1.0 + v) / 2.0)) * h * h

    shrunk = (area(trajectory[0]) - area(trajectory[-1])) / math.pi
    assert shrunk == pytest.approx(2 * nu * steps * dt, rel=0.01)
    sphere = ia.phase_field(_sphere(r), shape=32)
    ch = ex.stepper.reaction.CahnHilliard(**ia.spectral_grid(sphere), dt=1e-3, diffusivity=0.01,
                                          gamma=sphere.metadata["epsilon"] ** 2)  # fmt: skip
    end = ex.repeat(ch, 100)(ia.to_jax(ia.state(sphere)))
    assert float(end.mean()) == pytest.approx(float(sphere.data.mean()), abs=1e-5)
    model = model_from_config({"type": "boolean_spheres", "radius": 0.2, "volume_fraction": 0.3})
    batch = ia.realizations(model, Domain(size=1.0, periodic=True), [1, 2], shape=16)
    grid = {"num_spatial_dims": 3, "domain_extent": 1.0, "num_points": 16}
    ac = ex.stepper.reaction.AllenCahn(**grid, dt=0.05, diffusivity=(1.5 / 16) ** 2)
    stepped = jax.vmap(ac)(ia.to_jax(batch))
    assert stepped.shape == (2, 1, 16, 16, 16)
