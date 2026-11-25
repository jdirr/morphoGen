"""Voxel images as the arrays of other libraries without copies, and the phase fields of
the realizations for spectral solvers of phase-field equations.

**DLPack.** A :class:`~morphogen.image.VoxelImage` hands its array over by the DLPack
protocol (``image.__dlpack__``): ``jax.numpy.from_dlpack(image)``,
``torch.from_dlpack(image)``, ``cupy.from_dlpack(image)`` and ``numpy.from_dlpack(image)``
read the voxels where they are. JAX on the CPU shares a buffer only when it is aligned on
64 bytes (that of XLA), and copies it otherwise: :func:`aligned` gives such a buffer (the
array itself when it is one), :func:`aligned_empty` makes one, and :func:`to_jax` gives the
image an aligned buffer once, which the image and the JAX array then share. JAX works in
single precision unless ``jax_enable_x64`` is set: a field in double precision is copied
into single; the phase fields are single by default. :func:`from_dlpack` brings an array
of any of them back as an image of a grid (a state of a solver, its channel), without a
copy on the CPU.

**Phase fields.** :func:`phase_field` turns a phase of a realization into a diffuse
interface, ``u = tanh(-d / (sqrt(2) epsilon))`` (1 in the phase, -1 outside), ``d`` the
signed distance to its boundary (the implicit function of the phase over the norm of its
gradient, exact for the signed distances and for the functions whose gradient is constant
across the interface): the profile of the plane interface at equilibrium of Allen–Cahn's
equation ``u_t = nu Lap u + u - u^3`` for ``epsilon = sqrt(nu)``, and of Cahn–Hilliard's
``u_t = nu Lap(u^3 - u - gamma Lap u)`` for ``epsilon = sqrt(gamma)`` (the defaults of
Exponax's ``AllenCahn`` and ``CahnHilliard``; docs/theory.md). :func:`realizations` stacks
those of several realizations (batch, channel, voxels) for ensemble rollouts.

**Grids.** The values are those of the centres of the voxels. A spectral solver of a
periodic box whose grid starts at 0 (Exponax: ``x_i = i L / N``, the right face left out)
takes them as they are on its grid shifted by half a voxel, its origin at the centre of the
first voxel: nothing is interpolated, and :func:`from_dlpack` with ``like`` gives the state
back on the voxels. :func:`spectral_grid` gives the arguments of such a solver (the number
of dimensions, the length of the box, the points per axis) and checks that the image is one
(periodic, a cube of cubic voxels; a planar image, one voxel thick, a square in 2D);
:func:`state` the array in its layout (channel, voxels).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from morphogen.image import VoxelImage

#: The alignment of the buffers that JAX shares on the CPU without a copy (bytes)
ALIGNMENT = 64


def is_aligned(array, alignment: int = ALIGNMENT) -> bool:
    """Whether an array is C-contiguous and starts on a multiple of ``alignment`` bytes."""
    a = np.asarray(array.data if isinstance(array, VoxelImage) else array)
    return bool(a.flags.c_contiguous) and a.ctypes.data % int(alignment) == 0


def aligned_empty(shape, dtype=np.float32, alignment: int = ALIGNMENT) -> np.ndarray:
    """An empty C-contiguous array whose buffer starts on a multiple of ``alignment``
    bytes (a view of a larger buffer, which it keeps alive)."""
    dtype = np.dtype(dtype)
    shape = tuple(int(n) for n in np.atleast_1d(shape))
    nbytes = int(np.prod(shape)) * dtype.itemsize
    buffer = np.empty(nbytes + int(alignment), dtype=np.uint8)
    start = (-buffer.ctypes.data) % int(alignment)
    return buffer[start : start + nbytes].view(dtype).reshape(shape)


def aligned(array, dtype=None, alignment: int = ALIGNMENT) -> np.ndarray:
    """The array (of an image) on a C-contiguous buffer aligned on ``alignment`` bytes, in
    ``dtype`` if given: the array itself when it is already one, else a copy."""
    a = np.asarray(array.data if isinstance(array, VoxelImage) else array)
    dtype = a.dtype if dtype is None else np.dtype(dtype)
    if a.dtype == dtype and is_aligned(a, alignment):
        return a
    out = aligned_empty(a.shape, dtype, alignment)
    out[...] = a
    return out


def _import(name: str, extra: str):
    try:
        return __import__(name, fromlist=["_"])
    except ImportError:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"this needs the optional package {name!r}: pip install {extra}"
        ) from None


def to_jax(obj, *, dtype=None, device=None):
    """A JAX array of an image (or of an array: a state, an ensemble) sharing its buffer
    on the CPU (DLPack): the image is given an aligned buffer of ``dtype`` first if its
    array is not one (one copy, after which the image and the JAX array share it);
    ``device`` a JAX device to put it on (a copy to a GPU)."""
    jnp = _import("jax.numpy", "jax")
    a = aligned(obj, dtype)
    if isinstance(obj, VoxelImage) and a is not obj.data:
        obj.data = a  # the same values on the aligned buffer, shared from now on
    out = jnp.from_dlpack(a)
    return out if device is None else __import__("jax").device_put(out, device)


def to_torch(obj, *, dtype=None):
    """A PyTorch tensor of an image (or of an array) sharing its buffer (DLPack); a copy
    in ``dtype`` if given."""
    torch = _import("torch", "torch")
    a = np.asarray(obj.data if isinstance(obj, VoxelImage) else obj)
    if dtype is not None and a.dtype != np.dtype(dtype):
        a = aligned(a, dtype)
    return torch.from_dlpack(np.ascontiguousarray(a))


def from_dlpack(
    x,
    like: VoxelImage | None = None,
    *,
    channel: int | None = None,
    spacing=None,
    origin=None,
    periodic=None,
    name: str = "field",
) -> VoxelImage:
    """A voxel image of an array of another library on the CPU (DLPack, without a copy:
    JAX, PyTorch, CuPy on the host, NumPy): a field of three dimensions (or two: a planar
    image, one voxel thick), or the ``channel`` of a state (channel, voxels...); on the
    grid of ``like`` (the image it came from), or of ``spacing``, ``origin`` and
    ``periodic`` (by default unit voxels at the origin, periodic). An array that NumPy
    cannot read in place (on a GPU) is copied to the host."""
    try:
        a = np.from_dlpack(x)
    except (TypeError, BufferError, RuntimeError, ValueError):
        a = np.asarray(x)
    if channel is not None:
        a = a[int(channel)]
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError(
            f"from_dlpack: a field of 3 (or 2) dimensions, got shape {a.shape}; give "
            "channel= for a state of channels"
        )
    ref = like
    return VoxelImage(
        a,
        spacing=spacing if spacing is not None else (ref.spacing if ref else (1.0, 1.0, 1.0)),
        origin=origin if origin is not None else (ref.origin if ref else (0.0, 0.0, 0.0)),
        periodic=periodic if periodic is not None else (ref.periodic if ref else True),
        name=name,
        metadata=dict(ref.metadata) if ref else {},
    )


def _gradient_norm(phi: np.ndarray, spacing, periodic: bool) -> np.ndarray:
    """The norm of the gradient by central differences: of the fourth order on periodic
    axes of five voxels or more (the slope of a curved distance within ``(h / rho)^4``),
    of the second order otherwise (one-sided at the faces of a window); the axes of one
    voxel left out."""
    total = np.zeros(phi.shape)
    for axis, h in enumerate(spacing):
        n = phi.shape[axis]
        if n < 3:
            continue
        if periodic and n >= 5:

            def at(k, a=axis):
                return np.roll(phi, -k, a)

            g = (8.0 * (at(1) - at(-1)) - (at(2) - at(-2))) / (12.0 * h)
        elif periodic:
            g = (np.roll(phi, -1, axis) - np.roll(phi, 1, axis)) / (2.0 * h)
        else:
            g = np.gradient(phi, h, axis=axis)
        total += g * g
    return np.sqrt(total)


#: Points of the implicit functions evaluated at once (memory)
_SLAB_POINTS = 1 << 20


def _implicit_on_grid(micro, phase: int, grid, reach: float) -> np.ndarray | None:
    """The point-wise implicit function of a phase at the centres of the voxels of a grid,
    exact within ``reach`` of the interface (by slabs normal to z); None for the
    structures whose functions are those of grids only."""
    nx, ny, nz = grid.shape
    out = np.empty(grid.shape)
    per = max(1, _SLAB_POINTS // max(nx * ny, 1))
    for k0 in range(0, nz, per):
        k1 = min(nz, k0 + per)
        values = micro.implicit_at(phase, grid.center_points(k0, k1), reach)
        if values is None:
            return None
        out[:, :, k0:k1] = np.asarray(values, dtype=float).reshape(nx, ny, k1 - k0)
    return out


def signed_distance(
    obj, phase: int = 1, *, shape=None, voxel_size=None, reach: float | None = None
) -> VoxelImage:
    """An estimate of the signed distance to the boundary of a phase at the centres of the
    voxels (negative inside): from the point-wise implicit function of the phase, exact
    within ``reach`` of the interface (12 voxels by default) for the structures that know
    their distances, over the norm of its gradient — exact for the signed distances, and
    near the interface for the smooth functions (TPMS, Gaussian fields) whose gradient
    varies little across it; the structures without point-wise functions from their
    functions on the grid, their clipped values (the edges of their bands) taken as far
    from the interface; a phase image from its indicator smoothed over a voxel (Gaussian),
    then the same."""
    from morphogen.models.base import Microstructure

    far = None
    if isinstance(obj, Microstructure):
        if shape is None and voxel_size is None:
            shape = 64
        grid = obj.grid(shape, voxel_size=voxel_size)
        reach = 12.0 * float(grid.spacing.max()) if reach is None else float(reach)
        phi = _implicit_on_grid(obj, phase, grid, reach)
        if phi is not None:  # beyond the reach: far from the interface
            far = np.abs(phi) >= (1.0 - 1e-9) * reach
        else:  # the function of the grid, clipped to its band
            phi = np.asarray(obj.phase_level_set(phase, grid=grid).data, dtype=float)
            top = float(np.abs(phi).max()) if phi.size else 0.0
            far = np.abs(phi) >= (1.0 - 1e-6) * top if top > 0 else None
        image = VoxelImage.from_grid(phi, grid, name="level_set", metadata=obj.image_metadata())
    elif isinstance(obj, VoxelImage):
        image = obj
        data = np.asarray(obj.data)
        if data.dtype.kind in "iub":
            mode = "wrap" if obj.periodic else "nearest"
            phi = ndimage.gaussian_filter(0.5 - (data == phase).astype(float), 1.0, mode=mode)
        elif obj.name == "fraction":
            phi = 0.5 - data.astype(float)
        else:
            phi = data.astype(float)
    else:
        raise TypeError("signed_distance: a microstructure or a voxel image")
    norm = _gradient_norm(phi, image.spacing, image.periodic)
    interface = np.abs(phi) < 1.5 * float(max(image.spacing))
    if interface.any() and abs(float(np.median(norm[interface])) - 1.0) < 0.01:
        distance = phi  # a distance already: its slope one but at its ridges (medial axes)
    else:
        floor = 1e-6 * float(np.max(norm)) if norm.size and np.max(norm) > 0 else 1.0
        distance = phi / np.maximum(norm, floor)
    if far is not None:  # the edges of the band of the grid: far from the interface
        distance = np.where(far, np.sign(phi) * np.inf, distance)
    return image.with_data(distance, name="signed_distance")


def phase_field(
    obj,
    phase: int = 1,
    *,
    shape=None,
    voxel_size=None,
    epsilon: float | None = None,
    inside: float = 1.0,
    outside: float = -1.0,
    dtype=np.float32,
) -> VoxelImage:
    """The diffuse interface of a phase of a realization (or of an image):
    ``u = m + a tanh(-d / (sqrt(2) epsilon))`` at the centres of the voxels, ``d`` the
    signed distance to its boundary (:func:`signed_distance`), ``m`` and ``a`` such that
    ``u`` is ``inside`` deep in the phase and ``outside`` far from it; the profile of the
    plane interface at equilibrium of Allen–Cahn's equation for ``epsilon = sqrt(nu)`` and
    of Cahn–Hilliard's for ``epsilon = sqrt(gamma)`` (see the module). ``epsilon`` by
    default one voxel and a half, so that the interface spans four to five voxels and the
    spectral solvers resolve it. On an aligned buffer of ``dtype`` (single precision by
    default, JAX's), shared with JAX by :func:`to_jax`."""
    if epsilon is not None and not float(epsilon) > 0:
        raise ValueError("epsilon: a positive length (the width of the interface)")
    reach = None if epsilon is None else 8.0 * float(epsilon)  # tanh(8 / sqrt 2): 1 - 2e-5
    distance = signed_distance(obj, phase, shape=shape, voxel_size=voxel_size, reach=reach)
    h = float(max(distance.spacing))
    eps = 1.5 * h if epsilon is None else float(epsilon)
    middle, half = 0.5 * (inside + outside), 0.5 * (inside - outside)
    out = aligned_empty(distance.shape, dtype)
    out[...] = middle + half * np.tanh(-np.asarray(distance.data) / (math.sqrt(2.0) * eps))
    image = distance.with_data(out, name="phase_field")
    image.metadata.update({"phase": int(phase), "epsilon": eps, "inside": inside,
                           "outside": outside})  # fmt: skip
    return image


def spectral_grid(image: VoxelImage) -> dict:
    """The arguments of a spectral solver of a periodic box for an image (those of
    Exponax's steppers): ``num_spatial_dims`` (3, or 2 for a planar image one voxel
    thick), ``domain_extent`` (the length of the box) and ``num_points`` (per axis). The
    image must be periodic, of cubic voxels and as many along each axis (a cube, or a
    square in 2D): such a solver has one length and one number of points."""
    if not image.periodic:
        raise ValueError("spectral_grid: the image of a periodic box (periodic: true)")
    shape = [n for n in image.shape if n > 1] if image.shape[2] == 1 else list(image.shape)
    spacing = image.spacing[: len(shape)] if image.shape[2] == 1 else image.spacing
    if len(set(shape)) != 1 or not np.allclose(spacing, spacing[0], rtol=1e-9):
        raise ValueError(
            "spectral_grid: as many voxels along each axis, all cubic (a cube; a square "
            f"for a planar image), got {image.shape} voxels of {image.spacing}"
        )
    n = int(shape[0])
    return {
        "num_spatial_dims": len(shape),
        "domain_extent": float(n * spacing[0]),
        "num_points": n,
    }


def state(image: VoxelImage) -> np.ndarray:
    """The array of an image in the layout of the states of spectral solvers, ``(1,
    *voxels)``: one channel and the voxels (those of the plane of a planar image), a view
    of the image (no copy)."""
    grid = spectral_grid(image)
    data = np.asarray(image.data)
    return (data[:, :, 0] if grid["num_spatial_dims"] == 2 else data)[None]


def realizations(
    model,
    domain,
    seeds,
    *,
    shape,
    phase: int = 1,
    epsilon: float | None = None,
    dtype=np.float32,
) -> np.ndarray:
    """The phase fields (:func:`phase_field`) of the realizations of a model in a domain,
    one per seed of ``seeds`` (``derive_seed`` of a run: a sequence of integers), on
    ``shape`` voxels along the longest edge: an aligned array ``(batch, 1, *voxels)`` (a
    planar domain, one voxel thick: the voxels of the plane), the initial conditions of an
    ensemble rollout (``jax.vmap`` of a stepper over the first axis)."""
    seeds = list(seeds)
    if not seeds:
        raise ValueError("realizations: at least one seed")
    out = None
    for b, seed in enumerate(seeds):
        u = phase_field(model.generate(domain, seed), phase, shape=shape, epsilon=epsilon,
                        dtype=dtype)  # fmt: skip
        s = state(u)
        if out is None:
            out = aligned_empty((len(seeds), *s.shape), dtype)
        out[b] = s
    return out
