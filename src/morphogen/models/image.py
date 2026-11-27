"""The microstructure of an image: a volume (a tomography, a stack of sections) or a section,
segmented into phases and described by implicit functions at any point.

The phases of a grey-level image are the classes of its levels split by thresholds (given,
or Otsu's: :mod:`morphogen.morphology.segmentation`), after an optional Gaussian smoothing;
an image of integers already holds its phases. Each phase ``k`` has a function of the
points of space, negative inside it, from the image interpolated trilinearly between the
centres of its voxels:

* *grey levels* ``g`` and thresholds ``t_1 < ... < t_{n-1}``: the phase ``k`` is ``t_k <= g <
  t_{k+1}`` and its function ``max(t_k - g, g - t_{k+1}) / s``, ``s`` the median of the
  gradient of the levels at the voxels next to a threshold (the function is then about the
  distance to the interface near it). The interfaces are the iso-surfaces of the
  interpolated image at the thresholds: those that marching cubes draws, within a fraction
  of a voxel;
* *phases*: the indicator ``χ_k`` of each phase (smoothed by ``sigma`` voxels if given),
  interpolated, and the function ``(max_{l != k} χ_l - χ_k) h / 2`` (``h`` the smallest
  spacing): negative where the phase ``k`` has the largest share, zero half-way between
  the centres of two voxels of different phases.

The functions of the phases partition space (a point lies in the phase of the smallest
one), and their slopes are bounded by their largest differences between neighbouring
voxels, so that the exact meshes, the probes of :mod:`morphogen.morphology`, the voxel
images at any resolution and the exports take the image as they take a model. A
realization is a window of the image of the size of the domain, at the first voxels of the
image (``window: origin``) or drawn at random in it for each seed (``window: random``): the
windows of a large image are realizations of the medium it shows, for the size of its
representative volume (as Kanit et al. 2003 with windows of one image). A periodic domain
takes the whole image, its faces wrapped.
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy import ndimage

from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.models.base import Microstructure, Model, param, register, register_kind
from morphogen.seeding import SeedLike, make_rng

WINDOWS = ("origin", "random")
#: The image that comes with morphoGen (see :func:`example_image`)
EXAMPLE = "example_image.npz"


def example_image(name: str = EXAMPLE):
    """The path of an image that comes with morphoGen: ``example_image.npz``, a volume of 48³
    voxels of 0.25 (a box of 12) of grey levels as a tomography of a particulate composite
    would give them: hard spheres of radius 1.2 at a fraction of 0.2 (levels 170) in a
    matrix (levels 60), blurred by the partial volumes and the spread of the beam (a
    Gaussian of 0.8 voxel) and noisy (a Gaussian noise of 12 levels), in bytes; written by
    :func:`write_example_image`."""
    from importlib import resources

    path = str(resources.files("morphogen") / "templates" / name)
    if not os.path.isfile(path):
        raise ValueError(f"example_image: no image {name!r} comes with morphoGen")
    return path


def synthetic_tomography(seed: int = 7) -> VoxelImage:
    """The volume of :func:`example_image`, made again from its seed."""
    from morphogen.models.hardcore import HardcoreSpheres

    micro = HardcoreSpheres(radius=1.2, volume_fraction=0.2).generate(
        Domain.cube(12.0, periodic=True), seed=seed
    )
    fraction = micro.voxelize(48, supersampling=4).data.astype(float)
    grey = 60.0 + 110.0 * fraction
    grey = ndimage.gaussian_filter(grey, 0.8, mode="wrap")
    grey += np.random.default_rng(seed).normal(0.0, 12.0, grey.shape)
    data = np.clip(np.rint(grey), 0, 255).astype(np.uint8)
    return VoxelImage(
        data,
        spacing=(0.25, 0.25, 0.25),
        periodic=False,
        name="grey",
        metadata={
            "description": "grey levels of a synthetic tomography of hard spheres (radius "
            "1.2, fraction 0.2) in a matrix",
            "seed": int(seed),
        },
    )


def write_example_image(path) -> str:
    """Write the image of :func:`example_image` (a compressed NumPy archive)."""
    from morphogen.io.native import write_npz

    return str(write_npz(path, synthetic_tomography()))


@functools.lru_cache(maxsize=8)
def _read(path: str, mtime: float, array: str | None) -> VoxelImage:
    from morphogen.io import load_image

    return load_image(path, array)


@functools.lru_cache(maxsize=8)
def _grey(path: str, mtime: float, array: str | None, classes: int, thresholds, sigma: float):
    """The smoothed grey levels of an image and its thresholds (Otsu's if none are given)."""
    from morphogen.morphology.segmentation import multi_otsu, smooth

    image = _read(path, mtime, array)
    grey = smooth(image, sigma) if sigma > 0.0 else image.data.astype(float)
    if thresholds is None:
        thresholds = tuple(float(t) for t in multi_otsu(grey, classes))
    return grey.astype(np.float32), tuple(thresholds)


@register
@dataclass(frozen=True)
class ImageModel(Model):
    """The microstructure of an image (a volume, a stack of sections, or a section),
    segmented into its phases and described by implicit functions (see the module): a
    window of the size of the domain, at the origin of the image or drawn at random."""

    name = "image"
    phases: ClassVar[dict[int, str]] = {0: "phase 0", 1: "phase 1"}
    count_label = ""
    path_parameters: ClassVar[tuple[str, ...]] = ("file",)

    file: str = param(
        EXAMPLE,
        doc="The image: a file of voxels (.npz, .npy, .vti, .vtk, .mhd, .xdmf, .tif stacks; "
        "from the folder of the configuration file), or the image that comes with "
        "morphoGen (example_image.npz)",
        kind="str",
    )
    array: str | None = param(None, doc="The array of the file (default: the first)", kind="str")
    classes: int = param(
        2,
        doc="The number of phases (an image of integers from 0 to classes - 1 holds its "
        "phases; the others are grey levels, split into this number of phases)",
        kind="int",
    )
    thresholds: tuple[float, ...] | None = param(
        None,
        doc="The grey levels between the phases, increasing (default: Otsu's thresholds)",
        kind="floats",
    )
    sigma: float = param(
        0.0,
        doc="Gaussian smoothing of the grey levels (or of the indicators of the phases) "
        "before segmentation, in voxels",
    )
    spacing: tuple[float, ...] | float | None = param(
        None, doc="The size of the voxels (default: that of the file)", kind="vector",
        unit="length",
    )  # fmt: skip
    window: str = param(
        "origin",
        doc="origin (the window of the size of the domain at the first voxels of the image) "
        "or random (a window drawn in the image for each realization)",
        kind="str",
    )

    def _validate(self) -> None:
        if not self.file:
            raise ValueError("file: the image")
        if self.classes < 2:
            raise ValueError("classes: at least 2 phases")
        if self.thresholds is not None and len(self.thresholds) != self.classes - 1:
            raise ValueError("thresholds: one fewer than the classes")
        if self.thresholds is not None and list(self.thresholds) != sorted(self.thresholds):
            raise ValueError("thresholds: increasing")
        if self.sigma < 0.0:
            raise ValueError("sigma: non-negative")
        if self.window not in WINDOWS:
            raise ValueError(f"window: one of {WINDOWS}")
        if self.spacing is not None and np.any(np.asarray(self.spacing, dtype=float) <= 0):
            raise ValueError("spacing: positive sizes")

    @property
    def has_level_set(self) -> bool:
        return self.classes == 2

    def active_phases(self) -> dict[int, str]:
        return {k: f"phase {k}" for k in range(self.classes)}

    def path(self) -> str:
        """The file of the image: ``file``, or the image of its name that comes with
        morphoGen where there is no such file."""
        path = os.path.abspath(os.path.expanduser(self.file))
        if not os.path.isfile(path):
            try:
                path = example_image(os.path.basename(path))
            except ValueError:
                raise ValueError(f"image: no file {self.file!r}") from None
        return path

    def image(self) -> VoxelImage:
        """The image of the file, with the spacing of the model if it gives one."""
        path = self.path()
        img = _read(path, os.path.getmtime(path), self.array)
        if self.spacing is not None:
            h = np.broadcast_to(np.asarray(self.spacing, dtype=float), (3,))
            img = VoxelImage(img.data, spacing=tuple(h), origin=img.origin, name=img.name,
                             metadata=img.metadata)  # fmt: skip
        return img

    def _fields(self, img: VoxelImage):
        """``("grey", levels, thresholds)`` or ``("phases", phases)`` of the image."""
        if self._holds_phases(img):
            return ("phases", np.asarray(img.data).astype(np.uint8))
        path = self.path()
        grey, thresholds = _grey(path, os.path.getmtime(path), self.array, self.classes,
                                 self.thresholds, float(self.sigma))  # fmt: skip
        return ("grey", grey, thresholds)

    def _holds_phases(self, img: VoxelImage) -> bool:
        """Whether the image holds its phases: integers from 0 to ``classes - 1`` (and no
        thresholds given); other images are grey levels."""
        data = img.data
        if self.thresholds is not None or data.dtype.kind not in "iub":
            return False
        return int(data.min()) >= 0 and int(data.max()) < self.classes

    def _window(self, domain: Domain, img: VoxelImage, rng) -> np.ndarray:
        """The lower corner, in the image, of the window that the domain covers."""
        box = np.asarray(img.data.shape, dtype=float) * np.asarray(img.spacing, dtype=float)
        size = domain.lengths
        planar = np.asarray(img.data.shape) == 1
        fits = (size <= box * (1.0 + 1e-9)) | planar
        if not np.all(fits):
            raise ValueError(
                f"image: the domain ({size.tolist()}) is larger than the image ({box.tolist()})"
            )
        if domain.periodic and not np.allclose(size[~planar], box[~planar], rtol=1e-9):
            raise ValueError("image: a periodic domain takes the whole image (its box)")
        room = np.where(planar, 0.0, np.maximum(box - size, 0.0))
        if self.window == "random":
            return np.asarray(img.origin) + rng.random(3) * room
        return np.asarray(img.origin, dtype=float)

    def generate(self, domain: Domain, seed: SeedLike = None) -> ImageStructure:
        rng, seed_info = make_rng(seed)
        img = self.image()
        corner = self._window(domain, img, rng)
        return ImageStructure(
            domain, img, self._fields(img), corner, model=self, seed=seed_info,
            sigma=float(self.sigma) if self._holds_phases(img) else 0.0,
        )  # fmt: skip

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        """The fractions of the phases in the whole image (no closed form: the measures of
        the image itself)."""
        img = self.image()
        fields = self._fields(img)
        if fields[0] == "phases":
            phases = fields[1]
        else:
            phases = np.searchsorted(np.asarray(fields[2]), fields[1], side="right")
        counts = np.bincount(phases.ravel(), minlength=self.classes) / phases.size
        out = {f"fraction_{k}": float(c) for k, c in enumerate(counts)}
        if self.classes == 2:
            out["volume_fraction"] = float(counts[1])
        return out


@register_kind
class ImageStructure(Microstructure):
    """The phases of an image in a window, by their implicit functions (see the module)."""

    kind = "image"

    def __init__(self, domain, image: VoxelImage, fields, corner, *, sigma: float = 0.0,
                 **kwargs) -> None:  # fmt: skip
        super().__init__(domain, **kwargs)
        self.image = image
        self.corner = np.asarray(corner, dtype=float)
        self.spacing = np.asarray(image.spacing, dtype=float)
        self.shape = np.asarray(image.data.shape)
        self.mode = fields[0]
        planar = self.shape == 1
        self._planar = planar
        if self.mode == "grey":
            grey, thresholds = fields[1], fields[2]
            self.thresholds = tuple(float(t) for t in thresholds)
            self.n_phases = len(self.thresholds) + 1
            self._arrays = [np.ascontiguousarray(grey, dtype=np.float64)]
            self._scale = self._grey_scale(grey)
            slope = self._slope(grey) / self._scale
        else:
            phases = fields[1]
            self.thresholds = ()
            self.n_phases = int(self.model.classes) if self.model is not None else int(
                phases.max()) + 1  # fmt: skip
            self._arrays = []
            for k in range(self.n_phases):
                chi = (phases == k).astype(np.float64)
                if sigma > 0.0:
                    sig = [sigma if n > 1 else 0.0 for n in chi.shape]
                    chi = ndimage.gaussian_filter(chi, sig, mode=self._edge_mode())
                self._arrays.append(chi)
            self._scale = 2.0 / float(self.spacing[~planar].min() if (~planar).any() else 1.0)
            slope = 2.0 * max(self._slope(a) for a in self._arrays) / self._scale
        self.lipschitz = max(float(slope), 1e-12)

    # -- the functions -------------------------------------------------------------------

    def _edge_mode(self) -> str:
        return "wrap" if self.domain.periodic else "nearest"

    def _slope(self, array: np.ndarray) -> float:
        """A bound of the slope of the trilinear interpolation of an array: the largest
        differences between neighbours along each axis over the spacing, combined."""
        parts = []
        for axis in range(3):
            if array.shape[axis] < 2:
                parts.append(0.0)
                continue
            d = np.abs(np.diff(array, axis=axis)).max() / self.spacing[axis]
            parts.append(float(d))
        return math.sqrt(sum(p * p for p in parts))

    def _grey_scale(self, grey: np.ndarray) -> float:
        """The median of the gradient of the levels at the voxels next to a threshold."""
        grads = np.gradient(grey, *[h if n > 1 else 1.0 for h, n in
                                    zip(self.spacing, grey.shape, strict=True)])  # fmt: skip
        norm = np.sqrt(sum(g * g for g, n in zip(grads, grey.shape, strict=True) if n > 1))
        near = np.zeros(grey.shape, dtype=bool)
        step = float(np.ptp(grey)) / 64.0 if np.ptp(grey) > 0 else 1.0
        for t in self.thresholds:
            near |= np.abs(grey - t) <= max(step, 0.5 * float(np.median(norm)) *
                                            float(self.spacing.min()))  # fmt: skip
        values = norm[near] if near.any() else norm.ravel()
        scale = float(np.median(values)) if values.size else 1.0
        return scale if scale > 0.0 else 1.0

    def _coordinates(self, points) -> np.ndarray:
        """The positions of points in the index space of the image (voxel centres at
        integers)."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if self.domain.periodic:
            pts = self.domain.wrap(pts)
        local = pts - self.domain.lower + self.corner
        idx = (local - np.asarray(self.image.origin)) / self.spacing - 0.5
        idx[:, self._planar] = 0.0
        return idx.T

    def _interpolate(self, array: np.ndarray, coords: np.ndarray) -> np.ndarray:
        mode = "grid-wrap" if self.domain.periodic else "nearest"
        return ndimage.map_coordinates(array, coords, order=1, mode=mode, prefilter=False)

    def phase_level_sets_at(self, points, reach: float = np.inf) -> dict[int, np.ndarray]:
        coords = self._coordinates(points)
        reach = float(reach)
        out: dict[int, np.ndarray] = {}
        if self.mode == "grey":
            g = self._interpolate(self._arrays[0], coords)
            edges = [-np.inf, *self.thresholds, np.inf]
            for k in range(self.n_phases):
                f = np.maximum(edges[k] - g, g - edges[k + 1]) / self._scale
                out[k] = np.clip(f, -reach, reach)
            return out
        chis = np.stack([self._interpolate(a, coords) for a in self._arrays])
        for k in range(self.n_phases):
            others = np.delete(chis, k, axis=0).max(axis=0)
            out[k] = np.clip((others - chis[k]) / self._scale, -reach, reach)
        return out

    def implicit_phases(self) -> list[int]:
        return list(range(self.n_phases))

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        if self.n_phases != 2:
            raise NotImplementedError("an image of more than two phases: phase_level_sets_at")
        return self.phase_level_sets_at(points, reach)[1]

    @property
    def phase_names(self) -> dict[int, str]:
        return {k: f"phase {k}" for k in range(self.n_phases)}

    def _centres(self, grid: Grid) -> np.ndarray:
        return grid.center_points()

    def phases_at(self, points) -> np.ndarray:
        """The phase of each point."""
        values = self.phase_level_sets_at(points, 1.0)
        stack = np.stack([values[k] for k in range(self.n_phases)])
        return np.argmin(stack, axis=0).astype(np.uint8)

    def contains(self, points) -> np.ndarray:
        return self.phases_at(points) == 1

    # -- discretizations -----------------------------------------------------------------

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        s = int(supersampling)
        if s <= 1:
            data = self.phases_at(self._centres(grid)).reshape(grid.shape)
            return VoxelImage.from_grid(data, grid, name="phase", metadata=self.image_metadata())
        acc = np.zeros(grid.shape, dtype=np.float64)
        base = self._centres(grid)
        h = grid.spacing
        offsets = (np.arange(s) + 0.5) / s - 0.5
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    pts = base + np.array([dx, dy, dz]) * h
                    acc += (self.phases_at(pts) == 1).reshape(grid.shape)
        data = (acc / s**3).astype(np.float32)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=self.image_metadata())

    def phase_level_set(self, phase: int, shape=None, *, voxel_size=None, grid=None):
        from morphogen.models.implicit import band_of

        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        phase = int(phase)
        if phase not in self.implicit_phases():
            raise ValueError(f"phase {phase}: the phases of this structure are "
                             f"{self.implicit_phases()}")  # fmt: skip
        band = band_of(grid)
        f = self.phase_level_sets_at(self._centres(grid), band)[phase]
        data = f.reshape(grid.shape).astype(np.float32)
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        if self.n_phases != 2:
            raise NotImplementedError("an image of more than two phases: phase_level_set")
        return self.phase_level_set(1, shape, voxel_size=voxel_size, grid=grid)

    # -- description ---------------------------------------------------------------------

    def summary(self) -> dict:
        out = {
            "image_shape": [int(n) for n in self.shape],
            "window_corner": [float(v) for v in self.corner],
        }
        if self.thresholds:
            out["thresholds"] = list(self.thresholds)
        return out

    def _geometry(self) -> dict:
        return {"corner": self.corner.tolist()}

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        model = kwargs.get("model")
        if model is None:
            raise ValueError("an image structure is read back with its model (its file)")
        img = model.image()
        sigma = float(model.sigma) if model._holds_phases(img) else 0.0
        return cls(domain, img, model._fields(img), geometry["corner"], sigma=sigma, **kwargs)


__all__ = ["WINDOWS", "ImageModel", "ImageStructure", "example_image", "synthetic_tomography",
           "write_example_image"]  # fmt: skip
