"""Affine anisotropy of morphological models.

A model given an ``anisotropy`` generates the image by the linear map ``x = A y`` of a
realization of its isotropic version: ``A = g^T diag(s) g`` stretches by ``s`` along
principal axes, the rows of the Bunge rotation ``g`` of the angles ``euler`` (degrees, the
convention of :func:`morphogen.orientations.euler_to_matrix`). The stretches are normalized
to a unit product, so that ``A`` preserves volumes: volume fractions, numbers of objects
per unit volume, object volumes, integral ranges, Euler densities and other topological
numbers keep their values and meaning, and only shapes change. A sphere of radius ``r``
becomes an ellipsoid of semi-axes ``s_i r`` (and of the same volume), a hard-core packing
of spheres a hard-core packing of aligned ellipsoids, a Voronoi or Laguerre tessellation
the tessellation computed in the stretched space and mapped back (cells elongated along
the axes of the largest stretches), an excursion of a Gaussian field the excursion of a
field of covariance ``rho(|A^-1 h|)``. The mean intercept length along a unit direction
``d`` becomes ``MIL_iso / |A^-1 d|``: its fabric tensor (:func:`morphogen.analysis.
fabric_tensor`) has the principal axes of ``A`` and principal MILs in the ratios of ``s``.

Configuration, the same for every model::

    anisotropy: {stretch: [1, 1, 2], euler: [0, 0, 0]}   # elongated twice as much along z
    anisotropy: [1, 1, 2]                                  # the same: 3 stretches, or 6
                                                           # numbers with the Euler angles
    anisotropy: 2                                          # an elongation 2 along z

Periodicity. The image of a periodic box by ``A`` is periodic in the same box only when
``A`` is diagonal in the axes of the box: principal axes along the box axes (Euler angles
multiples of 90 degrees, or turning about the axis of two equal stretches). The isotropic
model is then generated in the box of edges ``L_i / t_i`` (``t = diag A``), periodic like
the domain, and its voxel images on the grid of the same shape are those of the
anisotropic structure (an exact map of grids). Other principal axes need a non-periodic
domain (an error otherwise): the isotropic model is generated in the bounding box of the
preimage of the domain (numbers of cells or objects then refer to that box). Two-phase
structures are voxelized exactly, by point queries at the preimages of the voxel centres;
labels, several phases and level sets by sampling at those preimages a grid of that box of
spacing the voxel size over the largest stretch (nearest voxel; trilinear for level sets
and partial volume fractions: interfaces within half a voxel). Models that apply the
metric themselves (the Gaussian field) accept any rotation in periodic boxes.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import re

import numpy as np
from scipy import ndimage

from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.models.base import Microstructure, register_kind
from morphogen.orientations import euler_to_matrix

#: Relative size of the off-diagonal entries of ``A`` below which its principal axes are
#: taken along the box axes.
ALIGNED = 1e-12


@dataclasses.dataclass(frozen=True)
class Anisotropy:
    """Stretches ``stretch`` along the principal axes turned by the Bunge angles ``euler``
    (degrees); ``elongation``: the number it was given as (an elongation along z)."""

    stretch: tuple[float, float, float] = (1.0, 1.0, 1.0)
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    elongation: float | None = None

    def __post_init__(self) -> None:
        s = tuple(float(v) for v in np.ravel(self.stretch))
        e = tuple(float(v) for v in np.ravel(self.euler))
        if len(s) != 3 or not all(math.isfinite(v) and v > 0.0 for v in s):
            raise ValueError(f"anisotropy: stretch: three positive numbers, got {self.stretch}")
        if len(e) != 3 or not all(math.isfinite(v) for v in e):
            raise ValueError(f"anisotropy: euler: three Bunge angles (degrees), got {self.euler}")
        object.__setattr__(self, "stretch", s)
        object.__setattr__(self, "euler", e)

    @property
    def principal(self) -> np.ndarray:
        """The stretches normalized to a unit product (a map preserving volumes)."""
        s = np.asarray(self.stretch)
        return s / math.prod(self.stretch) ** (1.0 / 3.0)

    @property
    def axes(self) -> np.ndarray:
        """The principal axes (columns) in the axes of the sample."""
        return euler_to_matrix(self.euler)[0].T

    @property
    def matrix(self) -> np.ndarray:
        """``A = g^T diag(s) g``: the image of a point ``y`` of the isotropic structure."""
        g = euler_to_matrix(self.euler)[0]
        return g.T @ np.diag(self.principal) @ g

    @property
    def is_identity(self) -> bool:
        return bool(np.all(np.abs(self.principal - 1.0) <= 1e-12))

    def box_stretch(self) -> np.ndarray | None:
        """The stretches ``t`` along x, y and z when ``A`` is diagonal in the box axes, else
        None."""
        a = self.matrix
        if np.abs(a - np.diag(np.diag(a))).max() > ALIGNED * np.abs(a).max():
            return None
        if not np.any(self.euler):
            return self.principal  # the exact diagonal
        return np.diag(a).copy()

    def to_config(self):
        if self.elongation is not None:
            return self.elongation
        out: dict = {"stretch": list(self.stretch)}
        if any(self.euler):
            out["euler"] = list(self.euler)
        return out


def anisotropy_from_config(value, name: str = "anisotropy") -> Anisotropy | None:
    """An :class:`Anisotropy` from a configuration value: None (isotropic), a mapping with
    ``stretch`` and ``euler``, a list of 3 stretches or of 6 numbers (stretches, then
    angles), or a number (an elongation along z)."""
    if value is None or isinstance(value, Anisotropy):
        return value
    try:
        if isinstance(value, bool):
            raise TypeError("expected numbers")
        if isinstance(value, (int, float)):
            return Anisotropy(stretch=(1.0, 1.0, float(value)), elongation=float(value))
        if isinstance(value, dict):
            unknown = sorted(set(value) - {"stretch", "euler"})
            if unknown:
                raise ValueError(f"unknown keys {unknown} (stretch, euler)")
            return Anisotropy(
                stretch=value.get("stretch", (1.0, 1.0, 1.0)),
                euler=value.get("euler", (0.0, 0.0, 0.0)),
            )
        numbers = [float(v) for v in value]
        if len(numbers) not in (3, 6):
            raise ValueError("3 stretches, or 3 stretches and 3 Euler angles")
        return Anisotropy(stretch=numbers[:3], euler=numbers[3:] or (0.0, 0.0, 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name}: expected a mapping {{stretch: [sx, sy, sz], euler: [phi1, Phi, phi2]}}, "
            f"3 or 6 numbers, or an elongation along z, got {value!r} ({exc})"
        ) from None


#: The documentation of the ``anisotropy`` field of the models that accept it (declared
#: ``param(None, doc=ANISOTROPY, kind="anisotropy", kw_only=True)``; None: isotropic).
ANISOTROPY = (
    "Affine anisotropy: stretches along principal axes turned by Bunge angles, "
    "{stretch: [sx, sy, sz], euler: [phi1, Phi, phi2]} (degrees), or 3 or 6 numbers, or an "
    "elongation along z; normalized to preserve volumes; periodic boxes: principal axes "
    "along the box axes (none: isotropic)"
)


#: Theoretical characteristics changed by a map preserving volumes (lengths, areas and
#: curvatures); volumes, fractions, numbers per unit volume and topological numbers stay.
_NOT_INVARIANT = re.compile(r"surface|curvature|radius|length|chord|thickness|diameter|size")


def invariant_theory(values: dict[str, float]) -> dict[str, float]:
    """The characteristics of :meth:`Model.theory` that a volume-preserving map keeps."""
    return {k: v for k, v in values.items() if not _NOT_INVARIANT.search(k)}


def _corners(domain: Domain) -> np.ndarray:
    lo, hi = domain.lower, domain.upper
    return np.array([np.where(c, hi, lo) for c in itertools.product((0, 1), repeat=3)])


def _preimage_box(domain: Domain, t: np.ndarray) -> Domain:
    """The box mapped onto ``domain`` by ``diag(t)``, periodic like it."""
    return Domain(
        size=tuple(domain.lengths / t),
        origin=tuple(domain.lower / t),
        periodic=domain.periodic,
        unit=domain.unit,
    )


def reference_domain(anisotropy: Anisotropy, domain: Domain, name: str = "model") -> Domain:
    """The domain where the isotropic model is generated (see the module)."""
    t = anisotropy.box_stretch()
    if t is not None:
        return _preimage_box(domain, t)
    if domain.periodic:
        raise ValueError(
            f"{name}: anisotropy: principal axes other than the box axes break the "
            "periodicity of the box; use a non-periodic domain (periodic: false), or Euler "
            "angles that keep the principal axes along the box axes (multiples of 90 degrees)"
        )
    y = _corners(domain) @ np.linalg.inv(anisotropy.matrix).T
    lo, hi = y.min(axis=0), y.max(axis=0)
    return Domain(size=tuple(hi - lo), origin=tuple(lo), periodic=False, unit=domain.unit)


def anisotropic(cls):
    """Class decorator of a model with an ``anisotropy`` field: its realizations are those
    of its isotropic version, generated in the preimage of the domain and mapped
    (:class:`AffineStructure`), and its theoretical characteristics those the map keeps.
    Without anisotropy the model is unchanged."""
    generate, theory = cls.generate, cls.theory

    @functools.wraps(generate)
    def generate_mapped(self, domain, seed=None):
        a = self.anisotropy
        if a is None or a.is_identity:
            return generate(self, domain, seed)
        self._check_domain(domain)
        iso = dataclasses.replace(self, anisotropy=None)
        ref = reference_domain(a, domain, self.name)
        try:
            inner = generate(iso, ref, seed)
        except ValueError as exc:
            lengths = ", ".join(f"{v:.6g}" for v in ref.lengths)
            raise ValueError(
                f"{exc} (in the box of edges {lengths}, the preimage of the domain by the "
                "anisotropy, where the isotropic model is generated)"
            ) from None
        return AffineStructure(
            domain, inner, a, model=self, seed=inner.seed, metadata=dict(inner.metadata)
        )

    @functools.wraps(theory)
    def theory_mapped(self, domain=None):
        a = self.anisotropy
        if a is None or a.is_identity:
            return theory(self, domain)
        iso = dataclasses.replace(self, anisotropy=None)
        if domain is None or (domain.periodic and a.box_stretch() is None):
            ref = None  # (no realization in such a domain)
        else:
            ref = reference_domain(a, domain, self.name)
        return invariant_theory(theory(iso, ref))

    cls.generate, cls.theory = generate_mapped, theory_mapped
    return cls


@register_kind
class AffineStructure(Microstructure):
    """The image by ``x = A y`` (:class:`Anisotropy`) of a realization ``inner`` of the
    isotropic model, generated in the preimage of ``domain``."""

    kind = "affine"

    #: Attributes of the inner realization that do not depend on positions (data of its
    #: cells and grains), forwarded.
    FORWARDED = ("orientations", "cell_phases")

    def __init__(self, domain: Domain, inner: Microstructure, anisotropy: Anisotropy, **kwargs):
        super().__init__(domain, **kwargs)
        self.inner = inner
        self.anisotropy = anisotropy
        self.matrix = anisotropy.matrix
        self.box_stretch = anisotropy.box_stretch()

    def __getattr__(self, name):
        if name in AffineStructure.FORWARDED and "inner" in self.__dict__:
            return getattr(self.inner, name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    @property
    def has_labels(self) -> bool:
        return self.inner.has_labels

    def summary(self) -> dict:
        return self.inner.summary()

    # -- discretization ------------------------------------------------------------
    def _image(self, method: str, grid: Grid, order: int, **kwargs) -> VoxelImage:
        """``inner.<method>`` on the preimage of ``grid``, as an image of ``grid``."""
        if self.box_stretch is not None:
            ref = Grid(_preimage_box(grid.domain, self.box_stretch), grid.shape)
            image = getattr(self.inner, method)(grid=ref, **kwargs)
            data = image.data
        else:
            box = self.inner.domain
            h = float(np.min(grid.spacing)) / float(self.anisotropy.principal.max())
            ref = Grid.for_domain(box, voxel_size=np.minimum(h, box.lengths))
            image = getattr(self.inner, method)(grid=ref, **kwargs)
            inverse = np.linalg.inv(self.matrix)
            nx, ny, nz = grid.shape
            data = np.empty(grid.shape, dtype=image.data.dtype)
            k_step = max(1, 2**21 // (nx * ny))
            for k0 in range(0, nz, k_step):
                k1 = min(nz, k0 + k_step)
                y = grid.center_points(k0, k1) @ inverse.T
                index = (y - ref.origin) / ref.spacing - 0.5
                values = ndimage.map_coordinates(image.data, index.T, order=order, mode="nearest")
                data[:, :, k0:k1] = values.reshape(nx, ny, k1 - k0)
        return VoxelImage.from_grid(data, grid, name=image.name, metadata=self.image_metadata())

    def _exact(self, grid: Grid, supersampling: int) -> VoxelImage:
        """Phases 0 and 1 by point queries at the preimages of the voxel centres (and of
        ``supersampling**3`` sub-voxel points)."""
        ss = int(supersampling)
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        fine = grid if ss == 1 else Grid(grid.domain, tuple(n * ss for n in grid.shape))
        inverse = np.linalg.inv(self.matrix)
        nx, ny, nz = fine.shape
        inside = np.empty(fine.shape, dtype=bool)
        k_step = max(1, 2**20 // (nx * ny))
        for k0 in range(0, nz, k_step):
            k1 = min(nz, k0 + k_step)
            y = fine.center_points(k0, k1) @ inverse.T
            inside[:, :, k0:k1] = self.inner.contains(y).reshape(nx, ny, k1 - k0)
        if ss == 1:
            data, name = inside.astype(np.uint8), "phase"
        else:
            n = grid.shape
            data = inside.reshape(n[0], ss, n[1], ss, n[2], ss).mean(
                axis=(1, 3, 5), dtype=np.float32
            )
            name = "fraction"
        return VoxelImage.from_grid(data, grid, name=name, metadata=self.image_metadata())

    def voxelize(
        self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling: int = 1
    ) -> VoxelImage:
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        two_phases = self.model is not None and set(self.model.active_phases()) <= {0, 1}
        if self.box_stretch is None and two_phases and not self.inner.has_labels:
            try:
                return self._exact(grid, supersampling)
            except NotImplementedError:
                pass
        order = 0 if int(supersampling) == 1 else 1
        return self._image("voxelize", grid, order, supersampling=supersampling)

    def labels(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        return self._image("labels", self.grid(shape, voxel_size=voxel_size, grid=grid), 0)

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """The level set of the inner realization at the preimages (its zero set is the
        boundary; its values are distances in the isotropic space)."""
        return self._image("level_set", self.grid(shape, voxel_size=voxel_size, grid=grid), 1)

    def implicit_phases(self) -> list[int]:
        return self.inner.implicit_phases()

    def phase_level_set(
        self, phase: int, shape=None, *, voxel_size=None, grid: Grid | None = None
    ) -> VoxelImage:
        """The implicit function of a phase of the inner realization at the preimages."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        return self._image("phase_level_set", grid, 1, phase=int(phase))

    def contains(self, points) -> np.ndarray:
        y = np.asarray(points, dtype=float).reshape(-1, 3) @ np.linalg.inv(self.matrix).T
        return self.inner.contains(y)

    @property
    def lipschitz(self) -> float:
        """That of the inner realization, times the largest stretch of the inverse map."""
        stretch = float(np.linalg.norm(np.linalg.inv(self.matrix), 2))
        return float(getattr(self.inner, "lipschitz", 1.0)) * stretch

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """The level set of the inner realization at the preimages of the points (its zero
        set is the boundary; distances of the isotropic space, within the stretch ratios of
        those of the mapped one)."""
        y = np.asarray(points, dtype=float).reshape(-1, 3) @ np.linalg.inv(self.matrix).T
        return self.inner.level_set_at(y, reach)

    def phase_level_sets_at(self, points, reach: float = np.inf) -> dict[int, np.ndarray]:
        """The functions of the phases of the inner realization at the preimages of the
        points (:meth:`phase_level_set` at any point)."""
        y = np.asarray(points, dtype=float).reshape(-1, 3) @ np.linalg.inv(self.matrix).T
        return self.inner.phase_level_sets_at(y, reach)

    # -- serialization -------------------------------------------------------------
    def _geometry(self) -> dict:
        return {"anisotropy": self.anisotropy.to_config(), "inner": self.inner.to_dict()}

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        inner = Microstructure.from_dict(geometry["inner"])
        return cls(domain, inner, anisotropy_from_config(geometry["anisotropy"]), **kwargs)
