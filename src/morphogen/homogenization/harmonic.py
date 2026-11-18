"""Harmonic (frequency-domain) homogenization of periodic voxel images: effective
properties of phases whose laws are complex, in the quasi-static regime where the
wavelengths are much longer than the microstructure.

A harmonic excitation ``exp(i omega t)`` of linear dissipative phases has complex laws:

* **viscoelastic solids** (the correspondence principle): the complex moduli ``K*(omega)``
  and ``G*(omega)`` of the Prony series of each phase (:class:`~morphogen.homogenization.
  laws.Viscoelasticity`) give the effective complex stiffness ``C*(omega) = C' + i C''``:
  storage and loss moduli, loss factors (:func:`dynamic_stiffness`);
* **conductors and dielectrics** (quasi-static electromagnetism): the complex conductivity
  ``sigma* = sigma + i omega eps`` of each phase gives the effective conductivity and
  permittivity of the composite at each frequency, with the interfacial polarization of
  Maxwell, Wagner and Sillars where conducting inclusions charge their interfaces
  (:func:`complex_conductivity`);
* **optics** in the long-wavelength limit: the complex permittivities ``(n + i k)^2`` of
  the phases at a wavelength give the effective permittivity tensor of the composite, its
  refractive indices and its absorption (form birefringence, effective media of
  nanocomposites: :func:`effective_index`).

All are the same problem (:func:`harmonic_tensor`): the periodic problem of a field with
a complex symmetric law ``L = L' + i L''``, whose real form doubles the field,

    [ L'  -L''] [Z']   [ S']
    [-L'' -L' ] [Z''] = [-S''],

a real symmetric law that is not definite, solved by the coupled solver of
:mod:`morphogen.homogenization.coupled` (MINRES preconditioned in Fourier space) for the
unit real loadings. The laws are first turned by a phase ``exp(i theta)`` that centres
them about the real axis, so that their real parts are positive definite whenever the
laws of the phases lie in an open half of the complex plane (dielectrics and metals of
negative permittivity together, lossy phases with lossless ones). The effective tensors
are complex symmetric; :func:`~morphogen.homogenization.coupled.laminate` gives those of
layers exactly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from morphogen.homogenization.coupled import CoupledProblem, layout_sizes
from morphogen.homogenization.materials import (
    Material,
    bulk_shear,
    isotropic_matrix,
    material_table,
)
from morphogen.image import VoxelImage

log = logging.getLogger("morphogen")


def _half_plane(laws) -> float:
    """The angle ``theta`` whose phase ``exp(i theta)`` centres the eigenvalues of complex
    symmetric laws about the positive real axis."""
    values = np.concatenate([np.linalg.eigvals(law) for law in laws])
    values = values[np.abs(values) > 0]
    if not values.size:
        return 0.0
    args = np.sort(np.angle(values))
    # the widest gap of the arguments around the circle: the half plane opposite to it
    gaps = np.diff(np.concatenate([args, [args[0] + 2.0 * np.pi]]))
    k = int(np.argmax(gaps))
    if gaps[k] <= np.pi * (1.0 + 1e-12):
        raise ValueError(
            "harmonic laws: the eigenvalues of the laws of the phases must lie in an open "
            "half of the complex plane (a gain medium beside a lossy one, or opposite signs "
            "without losses, have no such problem)"
        )
    centre = args[k] + 0.5 * gaps[k] + np.pi  # the middle of the arc that holds them all
    return float(-centre)


def harmonic_tensor(image: VoxelImage, laws, mat, layout=("potential",), *,
                    tol: float = 1e-8, maxiter: int | None = None):  # fmt: skip
    """The effective complex symmetric tensor of a periodic voxel image whose materials
    have the complex symmetric laws ``laws[m]`` (the components of the ``layout`` of
    fields: 3 x 3 for a potential, 6 x 6 in the Voigt order for a displacement), ``mat``
    the material of each voxel (-1: none); see the module. Returns the tensor and the
    :class:`~morphogen.homogenization.coupled.CoupledProblem` solved (its iterations)."""
    layout = tuple(layout)
    _, nz = layout_sizes(layout)
    laws = np.asarray(laws, dtype=complex).reshape(-1, nz, nz)
    if not np.allclose(laws, np.swapaxes(laws, 1, 2), rtol=1e-10,
                       atol=1e-12 * max(np.abs(laws).max(initial=0.0), 1e-300)):  # fmt: skip
        raise ValueError("harmonic laws: complex symmetric matrices (not Hermitian ones)")
    mat = np.asarray(mat).ravel()
    used = [laws[m] for m in np.unique(mat[mat >= 0])]
    theta = _half_plane(used)
    turned = np.exp(1j * theta) * laws
    real = np.zeros((len(laws), 2 * nz, 2 * nz))
    real[:, :nz, :nz] = turned.real
    real[:, nz:, nz:] = -turned.real
    real[:, :nz, nz:] = real[:, nz:, :nz] = -turned.imag
    for m in np.unique(mat[mat >= 0]):
        if np.linalg.eigvalsh(turned[m].real).min() <= 0:
            raise ValueError(
                "harmonic laws: an anisotropic law whose real part is not definite once "
                "centred with the others (its principal axes of storage and loss differ too "
                "much)"
            )
    problem = CoupledProblem(image, real, layout + layout, mat, tol=tol, maxiter=maxiter,
                             pairs=True)  # fmt: skip
    out = np.zeros((nz, nz), dtype=complex)
    for j in range(nz):
        unit = np.zeros(2 * nz)
        unit[j] = 1.0
        flux = problem.solve(unit).flux
        out[:, j] = flux[:nz] - 1j * flux[nz:]
    out *= np.exp(-1j * theta)
    return 0.5 * (out + out.T), problem


@dataclass
class HarmonicResponse:
    """The effective complex tensors of a sample at angular ``frequencies`` (rad per unit
    time; wavelengths for :func:`effective_index`): ``tensors`` (frequencies x d x d,
    complex symmetric; 6 x 6 in the Voigt order 11 22 33 23 31 12 with engineering shears
    for a stiffness, 3 x 3 for a transport), the ``kind`` of the law (``"stiffness"``,
    ``"conductivity"`` or ``"permittivity"``) and the iterations of the solver at each
    frequency."""

    kind: str
    frequencies: np.ndarray
    tensors: np.ndarray
    iterations: np.ndarray

    @property
    def storage(self) -> np.ndarray:
        """The real parts of the tensors (the storage moduli of a stiffness)."""
        return self.tensors.real

    @property
    def loss(self) -> np.ndarray:
        """The imaginary parts of the tensors (the loss moduli of a stiffness)."""
        return self.tensors.imag

    # -- stiffness ---------------------------------------------------------------------

    def _need(self, kind: str) -> None:
        if self.kind != kind:
            raise ValueError(f"a {self.kind} has no such quantity (a {kind} has)")

    @property
    def bulk(self) -> np.ndarray:
        """The complex bulk moduli of the isotropic parts of the stiffnesses."""
        self._need("stiffness")
        c = self.tensors
        return (c[:, :3, :3].sum(axis=(1, 2))) / 9.0

    @property
    def shear(self) -> np.ndarray:
        """The complex shear moduli of the isotropic parts of the stiffnesses."""
        self._need("stiffness")
        c = self.tensors
        d = c[:, 0, 0] + c[:, 1, 1] + c[:, 2, 2]
        o = c[:, 0, 1] + c[:, 0, 2] + c[:, 1, 2]
        s = c[:, 3, 3] + c[:, 4, 4] + c[:, 5, 5]
        return (d - o + 3.0 * s) / 15.0

    def young(self, axis: int = 0) -> np.ndarray:
        """The complex Young's moduli along an axis: ``1 / S*_ii`` of the compliances."""
        self._need("stiffness")
        return np.array([1.0 / np.linalg.inv(c)[axis, axis] for c in self.tensors])

    @staticmethod
    def loss_factor(modulus) -> np.ndarray:
        """``tan delta``: the loss over the storage of complex moduli."""
        modulus = np.asarray(modulus)
        return modulus.imag / modulus.real

    # -- transport ---------------------------------------------------------------------

    @property
    def conductivity(self) -> np.ndarray:
        """The effective conductivity tensors (the real parts of ``sigma*``)."""
        self._need("conductivity")
        return self.tensors.real

    @property
    def permittivity(self) -> np.ndarray:
        """The effective permittivity tensors: ``Im(sigma*) / omega`` of a complex
        conductivity; the complex tensors themselves for :func:`effective_index`."""
        if self.kind == "permittivity":
            return self.tensors
        self._need("conductivity")
        return self.tensors.imag / self.frequencies[:, None, None]

    @property
    def index(self) -> np.ndarray:
        """The complex refractive indices ``n + i k`` along the three axes (frequencies x
        3): the square roots of the diagonal of the permittivity tensors (non-magnetic
        media; ``k >= 0``: absorption)."""
        self._need("permittivity")
        diag = np.einsum("nii->ni", self.tensors)
        n = np.sqrt(diag)
        return np.where(n.imag < 0, -n, n)

    @property
    def absorption(self) -> np.ndarray:
        """The absorption coefficients ``4 pi k / lambda`` along the three axes, for the
        wavelengths of :func:`effective_index` (the inverse of their unit)."""
        return 4.0 * np.pi * self.index.imag / self.frequencies[:, None]

    def to_dict(self) -> dict:
        """The response as plain arrays (for JSON) and the scalar :meth:`entries`: the
        ``frequencies`` (``wavelengths`` of a permittivity), the ``storage`` and ``loss``
        parts of the tensors, the ``iterations``; of a conductivity, its ``conductivity``
        and ``permittivity`` tensors; of a permittivity, the ``index``, ``extinction`` and
        ``absorption`` along the axes."""
        name = "wavelengths" if self.kind == "permittivity" else "frequencies"
        out = {"kind": self.kind, name: self.frequencies, "storage": self.storage,
               "loss": self.loss, "iterations": self.iterations}  # fmt: skip
        if self.kind == "conductivity":
            out.update(conductivity=self.conductivity, permittivity=self.permittivity)
        elif self.kind == "permittivity":
            out.update(index=self.index.real, extinction=self.index.imag,
                       absorption=self.absorption)  # fmt: skip
        out.update(self.entries())
        return out

    def entries(self) -> dict[str, float]:
        """Scalar entries for the tables of results, the frequencies numbered from 1:
        of a stiffness ``bulk_storage_1``, ``bulk_loss_1``, ``shear_storage_1``,
        ``shear_loss_1``, ``shear_tan_delta_1``, ``young_storage_1``, ``young_loss_1``
        (along x); of a conductivity ``conductivity_1``, ``permittivity_1`` (the means of
        their diagonals); of a permittivity ``index_1``, ``extinction_1`` (the means of
        ``n`` and ``k``)."""
        out: dict[str, float] = {}
        for i in range(len(self.frequencies)):
            k = i + 1
            if self.kind == "stiffness":
                for name, value in (("bulk", self.bulk[i]), ("shear", self.shear[i]),
                                    ("young", self.young()[i])):  # fmt: skip
                    out[f"{name}_storage_{k}"] = float(value.real)
                    out[f"{name}_loss_{k}"] = float(value.imag)
                out[f"shear_tan_delta_{k}"] = float(self.shear[i].imag / self.shear[i].real)
            elif self.kind == "conductivity":
                out[f"conductivity_{k}"] = float(np.trace(self.conductivity[i]) / 3.0)
                out[f"permittivity_{k}"] = float(np.trace(self.permittivity[i]) / 3.0)
            else:
                out[f"index_{k}"] = float(self.index[i].real.mean())
                out[f"extinction_{k}"] = float(self.index[i].imag.mean())
        return out


#: The kind of the tensors of each harmonic property
KINDS = {
    "dynamic_stiffness": "stiffness",
    "complex_conductivity": "conductivity",
    "refractive_index": "permittivity",
}


def entry_names(prop: str, count: int) -> tuple[str, ...]:
    """The names of the scalar entries (:meth:`HarmonicResponse.entries`) of a harmonic
    property (``dynamic_stiffness``, ``complex_conductivity``, ``refractive_index``) at
    ``count`` frequencies or wavelengths, numbered from 1: the columns of the tables of
    results and the quantities of the statistics of RVE studies."""
    if prop not in KINDS:
        raise ValueError(f"harmonic property: one of {tuple(KINDS)}, got {prop!r}")
    stems = {
        "stiffness": ("bulk_storage", "bulk_loss", "shear_storage", "shear_loss",
                      "young_storage", "young_loss", "shear_tan_delta"),
        "conductivity": ("conductivity", "permittivity"),
        "permittivity": ("index", "extinction"),
    }[KINDS[prop]]  # fmt: skip
    return tuple(f"{stem}_{k}" for k in range(1, int(count) + 1) for stem in stems)


def _frequencies(values, name: str = "frequencies") -> np.ndarray:
    f = np.atleast_1d(np.asarray(values, dtype=float))
    if f.ndim != 1 or not f.size or np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError(f"{name}: positive values")
    return f


def _image(image) -> np.ndarray:
    if not isinstance(image, VoxelImage):
        raise TypeError("harmonic homogenization: a periodic VoxelImage (the FFT solver)")
    if image.data.dtype.kind not in "iub":
        raise ValueError("homogenization needs an integer phase image")
    return np.asarray(image.data).ravel().astype(np.int64)


def dynamic_stiffness(image: VoxelImage, materials: dict[int, Material], frequencies, *,
                      dtemp: float = 0.0, grains=None, orientations=None, tol: float = 1e-8,
                      maxiter: int | None = None) -> HarmonicResponse:  # fmt: skip
    """The effective complex stiffness of a periodic voxel image at the angular
    ``frequencies`` (rad per unit time): the phases with a ``viscoelasticity`` have the
    complex moduli of their Prony series (at the change of temperature ``dtemp`` from
    their reference: their shifted times), the others their elasticity (crystals turned
    by the orientations of their ``grains``). :class:`HarmonicResponse`: the storage and
    loss stiffnesses, the complex bulk, shear and Young's moduli, the loss factors."""
    phase = _image(image)
    freqs = _frequencies(frequencies)
    table = material_table(image.data, materials, "elastic", grains=grains,
                           orientations=orientations)  # fmt: skip
    viscous = {}
    for p in table.phases:
        m = materials[p]
        if m.void or m.viscoelasticity is None:
            continue
        rows = np.unique(table.mat[phase == p])
        viscous[p] = (rows[rows >= 0], m)
    if not viscous:
        log.info("dynamic stiffness: no viscoelastic phase, the stiffness is real")
    tensors = np.zeros((freqs.size, 6, 6), dtype=complex)
    iterations = np.zeros(freqs.size, dtype=int)
    for i, omega in enumerate(freqs):
        laws = table.matrices.astype(complex)
        for rows, m in viscous.values():
            k, g = m.viscoelasticity.complex_moduli(omega, *bulk_shear(m.young, m.poisson), dtemp)
            law = (k - 2.0 * g / 3.0) * isotropic_matrix(1.0, 0.0) + g * isotropic_matrix(0.0, 1.0)
            laws[rows] = law
        tensors[i], problem = harmonic_tensor(image, laws, table.mat, ("displacement",),
                                              tol=tol, maxiter=maxiter)  # fmt: skip
        iterations[i] = problem.iterations
    return HarmonicResponse("stiffness", freqs, tensors, iterations)


def _transport_laws(image, materials, physics, grains, orientations):
    """The real tensors of the phases for a transport physics (zero where a phase has no
    coefficient for it), one per phase or per grain of an oriented phase, on the material
    indices shared by the physics: (indices of the voxels, tensors)."""
    phase = _image(image)
    present = [int(p) for p in np.unique(phase)]
    missing = [p for p in present if p not in materials]
    if missing:
        raise ValueError(f"no material for phase(s) {missing}; phases present: {present}")
    oriented = grains is not None and orientations is not None
    if oriented:
        grains = np.asarray(grains.data if isinstance(grains, VoxelImage) else grains).ravel()
    mat = np.full(phase.size, -1, dtype=np.int64)
    tensors = {name: [] for name in physics}
    count = 0
    for p in present:
        m = materials[p]
        if m.void:
            continue
        sel = np.flatnonzero(phase == p)
        turn = oriented and any(m.has(ph) and m.oriented(ph) for ph in physics)
        if turn:
            ids, inverse = np.unique(grains[sel], return_inverse=True)
            mat[sel] = count + inverse
            euler = np.asarray(orientations, dtype=float)[ids]
            n = len(ids)
        else:
            mat[sel] = count
            euler, n = None, 1
        for name in physics:
            if not m.has(name) or m.inert(name):
                tensors[name].append(np.zeros((n, 3, 3)))
            else:
                tensors[name].append(np.asarray(m.matrix(name, euler)).reshape(n, 3, 3))
        count += n
    empty = np.zeros((0, 3, 3))
    return mat, {name: np.concatenate(v) if v else empty for name, v in tensors.items()}


def complex_conductivity(image: VoxelImage, materials: dict[int, Material], frequencies, *,
                         grains=None, orientations=None, tol: float = 1e-8,
                         maxiter: int | None = None) -> HarmonicResponse:  # fmt: skip
    """The effective complex conductivity ``sigma*(omega)`` of a periodic voxel image at
    the angular ``frequencies``: each phase has ``sigma* = sigma + i omega eps`` of its
    ``electrical_conductivity`` and ``permittivity`` (either may be missing or zero: a
    perfect insulator, a conductor without polarization). :class:`HarmonicResponse`: the
    effective conductivity ``Re(sigma*)`` and permittivity ``Im(sigma*) / omega`` at each
    frequency (the dispersion of Maxwell, Wagner and Sillars)."""
    freqs = _frequencies(frequencies)
    mat, parts = _transport_laws(image, materials, ("electrical", "dielectric"), grains,
                                 orientations)  # fmt: skip
    sigma, eps = parts["electrical"], parts["dielectric"]
    carried = np.abs(sigma).max(axis=(1, 2), initial=0.0)
    carried = carried + np.abs(eps).max(axis=(1, 2), initial=0.0)
    if len(carried) and np.any(carried == 0):
        dead = np.flatnonzero(carried == 0)
        mat = np.where(np.isin(mat, dead), -1, mat)  # neither conducts nor polarizes: a void
    tensors = np.zeros((freqs.size, 3, 3), dtype=complex)
    iterations = np.zeros(freqs.size, dtype=int)
    for i, omega in enumerate(freqs):
        tensors[i], problem = harmonic_tensor(image, sigma + 1j * omega * eps, mat,
                                              tol=tol, maxiter=maxiter)  # fmt: skip
        iterations[i] = problem.iterations
    return HarmonicResponse("conductivity", freqs, tensors, iterations)


def effective_index(image: VoxelImage, indices: dict, wavelengths=1.0, *, tol: float = 1e-8,
                    maxiter: int | None = None) -> HarmonicResponse:  # fmt: skip
    """The effective permittivity of a periodic voxel image in the long-wavelength
    (quasi-static) limit of optics: ``indices`` maps each phase to its complex refractive
    index ``n + i k`` (one value, or one per wavelength; ``k > 0``: absorption; or a
    :class:`~morphogen.homogenization.materials.Material` with a ``refractive_index``) and
    the permittivity of each phase is ``(n + i k)^2``. :class:`HarmonicResponse` (its
    ``frequencies`` are the ``wavelengths``): the complex permittivity tensors, the
    refractive indices along the axes and the absorption coefficients ``4 pi k /
    lambda``. Valid while the wavelengths are much longer than the microstructure;
    resonances of its size (photonic bands) are those of :mod:`morphogen.homogenization.
    waves`."""
    phase = _image(image)
    lengths = _frequencies(wavelengths, "wavelengths")
    present = [int(p) for p in np.unique(phase)]
    missing = [p for p in present if p not in indices]
    if missing:
        raise ValueError(f"no refractive index for phase(s) {missing}")
    values = {}
    for p in present:
        value = indices[p]
        if isinstance(value, Material):
            if value.refractive_index is None:
                raise ValueError(f"phase {p}: the material has no refractive_index")
            value = value.refractive_index
        n = np.atleast_1d(np.asarray(value, dtype=complex))
        if n.size not in (1, lengths.size):
            raise ValueError(f"indices[{p}]: one index, or one per wavelength")
        values[p] = np.broadcast_to(n, lengths.shape)
    mat = np.searchsorted(present, phase)
    tensors = np.zeros((lengths.size, 3, 3), dtype=complex)
    iterations = np.zeros(lengths.size, dtype=int)
    for i in range(lengths.size):
        laws = np.array([values[p][i] ** 2 * np.eye(3) for p in present])
        tensors[i], problem = harmonic_tensor(image, laws, mat, tol=tol, maxiter=maxiter)
        iterations[i] = problem.iterations
    return HarmonicResponse("permittivity", lengths, tensors, iterations)
