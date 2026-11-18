"""Electromagnetic Bloch waves of three-dimensional photonic crystals, by plane waves.

In a periodic medium of permittivity ``ε(x)`` and permeability ``μ(x)``, a harmonic field
without sources obeys ``curl(ε⁻¹ curl H) = ω² B`` with ``B = μ H`` and ``div B = 0``. At
the wave vector ``k`` the induction of a Bloch wave is a sum of plane waves over the
reciprocal vectors ``G`` of the cell,

    B(x) = Σ_G (b¹_G e¹_G + b²_G e²_G) exp(i (k + G)·x),

with two amplitudes per plane wave on the unit vectors ``e¹, e²`` normal to ``q = k + G``
(``(e¹, e², q/|q|)`` direct): the fields are divergence-free by construction, and the
gradients, which the curl ignores and which clutter the spectrum of the nodal
discretizations with zero frequencies, are left out (Ho, Chan & Soukoulis 1990; Johnson &
Joannopoulos 2001). The frequencies are those of the Hermitian problem

    Pᴴ μ⁻¹ C ε⁻¹ C μ⁻¹ P b = ω² Pᴴ μ⁻¹ P b,        C = i q ×,

``P`` the map of the amplitudes to the plane waves. The curls are taken on the plane
waves and the products by ``ε⁻¹`` and ``μ⁻¹`` on the grid of the cell, a pair of fast
Fourier transforms apart, so that the matrices are never formed: the lowest frequencies
are found by a block Davidson method (the Rayleigh-Ritz values of a subspace that the
preconditioned residuals of its lowest waves extend, restarted from them and from those
of the step before), from the waves of the previous wave vector; small cells are
diagonalized. The preconditioner is the operator of the inverse medium, ``Pᴴ μ C⁻¹ ε C⁻¹
μ P`` with ``C⁻¹ = i q × / |q|²`` on the transverse waves (Johnson & Joannopoulos 2001):
the inverse of the operator but for the projections, whatever the contrast of the
phases. A uniform permeability leaves the standard problem ``Pᴴ C ε⁻¹ C P h = ω² μ h`` of
the dielectric crystals. The Galerkin projection on the transverse waves is exact: the
residual of an eigenvector is transverse and orthogonal to the transverse waves in the
product of ``μ⁻¹``.

The grid of the plane waves is that of the voxels, or one ``coarsen`` times coarser
whose cells hold several voxels. A cell that an interface cuts then takes the tensor of
Meade et al. (1993), as averaged by Johnson & Joannopoulos (2001),

    ε̃⁻¹ = <ε⁻¹> n ⊗ n + <ε>⁻¹ (1 - n ⊗ n),

the harmonic mean for the field normal to the interface and the arithmetic mean along
it (the effective tensor of the layers that the interface draws in the cell), ``n`` the
sum of the gradients of the permittivity over the voxels of the cell. The stairs of the
voxels cost the frequencies an error of first order in the size of the cells; this
average brings it back to the second order: voxelize three or four times finer than
the plane waves.

With relative permittivities and permeabilities the frequencies are in units of the
speed of light over the unit of length (``ω a / 2π c``: :meth:`~morphogen.homogenization.
bloch.BandStructure.normalized`). A cell one voxel thick is a crystal invariant along
that axis: its waves in the plane are the two polarizations ``"tm"`` and ``"te"`` of
:func:`~morphogen.homogenization.bloch.scalar_band_structure` together, and those that
leave the plane (a wave vector with a component along the axis) are found here only.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import fft as sfft
from scipy.linalg import eigh

from morphogen import settings
from morphogen.image import VoxelImage

log = logging.getLogger(__name__)

#: Up to this number of unknowns (two per plane wave) the matrices are formed and
#: diagonalized
DENSE_LIMIT = 1800

#: The amplitudes of a block handled at once (bounds the memory of the transforms)
BLOCK_VALUES = 6_000_000

#: The subspace of the Davidson method, in blocks of the waves sought, before a restart
SUBSPACE = 5

_AXES = (-3, -2, -1)


def _tensor_field(values: dict, data: np.ndarray, name: str):
    """The field of a coefficient of the phases on the voxels: an array of the shape of
    the image (isotropic phases) or ``(3, 3) + shape`` (a tensor in one phase at least)."""
    phases = np.unique(data)
    missing = [int(p) for p in phases if int(p) not in values]
    if missing:
        raise ValueError(f"electromagnetic waves: no {name} for phase(s) {missing}")
    tensors = {}
    for p in phases:
        v = np.asarray(values[int(p)], dtype=float)
        if v.shape == (3,):
            v = np.diag(v)
        if v.shape not in ((), (3, 3)) or not np.all(np.isfinite(v)):
            raise ValueError(f"electromagnetic waves: a {name} or a 3 x 3 tensor (phase {int(p)})")
        low = float(v) if v.ndim == 0 else float(np.linalg.eigvalsh(0.5 * (v + v.T)).min())
        if not low > 0.0:
            raise ValueError(
                f"electromagnetic waves: a positive {name} (phase {int(p)}; 1 for the vacuum: "
                "perfect conductors and absorbing phases are not covered)"
            )
        tensors[int(p)] = v
    if all(v.ndim == 0 for v in tensors.values()):
        lookup = np.zeros(int(phases.max()) + 1)
        for p, v in tensors.items():
            lookup[p] = float(v)
        return lookup[data]
    lookup = np.zeros((int(phases.max()) + 1, 3, 3))
    for p, v in tensors.items():
        lookup[p] = v * np.eye(3) if v.ndim == 0 else 0.5 * (v + v.T)
    return np.moveaxis(lookup[data], (-2, -1), (0, 1))


def _invert(field: np.ndarray) -> np.ndarray:
    if field.ndim == 3:
        return 1.0 / field
    return np.moveaxis(np.linalg.inv(np.moveaxis(field, (0, 1), (-2, -1))), (-2, -1), (0, 1))


def _blocks(field: np.ndarray, factors) -> np.ndarray:
    """The mean of a field of the voxels over the cells of ``factors`` voxels per axis."""
    n = field.shape
    shape = (n[0] // factors[0], factors[0], n[1] // factors[1], factors[1], n[2] // factors[2])
    return field.reshape(*shape, factors[2]).mean(axis=(1, 3, 5))


def smoothed_inverse(field: np.ndarray, coarsen: int = 1) -> np.ndarray:
    """The inverse of an isotropic coefficient of the voxels (``field``: the permittivity)
    on a grid ``coarsen`` times coarser along each axis (the axes of one voxel are kept):
    ``1 / field`` on the voxels themselves, else the tensor ``<1/ε> n ⊗ n + (1 - n ⊗ n) /
    <ε>`` of each cell (``(3, 3) + shape``; see the module). The normal ``n`` is the sum
    of the gradients of the field over the voxels of the cell (central differences: the
    normal of a plane that cuts the cell, whatever its stairs), or, where they cancel (a
    layer thinner than the cell, with its two faces), their principal direction; the
    cells that hold one value keep it."""
    field = np.asarray(field, dtype=float)
    s = int(coarsen)
    if s < 1:
        raise ValueError("coarsen: a positive integer")
    if s == 1:
        return 1.0 / field
    factors = tuple(1 if n == 1 else s for n in field.shape)
    if any(n % f for n, f in zip(field.shape, factors, strict=True)):
        raise ValueError(f"coarsen: {s} does not divide the voxels {field.shape} of the cell")
    mean = _blocks(field, factors)
    harmonic = _blocks(1.0 / field, factors)
    gradient = [
        np.zeros_like(field)
        if field.shape[a] == 1
        else 0.5 * (np.roll(field, -1, axis=a) - np.roll(field, 1, axis=a))
        for a in range(3)
    ]
    total = np.stack([_blocks(g, factors) for g in gradient])
    length = np.sqrt((total**2).sum(axis=0))
    spread = _blocks(np.sqrt(sum(g * g for g in gradient)), factors)  # the mean norm
    structure = np.zeros((3, 3, *mean.shape))
    for i in range(3):
        for j in range(i, 3):
            structure[i, j] = structure[j, i] = _blocks(gradient[i] * gradient[j], factors)
    _, vectors = np.linalg.eigh(np.moveaxis(structure, (0, 1), (-2, -1)))
    normal = np.moveaxis(vectors[..., -1], -1, 0)  # that of the largest eigenvalue
    single = length > 0.3 * spread  # one interface: its gradients add up
    normal = np.where(single, total / np.where(length > 0.0, length, 1.0), normal)
    projector = normal[:, None] * normal[None, :]
    inverse = (harmonic - 1.0 / mean) * projector
    for i in range(3):
        inverse[i, i] += 1.0 / mean
    return inverse


def _multiply(tensor: np.ndarray, field: np.ndarray) -> np.ndarray:
    """A coefficient of the grid (isotropic, or ``(3, 3) + shape``) times fields ``(p, 3)
    + shape``."""
    if tensor.ndim == 3:
        return tensor * field
    return np.einsum("ijxyz,pjxyz->pixyz", tensor, field)


class PlaneWaves:
    """The plane waves of a periodic cell of edges ``size`` on the grid of the inverse
    coefficients ``inverse_permittivity`` and ``inverse_permeability`` (arrays of the
    shape of the grid, or ``(3, 3) + shape``; the permeability a number when uniform),
    and the Maxwell operators at a wave vector (:meth:`at`)."""

    def __init__(self, inverse_permittivity, size, inverse_permeability=1.0) -> None:
        self.inverse_permittivity = np.asarray(inverse_permittivity, dtype=float)
        self.permittivity = _invert(self.inverse_permittivity)
        self.shape = self.inverse_permittivity.shape[-3:]
        self.size = np.asarray(size, dtype=float)
        mu = np.asarray(inverse_permeability, dtype=float)
        self.uniform = mu.ndim == 0
        self.inverse_permeability = float(mu) if self.uniform else mu
        self.permeability = 1.0 / float(mu) if self.uniform else _invert(mu)
        self.g = np.stack(
            np.meshgrid(
                *[
                    2.0 * np.pi * sfft.fftfreq(n, d=length / n)
                    for n, length in zip(self.shape, self.size, strict=True)
                ],
                indexing="ij",
            )
        )
        #: the unit of the eigenvalues: the square of the first reciprocal vector
        self.scale = float((self.size.max() / (2.0 * np.pi)) ** 2)
        self.count = int(np.prod(self.shape))

    @property
    def unknowns(self) -> int:
        return 2 * self.count

    def at(self, wavevector) -> PlaneWaves:
        """Set the wave vector: the vectors ``q = k + G``, their norms and their
        transverse unit vectors ``e1`` and ``e2``."""
        k = np.asarray(wavevector, dtype=float).reshape(3, 1, 1, 1)
        self.q = self.g + k
        norm = np.sqrt((self.q**2).sum(axis=0))
        zero = norm == 0.0
        u = self.q / np.where(zero, 1.0, norm)
        e1 = np.stack([-u[1], u[0], np.zeros_like(norm)])  # z × u
        length = np.sqrt((e1**2).sum(axis=0))
        along = length < 1e-6  # u along z (or zero): x, less its part along u
        if np.any(along):
            e1[:, along] = np.array([1.0, 0.0, 0.0])[:, None] - u[0][along] * u[:, along]
            length[along] = np.sqrt((e1[:, along] ** 2).sum(axis=0))
        e1 /= length
        e2 = np.cross(u, e1, axis=0)
        e2[:, zero] = np.array([0.0, 1.0, 0.0])[:, None]
        self.norm, self.e1, self.e2 = norm, e1, e2
        self.inverse_norm = np.where(zero, 0.0, 1.0 / np.where(zero, 1.0, norm))
        return self

    # -- the maps between amplitudes, plane waves and the grid ----------------------------

    def _fields(self, c: np.ndarray) -> np.ndarray:
        return c[:, 0, None] * self.e1 + c[:, 1, None] * self.e2

    def _project(self, f: np.ndarray) -> np.ndarray:
        return np.stack([(f * self.e1).sum(axis=1), (f * self.e2).sum(axis=1)], axis=1)

    def _on_grid(self, tensor, f: np.ndarray) -> np.ndarray:
        if np.ndim(tensor) == 0:
            return tensor * f
        workers = settings.workers()
        f = sfft.ifftn(f, axes=_AXES, workers=workers)
        return sfft.fftn(_multiply(tensor, f), axes=_AXES, workers=workers)

    def _curl(self, f: np.ndarray) -> np.ndarray:
        q = self.q
        return 1j * np.stack(
            [
                q[1] * f[:, 2] - q[2] * f[:, 1],
                q[2] * f[:, 0] - q[0] * f[:, 2],
                q[0] * f[:, 1] - q[1] * f[:, 0],
            ],
            axis=1,
        )

    def _stiffness(self, c: np.ndarray) -> np.ndarray:
        if self.uniform:  # the curl of a transverse wave: i |q| (c1 e2 - c2 e1)
            d = self.norm * (c[:, 0, None] * self.e2 - c[:, 1, None] * self.e1)
            e = self._on_grid(self.inverse_permittivity, d)
            out = np.stack([(e * self.e2).sum(axis=1), -(e * self.e1).sum(axis=1)], axis=1)
            return (self.scale * self.inverse_permeability) * self.norm * out
        h = self._on_grid(self.inverse_permeability, self._fields(c))
        e = self._on_grid(self.inverse_permittivity, self._curl(h))
        return self.scale * self._project(self._on_grid(self.inverse_permeability, self._curl(e)))

    def _mass(self, c: np.ndarray) -> np.ndarray:
        return self._project(self._on_grid(self.inverse_permeability, self._fields(c)))

    def _precondition(self, c: np.ndarray) -> np.ndarray:
        inverse = self.inverse_norm
        if self.uniform:  # the inverse curls of the transverse waves, as the curls above
            f = inverse * (c[:, 0, None] * self.e2 - c[:, 1, None] * self.e1)
            f = self._on_grid(self.permittivity, f)
            out = np.stack([(f * self.e2).sum(axis=1), -(f * self.e1).sum(axis=1)], axis=1)
            return (self.permeability / self.scale) * inverse * out
        f = self._curl(self._on_grid(self.permeability, self._fields(c))) * inverse**2
        f = self._curl(self._on_grid(self.permittivity, f)) * inverse**2
        return self._project(self._on_grid(self.permeability, f)) / self.scale

    def _blockwise(self, function, x: np.ndarray) -> np.ndarray:
        """``function`` of the columns of ``x`` (unknowns x p), a few at a time."""
        x = np.asarray(x, dtype=complex)
        single = x.ndim == 1
        columns = x.reshape(self.unknowns, -1).T.reshape(-1, 2, *self.shape)
        step = max(1, BLOCK_VALUES // (3 * self.count))
        out = np.concatenate(
            [function(columns[i : i + step]) for i in range(0, len(columns), step)]
        )
        out = out.reshape(len(columns), self.unknowns).T
        return out[:, 0] if single else out

    def stiffness(self, x: np.ndarray) -> np.ndarray:
        """``Pᴴ μ⁻¹ C ε⁻¹ C μ⁻¹ P`` times the amplitudes ``x`` (unknowns, or unknowns x p),
        in units of the square of the first reciprocal vector."""
        return self._blockwise(self._stiffness, x)

    def mass(self, x: np.ndarray) -> np.ndarray:
        """``Pᴴ μ⁻¹ P`` times the amplitudes (the identity for a uniform permeability,
        which the stiffness then carries)."""
        return np.asarray(x, dtype=complex) if self.uniform else self._blockwise(self._mass, x)

    def precondition(self, x: np.ndarray) -> np.ndarray:
        """The operator of the inverse medium, ``Pᴴ μ C⁻¹ ε C⁻¹ μ P``, times the amplitudes:
        the inverse of the stiffness but for the projections on the transverse waves."""
        return self._blockwise(self._precondition, x)

    # -- the frequencies -------------------------------------------------------------------

    def _orthonormal(self, x: np.ndarray, mx: np.ndarray):
        """The columns of ``x`` made orthonormal in the product of the mass (``mx``: the
        mass times them), those that depend on the others left out."""
        gram = x.conj().T @ mx
        values, vectors = eigh(0.5 * (gram + gram.conj().T))
        keep = values > 1e-12 * values[-1]
        t = vectors[:, keep] / np.sqrt(values[keep])
        return x @ t, mx @ t

    def _davidson(self, x: np.ndarray, bands: int, tol: float, maxiter: int):
        """The lowest waves of the pencil of the stiffness and the mass from the block
        ``x``: the Rayleigh-Ritz values of a subspace that grows by the preconditioned
        residuals of the waves not yet within ``tol`` (those asked for: the first
        ``bands``), restarted from the waves and from those of the step before."""
        block = x.shape[1]
        v, mv = self._orthonormal(x, self.mass(x))
        kv = self.stiffness(v)
        before = None  # the waves of the step before, in the basis
        for iteration in range(1, maxiter + 1):
            h = v.conj().T @ kv
            values, c = eigh(0.5 * (h + h.conj().T))
            values, c = values[:block], c[:, :block]
            x, kx, mx = v @ c, kv @ c, mv @ c
            norms = np.sqrt((np.abs(kx - mx * values) ** 2).sum(axis=0))
            worst = float(norms[:bands].max())
            active = norms > tol
            if worst <= tol or iteration == maxiter or not np.any(active):
                break
            w = self.precondition((kx - mx * values)[:, active])
            if v.shape[1] + w.shape[1] > SUBSPACE * block:  # restart
                both = c if before is None else np.hstack([c, before])
                t, _ = np.linalg.qr(both)
                v, kv, mv = v @ t, kv @ t, mv @ t
                c = t.conj().T @ c
            before = np.vstack([c, np.zeros((w.shape[1], c.shape[1]), dtype=c.dtype)])
            for _ in range(2):  # against the subspace, twice (Gram-Schmidt)
                w -= v @ (mv.conj().T @ w)
            w, mw = self._orthonormal(w, self.mass(w))
            if w.shape[1] == 0:
                break
            if w.shape[1] < before.shape[0] - c.shape[0]:
                before = before[: c.shape[0] + w.shape[1]]
            v, mv = np.hstack([v, w]), np.hstack([mv, mw])
            kv = np.hstack([kv, self.stiffness(w)])
        return values, x, worst, iteration

    def frequencies(self, wavevector, bands: int, *, guess=None, tol: float = 1e-4,
                    maxiter: int = 300):  # fmt: skip
        """The lowest ``bands`` angular frequencies at a wave vector and their amplitudes
        (unknowns x block; a ``guess`` for them: those of a neighbouring wave vector)."""
        self.at(wavevector)
        n = self.unknowns
        if not 0 < bands <= n:
            raise ValueError(f"bands: between 1 and {n} (two per plane wave), got {bands}")
        if n <= DENSE_LIMIT or bands > n // 8:
            identity = np.eye(n, dtype=complex)
            k = self.stiffness(identity)
            m = None if self.uniform else self.mass(identity)
            values, vectors = eigh(
                0.5 * (k + k.conj().T), None if m is None else 0.5 * (m + m.conj().T)
            )
            return self._frequency(values[:bands]), vectors[:, :bands]
        block = bands + max(3, bands // 4)
        rng = np.random.default_rng(0)
        x = rng.standard_normal((n, block)) + 1j * rng.standard_normal((n, block))
        long_waves = 1.0 / (self.norm**2 * self.scale + 1.0)
        x *= np.concatenate([long_waves.ravel(), long_waves.ravel()])[:, None]
        if guess is not None and guess.shape == (n, block):
            # the waves of the neighbouring wave vector, with a part of all the others: a
            # block that misses one of the lowest waves altogether would never find it
            x = guess + 0.1 * x * (np.linalg.norm(guess, axis=0) / np.linalg.norm(x, axis=0))
        values, x, worst, iterations = self._davidson(x, bands, tol, maxiter)
        if worst > tol:
            log.warning(
                "electromagnetic waves: the bands are within %.1e after %d iterations (asked: "
                "%.1e)", worst, iterations, tol,
            )  # fmt: skip
        return self._frequency(values[:bands]), x

    def _frequency(self, values: np.ndarray) -> np.ndarray:
        return np.sqrt(np.clip(values.real, 0.0, None) / self.scale)


def _crystal(sample, permittivity: dict, permeability, coarsen: int) -> PlaneWaves:
    if not isinstance(sample, VoxelImage):
        raise TypeError(
            "electromagnetic waves: a voxel image (the plane waves are those of its grid; "
            "voxelize a morphology, rather than mesh it)"
        )
    if not sample.periodic:
        log.info("electromagnetic waves: the image is taken as the cell of a periodic medium")
    data = np.asarray(sample.data)
    if data.dtype.kind not in "iu":
        raise ValueError("electromagnetic waves: an image of phases (integers)")
    eps = _tensor_field(permittivity, data, "permittivity")
    if eps.ndim == 3:
        inverse = smoothed_inverse(eps, coarsen)
    elif int(coarsen) != 1:
        raise ValueError(
            "coarsen: the average of the cells that the interfaces cut is that of isotropic "
            "phases; with a tensor of permittivity the plane waves are those of the voxels"
        )
    else:
        inverse = _invert(eps)
    if isinstance(permeability, dict):
        present = [permeability.get(int(p)) for p in np.unique(data)]
        if all(v is not None and np.ndim(v) == 0 for v in present) and len(set(present)) == 1:
            permeability = float(present[0])  # the same in every phase
    if isinstance(permeability, dict):
        mu = _tensor_field(permeability, data, "permeability")
        if mu.ndim == 3:
            inverse_mu = smoothed_inverse(mu, coarsen)
        elif int(coarsen) != 1:
            raise ValueError("coarsen: isotropic phases (see the permittivity)")
        else:
            inverse_mu = _invert(mu)
    else:
        if not float(permeability) > 0.0:
            raise ValueError("electromagnetic waves: a positive permeability")
        inverse_mu = 1.0 / float(permeability)
    return PlaneWaves(inverse, sample.size, inverse_mu)


def electromagnetic_band_structure(
    sample,
    permittivity: dict,
    path=("G", "X", "M", "G", "R"),
    *,
    permeability=1.0,
    bands: int = 8,
    points: int = 12,
    coarsen: int = 1,
    tol: float = 1e-4,
):
    """The lowest ``bands`` frequencies of the electromagnetic Bloch waves of a periodic
    cell (a :class:`~morphogen.image.VoxelImage` of phases) along ``path`` (corners of
    the Brillouin zone, ``points`` per segment; or an array of wave vectors), by plane
    waves (see the module): ``permittivity``, that of each phase (a number, three
    principal values or a 3 x 3 tensor in the axes of the cell), and ``permeability`` (a
    number for the whole cell, or those of the phases likewise). ``coarsen``: the plane
    waves on a grid that many times coarser than the voxels, whose cells cut by the
    interfaces take the averaged tensor of Meade et al. (isotropic phases). ``tol``: the
    residual of the bands, in units of the square of the first reciprocal vector (the
    squares of the frequencies are then within about its square). Returns a
    :class:`~morphogen.homogenization.bloch.BandStructure`; at the centre of the zone the
    first two frequencies are zero (the uniform fields)."""
    from morphogen.homogenization.bloch import BandStructure, brillouin_path

    crystal = _crystal(sample, permittivity, permeability, coarsen)
    if isinstance(path, np.ndarray) and path.ndim == 2:
        ks = np.asarray(path, dtype=float)
        dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ks, axis=0), axis=1))])
        labels = []
    else:
        ks, dist, labels = brillouin_path(crystal.size, path, points)
    freqs = np.zeros((len(ks), int(bands)))
    guess = None
    for i, k in enumerate(ks):
        freqs[i], guess = crystal.frequencies(k, int(bands), guess=guess, tol=tol)
    return BandStructure(ks, dist, freqs, labels)
