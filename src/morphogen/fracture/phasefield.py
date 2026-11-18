"""Brittle fracture of voxel images by the phase-field method: cracks that start, branch
and merge in a periodic microstructure without being meshed or tracked.

A field ``d`` (0: sound, 1: broken) regularizes the cracks over the length ``l``
(Francfort & Marigo 1998; Bourdin, Francfort & Marigo 2000): the energy of the cell is

    E(u, d) = int g(d) psi(eps(u)) dV + int G_c gamma(d, grad d) dV,   g(d) = (1 - d)^2 + k,

with the crack density ``gamma = d^2 / (2 l) + l |grad d|^2 / 2`` (the model ``AT2``: the
damage grows from the first load) or ``gamma = 3/8 (d / l + l |grad d|^2)`` (``AT1``:
elastic up to the stress ``sqrt(3 G_c E / (8 l))`` in uniaxial tension), whose integral
tends to the area of the cracks as ``l`` vanishes: the work to break the cell is the
toughness ``G_c`` times that area, whatever the voxels, provided they resolve ``l``
(linear elements over-estimate it by the factor ``1 + h / (2 l)``, ``h`` their size: at
least two voxels per length).

The problem is solved by alternate minimization (Bourdin et al. 2000): at each step of a
loading path, until the damage no longer moves,

1. the displacement at fixed damage: the periodic elastic problem of the voxels of the
   FFT solver (:mod:`morphogen.homogenization.fft`), each voxel with its stiffness times
   ``g`` of its damage (conjugate gradients preconditioned in Fourier space, the
   reference medium scaled by the degradation of the nodes; the macroscopic components
   prescribed as stresses are condensed);
2. the damage at fixed displacement: ``(G_c / l + 2 psi+) d - G_c l lap(d) = 2 psi+``
   (``AT2``) with ``psi+`` the tensile energy of each voxel, under the bound that the
   damage does not decrease (irreversibility, by a primal-dual active set: the obstacle
   problem of ``AT1``, whose damage is zero below its elastic limit and whose cracks have
   the finite width ``4 l``); trilinear elements with a lumped reaction, conjugate
   gradients. With ``irreversibility="history"`` the bound is replaced by the largest
   ``psi+`` reached at each voxel (the history field of Miehe, Hofacker & Welschinger
   2010): simpler, but it widens the cracks and over-estimates their energy.

The stress is degraded as a whole (``sigma = g(d) C : eps``: the hybrid formulation of
Ambati, Gerasimov & De Lorenzis 2015, linear at fixed damage) while the energy that
drives the damage is its tensile part only: all of it (``split="none"``), the spectral
split of Miehe et al. (``"spectral"``: ``lambda <tr eps>+^2 / 2 + mu sum <eps_i>+^2``, by
default) or the volumetric one of Amor et al. (``"volumetric"``: ``K <tr eps>+^2 / 2 + mu
dev eps : dev eps``), so that compression does not break. Each phase has its toughness;
the phases without one do not break (their stiffness is not degraded), the voids carry
neither stress nor cracks. Initial cracks are voxels given as broken: the damage of their
nodes is held at 1.

The results are those of a loading path (the macroscopic stress and strain in time),
with the mean damage, the regularized area of the cracks and the energies; the strength
of the cell (its largest stress) and its work of fracture are quantities for the
statistics of representative volumes (:meth:`PhaseFieldResult.strength`,
:meth:`PhaseFieldResult.work`).

In finite strains (rubbers, foams, soft tissues): :mod:`morphogen.fracture.
phasefield_finite`, the same damage problem with the energy of the deformation gradient
and the nonlinear FFT solver for the displacement.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from morphogen.homogenization import kernels
from morphogen.homogenization.fft import FFTElasticProblem, FourierPreconditioner
from morphogen.homogenization.materials import Material
from morphogen.homogenization.paths import LoadingPath, PathResult
from morphogen.homogenization.solvers import conjugate_gradient
from morphogen.image import VoxelImage
from morphogen.stopping import checkpoint

log = logging.getLogger("morphogen")

#: Models of the crack density: (c_w, the local term) of ``gamma = (w(d) / l + l |grad
#: d|^2) / c_w``
MODELS = ("AT2", "AT1")
#: Splits of the energy that drives the damage
SPLITS = ("spectral", "volumetric", "none")
#: Engineering shears of the tensor components of the strains
_ENGINEERING = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])


@dataclass(frozen=True)
class PhaseField:
    """The fracture of the phases of a sample: ``toughness``, the critical energy release
    rate ``G_c`` of all the phases (one value) or of each (a mapping phase -> ``G_c``; the
    phases left out do not break), the regularization ``length`` ``l`` (at least two
    voxels), the ``model`` (:data:`MODELS`), the ``split`` of the driving energy
    (:data:`SPLITS`), the ``residual`` stiffness ``k`` of a broken voxel and the
    ``irreversibility``: ``"bound"`` (the damage of each node never decreases) or
    ``"history"`` (the largest driving energy reached; see the module)."""

    toughness: float | dict
    length: float
    model: str = "AT2"
    split: str = "spectral"
    residual: float = 1e-6
    irreversibility: str = "bound"

    def __post_init__(self) -> None:
        if isinstance(self.toughness, dict):
            values = {int(p): float(g) for p, g in self.toughness.items()}
            if not values or any(not (math.isfinite(g) and g > 0) for g in values.values()):
                raise ValueError("toughness: positive values (phase -> G_c)")
            object.__setattr__(self, "toughness", tuple(sorted(values.items())))
        elif not (math.isfinite(self.toughness) and self.toughness > 0):
            raise ValueError("toughness: a positive critical energy release rate")
        else:
            object.__setattr__(self, "toughness", float(self.toughness))
        if not (math.isfinite(self.length) and self.length > 0):
            raise ValueError("length: a positive regularization length")
        if self.model not in MODELS:
            raise ValueError(f"model: one of {MODELS}, got {self.model!r}")
        if self.split not in SPLITS:
            raise ValueError(f"split: one of {SPLITS}, got {self.split!r}")
        if not 0.0 < self.residual < 1e-2:
            raise ValueError("residual: a small positive fraction of the stiffness")
        if self.irreversibility not in ("bound", "history"):
            raise ValueError("irreversibility: 'bound' (the damage never decreases) or 'history'")

    @classmethod
    def from_config(cls, spec) -> PhaseField:
        """A mapping of the parameters of :class:`PhaseField`."""
        if isinstance(spec, PhaseField):
            return spec
        if not isinstance(spec, dict):
            raise ValueError("phase field: a mapping (toughness, length, model, split)")
        unknown = sorted(set(spec) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"phase field: unknown keys {unknown}")
        return cls(**spec)

    def of(self, phase: int) -> float | None:
        """The toughness of a phase (None: it does not break)."""
        if isinstance(self.toughness, tuple):
            return dict(self.toughness).get(int(phase))
        return self.toughness

    def critical_stress(self, modulus: float) -> float:
        """The largest stress of a homogeneous bar of modulus ``modulus`` (that of its
        loading: Young's modulus in uniaxial stress) and of this toughness (one value):
        ``9/16 sqrt(E G_c / (3 l))`` (``AT2``), ``sqrt(3 E G_c / (8 l))`` (``AT1``)."""
        gc = self.toughness if not isinstance(self.toughness, tuple) else None
        if gc is None:
            raise ValueError("critical_stress: one toughness for all the phases")
        if self.model == "AT2":
            return 9.0 / 16.0 * math.sqrt(modulus * gc / (3.0 * self.length))
        return math.sqrt(3.0 * modulus * gc / (8.0 * self.length))


# -- kernels -----------------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _apply_scaled(nx, ny, nz, nd, group, scale, mats, w, out):
    """``out = K w`` on the periodic grid, the element matrix of the voxel ``e`` being
    ``scale[e] * mats[group[e]]`` (-1: none)."""
    corners = kernels.CORNERS
    size = 8 * nd
    for n in prange(nx * ny * nz):
        k = n % nz
        j = (n // nz) % ny
        i = n // (ny * nz)
        acc = np.zeros(nd)
        we = np.empty(size)
        for a in range(2):
            ei = i - a if i - a >= 0 else nx - 1
            for b in range(2):
                ej = j - b if j - b >= 0 else ny - 1
                for c in range(2):
                    ek = k - c if k - c >= 0 else nz - 1
                    e = (ei * ny + ej) * nz + ek
                    g = group[e]
                    if g < 0:
                        continue
                    ln = kernels.CORNER_INDEX[a, b, c]
                    for q in range(8):
                        qi = ei + corners[q, 0]
                        qj = ej + corners[q, 1]
                        qk = ek + corners[q, 2]
                        if qi == nx:
                            qi = 0
                        if qj == ny:
                            qj = 0
                        if qk == nz:
                            qk = 0
                        m = (qi * ny + qj) * nz + qk
                        for c2 in range(nd):
                            we[q * nd + c2] = w[m * nd + c2]
                    mat = mats[g]
                    factor = scale[e]
                    for r in range(nd):
                        row = ln * nd + r
                        s = 0.0
                        for col in range(size):
                            s += mat[row, col] * we[col]
                        acc[r] += factor * s
        for r in range(nd):
            out[n * nd + r] = acc[r]


@njit(parallel=True, cache=True)
def _load_scaled(nx, ny, nz, nd, group, scale, loads):
    """``-sum_e scale[e] loads[group[e]]``: the nodal forces of element loads."""
    out = np.zeros(nx * ny * nz * nd)
    for n in prange(nx * ny * nz):
        k = n % nz
        j = (n // nz) % ny
        i = n // (ny * nz)
        for a in range(2):
            ei = i - a if i - a >= 0 else nx - 1
            for b in range(2):
                ej = j - b if j - b >= 0 else ny - 1
                for c in range(2):
                    ek = k - c if k - c >= 0 else nz - 1
                    e = (ei * ny + ej) * nz + ek
                    g = group[e]
                    if g < 0:
                        continue
                    ln = kernels.CORNER_INDEX[a, b, c]
                    for r in range(nd):
                        out[n * nd + r] -= scale[e] * loads[g, ln * nd + r]
    return out


def _to_nodes(values: np.ndarray) -> np.ndarray:
    """The sum over the eight voxels around each node of a periodic grid of the values of
    the voxels (the node ``(i, j, k)`` is the lower corner of the voxel ``(i, j, k)``)."""
    out = np.zeros_like(values)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                out += np.roll(values, (a, b, c), axis=(0, 1, 2))
    return out


def _to_voxels(values: np.ndarray) -> np.ndarray:
    """The mean over the eight corners of each voxel of nodal values."""
    out = np.zeros_like(values)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                out += np.roll(values, (-a, -b, -c), axis=(0, 1, 2))
    return out / 8.0


# -- results -----------------------------------------------------------------------------------


@dataclass
class PhaseFieldResult(PathResult):
    """The response of a cell that breaks along a loading path: a
    :class:`~morphogen.homogenization.paths.PathResult` (the macroscopic ``strain`` and
    ``stress`` at the ``times``; its variable ``plastic`` is the mean damage of the phases
    that break, as that of each phase in ``phases``) with the ``damage`` (the same), the
    regularized area of the cracks per unit volume (``crack_surface``), the elastic and
    fracture energies per unit volume, and the alternate minimizations of each step
    (``iterations``). With ``fields``: the damage, strain and stress of the voxels at chosen
    times. :meth:`~morphogen.homogenization.paths.PathResult.strength` is the strength of
    the cell and :meth:`~morphogen.homogenization.paths.PathResult.work` the work done on
    it: with the stored energy released, more than the fracture energy when the cell
    breaks at once (the cracks of a brittle cell run faster than its loading)."""

    damage: np.ndarray | None = None
    crack_surface: np.ndarray | None = None
    elastic_energy: np.ndarray | None = None
    fracture_energy: np.ndarray | None = None

    def _row(self, k: int) -> dict:
        row = super()._row(k)
        row.update(
            damage=float(self.damage[k]),
            crack_surface=float(self.crack_surface[k]),
            elastic_energy=float(self.elastic_energy[k]),
            fracture_energy=float(self.fracture_energy[k]),
        )
        return row


# -- the problem -------------------------------------------------------------------------------


class PhaseFieldProblem:
    """The brittle fracture of a periodic voxel image (see the module): ``materials`` of
    its phases (elastic; the crystals turned by the orientations of their ``grains``),
    ``fracture`` (:class:`PhaseField`, or a mapping of its parameters), ``initial``: the
    voxels broken from the start (a boolean array of the shape of the image). ``tol``:
    the relative residual of the conjugate gradients; ``staggered_tol``: the largest change
    of the damage that ends the alternate minimization of a step, of at most
    ``max_staggered`` iterations."""

    def __init__(self, image: VoxelImage, materials: dict[int, Material], fracture, *,
                 initial=None, grains=None, orientations=None, tol: float = 1e-7,
                 staggered_tol: float = 1e-3, max_staggered: int = 200):  # fmt: skip
        self.fracture = PhaseField.from_config(fracture)
        self.elastic = FFTElasticProblem(image, materials, tol=tol, grains=grains,
                                         orientations=orientations)  # fmt: skip
        p = self.elastic
        self.image, self.shape = image, image.shape
        self.tol = float(tol)
        self.staggered_tol, self.max_staggered = float(staggered_tol), int(max_staggered)
        frac = self.fracture
        h = float(max(image.spacing[d] for d in range(3) if image.shape[d] > 1))
        if frac.length < 1.5 * h:
            log.warning(
                "phase field: the length %.3g is less than two voxels (%.3g): the cracks "
                "are not resolved and the work of fracture over-estimated", frac.length, h
            )  # fmt: skip
        phase = np.asarray(image.data).reshape(self.shape)
        # toughness of each voxel (0: it does not break) and its Lame coefficients
        gc = np.zeros(self.shape)
        lam = np.zeros(self.shape)
        mu = np.zeros(self.shape)
        for ph in np.unique(phase):
            m = materials[int(ph)]
            if m.void:
                continue
            value = frac.of(int(ph))
            if value is not None:
                gc[phase == ph] = value
            if frac.split != "none":
                if not m.isotropic or m.stiffness is not None:
                    raise ValueError(
                        f"phase {int(ph)}: the {frac.split} split needs isotropic phases "
                        "(split='none' for crystals)"
                    )
                lam[phase == ph], mu[phase == ph] = m.lame
        self.gc, self.lam, self.mu = gc, lam, mu
        self.solid = (p.mat >= 0).reshape(self.shape)
        self.breakable = gc > 0
        if not self.breakable.any():
            raise ValueError("phase field: no phase with a toughness")
        self.volume = float(image.volume)
        self.voxel = float(np.prod(image.spacing))
        # the damage equation: diffusion a, reaction r = r0 + 2 H, source 2 H - s0
        length = frac.length
        if frac.model == "AT2":
            self.diffusion, self.reaction0, self.source0 = gc * length, gc / length, 0.0 * gc
            self.floor = 0.0 * gc
        else:
            self.diffusion = 0.75 * gc * length
            self.reaction0 = 0.0 * gc
            self.source0 = 0.375 * gc / length
            self.floor = 0.0 * gc
        self.k_d = np.ascontiguousarray(p.element.k_thermal[None])
        self.group_d = np.where(self.breakable.ravel(), 0, -1).astype(np.int64)
        self.history = self.floor.copy()
        #: the nodes held at d = 1: those of the voxels broken from the start
        self.fixed = np.zeros(self.shape, dtype=bool)
        if initial is not None:
            broken = np.asarray(initial, dtype=bool)
            if broken.shape != self.shape:
                raise ValueError("initial: a boolean array of the shape of the image")
            self.fixed = _to_nodes((broken & self.breakable).astype(float)) > 0
        self.d = np.zeros(self.shape)
        self._preconditioner = None
        self._warm: dict = {}
        if initial is not None:
            self.d = self._damage(self.history)

    # -- the displacement at fixed damage ------------------------------------------------

    def degradation(self, d: np.ndarray | None = None) -> np.ndarray:
        """``g`` of each voxel for the nodal damage ``d`` (the current one by default): 1
        in the phases that do not break."""
        d = self.d if d is None else d
        mean = np.clip(_to_voxels(d), 0.0, 1.0)
        g = (1.0 - mean) ** 2 + self.fracture.residual
        return np.where(self.breakable, g, 1.0)

    def _mechanics(self, g: np.ndarray, strain: np.ndarray, dt: float, key):
        """The periodic fluctuation of the macroscopic strain ``strain`` and the change of
        temperature ``dt`` in the cell degraded by ``g`` (warm start: ``key``)."""
        p = self.elastic
        nx, ny, nz = self.shape
        scale = np.ascontiguousarray(g.ravel())
        loads = p.element_matrices @ p.element.affine_displacement(strain)
        if dt:
            loads = loads - dt * p._thermal_loads()
        rhs = _load_scaled(nx, ny, nz, 3, p.mat, scale, np.ascontiguousarray(loads))
        norm = float(np.linalg.norm(rhs))
        reference = float(np.sqrt(np.sum(p.counts * np.sum(loads * loads, axis=1))))
        if norm <= 1e-12 * reference:
            return np.zeros_like(rhs), 0
        if self._preconditioner is None:
            self._preconditioner = FourierPreconditioner.from_element(
                self.shape, p._reference_element(), 3
            )
        mats = p.element_matrices

        def apply(v):
            out = np.empty_like(v)
            _apply_scaled(nx, ny, nz, 3, p.mat, scale, mats, v, out)
            return out

        # the reference medium scaled by the degradation of the nodes: the stiffness of a
        # damaged cell is that of the sound one times g, which the Fourier inverse of the
        # sound reference ignores (thousands of iterations across broken voxels)
        nodal = _to_nodes(g) / 8.0
        weight = np.repeat(1.0 / np.sqrt(nodal.ravel()), 3)
        reference = self._preconditioner

        def precondition(r):
            return weight * reference(weight * r)

        start = self._warm.get(key)
        maxiter = 40 * int(max(self.shape)) + 2000
        if start is None:
            w, its, res = conjugate_gradient(apply, precondition, rhs, self.tol, maxiter)
        else:  # from the solution of the last iteration
            r = rhs - apply(start)
            ratio = float(np.linalg.norm(r)) / norm
            if ratio <= self.tol:
                w, its, res = start, 0, ratio
            else:
                dw, its, res = conjugate_gradient(apply, precondition, r, self.tol / ratio,
                                                  maxiter)  # fmt: skip
                w, res = start + dw, res * ratio
        if res > 10.0 * self.tol:
            log.warning("phase field: the elastic solve stopped at the residual %.2e", res)
        self._warm[key] = w
        return w, its

    def _voxel_fields(self, w, strain, dt):
        """Voxel means of the strain (tensor components), of the sound stress ``C : eps``
        and twice the sound energy density (zero in the voids)."""
        p = self.elastic
        nx, ny, nz = self.shape
        el = p.element
        eps, sig, energy = kernels.elastic_fields(
            nx, ny, nz, True, p.mat, p.table.matrices, p.table.eigen, float(dt), w,
            el.affine_displacement(strain), el.b_mean, el.b_gauss,
        )  # fmt: skip
        return np.nan_to_num(eps), np.nan_to_num(sig), np.nan_to_num(energy)

    def _driving(self, eps, energy, dt) -> np.ndarray:
        """The tensile energy density ``psi+`` of the voxels."""
        split = self.fracture.split
        if split == "none":
            return 0.5 * energy.reshape(self.shape)
        e = eps
        if dt:
            p = self.elastic
            thermal = p.table.eigen[np.maximum(p.mat, 0)] * np.array([1, 1, 1, 0.5, 0.5, 0.5])
            e = eps - dt * np.where((p.mat >= 0)[:, None], thermal, 0.0)
        tr = e[:, :3].sum(axis=1)
        lam, mu = self.lam.ravel(), self.mu.ravel()
        if split == "volumetric":
            dev2 = np.sum((e[:, :3] - tr[:, None] / 3.0) ** 2, axis=1) + 2.0 * np.sum(
                e[:, 3:] ** 2, axis=1
            )
            bulk = lam + 2.0 * mu / 3.0
            psi = 0.5 * bulk * np.maximum(tr, 0.0) ** 2 + mu * dev2
        else:
            t = np.empty((len(e), 3, 3))
            for k, (a, b) in enumerate(((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))):
                t[:, a, b] = t[:, b, a] = e[:, k]
            principal = np.linalg.eigvalsh(t)
            psi = 0.5 * lam * np.maximum(tr, 0.0) ** 2 + mu * np.sum(
                np.maximum(principal, 0.0) ** 2, axis=1
            )
        return psi.reshape(self.shape)

    # -- the damage at fixed displacement ------------------------------------------------

    def _damage(self, history: np.ndarray, lower: np.ndarray | None = None) -> np.ndarray:
        """The nodal damage of a history field: the minimum of the energy at fixed
        displacement, ``(r0 + 2 H) d - div(a grad d) = 2 H - s0`` (the reaction and the
        source lumped at the nodes), under the bound ``d >= lower`` (the damage of the last
        step: irreversibility; zero by default) by a primal-dual active set: the nodes at
        their bound are held there while the multiplier of the bound pushes them down."""
        nx, ny, nz = self.shape
        on = self.breakable
        reaction = _to_nodes(np.where(on, self.reaction0 + 2.0 * history, 0.0)) * self.voxel / 8.0
        source = _to_nodes(np.where(on, 2.0 * history - self.source0, 0.0)) * self.voxel / 8.0
        source = source.ravel()
        scale = np.ascontiguousarray(self.diffusion.ravel())
        diag = (_to_nodes(self.diffusion) * float(self.k_d[0, 0, 0]) + reaction).ravel()
        alive = diag > 0
        fixed = self.fixed.ravel()
        bound = np.zeros(diag.size) if lower is None else lower.ravel()
        held = np.where(fixed, 1.0, bound)
        mass = reaction.ravel()

        def operator(v):
            out = np.empty_like(v)
            _apply_scaled(nx, ny, nz, 1, self.group_d, scale, self.k_d, v, out)
            return out + mass * v

        d = np.where(fixed, 1.0, np.maximum(self.d.ravel(), bound))
        active = fixed.copy()
        for _ in range(50):
            free = alive & ~active
            # the unknowns scaled by the diagonal: the residuals are then errors of the
            # damage, whatever the driving energies (those of a crack tip are far above)
            s = np.where(free, 1.0 / np.sqrt(np.where(free, diag, 1.0)), 0.0)
            d = np.where(active, held, np.where(free, d, 0.0))
            r = s * (source - operator(d))
            norm = math.sqrt(float(free.sum()))
            residual = float(np.linalg.norm(r))
            if residual > self.tol * norm:
                y, _, _ = conjugate_gradient(lambda v, s=s: s * operator(s * v), lambda v: v, r,
                                             self.tol * norm / residual, 5000)  # fmt: skip
                d = d + s * y
            multiplier = np.where(active & ~fixed, operator(d) - source, 0.0)
            wanted = fixed | (alive & (multiplier + diag * (bound - d) > 1e-12 * diag))
            if np.array_equal(wanted, active):
                break
            active = wanted
        return np.clip(np.maximum(d, bound), 0.0, 1.0).reshape(self.shape)

    def crack_surface(self, d: np.ndarray | None = None, weight: np.ndarray | None = None) -> float:
        """The regularized area of the cracks, ``int gamma(d, grad d) dV`` over the phases
        that break; with ``weight`` (the toughness of the voxels), the fracture energy."""
        d = self.d if d is None else d
        nx, ny, nz = self.shape
        frac = self.fracture
        on = self.breakable.astype(float) if weight is None else weight
        length = frac.length
        nodal = _to_nodes(on) * self.voxel / 8.0
        grad = np.empty(d.size)
        _apply_scaled(nx, ny, nz, 1, self.group_d, np.ascontiguousarray(on.ravel()), self.k_d,
                      np.ascontiguousarray(d.ravel()), grad)  # fmt: skip
        gradient = float(d.ravel() @ grad)
        if frac.model == "AT2":
            return float(np.sum(nodal * d * d) / (2.0 * length) + 0.5 * length * gradient)
        return float(0.375 * (np.sum(nodal * d) / length + length * gradient))

    # -- loading paths -------------------------------------------------------------------

    def _state(self, g, values, controls, dt):
        """The macroscopic strain and stress, the fluctuation and the voxel fields of the
        cell degraded by ``g`` under the components ``values`` (strains where ``controls``,
        stresses elsewhere)."""
        unknown = np.flatnonzero(~controls)
        strain = np.where(controls, values, 0.0)
        w, its = self._mechanics(g, strain, dt, "base")
        fields = self._voxel_fields(w, strain, dt)
        gv = g.ravel() * self.voxel / self.volume

        def mean_stress(sig):
            return gv @ sig

        stress = mean_stress(fields[1])
        if unknown.size:
            # the cell is linear at fixed damage: the unit strains of the free components
            columns, units = np.zeros((6, unknown.size)), []
            for k, j in enumerate(unknown):
                unit = np.zeros(6)
                unit[j] = 1.0
                wj, n = self._mechanics(g, unit, 0.0, int(j))
                its += n
                units.append(wj)
                columns[:, k] = mean_stress(self._voxel_fields(wj, unit, 0.0)[1])
            free = np.linalg.solve(columns[unknown], values[unknown] - stress[unknown])
            strain[unknown] = free
            w = w + sum(c * wj for c, wj in zip(free, units, strict=True))
            fields = self._voxel_fields(w, strain, dt)
            stress = mean_stress(fields[1])
        return strain, stress, w, fields, its

    def run(self, path, *, name: str = "path", fields=False, monitor=None) -> PhaseFieldResult:
        """Follow a loading path (:class:`~morphogen.homogenization.paths.LoadingPath`, or
        a mapping for it; its components prescribed as strains or as stresses, with a
        change of temperature): :class:`PhaseFieldResult`. ``fields``: true for the damage,
        strain and stress of the voxels at the end of each segment of the path, ``"steps"``
        at the end of each step, or a list of times of its steps. ``monitor(time, records
        so far)`` after each step; a true value ends the path there."""
        path = LoadingPath.from_spec(path)
        t0 = time.perf_counter()
        controls = path.strain_controlled
        ends = path.schedule()
        if fields is True:
            wanted = np.asarray(path.times[1:], dtype=float)
        elif isinstance(fields, str):
            if fields != "steps":
                raise ValueError("fields: true, 'steps' or a list of times of the path")
            wanted = np.asarray(ends, dtype=float)
        else:
            wanted = np.asarray(fields if fields else [], dtype=float)
        names = ("t", "e", "s", "T", "d", "gamma", "elastic", "fracture", "its", "phases")
        rec = {k: [] for k in names}
        rec["t"].append(float(path.times[0]))
        rec["e"].append(np.zeros(6))
        rec["s"].append(np.zeros(6))
        rec["T"].append(0.0)
        rec["d"].append(self._mean_damage())
        rec["gamma"].append(self.crack_surface() / self.volume)
        rec["elastic"].append(0.0)
        rec["fracture"].append(self.crack_surface(weight=self.gc) / self.volume)
        rec["phases"].append(self._phase_means(None, None))
        kept, solves = {}, 0
        for t in ends:
            checkpoint()
            values, dt = path.at(float(t))
            history, d = self.history, self.d
            committed = self.d
            for it in range(1, self.max_staggered + 1):  # noqa: B007
                g = self.degradation(d)
                strain, stress, _, (eps, sig, energy), n = self._state(g, values, controls, dt)
                solves += n
                history = self._driving(eps, energy, dt)
                if self.fracture.irreversibility == "history":
                    history = np.maximum(self.history, history)
                new = self._damage(history, committed)
                change = float(np.abs(new - d).max())
                d = new
                self.d = new  # the next damage solve starts from it
                if change < self.staggered_tol:
                    break
            else:
                log.warning(
                    "phase field: the alternate minimization of the step to t = %g stopped at "
                    "a change of the damage of %.2e", t, change
                )  # fmt: skip
            self.history, self.d = history, d
            g = self.degradation(d)
            rec["t"].append(float(t))
            rec["e"].append(strain)
            rec["s"].append(stress)
            rec["T"].append(float(dt))
            rec["phases"].append(self._phase_means(eps, g.ravel()[:, None] * sig))
            rec["d"].append(self._mean_damage())
            rec["gamma"].append(self.crack_surface() / self.volume)
            rec["elastic"].append(0.5 * float(g.ravel() @ energy) * self.voxel / self.volume)
            rec["fracture"].append(self.crack_surface(weight=self.gc) / self.volume)
            rec["its"].append(it)
            if wanted.size and np.any(np.isclose(t, wanted, rtol=1e-12, atol=0.0)):
                kept[float(t)] = {
                    "damage": _to_voxels(d),
                    "p": _to_voxels(d),
                    "strain": eps.reshape(*self.shape, 6),
                    "stress": (g.ravel()[:, None] * sig).reshape(*self.shape, 6),
                }
            if monitor is not None and monitor(float(t), rec):
                break
        phases = {}
        for ph in rec["phases"][0]:
            phases[ph] = {key: np.array([step[ph][key] for step in rec["phases"]])
                          for key in ("stress", "strain", "plastic")}  # fmt: skip
        return PhaseFieldResult(
            name=name,
            bc="periodic",
            path=path,
            times=np.array(rec["t"]),
            strain=np.array(rec["e"]),
            stress=np.array(rec["s"]),
            temperature=np.array(rec["T"]),
            plastic=np.array(rec["d"]),
            phases=phases,
            iterations=np.array(rec["its"], dtype=int),
            info={
                "linear_iterations": int(solves),
                "time_s": round(time.perf_counter() - t0, 3),
            },
            fields=kept,
            support=self.image if kept else None,
            damage=np.array(rec["d"]),
            crack_surface=np.array(rec["gamma"]),
            elastic_energy=np.array(rec["elastic"]),
            fracture_energy=np.array(rec["fracture"]),
        )

    def _mean_damage(self) -> float:
        """The mean damage of the voxels that may break."""
        d = np.clip(_to_voxels(self.d), 0.0, 1.0)
        return float(d[self.breakable].mean())

    def _phase_means(self, eps, sig) -> dict:
        """The mean stress, strain (engineering shears) and damage of each phase that
        carries a material."""
        p = self.elastic
        damage = np.clip(_to_voxels(self.d), 0.0, 1.0).ravel()
        out = {}
        for g, phase in enumerate(p.table.phases):
            sel = (p.stat == g) & (p.mat >= 0)
            if not sel.any():
                continue
            zero = np.zeros(6)
            out[int(phase)] = {
                "stress": zero if sig is None else sig[sel].mean(axis=0),
                "strain": zero if eps is None else eps[sel].mean(axis=0) * _ENGINEERING,
                "plastic": float(damage[sel].mean()),
            }
        return out


def phase_field_fracture(
    image: VoxelImage,
    materials: dict[int, Material],
    fracture,
    path,
    *,
    name: str = "path",
    fields=False,
    kinematics: str = "small",
    **options,
) -> PhaseFieldResult:
    """The response of a periodic voxel image that breaks along a loading path: a
    :class:`PhaseFieldProblem` (its ``options``: ``initial``, ``grains``, ``orientations``,
    ``tol``, ``staggered_tol``, ``max_staggered``) run along ``path``; with
    ``kinematics="finite"``, in finite strains (:class:`~morphogen.fracture.
    phasefield_finite.FinitePhaseFieldProblem`: hyperelastic or elastic phases, nominal
    strains and stresses, its options)."""
    if kinematics not in ("small", "finite"):
        raise ValueError(f"kinematics: 'small' or 'finite', got {kinematics!r}")
    if kinematics == "finite":
        from morphogen.fracture.phasefield_finite import FinitePhaseFieldProblem

        problem = FinitePhaseFieldProblem(image, materials, fracture, **options)
    else:
        problem = PhaseFieldProblem(image, materials, fracture, **options)
    return problem.run(path, name=name, fields=fields)
