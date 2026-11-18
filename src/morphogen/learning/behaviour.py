"""Learned laws in the solvers of morphoGen: a material whose ``learned`` names a model
(:class:`LearnedBehaviour`) has its points integrated by the model, along the seam of the
MFront behaviours (points of the kind ``MFRONT``, which the kernels skip; their
integration after them): at a material point, by finite elements and on voxels. The
rest of the solvers is unchanged (the periodic element, Newton's method, the outputs);
the elastic constants of the material remain those of the linear problems and of the
reference medium of the FFT solver (:func:`learned_material` takes those of the initial
tangent of the model).

A step of a point gives the model the strain at its end, the strain and the state of the
model at its start (the internal variables of the point) and the time step: the stress
and the consistent tangent, and the state at its end. The tangent is exact at a material
point; in volume elements it is symmetrized for the conjugate gradients when the laws of
all the phases have symmetric tangents (those of the potential and of the standard model
are), and kept whole for GMRES otherwise (the perceptron, the recurrent network:
:attr:`LearnedLaw.symmetric`). In finite strains, the model of the Green-Lagrange strain
``E`` and the second Piola-Kirchhoff stress ``S`` gives the first Piola-Kirchhoff stress
``P = F S`` and its tangent ``dP/dF = I (x) S + F . dS/dE . dE/dF``.

A learned law depends on the temperature through the thermal expansion of its material
only, as morphoGen's own behaviours do: the model sees the mechanical strain, the strain
less the stress-free thermal strain ``alpha dT`` (``dT`` from the stress-free state, the
expansion tensor in the axes of the material). In finite strains the split is that of
morphoGen's Hencky points (:mod:`morphogen.homogenization.finite`): the mechanical
logarithmic strain ``H_m = 1/2 ln C - alpha dT``, the stress ``T`` conjugate to ``H = 1/2
ln C`` and the free energy per unit reference volume (no factor ``J_th``). The model sees
the Green-Lagrange strain of ``H_m``, ``E_m = (exp(2 H_m) - I)/2`` (a law of ``ln C``
sees ``H_m`` itself), its stress gives ``T = J_m^-T S_m`` and ``K = dT/dH = J_m^-T
(dS_m/dE_m - T : d2H/dE2 (E_m)) J_m^-1`` (``J = dH/dE``, ``J_m`` at ``E_m``), and ``S =
J^T T``, ``dS/dE = J^T K J + T : d2H/dE2`` (:mod:`morphogen.learning.logarithmic`): a law
of ``E`` that is Hencky's elasticity gives the stresses of the Hencky points. For an
isotropic expansion ``E_m = A (E + I/2) A - I/2`` and ``S = A S_m A`` (``A = exp(-alpha
dT)``: the multiplicative split ``F = F_m F_th``, ``F_th = exp(alpha dT)``); for an
anisotropic one that split is not the Hencky points' (the logarithms add only where ``C``
and ``alpha`` commute). Without an expansion, a change of temperature leaves the law
unchanged.

A learned law runs in the axes of its material, which the orientation of its point turns
as it turns crystals (``crystal = g sample``: Bunge angles at a material point, those of
the grain of each point in polycrystals): the model sees the strain ``g eps g^T`` (``g E
g^T`` in finite strains) and its stress and tangent are turned back, ``sigma = g^T
sigma_g g`` and ``C = T^T C_g T`` (``T``: the 6 x 6 matrix of ``g`` on the strains); a law
learned once in the axes of a crystal serves every grain. Without an orientation, or with
``oriented: false``, it runs in the axes of the sample. Order of the components at every
interface: morphoGen's Voigt order 11, 22, 33, 23, 31, 12 (engineering shears of the
strains) in the data, the models, the arrays of the points and the 6 x 6 rotations; the
learned points never pass TFEL's order 12, 13, 23 of the MFront points beside them.

Slots of the internal variables of a point: the plastic strain and the cumulated plastic
strain of the outputs (zero: the model has no plastic strain), the strain (or ``E``) of
the model at the end of the step (in the axes of its material), then the state of the
model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from morphogen.homogenization.behaviours import MFRONT, STATE_P
from morphogen.learning.data import PAIRS, _tensor, _voigt
from morphogen.learning.logarithmic import _measures, _pullback
from morphogen.learning.networks import Model, load_model

_STRAIN = STATE_P + 1
_STATE = _STRAIN + 6
#: The models loaded from their files (path -> (modification time, model))
_LOADED: dict[str, tuple[float, Model]] = {}


@dataclass(frozen=True)
class LearnedBehaviour:
    """A learned law of a material: ``model``, the file of a model of
    :mod:`morphogen.learning` (``.npz``, :meth:`~morphogen.learning.networks.Model.save`)
    or the model itself; ``oriented``: the law in the axes of its material, turned by the
    orientation of its point (that of its grain in a polycrystal) as a crystal is, or
    (false) in the axes of the sample everywhere."""

    model: object
    oriented: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "oriented", bool(self.oriented))
        if isinstance(self.model, (str, Path)):
            object.__setattr__(self, "model", str(self.model))
        elif not isinstance(self.model, Model):
            raise ValueError("learned: the file of a model of morphogen.learning, or the model")
        elif not self.model.ready:
            raise ValueError("learned: the model is not trained")

    @classmethod
    def from_config(cls, spec) -> LearnedBehaviour:
        """The file of a model, or a mapping ``model: <file>`` (and ``oriented``)."""
        if isinstance(spec, LearnedBehaviour):
            return spec
        if isinstance(spec, dict):
            unknown = sorted(set(spec) - {"model", "oriented"})
            if unknown or "model" not in spec:
                raise ValueError(f"learned: a mapping of model (and oriented), got {sorted(spec)}")
            return cls(spec["model"], spec.get("oriented", True))
        return cls(spec)

    def to_config(self) -> dict:
        model = self.model
        path = model.path if isinstance(model, Model) else model
        out = {"model": None if path is None else str(path)}
        if not self.oriented:
            out["oriented"] = False
        return out

    def load(self) -> Model:
        """The model (loaded once from its file while the file does not change)."""
        if isinstance(self.model, Model):
            return self.model
        path = Path(self.model)
        if not path.is_file():
            raise FileNotFoundError(f"learned: no model {path}")
        key, stamp = str(path.resolve()), path.stat().st_mtime
        cached = _LOADED.get(key)
        if cached is None or cached[0] != stamp:
            cached = (stamp, load_model(path))
            _LOADED[key] = cached
        return cached[1]


@dataclass
class LearnedLaw:
    """A learned law loaded for the points of a material: its model and kinematics, and
    the thermal expansion of its material (3 x 3, per unit change of temperature, in the
    axes of the material; None without one)."""

    spec: LearnedBehaviour
    model: Model
    finite: bool
    expansion: np.ndarray | None = None

    @property
    def width(self) -> int:
        """Internal variables of a point in morphoGen's arrays."""
        return _STATE + self.model.n_state

    @property
    def symmetric(self) -> bool:
        """Whether the consistent tangent of the law is symmetric: that of a potential (the
        Hessian of its free energy) and of a standard model (of its two convex
        potentials), not those of the perceptron and of the recurrent network; a law that
        says so itself (``model.symmetric``: those of
        :mod:`morphogen.homogenization.laws`)."""
        own = getattr(self.model, "symmetric", None)
        return self.model.kind in ("potential", "standard") if own is None else bool(own)


def learned_law(spec, *, finite: bool, expansion=None) -> LearnedLaw:
    """The learned law of ``spec`` for the points of a solver, its model checked against
    the kinematics (small strains, or the Green-Lagrange measures in finite strains);
    ``expansion``: the thermal expansion tensor of its material (in its axes)."""
    spec = LearnedBehaviour.from_config(spec)
    model = spec.load()
    wanted = "green-lagrange" if finite else "small"
    if model.measure != wanted:
        kind = "finite" if finite else "small"
        raise ValueError(f"learned: a model of {model.measure} data; the paths in {kind} "
                         f"strains need one of {wanted} data")  # fmt: skip
    if expansion is not None:
        expansion = np.asarray(expansion, dtype=float).reshape(3, 3)
        expansion = 0.5 * (expansion + expansion.T) if np.any(expansion) else None
    return LearnedLaw(spec, model, bool(finite), expansion)


def finite_stress(f, s6, c6) -> tuple[np.ndarray, np.ndarray]:
    """The first Piola-Kirchhoff stresses (n, 9) and their tangents ``dP/dF`` (n, 9, 9;
    tensors row by row) of the gradients ``f`` (n, 3, 3), of the second Piola-Kirchhoff
    stresses ``s6`` (n, 6) and of their tangents with respect to the Green-Lagrange
    strain ``c6`` (n, 6, 6; engineering shears)."""
    n = f.shape[0]
    s = _tensor(s6, engineering=False)
    # dE_a/dF_kL: E_AA = (F_mA F_mA - 1) / 2, 2 E_AB = F_mA F_mB
    de = np.zeros((n, 6, 3, 3))
    for a, (i, j) in enumerate(PAIRS):
        de[:, a, :, i] += f[:, :, j]
        if i != j:
            de[:, a, :, j] += f[:, :, i]
    ds6 = np.einsum("nba,nakl->nbkl", c6, de)
    ds = np.empty((n, 3, 3, 3, 3))
    for b, (i, j) in enumerate(PAIRS):
        ds[:, i, j] = ds6[:, b]
        ds[:, j, i] = ds6[:, b]
    tangent = np.einsum("ik,nlj->nijkl", np.eye(3), s) + np.einsum("nim,nmjkl->nijkl", f, ds)
    return (f @ s).reshape(n, 9), tangent.reshape(n, 9, 9)


def turned(t, s, c) -> tuple[np.ndarray, np.ndarray]:
    """The stresses (n, 6) and tangents (n, 6, 6) of a law in the axes of its material
    (``s``, ``c``) turned back to the axes of the sample by the 6 x 6 matrices ``t`` (n, 6,
    6) of its orientations on the strains (engineering shears): ``T^T s`` and ``T^T c T``,
    the work and the energy unchanged."""
    return np.einsum("nji,nj->ni", t, s), np.einsum("nki,nkl,nlj->nij", t, c, t)


def thermal_split(e, heat, dtemp) -> np.ndarray:
    """The mechanical Green-Lagrange strains (n, 6; engineering shears) of Green-Lagrange
    strains ``e`` (n, 6) heated by ``dtemp`` with the expansions ``heat`` (n, 3, 3), by the
    split of the Hencky points: the Green-Lagrange strain ``E_m = (exp(2 H_m) - I)/2`` of
    the mechanical logarithmic strain ``H_m = 1/2 ln C - dtemp alpha`` (``C = I + 2 E``;
    ``expm1`` of the eigenvalues of ``2 H_m``, exact to round-off near zero); NaN where
    ``C`` is not positive definite."""
    h = _measures(np.ascontiguousarray(e, dtype=float), 1.0)[0]
    h = h - float(dtemp) * _voigt(heat, engineering=True)
    out = np.full_like(h, np.nan)
    ok = np.all(np.isfinite(h), axis=1)
    if ok.any():
        w, v = np.linalg.eigh(_tensor(h[ok], engineering=True))
        out[ok] = _voigt(np.einsum("nij,nj,nkj->nik", v, 0.5 * np.expm1(2.0 * w), v),
                         engineering=True)  # fmt: skip
    return out


def thermal_conjugate(e, em, s, c) -> tuple[np.ndarray, np.ndarray]:
    """The stresses (n, 6) conjugate to the Green-Lagrange strains ``e`` (n, 6) and their
    tangents (n, 6, 6) of a law heated as the Hencky points are, of its stresses ``s`` and
    tangents ``c`` at the mechanical strains ``em`` of :func:`thermal_split`: the stress
    ``T = J_m^-T S_m`` conjugate to the logarithmic strain and its tangent ``K = dT/dH =
    J_m^-T (dS_m/dE_m - T : d2H/dE2 (E_m)) J_m^-1`` (``J = dH/dE``, ``J_m`` at ``em``),
    then ``S = J^T T`` and ``dS/dE = J^T K J + T : d2H/dE2`` at ``e`` (``S : dE = T : dH =
    S_m : dE_m``: the free energy of the law per unit reference volume); the tangent
    symmetrized if that of the law is symmetric to the bit (a potential, a standard
    model)."""
    e, em = (np.ascontiguousarray(a, dtype=float) for a in (e, em))
    s, c = np.asarray(s, dtype=float), np.asarray(c, dtype=float)
    n = e.shape[0]
    jm = _measures(em, 1.0)[1]
    # J_m^-T where C is positive definite (NaN elsewhere, as the stresses: a failed point)
    inverse = np.full((n, 6, 6), np.nan)
    ok = np.all(np.isfinite(jm), axis=(1, 2))
    if ok.any():
        inverse[ok] = np.linalg.inv(np.swapaxes(jm[ok], 1, 2))
    t = np.ascontiguousarray(np.einsum("nij,nj->ni", inverse, s))
    geometric = _pullback(em, t, np.zeros((n, 6, 6)), 1.0, False)[1]
    k = np.ascontiguousarray(inverse @ (c - geometric) @ np.swapaxes(inverse, 1, 2))
    return _pullback(e, t, k, 1.0, bool(np.array_equal(c, np.swapaxes(c, 1, 2))))


def _model_step(model, e, e_old, h_old, dt, dtemp=0.0):
    """The step of ``model`` at the points of finite strains ``e`` (n, 6); NaN at the others,
    where the model is not called: failed points (heated where ``C`` is not positive
    definite, as a Hencky point fails, or of a gradient that is not finite). A model whose
    law depends on the temperature (``model.thermal``: the shifted times of a
    viscoelastic law) also gets the change of temperature ``dtemp``."""
    extra = (dtemp,) if getattr(model, "thermal", False) else ()
    ok = np.all(np.isfinite(e), axis=1)
    if ok.all():
        return model.step(e, e_old, h_old, dt, *extra)
    n = e.shape[0]
    s, c = np.full((n, 6), np.nan), np.full((n, 6, 6), np.nan)
    h = np.full((n, model.n_state), np.nan)
    if ok.any():
        s[ok], c[ok], h[ok] = model.step(e[ok], e_old[ok], h_old[ok], dt, *extra)
    return s, c, h


class LearnedPoints:
    """The integration points of the materials with a learned law (those of kind
    ``MFRONT`` among ``kind``, of the laws ``pmat``), integrated by their models in the
    axes of their materials (``rotations``: the orientation of each row ``point_mat`` of
    the materials, ``crystal = g sample``; the laws not oriented in the axes of the
    sample); the interface of :class:`~morphogen.homogenization.mfront.MFrontPoints`."""

    def __init__(self, laws, kind, pmat, point_mat, rotations, finite: bool, threads=None,
                 symmetric: bool = True):  # fmt: skip
        from morphogen.learning.training import voigt_rotations

        self.finite = bool(finite)
        self.symmetric = bool(symmetric)
        self.points = np.flatnonzero(np.asarray(kind) == MFRONT)
        pmat = np.asarray(pmat)[self.points]
        point_mat = np.asarray(point_mat)[self.points]
        self.groups = []
        for c, law in enumerate(laws):
            if law.finite != self.finite:
                raise ValueError("learned: the law and the solver have other kinematics")
            local = np.flatnonzero(pmat == c)
            if not local.size:
                continue
            # the 6 x 6 matrices of the orientations of the points on the strains, if any
            t = None
            g = np.repeat(np.eye(3)[None], local.size, axis=0)
            if rotations is not None and len(rotations):
                g = np.asarray(rotations, dtype=float)[point_mat[local]]
            turn = law.spec.oriented and not np.allclose(g, np.eye(3), rtol=0.0, atol=1e-14)
            if turn:
                t = voigt_rotations(g)[0]
            # the expansions in the axes of the model: its material's, or (a law in the axes
            # of the sample) turned by the orientation of the point as the kernels turn it
            heat = None
            if law.expansion is not None:
                heat = np.repeat(law.expansion[None], local.size, axis=0)
                if not law.spec.oriented:
                    heat = np.einsum("npi,pq,nqj->nij", g, law.expansion, g)
            self.groups.append((law, local, t, heat))

    def integrate(self, grads, dtemp, dt, old, new, stress, tangent) -> int:
        """Integration of the points of learned laws among the arrays of all the points;
        returns the number of points whose stress is not finite."""
        idx = self.points
        out = [new[idx], stress[idx], tangent[idx]]
        failed = self.integrate_compact(grads[idx], dtemp, dt, old[idx], *out)
        new[idx], stress[idx], tangent[idx] = out
        return failed

    def integrate_compact(self, grads, dtemp, dt, old, new, stress, tangent) -> int:
        """The same for the arrays of these points only (in the order of :attr:`points`)."""
        failed = 0
        dtemp = float(dtemp)
        for law, local, t, heat in self.groups:
            o = old[local]
            n_state = law.model.n_state
            if self.finite:  # E = (H + H^T + H^T H) / 2: no cancellation of 1
                hg = np.asarray(grads[local]).reshape(-1, 3, 3)
                f = np.eye(3) + hg
                gl = hg + np.swapaxes(hg, 1, 2) + np.einsum("nki,nkj->nij", hg, hg)
                e = _voigt(0.5 * gl, engineering=True)
            else:
                e = np.asarray(grads[local])
            if t is not None:  # in the axes of the material: g eps g^T (g E g^T)
                e = np.einsum("nij,nj->ni", t, e)
            total = None
            if heat is not None and dtemp != 0.0:  # the mechanical strain
                if self.finite:  # of the Hencky points: 1/2 ln C - alpha dT
                    total, e = e, thermal_split(e, heat, dtemp)
                else:
                    e = e - dtemp * _voigt(heat, engineering=True)
            s, c, h = _model_step(law.model, e, o[:, _STRAIN:_STATE],
                                  o[:, _STATE : _STATE + n_state], dt, dtemp)  # fmt: skip
            if total is not None:  # S = J^T T, T = J_m^-T S_m: conjugate to E
                s, c = thermal_conjugate(total, e, s, c)
            if t is not None:
                s, c = turned(t, s, c)
            if self.finite:
                s, c = finite_stress(f, s, c)
            if self.symmetric:
                c = 0.5 * (c + np.swapaxes(c, 1, 2))
            bad = ~(np.all(np.isfinite(s), axis=1) & np.all(np.isfinite(c), axis=(1, 2)))
            failed += int(bad.sum())
            stress[local] = s
            tangent[local] = c
            state = np.zeros((local.size, new.shape[1]))
            state[:, _STRAIN:_STATE] = e
            state[:, _STATE : _STATE + n_state] = h
            if hasattr(law.model, "scalar"):  # the variable p of the outputs (a damage)
                state[:, STATE_P] = law.model.scalar(h)
            new[local] = state
        return failed


def isotropic_constants(stiffness) -> tuple[float, float]:
    """Young's modulus and Poisson's ratio of the isotropic projection (Voigt's means of
    the bulk and shear moduli) of a 6 x 6 stiffness (engineering shears)."""
    c = np.asarray(stiffness, dtype=float)
    normal = np.trace(c[:3, :3])
    off = c[0, 1] + c[0, 2] + c[1, 2]
    bulk = (normal + 2.0 * off) / 9.0
    shear = (normal - off + 3.0 * np.trace(c[3:, 3:])) / 15.0
    if not (bulk > 0 and shear > 0):
        raise ValueError("the stiffness has no positive isotropic part")
    young = 9.0 * bulk * shear / (3.0 * bulk + shear)
    poisson = (3.0 * bulk - 2.0 * shear) / (2.0 * (3.0 * bulk + shear))
    return float(young), float(min(max(poisson, -0.99), 0.49))


def initial_tangent(model: Model) -> np.ndarray:
    """The tangent of a model at the natural state (of its first step from it)."""
    zero = np.zeros((1, 6))
    return model.step(zero, zero, np.zeros((1, model.n_state)), 1.0)[1][0]


def learned_material(
    model, *, young: float | None = None, poisson: float | None = None, expansion=None
):
    """A :class:`~morphogen.homogenization.materials.Material` of the learned law of
    ``model`` (a model, its file, or a :class:`LearnedBehaviour`), its elastic constants
    (the linear problems and the reference media) those given or of the isotropic
    projection of its initial tangent; ``expansion``: its thermal expansion (one
    coefficient, principal values or a tensor in the axes of the material), the
    stress-free thermal strain taken out of the strain the model sees."""
    from morphogen.homogenization.materials import Material

    spec = LearnedBehaviour.from_config(model)
    if young is None or poisson is None:
        e, nu = isotropic_constants(initial_tangent(spec.load()))
        young = e if young is None else young
        poisson = nu if poisson is None else poisson
    if not (math.isfinite(young) and young > 0):
        raise ValueError(f"young: positive, got {young}")
    return Material(young=float(young), poisson=float(poisson), expansion=expansion,
                    learned=spec)  # fmt: skip
