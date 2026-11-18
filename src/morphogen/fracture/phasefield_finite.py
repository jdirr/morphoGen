"""Brittle fracture of voxel images by the phase-field method in finite strains: the cell
of :mod:`morphogen.fracture.phasefield` whose phases stretch much before they break
(rubbers, foams, soft tissues).

The phases are hyperelastic (the strain energies of :mod:`morphogen.homogenization.laws`)
or elastic (the energy of Hencky of the logarithmic strain, the elasticity of every
finite-strain solver of the package), and the energy of the cell is that of the small
strains with the energy of the deformation gradient, over the reference configuration,

    E(u, d) = int g(d) psi(F) dV0 + int G_c gamma(d, grad0 d) dV0,   g(d) = (1 - d)^2 + k:

the damage and its gradient are Lagrangian fields, and the toughness is an energy per unit
area of the reference configuration (Miehe & Schänzel 2014). The crack densities
(``AT2``, ``AT1``), the regularization length and the irreversibility are those of the
small strains (:class:`~morphogen.fracture.phasefield.PhaseField`).

A step of a loading path is solved by alternate minimization, as in small strains: until
the damage no longer moves,

1. the displacement at fixed damage: the periodic problem of the voxels of the nonlinear
   FFT solver (:class:`~morphogen.homogenization.fft_nonlinear.NonlinearFFTProblem`: total
   Lagrangian, F-bar voxels, Newton's method, the macroscopic components prescribed as
   stresses among its unknowns), the nominal stress and the tangent of each point times the
   degradation of its voxel (the stress is degraded as a whole, as in small strains: the
   hybrid formulation), the Fourier preconditioner of its Krylov solves scaled by the
   degradation of the nodes;
2. the damage of the driving energy ``psi+`` of the state reached: the problem of the
   small strains (the same equation, its bound or its history field).

The solver of the paths divides a step where Newton's method fails (a cell that breaks at
once): the damage then starts again from that of the last converged step. Where a crack
runs at a fixed load, the F-bar voxels of its band, sheared far, may turn inside out under
Newton's corrections from an equilibrium where none is near it, and dividing the step only
replays the run of the crack: the compressible solids, which do not lock, are better solved
on fully integrated voxels (``integration="full"``).

The energy that drives the damage (``split``) is all the energy (``"none"``); or its
isochoric part and its volumetric part in expansion, ``J > 1`` (``"volumetric"``: the split
of Amor et al. between the two terms of the strain energies, so that a hydrostatic
compression does not break); or, for the elastic phases only, the tensile part of the
energy of Hencky in the principal logarithmic strains (``"spectral"``, as in small
strains: ``lambda <tr e>+^2 / 2 + mu sum <e_i>+^2``). The compressible energies that are
not split (Blatz-Ko, hyperfoam) drive the damage as a whole.
"""

from __future__ import annotations

import logging

import numpy as np

from morphogen.fracture.phasefield import (
    PhaseField,
    PhaseFieldProblem,
    PhaseFieldResult,
    _to_nodes,
    _to_voxels,
)
from morphogen.homogenization import finite as fin
from morphogen.homogenization.fft_nonlinear import NonlinearFFTProblem, _VoxelPaths
from morphogen.homogenization.materials import Material
from morphogen.homogenization.paths import LoadingPath
from morphogen.image import VoxelImage

log = logging.getLogger("morphogen")

__all__ = ["FinitePhaseFieldProblem", "green_lagrange"]


def green_lagrange(grads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The Green-Lagrange strains (n, 6; engineering shears) and the Jacobians ``det F``
    of displacement gradients (n, 9; row by row)."""
    h = np.asarray(grads, dtype=float).reshape(-1, 3, 3)
    e = 0.5 * (h + np.swapaxes(h, 1, 2) + np.swapaxes(h, 1, 2) @ h)
    out = fin._voigt(e)
    out[:, 3:] *= 2.0
    return out, np.linalg.det(np.eye(3) + h)


class _DegradedVoxels(_VoxelPaths):
    """The voxels of a cell whose points are degraded: their stresses and tangents times
    the degradation of their voxel, and the preconditioner of the Krylov solves scaled by
    that of the nodes (the stiffness of a damaged cell is that of the sound one times
    ``g``, which the Fourier inverse of the sound reference medium ignores)."""

    def _buffers(self, n_pts: int, nc: int) -> None:
        super()._buffers(n_pts, nc)
        reference, problem = self.precondition, self.problem

        def scaled(r):
            w = problem.node_weight
            return w * reference(w * r)

        self.precondition = scaled

    def integrate(self, grads, dtemp, dt, old, new, stress, tangent, guess, ratio) -> int:
        failed = super().integrate(grads, dtemp, dt, old, new, stress, tangent, guess, ratio)
        g = self.problem.point_scale
        stress *= g[:, None]
        tangent *= g[:, None, None]
        return failed


class _DegradedFFTProblem(NonlinearFFTProblem):
    """The loading paths of a voxel image whose voxels are degraded by ``point_scale``
    (one value per integration point) and ``node_weight`` (one per nodal unknown), the
    damage of its ``owner`` settled on each state reached (the hooks of
    :meth:`~morphogen.homogenization.nonlinear._PathSolver.run`)."""

    def __init__(self, image, materials, owner, **options):
        super().__init__(image, materials, "periodic", kinematics="finite", **options)
        self.owner = owner
        self.point_scale = np.ones(0 if self.empty else self.kind.size)
        self.node_weight = np.ones(3 * int(np.prod(self.shape)))

    def _discretization(self) -> _VoxelPaths:
        return _DegradedVoxels(self)

    def _settle(self, state) -> bool:
        return self.owner._settle(state)

    def _rollback(self) -> None:
        self.owner._rollback()


class FinitePhaseFieldProblem:
    """The brittle fracture of a periodic voxel image in finite strains (see the module):
    ``materials`` of its phases, those that break hyperelastic or elastic (isotropic);
    ``fracture`` (:class:`~morphogen.fracture.phasefield.PhaseField`, or a mapping of its
    parameters); ``initial``: the voxels broken from the start. ``tol``: the relative
    residual of the linear solves; ``staggered_tol``: the largest change of the damage
    that ends the alternate minimization of a step, of at most ``max_staggered``
    iterations; ``nonlinear``: the other options of
    :class:`~morphogen.homogenization.fft_nonlinear.NonlinearFFTProblem` (``newton_tol``,
    ``max_newton``, ``max_cuts``, ``integration``, ``tangent``)."""

    def __init__(self, image: VoxelImage, materials: dict[int, Material], fracture, *,
                 initial=None, tol: float = 1e-7, staggered_tol: float = 1e-3,
                 max_staggered: int = 200, **nonlinear):  # fmt: skip
        fracture = PhaseField.from_config(fracture)
        self.staggered_tol, self.max_staggered = float(staggered_tol), int(max_staggered)
        for name in ("kinematics", "device", "grains", "orientations"):
            if name in nonlinear:
                raise ValueError(
                    f"phase field in finite strains: no option {name!r} (isotropic phases, "
                    "on the CPU)"
                )
        #: the damage problem (that of the small strains) and the state of the damage
        self.damage = PhaseFieldProblem(image, materials, fracture, initial=initial, tol=tol)
        self.fracture = fracture
        self.image, self.shape = image, image.shape
        self.paths = _DegradedFFTProblem(image, materials, self, tol=tol, **nonlinear)
        self.materials = materials
        self._energies = self._energy_laws()
        #: the damage and the history of the last converged step, the alternate
        #: minimizations of the step under way and what the last of them found
        self._committed = (self.damage.d, self.damage.history)
        self._staggered, self._trial = 0, (self.damage.history, 0.0, 0.0)
        self._degrade()

    # -- the energies of the points ------------------------------------------------------

    def _energy_laws(self) -> list:
        """For each phase whose energy is known (elastic or hyperelastic), its points and
        the function of their displacement gradients that gives the driving energy and
        the whole energy; the phases that break must be among them."""
        from morphogen.homogenization.laws import analytic_model

        paths, split = self.paths, self.fracture.split
        phase = np.asarray(self.image.data).ravel()[paths.elem]
        out = []
        for ph in np.unique(phase):
            m = self.materials[int(ph)]
            breaks = self.fracture.of(int(ph)) is not None
            points = np.flatnonzero(np.repeat(phase == ph, paths.ng))
            if m.hyperelasticity is not None and m.viscoelasticity is None:
                if breaks and split == "spectral":
                    raise ValueError(
                        f"phase {int(ph)}: the spectral split is that of the logarithmic "
                        "strain of elastic phases; a hyperelastic phase takes split "
                        "'volumetric' (its isochoric energy, and the volumetric one in "
                        "expansion) or 'none'"
                    )
                out.append((points, _hyperelastic_energy(analytic_model(m, finite=True), split)))
            elif not m.nonlinear and m.isotropic and m.stiffness is None:
                out.append((points, _hencky_energy(*m.lame, split)))
            elif breaks:
                raise ValueError(
                    f"phase {int(ph)}: in finite strains the phases that break are elastic "
                    "(isotropic) or hyperelastic"
                )
        return out

    def _energy(self, grads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The driving energy ``psi+`` and the whole energy of the voxels (the means of
        their points; zero in the voids and in the phases whose energy is not known)."""
        paths = self.paths
        plus, whole = np.zeros(paths.kind.size), np.zeros(paths.kind.size)
        for points, law in self._energies:
            plus[points], whole[points] = law(grads[points])
        out = []
        for values in (plus, whole):
            voxels = np.zeros(int(np.prod(self.shape)))
            voxels[paths.elem] = values.reshape(-1, paths.ng).mean(axis=1)
            out.append(voxels.reshape(self.shape))
        return out[0], out[1]

    def _degrade(self) -> np.ndarray:
        """The degradation of the voxels for the damage reached, given to the points of
        the mechanical problem and to its preconditioner."""
        g = self.damage.degradation()
        paths = self.paths
        paths.point_scale = np.ascontiguousarray(np.repeat(g.ravel()[paths.elem], paths.ng))
        nodal = _to_nodes(g) / 8.0
        paths.node_weight = np.repeat(1.0 / np.sqrt(nodal.ravel()), 3)
        return g

    # -- the damage of a state: the hooks of the solver of the paths ----------------------

    def _settle(self, state) -> bool:
        """The damage of the driving energy of the state reached, from that of the last
        converged step (its bound); true while it moves: the step is solved again."""
        dmg = self.damage
        plus, whole = self._energy(state.grads)
        history = plus
        if self.fracture.irreversibility == "history":
            history = np.maximum(self._committed[1], plus)
        g = dmg.degradation()  # of the damage the state was solved with
        elastic = float(np.sum(g * whole)) * dmg.voxel / dmg.volume
        new = dmg._damage(history, self._committed[0])
        change = float(np.abs(new - dmg.d).max())
        dmg.d = new
        self._trial = (history, elastic, change)
        self._degrade()
        self._staggered += 1
        return change >= self.staggered_tol and self._staggered < self.max_staggered

    def _rollback(self) -> None:
        """A step divided: the damage of the last converged step again."""
        self.damage.d = self._committed[0]
        self._staggered = 0
        self._degrade()

    # -- loading paths -------------------------------------------------------------------

    def run(self, path, *, name: str = "path", fields=False, monitor=None) -> PhaseFieldResult:
        """Follow a loading path (:class:`~morphogen.homogenization.paths.LoadingPath`, or
        a mapping for it; nominal strains and stresses, as for the loading paths in finite
        strains): :class:`~morphogen.fracture.phasefield.PhaseFieldResult` with the nominal
        and true measures of the path. ``fields``: true for the damage, strain and stress
        of the voxels at the end of each segment of the path, ``"steps"`` at the end of
        each step, or a list of times of its steps. ``monitor(time, records so far)``
        after each step; a true value ends the path there."""
        path = LoadingPath.from_spec(path)
        if np.any(np.asarray(path.temperature) != 0):
            raise ValueError("phase field in finite strains: no change of temperature")
        dmg, paths = self.damage, self.paths
        wanted = paths._field_times(path, fields)
        volume = dmg.volume
        rec = {k: [] for k in ("t", "d", "gamma", "elastic", "fracture", "its", "phases")}
        kept: dict = {}

        def record(t: float, elastic: float) -> None:
            rec["t"].append(float(t))
            rec["d"].append(dmg._mean_damage())
            rec["gamma"].append(dmg.crack_surface() / volume)
            rec["elastic"].append(elastic)
            rec["fracture"].append(dmg.crack_surface(weight=dmg.gc) / volume)
            means = dmg._phase_means(None, None)
            rec["phases"].append({ph: v["plastic"] for ph, v in means.items()})

        record(float(path.times[0]), 0.0)

        def after(state) -> bool:
            """A converged step: its damage and its history kept."""
            history, elastic, change = self._trial
            if change >= self.staggered_tol:
                log.warning(
                    "phase field: the alternate minimization of the step to t = %g stopped at "
                    "a change of the damage of %.2e", state.time, change
                )  # fmt: skip
            dmg.history = history
            self._committed = (dmg.d, history)
            rec["its"].append(self._staggered)
            self._staggered = 0
            record(state.time, elastic)
            if wanted.size and np.any(np.isclose(state.time, wanted, rtol=1e-12, atol=0.0)):
                kept[float(state.time)] = np.clip(_to_voxels(dmg.d), 0.0, 1.0)
            return bool(monitor is not None and monitor(float(state.time), rec))

        res = paths.run(path, name=name, fields=fields, monitor=after)
        for t, values in res.fields.items():  # the damage with the fields of the voxels
            values["damage"] = values["p"] = kept[float(t)]
        phases = {}
        for ph, means in res.phases.items():
            phases[ph] = dict(means)
            if ph in rec["phases"][0]:
                phases[ph]["plastic"] = np.array([step[ph] for step in rec["phases"]])
        info = dict(res.info)
        info["newton_iterations"] = int(np.sum(res.iterations))
        return PhaseFieldResult(
            name=res.name,
            bc=res.bc,
            path=res.path,
            times=res.times,
            strain=res.strain,
            stress=res.stress,
            temperature=res.temperature,
            plastic=np.array(rec["d"]),
            phases=phases,
            iterations=np.array(rec["its"], dtype=int),
            info=info,
            fields=res.fields,
            support=res.support,
            kinematics="finite",
            cauchy=res.cauchy,
            log_strain=res.log_strain,
            damage=np.array(rec["d"]),
            crack_surface=np.array(rec["gamma"]),
            elastic_energy=np.array(rec["elastic"]),
            fracture_energy=np.array(rec["fracture"]),
        )


def _hyperelastic_energy(law, split: str):
    """The driving and whole energies of the points of a hyperelastic law (its isochoric
    and volumetric parts) from their displacement gradients."""

    def energy(grads):
        e, j = green_lagrange(grads)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            iso, vol = law._parts(e)[4]
        iso, vol = np.nan_to_num(iso), np.nan_to_num(vol)
        whole = iso + vol
        if split == "none":
            return whole, whole
        return iso + np.where(j >= 1.0, vol, 0.0), whole

    return energy


def _hencky_energy(lam: float, mu: float, split: str):
    """The driving and whole energies of the points of an elastic phase in finite strains:
    the energy of Hencky ``lambda (tr e)^2 / 2 + mu e : e`` of the logarithmic strain."""
    bulk = lam + 2.0 * mu / 3.0

    def energy(grads):
        e = fin.log_strain(grads)
        e[:, 3:] *= 0.5  # tensor shears
        tr = e[:, :3].sum(axis=1)
        dev2 = np.sum((e[:, :3] - tr[:, None] / 3.0) ** 2, axis=1) + 2.0 * np.sum(
            e[:, 3:] ** 2, axis=1
        )
        whole = 0.5 * bulk * tr**2 + mu * dev2
        if split == "none":
            return whole, whole
        if split == "volumetric":
            return 0.5 * bulk * np.maximum(tr, 0.0) ** 2 + mu * dev2, whole
        t = np.empty((len(e), 3, 3))
        for k, (a, b) in enumerate(((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))):
            t[:, a, b] = t[:, b, a] = e[:, k]
        principal = np.linalg.eigvalsh(t)
        plus = 0.5 * lam * np.maximum(tr, 0.0) ** 2 + mu * np.sum(
            np.maximum(principal, 0.0) ** 2, axis=1
        )
        return plus, whole

    return energy
