"""The nonuniform transformation field analysis (NTFA; Michel & Suquet 2003, 2004): a
reduced model of the nonlinear response of a periodic voxel microstructure.

The plastic strain of each plastic phase ``r`` is sought as a combination of a few
modes, ``eps_p(x) = sum_k xi_k mu_k(x)`` (``k`` among the modes of ``r``, zero outside
it). Offline (:func:`train_ntfa`), full-field simulations along training paths (the
voxel solver of :class:`~morphogen.homogenization.fft_nonlinear.NonlinearFFTProblem`,
full integration) give snapshots of the plastic strain fields, and the modes are their
proper orthogonal decomposition phase by phase, orthogonal and normalized so that
``<mu_k : mu_k>_r = 3/2``. Linear elastic solves on the same voxels give the strain
fields ``A_j`` of the unit macroscopic strains and ``D_k`` of the modes taken as
eigenstrains, so that ``eps(x) = A(x) E + sum_k D_k(x) xi_k`` and, with
``sigma = C (eps - eps_p)``, the macroscopic stress and the reduced (generalized)
stresses ``tau_k = <sigma : mu_k>_r`` are affine in ``E`` and ``xi``:

    Sigma = L E + sum_k rho_k xi_k,     tau = a E + d xi.

Online (:meth:`NTFAModel.run`), the reduced variables follow the law of their phase
written for them (the von Mises model of Lemaitre & Chaboche with isotropic hardening):
``f_r = ||tau_r|| - R0 - R(p_r)``, ``dxi_r = dp_r tau_r / ||tau_r||``, ``p_r`` the
cumulated reduced plastic strain, with the consistency condition ``f_r = 0``
(rate-independent plasticity) or Norton's flow ``dp_r/dt = <f_r / K>^N``; for one
uniform mode this is exactly the law of the phase. Each step is integrated implicitly
(backward Euler) by Newton iterations on a few unknowns, with strain, stress or mixed
control, without any field computation.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from morphogen.homogenization import kernels
from morphogen.homogenization.behaviours import Plasticity
from morphogen.homogenization.fft import FFTElasticProblem
from morphogen.homogenization.materials import Material
from morphogen.homogenization.paths import LoadingPath, PathResult
from morphogen.image import VoxelImage

#: Tensor to engineering strain components (Voigt 11 22 33 23 31 12).
SHEAR = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
#: Weights of the double contraction of two engineering strains.
WEIGHTS = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])


def _contract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a : b`` of engineering strain fields (last axis: Voigt)."""
    return np.sum(a * b * WEIGHTS, axis=-1)


@dataclasses.dataclass
class NTFAModel:
    """Reduced model of a microstructure: the modes of its plastic phases and the
    affine relations ``Sigma = L E + rho^T xi`` and ``tau = a E + d xi`` (Voigt,
    engineering strains)."""

    stiffness: np.ndarray  # L, 6 x 6
    rho: np.ndarray  # modes x 6
    a: np.ndarray  # modes x 6
    d: np.ndarray  # modes x modes
    mode_phase: np.ndarray  # the phase of each mode
    plasticity: dict  # phase -> Plasticity
    fractions: dict  # phase -> volume fraction
    energy: dict = dataclasses.field(default_factory=dict)  # phase -> captured POD energy
    modes: np.ndarray | None = dataclasses.field(default=None, repr=False)  # modes x voxels x 6
    shape: tuple = ()
    #: for the local fields: the strain fields of the unit strains (6 x voxels x 6) and of
    #: the modes (modes x voxels x 6), the stiffness of each voxel
    #: the mean stress and strain of each phase: ``phase -> (stress of E (6 x 6), of xi
    #: (modes x 6), strain of E, of xi)``
    phase_means: dict = dataclasses.field(default_factory=dict, repr=False)
    unit_fields: np.ndarray | None = dataclasses.field(default=None, repr=False)
    mode_fields: np.ndarray | None = dataclasses.field(default=None, repr=False)
    voxel_stiffness: np.ndarray | None = dataclasses.field(default=None, repr=False)

    @property
    def n_modes(self) -> int:
        return len(self.mode_phase)

    def plastic_strain(self, xi) -> np.ndarray:
        """The plastic strain field (voxels x 6, engineering) of the reduced variables."""
        if self.modes is None:
            raise ValueError("the modes were not kept (train_ntfa(..., keep_modes=True))")
        return np.tensordot(np.asarray(xi, dtype=float), self.modes, axes=1)

    # -- online integration ---------------------------------------------------------------
    def _phase_slices(self):
        return {r: np.flatnonzero(self.mode_phase == r) for r in self.plasticity}

    def run(self, path, *, name: str = "ntfa") -> PathResult:
        """Macroscopic response along a loading path (:class:`LoadingPath`, or a mapping
        for :meth:`LoadingPath.from_spec`); strain, stress or mixed control, no change
        of temperature."""
        path = LoadingPath.from_spec(path)
        if np.any(np.asarray(path.temperature) != 0):
            raise ValueError("NTFA: no change of temperature")
        t0 = time.perf_counter()
        controls = path.strain_controlled
        free = np.flatnonzero(~controls)
        groups = self._phase_slices()
        xi = np.zeros(self.n_modes)
        p = dict.fromkeys(self.plasticity, 0.0)
        x = {
            r: [np.zeros(groups[r].size) for _ in law.kinematic]
            for r, law in self.plasticity.items()
        }
        e = np.zeros(6)
        times, strains, stresses, plastics, iterations = (
            [path.times[0]],
            [e.copy()],
            [np.zeros(6)],
            [0.0],
            [],
        )
        phase_p = {r: [0.0] for r in self.plasticity}
        phase_states = [(e.copy(), xi.copy())]
        t_old = path.times[0]
        for t in path.schedule():
            values, _ = path.at(t)
            e, xi, p, x, its = self._step(values, controls, free, e, xi, p, x, t - t_old, groups)
            t_old = t
            times.append(t)
            strains.append(e.copy())
            stresses.append(self.stiffness @ e + self.rho.T @ xi)
            plastics.append(sum(self.fractions[r] * p[r] for r in self.plasticity))
            for r in self.plasticity:
                phase_p[r].append(p[r])
            phase_states.append((e.copy(), xi.copy()))
            iterations.append(its)
        phases = {}
        for r, (ls, rs, le, rd) in self.phase_means.items():
            phases[r] = {
                "stress": np.array([ls @ ee + rs.T @ xx for ee, xx in phase_states]),
                "strain": np.array([le @ ee + rd.T @ xx for ee, xx in phase_states]),
                "plastic": np.asarray(phase_p.get(r, [0.0] * len(phase_states))),
            }
        return PathResult(
            name=name,
            bc="ntfa",
            path=path,
            times=np.asarray(times),
            strain=np.asarray(strains),
            stress=np.asarray(stresses),
            temperature=np.zeros(len(times)),
            plastic=np.asarray(plastics),
            phases=phases,
            iterations=np.asarray(iterations, dtype=int),
            info={"solver": "ntfa", "modes": self.n_modes, "time_s": time.perf_counter() - t0},
        )

    def _step(self, values, controls, free, e_old, xi_old, p_old, x_old, dt, groups):
        """One backward Euler step: the reduced variables, the cumulated reduced plastic
        strains, the reduced backstresses and the free macroscopic strains, by Newton's
        method (analytic Jacobian, backtracking) over the active plastic phases, the set
        of which is corrected until consistent. ``x_old``: phase -> list of the reduced
        backstresses of its kinematic terms (vectors over its modes)."""
        m, nf = self.n_modes, free.size
        known = np.where(controls, values, 0.0)
        y_ref = max(law.yield_stress for law in self.plasticity.values()) or 1.0
        e_ref = y_ref / float(np.abs(self.stiffness).max())

        def split(y, active):
            na = len(active)
            return y[:m], y[m : m + na], y[m + na :]

        def increment(law, q):
            """``dp`` and ``d dp / dq`` of the unknown ``q`` of a phase: ``dp`` itself, or
            the viscous overstress ``K (dp/dt)^(1/N)``."""
            if not law.viscous:
                return q, 1.0
            qq = max(q, 0.0)
            ratio = qq / law.viscosity
            return dt * ratio**law.exponent, dt * law.exponent * ratio ** (
                law.exponent - 1.0
            ) / law.viscosity

        def backstress(r, dxi_r, dp):
            """Armstrong-Frederick backstresses at the end of the step, their sum, and the
            derivatives of the sum by the increments of the reduced variables (a scalar
            times the identity) and by ``dp``."""
            terms, total = [], np.zeros(dxi_r.size)
            slope, by_dp = 0.0, np.zeros(dxi_r.size)
            for (c, d_k), x0 in zip(self.plasticity[r].kinematic, x_old[r], strict=True):
                factor = 1.0 / (1.0 + d_k * dp)
                x = (x0 + c * dxi_r) * factor
                terms.append(x)
                total = total + x
                slope += c * factor
                by_dp = by_dp - d_k * factor * x
            return terms, total, slope, by_dp

        def system(y, active):
            dxi, q, ef = split(y, active)
            e = known.copy()
            e[free] = ef
            xi = xi_old + dxi
            tau = self.a @ e + self.d @ xi
            n_a = len(active)
            res = np.zeros(m + n_a + nf)
            jac = np.zeros((m + n_a + nf, m + n_a + nf))
            res[:m] = dxi
            jac[:m, :m] = np.eye(m)
            for i, r in enumerate(active):
                idx = groups[r]
                law = self.plasticity[r]
                dp, ddp = increment(law, q[i])
                _, x, slope, x_dp = backstress(r, dxi[idx], dp)
                z = tau[idx] - x
                norm = float(np.linalg.norm(z))
                n = z / norm
                dn = (np.eye(idx.size) - np.outer(n, n)) / norm
                dz = self.d[idx].copy()  # d z / d dxi
                dz[:, idx] -= slope * np.eye(idx.size)
                dz_dq = -x_dp * ddp  # d z / d q
                res[idx] = dxi[idx] - dp * n
                jac[np.ix_(idx, np.arange(m))] -= dp * dn @ dz
                jac[idx, m + i] = -ddp * n - dp * dn @ dz_dq
                if nf:
                    jac[np.ix_(idx, m + n_a + np.arange(nf))] = -dp * dn @ self.a[np.ix_(idx, free)]
                p = p_old[r] + dp
                row = m + i
                visc = q[i] if law.viscous else 0.0
                res[row] = (norm - law.yield_stress - _hardening(law, p) - visc) / y_ref
                jac[row, :m] = n @ dz / y_ref
                jac[row, m + i] = (
                    n @ dz_dq - _slope(law, p) * ddp - (1.0 if law.viscous else 0.0)
                ) / y_ref
                if nf:
                    jac[row, m + n_a :] = n @ self.a[np.ix_(idx, free)] / y_ref
            if nf:
                sigma = self.stiffness @ e + self.rho.T @ xi
                rows = m + n_a + np.arange(nf)
                res[rows] = (sigma[free] - values[free]) / y_ref
                jac[np.ix_(rows, np.arange(m))] = self.rho[:, free].T / y_ref
                jac[np.ix_(rows, rows)] = self.stiffness[np.ix_(free, free)] / y_ref
            res[:m] /= e_ref
            jac[:m] /= e_ref
            return res, jac

        def newton(y, active):
            res, jac = system(y, active)
            norm = float(np.linalg.norm(res))
            for it in range(100):
                if norm <= 1e-11:
                    return y, it
                step = np.linalg.solve(jac, -res)
                t = 1.0
                while True:
                    trial = y + t * step
                    r_t, j_t = system(trial, active)
                    n_t = float(np.linalg.norm(r_t))
                    if n_t < (1.0 - 1e-4 * t) * norm or t < 1e-6:
                        break
                    t *= 0.5
                y, res, jac, norm = trial, r_t, j_t, n_t
            raise RuntimeError(f"NTFA: Newton's method did not converge (residual {norm:.2e})")

        # elastic trial: frozen plastic strains, the free macroscopic strains solved
        ef = e_old[free].copy()
        if nf:
            rhs = values[free] - (self.stiffness @ known + self.rho.T @ xi_old)[free]
            ef = np.linalg.solve(self.stiffness[np.ix_(free, free)], rhs)

        def overstress(e, xi, r):
            law = self.plasticity[r]
            tau = self.a @ e + self.d @ xi
            x = sum(x_old[r]) if x_old[r] else 0.0
            return (
                float(np.linalg.norm(tau[groups[r]] - x))
                - law.yield_stress
                - _hardening(law, p_old[r])
            )

        e = known.copy()
        e[free] = ef
        active = [r for r in groups if overstress(e, xi_old, r) > 1e-10 * y_ref]
        iterations = 0
        for _ in range(2 * len(groups) + 2):
            y = np.concatenate([np.zeros(m), np.zeros(len(active)), ef])
            if active:
                y, its = newton(y, active)
                iterations += its
            dxi, q, ef_new = split(y, active)
            e = known.copy()
            e[free] = ef_new
            xi = xi_old + dxi
            changed = False
            for i, r in enumerate(list(active)):
                if q[i] < 0.0:  # unloading: elastic
                    active.remove(r)
                    changed = True
            for r in groups:
                if r not in active and overstress(e, xi, r) > 1e-8 * y_ref:
                    active.append(r)
                    changed = True
            if not changed:
                p_new, x_new = dict(p_old), dict(x_old)
                for i, r in enumerate(active):
                    dp = increment(self.plasticity[r], q[i])[0]
                    p_new[r] = p_old[r] + dp
                    x_new[r] = backstress(r, dxi[groups[r]], dp)[0]
                return e, xi, p_new, x_new, iterations
            ef = ef_new
        raise RuntimeError("NTFA: no consistent set of plastic phases in the step")

    def local_fields(self, strain, xi) -> dict[str, np.ndarray]:
        """Strain and stress fields (voxels x 6; engineering strains, Voigt stresses) of
        the macroscopic strain ``strain`` and the reduced variables ``xi`` (trained with
        ``keep_fields=True``)."""
        if self.unit_fields is None:
            raise ValueError(
                "the influence fields were not kept (train_ntfa(..., keep_fields=True))"
            )
        eps = np.tensordot(np.asarray(strain, float), self.unit_fields, axes=1)
        eps = eps + np.tensordot(np.asarray(xi, float), self.mode_fields, axes=1)
        plastic = self.plastic_strain(xi)
        return {
            "strain": eps,
            "plastic_strain": plastic,
            "stress": np.einsum("nij,nj->ni", self.voxel_stiffness, eps - plastic),
        }

    # -- storage ----------------------------------------------------------------------------
    def save(self, path) -> None:
        """The reduced model (without the modes) in a ``.npz`` file."""
        laws = {
            str(r): [
                law.yield_stress,
                law.hardening,
                np.nan if law.viscosity is None else law.viscosity,
                np.nan if law.exponent is None else law.exponent,
            ]
            for r, law in self.plasticity.items()
        }
        iso = {
            str(r): np.asarray(law.isotropic, dtype=float).reshape(-1, 2)
            for r, law in self.plasticity.items()
        }
        kin = {
            str(r): np.asarray(law.kinematic, dtype=float).reshape(-1, 2)
            for r, law in self.plasticity.items()
        }
        np.savez(
            path,
            stiffness=self.stiffness,
            rho=self.rho,
            a=self.a,
            d=self.d,
            mode_phase=self.mode_phase,
            phases=np.array(sorted(self.plasticity), dtype=int),
            laws=np.array([laws[str(r)] for r in sorted(self.plasticity)]),
            fractions=np.array([self.fractions[r] for r in sorted(self.plasticity)]),
            **{f"iso_{k}": v for k, v in iso.items()},
            **{f"kin_{k}": v for k, v in kin.items()},
        )

    @classmethod
    def load(cls, path) -> NTFAModel:
        data = np.load(path)
        plasticity, fractions = {}, {}
        for r, law, frac in zip(data["phases"], data["laws"], data["fractions"], strict=True):
            r = int(r)
            visc = None if np.isnan(law[2]) else float(law[2])
            expo = None if np.isnan(law[3]) else float(law[3])
            iso = tuple(map(tuple, data[f"iso_{r}"])) if f"iso_{r}" in data else ()
            kin = tuple(map(tuple, data[f"kin_{r}"])) if f"kin_{r}" in data else ()
            plasticity[r] = Plasticity(float(law[0]), float(law[1]), iso, kin, visc, expo)
            fractions[r] = float(frac)
        return cls(
            data["stiffness"],
            data["rho"],
            data["a"],
            data["d"],
            data["mode_phase"],
            plasticity,
            fractions,
        )


def _hardening(law: Plasticity, p: float) -> float:
    r = law.hardening * p
    for q, b in law.isotropic:
        r += q * (1.0 - np.exp(-b * p))
    return float(r)


def _slope(law: Plasticity, p: float) -> float:
    h = law.hardening
    for q, b in law.isotropic:
        h += q * b * np.exp(-b * p)
    return float(h)


def _influence(
    problem: FFTElasticProblem, eigen: np.ndarray | None, macro: np.ndarray
) -> np.ndarray:
    """Mean strain of each voxel (engineering) of the periodic elastic solution for the
    macroscopic strain ``macro`` and the eigenstrain field ``eigen`` (voxels x 6)."""
    nx, ny, nz = problem.shape
    el = problem.element
    if eigen is None:
        w, _ = problem._field(macro, 0.0)
    else:
        n = nx * ny * nz
        mat = problem.mat
        c = problem.table.matrices[np.maximum(mat, 0)]
        sig = np.einsum("nij,nj->ni", c, eigen)
        sig[mat < 0] = 0.0
        loads = -el.weight * np.einsum("grc,nr->nc", el.b_gauss, sig)
        index = np.where(mat >= 0, np.arange(n), -1)
        dof = np.arange(3 * n, dtype=np.int64)
        rhs = kernels.element_load(
            nx, ny, nz, True, index, dof, 3, np.ascontiguousarray(loads), dof.size
        )
        rhs = rhs + _macro_load(problem, macro)
        scale = float(np.sqrt(np.sum(loads * loads))) or 1.0
        w, _ = problem._solve(rhs, scale)
    strain, _, _ = kernels.elastic_fields(
        nx,
        ny,
        nz,
        True,
        problem.mat,
        problem.table.matrices,
        problem.table.eigen,
        0.0,
        w,
        el.affine_displacement(macro),
        el.b_mean,
        el.b_gauss,
    )
    return np.nan_to_num(strain) * SHEAR


def _macro_load(problem: FFTElasticProblem, macro: np.ndarray) -> np.ndarray:
    if not np.any(macro):
        return 0.0
    nx, ny, nz = problem.shape
    loads = np.ascontiguousarray(
        problem.element_matrices @ problem.element.affine_displacement(macro)
    )
    dof = np.arange(3 * nx * ny * nz, dtype=np.int64)
    return kernels.element_load(nx, ny, nz, True, problem.mat, dof, 3, loads, dof.size)


def train_ntfa(
    image: VoxelImage,
    materials: dict[int, Material],
    paths,
    *,
    modes=4,
    tol: float = 1e-8,
    newton_tol: float = 1e-6,
    keep_modes: bool = True,
    keep_fields: bool = False,
    snapshots: list | None = None,
    normalization: str = "mean",
) -> NTFAModel:
    """Reduced model of a periodic voxel image: snapshots of the plastic strain fields
    along the training ``paths`` (full-field simulations, :class:`NonlinearFFTProblem`
    with full integration, at the end of every step), their POD in each plastic phase
    (``modes``: the number of modes, or a mapping phase -> number), and the elastic
    influence fields of the unit strains and of the modes (FFT solves). ``snapshots``:
    plastic strain fields already computed (voxels x 6, tensor components), used
    instead of the paths."""
    from morphogen.homogenization.fft_nonlinear import NonlinearFFTProblem

    if not image.periodic:
        raise ValueError("NTFA: a periodic voxel image")
    labels = np.asarray(image.data).ravel()
    plastic = {
        int(r): m.plasticity
        for r, m in materials.items()
        if m.plasticity is not None and np.any(labels == r)
    }
    for r, m in materials.items():  # not elastic phases in disguise
        if m.porous and np.any(labels == r):
            raise ValueError(
                f"NTFA: phase {r}: porous plasticity (its porosity and backstresses) is not "
                "reduced: isotropic von Mises phases only"
            )
    if not plastic:
        raise ValueError("NTFA: no plastic phase in the image")
    for r in plastic:
        if materials[r].stiffness is not None or materials[r].crystal_plasticity is not None:
            raise ValueError(f"NTFA: phase {r}: isotropic von Mises phases only")
        if materials[r].damage is not None:
            raise ValueError(f"NTFA: phase {r}: the damage of a plastic phase is not reduced")
        if materials[r].plasticity.hill is not None:
            raise ValueError(f"NTFA: phase {r}: von Mises phases only (not Hill's criterion)")
    if snapshots is None:
        snapshots = []
        for path in paths:
            problem = NonlinearFFTProblem(
                image, materials, "periodic", tol=tol, newton_tol=newton_tol, integration="full"
            )
            path = LoadingPath.from_spec(path)
            result = problem.run(path, fields=[float(t) for t in path.schedule()])
            for t in sorted(result.fields):
                snapshots.append(np.asarray(result.fields[t]["plastic_strain"]).reshape(-1, 6))
    snaps = np.stack([np.nan_to_num(s) * SHEAR for s in snapshots], axis=-1)  # voxels x 6 x S
    counts = {r: int(np.sum(labels == r)) for r in plastic}
    n_modes = modes if isinstance(modes, dict) else dict.fromkeys(plastic, int(modes))
    field_modes, mode_phase, energy = [], [], {}
    root_w = np.sqrt(WEIGHTS)[None, :, None]
    for r in sorted(plastic):
        sel = labels == r
        x = (snaps[sel] * root_w).reshape(-1, snaps.shape[-1])
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        keep = int(min(n_modes.get(r, 0), np.sum(s > 1e-10 * s[0]) if s.size and s[0] > 0 else 0))
        energy[r] = float(np.sum(s[:keep] ** 2) / np.sum(s**2)) if keep else 0.0
        for k in range(keep):
            mode = np.zeros((labels.size, 6))
            values = u[:, k].reshape(-1, 6) / np.sqrt(WEIGHTS)
            if normalization == "mean":  # Michel & Suquet: <mu_eq>_r = 1
                equivalent = np.sqrt(2.0 / 3.0 * _contract(values, values))
                values = values / equivalent.mean()
            else:  # <mu : mu>_r = 3/2
                values = values * np.sqrt(1.5 * counts[r])
            mode[sel] = values
            field_modes.append(mode)
            mode_phase.append(r)
    if not field_modes:
        raise ValueError("NTFA: the training paths stay elastic (no plastic strain to reduce)")
    mode_phase = np.asarray(mode_phase)
    linear = FFTElasticProblem(image, materials, "periodic", tol=min(tol, 1e-10))
    mat = linear.mat
    c = linear.table.matrices[np.maximum(mat, 0)]
    c[mat < 0] = 0.0
    unit_strains = [_influence(linear, None, np.eye(6)[j]) for j in range(6)]
    stress_e = [np.einsum("nij,nj->ni", c, a) for a in unit_strains]
    stiffness = np.stack([s.mean(axis=0) for s in stress_e], axis=1)
    m = len(field_modes)
    rho = np.zeros((m, 6))
    a_red = np.zeros((m, 6))
    d_red = np.zeros((m, m))
    stress_m = []
    mode_strains = []
    for mode in field_modes:
        strain = _influence(linear, mode, np.zeros(6))
        mode_strains.append(strain)
        stress_m.append(np.einsum("nij,nj->ni", c, strain - mode))
    for k, mode in enumerate(field_modes):
        n_r = counts[int(mode_phase[k])]
        rho[k] = stress_m[k].mean(axis=0)
        for j in range(6):
            a_red[k, j] = np.sum(stress_e[j] * mode) / n_r  # stress . engineering strain
        for ell in range(m):
            d_red[k, ell] = np.sum(stress_m[ell] * mode) / n_r
    fractions = {r: counts[r] / labels.size for r in plastic}
    phase_means = {}
    for r in np.unique(labels[mat >= 0]):
        sel = labels == r
        phase_means[int(r)] = (
            np.stack([s[sel].mean(axis=0) for s in stress_e], axis=1),
            np.array([s[sel].mean(axis=0) for s in stress_m]),
            np.stack([a[sel].mean(axis=0) for a in unit_strains], axis=1),
            np.array([d[sel].mean(axis=0) for d in mode_strains]),
        )
    return NTFAModel(
        stiffness=stiffness,
        rho=rho,
        a=a_red,
        d=d_red,
        mode_phase=mode_phase,
        plasticity=plastic,
        fractions=fractions,
        energy=energy,
        phase_means=phase_means,
        modes=np.stack(field_modes) if keep_modes or keep_fields else None,
        shape=tuple(image.shape),
        unit_fields=np.stack(unit_strains) if keep_fields else None,
        mode_fields=np.stack(mode_strains) if keep_fields else None,
        voxel_stiffness=c if keep_fields else None,
    )
