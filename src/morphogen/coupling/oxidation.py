"""Thermal oxidation of polymers: the closed-loop mechanistic scheme, homogeneous (thin
films) or limited by the diffusion of oxygen (thick samples).

The scheme of Colin, Verdu and co-workers, with the rate constants that Ernault et al.
(2019) identified on a DGEBA/IPDA epoxy at 110 °C in air (their Tables 1 and 2):

    POOH          -> 2 P + P=O + V          k1  (hydroperoxide decomposition)
    P + O2        -> POO                    k2
    POO + PH      -> POOH + P               k3  (propagation, consumes the sites PH)
    P + P         -> inactive products      k4
    POO + P       -> inactive products      k5
    POO + POO     -> inactive + g6 P=O + O2 k6

``P=O`` (amides here, followed by FTIR) traces the oxidation. In thick samples oxygen
enters by Fick's law, ``d[O2]/dt = D d2[O2]/dz2 - k2 [P][O2] + k6 [POO]^2``, at the
concentration ``S p`` (Henry) on the exposed faces: the oxidized layer is a few hundred
micrometres thick. Concentrations in mol/l, times in seconds, lengths in metres.

**In a microstructure** (:func:`oxidation_field`): the same scheme in every voxel of an
image, each phase with its own constants (a polymer that oxidizes, an inert phase that
lets oxygen through, an impermeable one: fibres, fillers), oxygen diffusing between the
voxels from the exposed faces of the image. As in :mod:`morphogen.coupling.diffusion`,
the activity of oxygen ``u`` (its partial pressure at equilibrium, ``[O2] = S u``) is
continuous across the interfaces and the flux between two voxels goes through the
harmonic mean of their permeabilities ``D S``:

    d[X]/dt = r_X([P], [POO], ..., [O2])        (the five species that do not move)
    S du/dt = div(D S grad u) + r_O2,           u = p on the exposed faces.

The radicals live for milliseconds and the oxidation for thousands of hours: the system
is integrated by the implicit one-step method TR-BDF2 (a trapezoidal stage and a
backward difference, second order, L-stable; Bank et al. 1985) with its embedded error
estimate for the steps (Hosea & Shampine 1996). Each Newton iteration eliminates the
five local species voxel by voxel (their 5 x 5 blocks), which leaves one symmetric
system for the oxygen of all the voxels, ``(diag(V S s) + a K) du = b`` with ``K`` the
conductances of the faces and ``s`` the local Schur complements, solved by conjugate
gradients (a sparse factorization for small images): the cost of a step grows as the
number of voxels, in one, two or three dimensions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.sparse import linalg as splinalg

log = logging.getLogger(__name__)

SPECIES = ("P", "POO", "POOH", "PH", "PO", "O2")


@dataclass(frozen=True)
class ClosedLoopOxidation:
    """Rate constants (s^-1, l mol^-1 s^-1), initial concentrations (mol/l) and oxygen
    transport (solubility mol l^-1 Pa^-1, diffusivity m^2/s, partial pressure Pa) of the
    closed-loop scheme; the defaults are those of Ernault et al. (2019) for DGEBA/IPDA at
    110 °C in air."""

    k1: float = 1e-5
    k2: float = 1e6
    k3: float = 2.25
    k4: float = 2.5e10
    k5: float = 2.25e10
    k6: float = 2e3
    gamma6: float = 0.5
    ph0: float = 5.8
    pooh0: float = 1e-3
    solubility: float = 2.3e-7
    diffusivity: float = 7.3e-13
    pressure: float = 0.21 * 101325.0

    @property
    def surface_oxygen(self) -> float:
        """``[O2] = S p`` at equilibrium with the atmosphere (Henry's law)."""
        return self.solubility * self.pressure

    def rates(self, y: np.ndarray) -> np.ndarray:
        """Reaction rates of the species ``SPECIES`` (rows of ``y``), oxygen included."""
        p, poo, pooh, ph, _, o2 = y
        r1 = self.k1 * pooh
        r2 = self.k2 * p * o2
        r3 = self.k3 * poo * ph
        r4 = self.k4 * p * p
        r5 = self.k5 * p * poo
        r6 = self.k6 * poo * poo
        return np.array(
            [
                2.0 * r1 - r2 + r3 - 2.0 * r4 - r5,
                r2 - r3 - r5 - 2.0 * r6,
                -r1 + r3,
                -r3,
                r1 + self.gamma6 * r6,
                -r2 + r6,
            ]
        )

    def initial(self, n: int = 1, oxygen: float = 0.0) -> np.ndarray:
        y = np.zeros((6, n))
        y[2], y[3], y[5] = self.pooh0, self.ph0, oxygen
        return y

    def thin_film(self, times) -> dict[str, np.ndarray]:
        """Homogeneous oxidation (oxygen at its surface concentration everywhere):
        concentrations of the species at ``times`` (s)."""
        times = np.asarray(times, dtype=float)
        o2 = self.surface_oxygen

        def f(_, y):
            z = np.append(y, o2)
            return self.rates(z)[:5]

        sol = solve_ivp(
            f,
            (0.0, times[-1]),
            self.initial(1)[:5, 0],
            method="Radau",
            t_eval=times,
            rtol=1e-8,
            atol=1e-14,
        )
        if not sol.success:
            raise RuntimeError(f"thin_film: {sol.message}")
        out = dict(zip(SPECIES[:5], sol.y, strict=True))
        out["O2"] = np.full(len(times), o2)
        return out

    def profile(self, thickness: float, times, *, nodes: int = 61) -> dict[str, np.ndarray]:
        """Diffusion-limited oxidation of a plate of ``thickness`` (m) exposed on both
        faces: ``depth`` (m, from a face to the middle) and the concentrations of the
        species (times x nodes) at ``times`` (s). Method of lines on a uniform grid of
        half the plate (zero flux in the middle), stiff integration (Radau)."""
        times = np.asarray(times, dtype=float)
        h = 0.5 * thickness / (nodes - 1)
        o2s = self.surface_oxygen
        n = nodes - 1  # the nodes below the surface (the surface holds S p)
        lap = sparse.diags([np.ones(n - 1), -2.0 * np.ones(n), np.ones(n - 1)], [-1, 0, 1]).tolil()
        lap[n - 1, n - 2] = 2.0  # symmetry in the middle
        lap = (lap / (h * h)).tocsr()
        source = np.zeros(n)
        source[0] = o2s / (h * h)
        d = self.diffusivity

        def f(_, y):
            z = y.reshape(6, n)
            dz = self.rates(z)
            dz[5] += d * (lap @ z[5] + source)
            return dz.ravel()

        block = sparse.identity(n) + abs(lap)  # coupling pattern: local species, O2 neighbours
        pattern = sparse.bmat([[sparse.identity(n)] * 6] * 5 + [[sparse.identity(n)] * 5 + [block]])
        sol = solve_ivp(
            f,
            (0.0, times[-1]),
            self.initial(n, oxygen=o2s).ravel(),  # stored in air
            method="Radau",
            t_eval=times,
            rtol=1e-6,
            atol=1e-12,
            jac_sparsity=pattern,
        )
        if not sol.success:
            raise RuntimeError(f"profile: {sol.message}")
        y = sol.y.reshape(6, n, -1)
        out = {"depth": np.arange(1, nodes) * h}
        for k, name in enumerate(SPECIES):
            out[name] = y[k].T
        return out

    def oxidized_layer(self, depth, amide, fraction: float = 0.1) -> float:
        """Thickness of the oxidized layer: the depth where the amide concentration falls
        to ``fraction`` of its value at the surface (linear interpolation)."""
        a = np.asarray(amide, dtype=float)
        target = fraction * a[0]
        below = np.flatnonzero(a < target)
        if below.size == 0:
            return float(depth[-1])
        i = below[0]
        return float(np.interp(target, [a[i], a[i - 1]], [depth[i], depth[i - 1]]))


# -- oxidation in a microstructure ---------------------------------------------------------

#: Absolute tolerances of the species (mol/l) in the steps of :func:`oxidation_field`:
#: the radicals are scarce (1e-11 to 1e-7 mol/l), the others are not
TOLERANCES = (1e-15, 1e-12, 1e-9, 1e-7, 1e-9, 1e-9)

#: Voxels up to which the oxygen of a Newton iteration is solved by a sparse
#: factorization; preconditioned conjugate gradients beyond
DIRECT_LIMIT = 512

_GAMMA = 2.0 - np.sqrt(2.0)  # the trapezoidal stage of TR-BDF2 ends at gamma h
_D = 0.5 * _GAMMA  # the diagonal coefficient of both stages
_W = 0.25 * np.sqrt(2.0)


@dataclass
class OxidationField:
    """The oxidation of an image: the concentrations of the species (``fields``: name ->
    array ``(times,) + shape``, mol/l; zero in the phases that hold none) at the
    ``times`` (s), and the number of ``steps`` of the integration."""

    times: np.ndarray
    fields: dict[str, np.ndarray]
    steps: int

    def tracer(self, species: str = "PO", time: int = -1) -> np.ndarray:
        """The field of a species at an output time (by default the oxidation products
        at the last one): the tracer of :func:`morphogen.coupling.ageing.
        graded_materials`."""
        return self.fields[species][time]


class _Kinetics:
    """The rates of the scheme in the voxels of an image, each with the constants of its
    phase, and their derivatives."""

    def __init__(self, schemes: list, index: np.ndarray) -> None:
        def table(name):
            return np.array([getattr(m, name) if m is not None else 0.0 for m in schemes])[index]

        self.k1, self.k2, self.k3 = table("k1"), table("k2"), table("k3")
        self.k4, self.k5, self.k6 = table("k4"), table("k5"), table("k6")
        self.gamma6 = table("gamma6")

    def rates(self, y: np.ndarray) -> np.ndarray:
        p, poo, pooh, ph, _, o2 = y
        r1 = self.k1 * pooh
        r2 = self.k2 * p * o2
        r3 = self.k3 * poo * ph
        r4 = self.k4 * p * p
        r5 = self.k5 * p * poo
        r6 = self.k6 * poo * poo
        return np.stack(
            [
                2.0 * r1 - r2 + r3 - 2.0 * r4 - r5,
                r2 - r3 - r5 - 2.0 * r6,
                -r1 + r3,
                -r3,
                r1 + self.gamma6 * r6,
                -r2 + r6,
            ]
        )

    def jacobian(self, y: np.ndarray) -> np.ndarray:
        """``d rates_i / d y_j`` (6 x 6 x n)."""
        p, poo, _, ph, _, o2 = y
        k1, k2, k3, k4, k5, k6 = self.k1, self.k2, self.k3, self.k4, self.k5, self.k6
        j = np.zeros((6, 6, p.size))
        j[0, 0] = -k2 * o2 - 4.0 * k4 * p - k5 * poo
        j[0, 1] = k3 * ph - k5 * p
        j[0, 2] = 2.0 * k1
        j[0, 3] = k3 * poo
        j[0, 5] = -k2 * p
        j[1, 0] = k2 * o2 - k5 * poo
        j[1, 1] = -k3 * ph - k5 * p - 4.0 * k6 * poo
        j[1, 3] = -k3 * poo
        j[1, 5] = k2 * p
        j[2, 1] = k3 * ph
        j[2, 2] = -k1
        j[2, 3] = k3 * poo
        j[3, 1] = -k3 * ph
        j[3, 3] = -k3 * poo
        j[4, 1] = 2.0 * self.gamma6 * k6 * poo
        j[4, 2] = k1
        j[5, 0] = -k2 * o2
        j[5, 1] = 2.0 * k6 * poo
        j[5, 5] = -k2 * p
        return j


def oxidation_field(
    image,
    schemes,
    times,
    *,
    exposed=("x-", "x+"),
    pressure: float | None = None,
    unit: float = 1.0,
    periodic=None,
    rtol: float = 1e-4,
    max_steps: int = 20000,
    progress=None,
) -> OxidationField:
    """The thermal oxidation of the phases of an image (see the module): the
    concentrations of the species in its voxels at ``times`` (s).

    ``schemes``: a :class:`ClosedLoopOxidation` for every phase, or a mapping phase ->
    scheme, each with the constants of its polymer and its transport of oxygen; a phase
    without reactive sites (``ph0 = pooh0 = 0``) lets oxygen through and does not
    oxidize, and a phase mapped to None is impermeable and inert (fibres, fillers).
    Oxygen enters through the ``exposed`` faces of the image (``"x-"``, ``"x+"``, ...),
    held at the partial ``pressure`` (Pa; by default that of the schemes, which must then
    agree); the other faces are impermeable, or periodic in a periodic image. The image
    is stored in air: its oxygen starts at equilibrium with the atmosphere. ``unit``:
    the unit of length of the image in metres (``1e-6`` for micrometres). ``rtol``: the
    relative tolerance of the time steps. ``progress``: a function called with the
    fraction of the last time reached, after each step."""
    from morphogen.analysis.basic import as_array
    from morphogen.coupling.diffusion import FACES, _operator
    from morphogen.image import VoxelImage

    labels = np.asarray(as_array(image)).astype(np.int64)
    spacing = np.asarray(image.spacing if isinstance(image, VoxelImage) else (1.0, 1.0, 1.0), float)
    spacing = spacing * float(unit)
    per = (
        bool(image.periodic)
        if periodic is None and isinstance(image, VoxelImage)
        else bool(periodic)
    )
    exposed = tuple(exposed)
    for face in exposed:
        if face not in FACES:
            raise ValueError(f"exposed: faces among {', '.join(FACES)}")
    phases = [int(q) for q in np.unique(labels)]
    if isinstance(schemes, ClosedLoopOxidation):
        schemes = dict.fromkeys(phases, schemes)
    missing = [q for q in phases if q not in schemes]
    if missing:
        raise ValueError(f"schemes: none for the phases {missing} (None: impermeable and inert)")
    models = [schemes[q] for q in phases]
    if any(m is not None and not isinstance(m, ClosedLoopOxidation) for m in models):
        raise ValueError("schemes: ClosedLoopOxidation, or None for an impermeable phase")
    live = [m for m in models if m is not None]
    if not live:
        raise ValueError("schemes: one phase at least takes oxygen")
    if pressure is None:
        if len({m.pressure for m in live}) > 1:
            raise ValueError("pressure: the schemes differ; give that of the atmosphere")
        pressure = live[0].pressure
    pressure = float(pressure)
    times = np.asarray(times, dtype=float).ravel()
    if times.size == 0 or np.any(np.diff(times) <= 0) or times[0] < 0:
        raise ValueError("times: increasing and non-negative")

    index = np.searchsorted(phases, labels.ravel())
    n = labels.size
    kinetics = _Kinetics(models, index)

    def table(name, default=0.0):
        return np.array([getattr(m, name) if m is not None else default for m in models])[index]

    solubility = table("solubility", 1.0)  # (an impermeable voxel: any, it exchanges nothing)
    permeability = (table("diffusivity") * table("solubility")).reshape(labels.shape)
    axes_exposed = {FACES.index(f) // 2 for f in exposed}
    periodic_axes = {a for a in range(3) if per and a not in axes_exposed}
    conductance, faces = _operator(permeability, spacing, periodic_axes, exposed)
    conductance = conductance.tocsr()
    volume = float(np.prod(spacing))
    weights = np.array(TOLERANCES)[:, None]

    y = np.zeros((6, n))
    y[2], y[3] = table("pooh0"), table("ph0")
    y[5] = np.where(permeability.ravel() > 0.0, solubility * pressure, 0.0)

    def rhs(state):
        out = kinetics.rates(state)
        flux = faces * pressure - conductance @ (state[5] / solubility)
        out[5] += flux / volume
        return out

    def scaled(delta, state):
        return float(np.sqrt(np.mean((delta / (weights + rtol * np.abs(state))) ** 2)))

    def solve(residual, state, a):
        """``(I - a J) delta = residual``: the local species eliminated in each voxel
        (Gauss's elimination of their 5 x 5 blocks, on all the voxels at once: the
        blocks of the kinetics need no pivoting), the oxygen of all the voxels solved
        together."""
        m = -a * kinetics.jacobian(state)
        m[range(6), range(6)] += 1.0
        r = residual.copy()
        for i in range(5):
            for j in range(i + 1, 6):
                factor = m[j, i] / m[i, i]
                m[j, i + 1 :] -= factor * m[i, i + 1 :]
                r[j] -= factor * r[i]
        schur = m[5, 5]
        matrix = (sparse.diags(volume * solubility * schur) + a * conductance).tocsr()
        if n <= DIRECT_LIMIT or np.any(schur <= 0.0):
            du = splinalg.spsolve(matrix.tocsc(), volume * r[5])
        else:
            diagonal = matrix.diagonal()
            jacobi = splinalg.LinearOperator(matrix.shape, matvec=lambda v: v / diagonal)
            du, info = splinalg.cg(matrix, volume * r[5], rtol=1e-10, maxiter=20000, M=jacobi)
            if info:
                raise RuntimeError("oxidation_field: the conjugate gradients did not converge")
        delta = np.empty_like(residual)
        delta[5] = solubility * du
        for i in range(4, -1, -1):
            delta[i] = (r[i] - np.einsum("jn,jn->n", m[i, i + 1 :], delta[i + 1 :])) / m[i, i]
        return delta

    def stage(guess, constant, a):
        """Newton's method for ``y - a f(y) = constant``; None when it does not converge."""
        state = guess.copy()
        for _ in range(12):
            delta = solve(constant + a * rhs(state) - state, state, a)
            state += delta
            if not np.all(np.isfinite(state)):
                return None
            if scaled(delta, state) < 0.03:
                return state
        return None

    fields = {name: np.zeros((times.size, *labels.shape)) for name in SPECIES}
    t, h, steps = 0.0, 1e-3, 0
    slope = rhs(y)
    for k, t_out in enumerate(times):
        while t < t_out:
            h = min(h, t_out - t)
            if steps >= max_steps:
                raise RuntimeError(f"oxidation_field: {max_steps} steps before t = {t_out:g} s")
            a = _D * h
            middle = stage(y + _GAMMA * h * slope, y + a * slope, a)
            new = None
            if middle is not None:
                c1 = 1.0 / (_GAMMA * (2.0 - _GAMMA))
                constant = c1 * middle - (1.0 - _GAMMA) ** 2 * c1 * y
                new = stage(middle, constant, a)
            if new is None:
                h *= 0.25
                continue
            slope_middle, slope_new = rhs(middle), rhs(new)
            estimate = h * (
                (_W - (1.0 - _W) / 3.0) * slope
                + (_W - (3.0 * _W + 1.0) / 3.0) * slope_middle
                + (_D - _D / 3.0) * slope_new
            )
            error = scaled(solve(estimate, new, a), new)  # (filtered: stiff components)
            steps += 1
            if error > 1.0:
                h *= max(0.2, 0.9 * error ** (-1.0 / 3.0))
                continue
            t += h
            y, slope = np.maximum(new, 0.0), slope_new
            if progress is not None:
                progress(t / float(times[-1]))
            if t < t_out:
                h *= min(5.0, max(0.2, 0.9 * max(error, 1e-10) ** (-1.0 / 3.0)))
        for row, name in enumerate(SPECIES):
            fields[name][k] = y[row].reshape(labels.shape)
    log.info("oxidation_field: %d steps to t = %g s on %d voxels", steps, times[-1], n)
    return OxidationField(times, fields, steps)
