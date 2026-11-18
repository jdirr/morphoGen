"""Topology optimization of periodic cells by inverse homogenization.

The cell is a grid of voxels of relative density ``rho`` (SIMP: stiffness
``eps + (1 - eps) rho^p`` times the base material, Bendsøe 1989). Its homogenized tensor
(FFT solver, :mod:`morphogen.optimization.cell`) and the exact derivatives give the
objective and its gradient (Sigmund 1994). Objectives:

``bulk_modulus``, ``shear_modulus``
    maximize the mean of the normal block of the stiffness tensor (bulk modulus of a
    cubic material) or of its shear moduli;
``poisson_ratio``
    minimize ``mean(C_ij, i != j) / mean(C_ii)`` over the normal block (``C_12 / C_11``
    for square or cubic symmetry: the Poisson's ratio under uniaxial strain), with the
    mean normal stiffness kept above ``min_stiffness`` times the Young's modulus of the
    base material (a second constraint, mma only): negative Poisson's ratios (auxetics);
``conductivity``
    maximize the mean of the diagonal of the conductivity tensor;
``target``
    reach a target stiffness tensor (least squares, NaN entries ignored).

Planar cells (one voxel through the thickness, prismatic microstructures) use the
in-plane components. The design is constrained to a volume fraction, regularized by a
density filter, a Heaviside projection with continuation and a symmetry group.

Methods: ``oc`` (optimality criteria, monotone objectives), ``mma`` (method of moving
asymptotes, any objective), ``beso`` (bi-directional evolutionary structural
optimization: solid or void voxels, the volume decreasing to the target, Huang and Xie
2010) and ``hca`` (hybrid cellular automaton, Tovar et al. 2006: every voxel adjusts its
density to bring the mean stimulus of its Moore neighbourhood to a common set point).
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from morphogen.automata.lattice import kernel_sum, offsets
from morphogen.homogenization.materials import Material
from morphogen.optimization.cell import PeriodicCell
from morphogen.optimization.filters import (
    SYMMETRIES,
    DensityFilter,
    Symmetry,
    heaviside,
    heaviside_derivative,
)
from morphogen.optimization.mma import MMA

OBJECTIVES = ("bulk_modulus", "shear_modulus", "poisson_ratio", "conductivity", "target")
METHODS = ("oc", "mma", "beso", "hca")
INITIAL = ("centered_void", "random", "uniform")
#: Monotone objectives (negative sensitivities), as needed by oc, beso and hca.
MONOTONE = ("bulk_modulus", "shear_modulus", "conductivity")


@dataclass
class CellDesign:
    """Result of a topology optimization."""

    density: np.ndarray
    design: np.ndarray
    tensor: np.ndarray
    binary_tensor: np.ndarray
    objective: float
    volume_fraction: float
    binary_fraction: float
    iterations: int
    converged: bool
    history: dict[str, list] = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    time_s: float = 0.0
    #: Constants on a finer grid (``refinement``, ``properties``), when checked.
    check: dict = field(default_factory=dict)

    @property
    def planar(self) -> bool:
        return self.density.shape[2] == 1

    @property
    def binary(self) -> np.ndarray:
        """Black-and-white design: voxels of density above 1/2."""
        return (self.density > 0.5).astype(np.uint8)

    @property
    def hinges(self) -> int:
        """Hinges of the black-and-white design (see :func:`count_hinges`)."""
        return count_hinges(self.binary)

    def properties(self, tensor: np.ndarray | None = None) -> dict[str, float]:
        """Engineering constants of the thresholded design (or of ``tensor``)."""
        c = self.binary_tensor if tensor is None else np.asarray(tensor)
        return tensor_properties(c, planar=self.planar)

    def refined_properties(self, factor: int = 2, *, tol: float = 1e-6) -> dict[str, float]:
        """Engineering constants of the design voxelized ``factor`` times finer from the
        level set of its densities, with true voids. Trilinear voxels overestimate thin
        members and hinges: moduli that drop on refinement call for a finer grid, a
        larger filter radius or a softer projection."""
        f = int(factor)
        if f < 1:
            raise ValueError("factor: a positive integer")
        shape = tuple(n * f if n > 1 else 1 for n in self.density.shape)
        image = self.to_microstructure().voxelize(shape=shape).data
        material = self.settings.get(
            "material", {"young": 1.0, "poisson": 0.3, "conductivity": 1.0}
        )
        cell = PeriodicCell(
            shape,
            (float(self.settings.get("voxel_size", 1.0)) / f,) * 3,
            physics="thermal" if self.settings.get("objective") == "conductivity" else "elastic",
            material=Material.from_config(material),
            tol=tol,
        )
        tensor, _ = cell.solve(image.astype(float))
        return tensor_properties(tensor, planar=self.planar)

    def to_microstructure(self, domain=None, *, smoothing: float = 0.0):
        """The design as a lattice microstructure of the periodic cell: phase 1 where the
        density exceeds 1/2, the density field giving smooth interfaces (level set),
        optionally smoothed by a Gaussian of ``smoothing`` voxels (rounding the staircases
        of nearly black-and-white designs, for conforming meshes and sections)."""
        from morphogen.domain import Domain
        from morphogen.models.lattice_field import LatticeField

        if domain is None:
            h = float(self.settings.get("voxel_size", 1.0))
            domain = Domain(size=tuple(n * h for n in self.density.shape), periodic=True)
        return LatticeField(
            domain,
            self.binary,
            field=self.density.astype(np.float32),
            threshold=0.5,
            smoothing=smoothing,
            attributes={"optimization": {k: v for k, v in self.settings.items()}},
        )


def _hinge_table(dims: int) -> np.ndarray:
    """For each pattern of a 2 x 2 (x 2) block (bit ``k``: voxel at the binary digits of
    ``k``), whether its solid voxels form several face-connected groups."""
    corners = list(itertools.product((0, 1), repeat=dims))
    table = np.zeros(2 ** len(corners), dtype=bool)
    for pattern in range(len(table)):
        solid = {k for k in range(len(corners)) if pattern >> k & 1}
        groups, seen = 0, set()
        for start in solid:
            if start in seen:
                continue
            groups += 1
            stack = [start]
            seen.add(start)
            while stack:
                a = corners[stack.pop()]
                for k in solid - seen:
                    if sum(abs(u - v) for u, v in zip(a, corners[k], strict=True)) == 1:
                        seen.add(k)
                        stack.append(k)
        table[pattern] = groups > 1
    return table


_HINGES = {2: _hinge_table(2), 3: _hinge_table(3)}


def count_hinges(binary) -> int:
    """Number of 2 x 2 blocks (2 x 2 x 2 in 3D) of a periodic black-and-white design
    whose solid voxels touch only along an edge or at a corner. Trilinear voxels make
    such contacts stiff hinges, which a continuum does not have (planar designs: one
    voxel through the thickness)."""
    b = np.asarray(binary) > 0
    if b.ndim != 3:
        raise ValueError("count_hinges: a 3D array ((nx, ny, 1) for planar designs)")
    axes = (0, 1) if b.shape[2] == 1 else (0, 1, 2)
    code = np.zeros(b.shape, dtype=np.int64)
    for k, shift in enumerate(itertools.product((0, 1), repeat=len(axes))):
        code |= np.roll(b, tuple(-s for s in shift), axis=axes).astype(np.int64) << k
    return int(np.count_nonzero(_HINGES[len(axes)][code]))


def tensor_properties(c: np.ndarray, *, planar: bool = False) -> dict[str, float]:
    """Moduli of a stiffness (6 x 6) or conductivity (3 x 3) tensor.

    Elastic: bulk modulus (mean of the normal block), shear modulus (mean of the shear
    diagonal), Young's moduli and Poisson's ratio ``nu_12`` from the compliance of the
    normal block (the full compliance for orthotropic cells); planar cells use the
    in-plane block (plane strain).
    """
    c = np.asarray(c, dtype=float)
    if c.shape == (3, 3):
        idx = [0, 1] if planar else [0, 1, 2]
        return {"conductivity": float(np.mean(np.diag(c)[idx]))}
    normal = [0, 1] if planar else [0, 1, 2]
    shear = [5] if planar else [3, 4, 5]
    out = {
        "bulk_modulus": float(c[np.ix_(normal, normal)].mean()),
        "shear_modulus": float(np.mean(np.diag(c)[shear])),
    }
    sub = c[np.ix_(normal, normal)]
    if np.all(np.isfinite(sub)):
        try:
            s = np.linalg.inv(sub)
            out["young_1"] = float(1.0 / s[0, 0])
            out["young_2"] = float(1.0 / s[1, 1])
            out["poisson_12"] = float(-s[0, 1] / s[0, 0])
        except np.linalg.LinAlgError:
            pass
    return out


def initial_density(micro, shape, *, phase: int = 1, supersampling: int = 4) -> np.ndarray:
    """The densities of a microstructure on the grid of a cell optimization (``shape``;
    ``nz = 1``: its middle section, a slab one voxel thick): the fraction of ``phase`` in
    each voxel (``supersampling**3`` samples), to start an optimization from it (the
    ``initial`` of :class:`CellOptimization`)."""
    from morphogen.domain import Domain, Grid

    shape = tuple(int(n) for n in shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("shape: three positive numbers of voxels")
    domain = micro.domain
    if shape[2] == 1:
        lower, lengths = np.asarray(domain.lower, float), np.asarray(domain.lengths, float)
        thickness = float(min(lengths[0] / shape[0], lengths[1] / shape[1]))
        middle = lower[2] + 0.5 * lengths[2] - 0.5 * thickness
        domain = Domain(
            size=(lengths[0], lengths[1], thickness),
            origin=(lower[0], lower[1], middle),
            periodic=domain.periodic,
        )
    image = micro.voxelize(grid=Grid(domain, shape), supersampling=int(supersampling))
    data = np.asarray(image.data)
    if data.dtype.kind == "f":  # the fraction of phase 1 in each voxel
        density = data.astype(float) if phase == 1 else 1.0 - data.astype(float)
    else:
        density = (data == phase).astype(float)
    return np.clip(density, 0.0, 1.0)


class CellOptimization:
    """Settings of the optimization of a periodic cell of ``shape`` voxels (``nz = 1``
    for planar cells), and :meth:`run`."""

    def __init__(
        self,
        shape,
        volume_fraction: float,
        *,
        objective: str = "bulk_modulus",
        material: Material | None = None,
        voxel_size: float = 1.0,
        method: str = "mma",
        penalty: float = 3.0,
        filter_radius: float = 1.5,
        projection: float = 0.0,
        symmetry: str = "cubic",
        stiffness_floor: float = 1e-3,
        iterations: int = 200,
        tolerance: float = 0.01,
        move: float = 0.2,
        target=None,
        min_stiffness: float = 0.05,
        evolution_rate: float = 0.02,
        initial="centered_void",
        solver_tol: float = 1e-4,
    ):
        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 3 or min(self.shape) < 1:
            raise ValueError("shape: three positive numbers of voxels")
        self.planar = self.shape[2] == 1
        if not 0.0 < volume_fraction < 1.0:
            raise ValueError("volume_fraction: in (0, 1)")
        if objective not in OBJECTIVES:
            raise ValueError(f"objective: one of {OBJECTIVES}")
        if method not in METHODS:
            raise ValueError(f"method: one of {METHODS}")
        if method != "mma" and objective not in MONOTONE:
            raise ValueError(f"method {method} needs a monotone objective {MONOTONE}; use mma")
        if symmetry not in SYMMETRIES:
            raise ValueError(f"symmetry: one of {SYMMETRIES}")
        if isinstance(initial, str):
            if initial not in INITIAL:
                raise ValueError(f"initial: one of {INITIAL}, or densities of the cell")
        else:  # the densities of a microstructure (initial_density)
            initial = np.asarray(initial, dtype=float)
            if initial.shape != self.shape:
                raise ValueError(f"initial: densities of the shape of the cell {self.shape}")
            if not np.all(np.isfinite(initial)) or initial.min() < 0.0 or initial.max() > 1.0:
                raise ValueError("initial: densities in [0, 1]")
        if penalty < 1:
            raise ValueError("penalty: at least 1")
        if not 0 < stiffness_floor < 0.1:
            raise ValueError("stiffness_floor: in (0, 0.1)")
        if projection < 0:
            raise ValueError("projection: a non-negative final sharpness (0: no projection)")
        self.volume_fraction = float(volume_fraction)
        self.objective = objective
        self.material = material or Material(young=1.0, poisson=0.3, conductivity=1.0)
        self.voxel_size = float(voxel_size)
        self.method = method
        self.penalty = float(penalty)
        self.filter_radius = float(filter_radius)
        self.projection = float(projection)
        self.symmetry = symmetry
        self.eps = float(stiffness_floor)
        self.iterations = int(iterations)
        self.tolerance = float(tolerance)
        self.move = float(move)
        if not 0.0 <= min_stiffness < 1.0:
            raise ValueError("min_stiffness: in [0, 1) times the Young's modulus")
        self.min_stiffness = float(min_stiffness)
        self.evolution_rate = float(evolution_rate)
        self.initial = initial
        self.solver_tol = float(solver_tol)
        self.physics = "thermal" if objective == "conductivity" else "elastic"
        if objective == "target":
            if target is None:
                raise ValueError("the target objective needs a target tensor (6 x 6)")
            self.target = np.asarray(target, dtype=float).reshape(6, 6)
        else:
            self.target = None
        self.normal = [0, 1] if self.planar else [0, 1, 2]
        self.shear = [5] if self.planar else [3, 4, 5]
        if self.physics == "thermal":
            self.cases = self.normal
        elif objective == "shear_modulus":
            self.cases = self.shear
        elif objective == "target":
            self.cases = [i for i in range(6) if np.any(np.isfinite(self.target[i]))]
        else:
            self.cases = self.normal
        self._scale = self._reference_scale()

    def settings(self) -> dict:
        return {
            "shape": list(self.shape),
            "volume_fraction": self.volume_fraction,
            "objective": self.objective,
            "method": self.method,
            "penalty": self.penalty,
            "filter_radius": self.filter_radius,
            "projection": self.projection,
            "symmetry": self.symmetry,
            "stiffness_floor": self.eps,
            "min_stiffness": self.min_stiffness,
            "voxel_size": self.voxel_size,
            "material": self.material.to_config(),
            "initial": self.initial if isinstance(self.initial, str) else "given",
        }

    def _reference_scale(self) -> float:
        """Modulus of the base material used to normalize the objective."""
        m = self.material
        if self.physics == "thermal":
            return float(m.conductivity)
        if self.objective == "shear_modulus":
            return float(m.shear)
        return float(m.young)

    # -- objective -----------------------------------------------------------------------

    def objective_value(self, tensor, cell: PeriodicCell, energies, iteration: int):
        """Objective (to minimize), its derivatives with respect to the voxel scales, and
        the extra constraints ``g <= 0`` as (value, derivatives) pairs."""
        ref = self._scale
        n_idx = self.normal
        if self.objective in ("bulk_modulus", "conductivity"):
            if self.physics == "thermal":
                pairs = [(i, i) for i in n_idx]
            else:
                pairs = [(i, j) for i in n_idx for j in n_idx]
            value = -np.mean([tensor[i, j] for i, j in pairs]) / ref
            grad = -sum(cell.derivative(energies, i, j) for i, j in pairs) / (len(pairs) * ref)
            return value, grad, []
        if self.objective == "shear_modulus":
            value = -np.mean([tensor[i, i] for i in self.shear]) / ref
            grad = -sum(cell.derivative(energies, i, i) for i in self.shear) / (
                len(self.shear) * ref
            )
            return value, grad, []
        if self.objective == "poisson_ratio":
            off = [(i, j) for i in n_idx for j in n_idx if i != j]
            num = float(np.mean([tensor[i, j] for i, j in off]))
            den = float(np.mean([tensor[i, i] for i in n_idx]))
            d_num = sum(cell.derivative(energies, i, j) for i, j in off) / len(off)
            d_den = sum(cell.derivative(energies, i, i) for i in n_idx) / len(n_idx)
            value = num / den
            grad = (d_num * den - num * d_den) / den**2
            c_min = self.min_stiffness * ref
            constraints = []
            if c_min > 0:
                constraints.append((1.0 - den / c_min, -d_den / c_min))
            return value, grad, constraints
        # target tensor
        t = self.target
        mask = np.isfinite(t)
        norm = float(np.sum(t[mask] ** 2)) or 1.0
        value = 0.0
        grad = np.zeros(cell.n_voxels)
        for i in self.cases:
            for j in self.cases:
                if not mask[i, j]:
                    continue
                diff = tensor[i, j] - t[i, j]
                value += diff * diff / norm
                grad += 2.0 * diff * cell.derivative(energies, i, j) / norm
        return value, grad, []

    # -- design --------------------------------------------------------------------------

    def initial_design(self, rng: np.random.Generator) -> np.ndarray:
        """``centered_void``: a void ball (disc) of radius 0.3 cell edges in a uniform
        matrix of the target volume; ``random``: noise around the volume fraction;
        ``uniform``: the volume fraction everywhere (a stationary point: only useful with
        asymmetric objectives); or the densities given (e.g. :func:`initial_density` of a
        microstructure), which the symmetry of the design then averages."""
        if not isinstance(self.initial, str):
            return np.array(self.initial, dtype=float)
        v = self.volume_fraction
        x = np.full(self.shape, v)
        if self.initial == "centered_void":
            c = [(np.arange(n) + 0.5) / n - 0.5 for n in self.shape]
            r2 = c[0][:, None, None] ** 2 + c[1][None, :, None] ** 2
            if not self.planar:
                r2 = r2 + c[2][None, None, :] ** 2
            hole = r2 < 0.3**2
            x[hole] = 0.0
            x[~hole] = min(1.0, v / max(1e-12, 1.0 - hole.mean()))
        elif self.initial == "random":
            x = np.clip(v + 0.25 * min(v, 1 - v) * (2 * rng.random(self.shape) - 1), 0, 1)
        return x

    def run(self, rng: np.random.Generator | None = None, callback=None) -> CellDesign:
        """Optimize; ``callback(iteration, density, objective)`` is called every iteration,
        and stops the optimization there when it returns True (the design so far, not
        converged)."""
        rng = np.random.default_rng() if rng is None else rng
        t0 = time.perf_counter()
        n = int(np.prod(self.shape))
        cell = PeriodicCell(
            self.shape,
            (self.voxel_size,) * 3,
            physics=self.physics,
            material=self.material,
            cases=self.cases,
            tol=self.solver_tol,
        )
        filt = DensityFilter(self.shape, self.filter_radius)
        sym = Symmetry(self.shape, self.symmetry)
        p, eps = self.penalty, self.eps
        beso = self.method == "beso"
        hca = self.method == "hca"
        use_filter = not (beso or hca)
        beta = 1.0 if (self.projection > 0 and not (beso or hca)) else 0.0
        x = sym(self.initial_design(rng))
        if beso:
            x = np.where(self.initial_design(rng) < self.volume_fraction, 1e-3, 1.0)
            x = sym(x)
            current_volume = float(x.mean())
        n_cons = 2 if (self.objective == "poisson_ratio" and self.min_stiffness > 0) else 1
        mma = MMA(n, n_cons, 0.0, 1.0, move=self.move) if self.method == "mma" else None
        hood = offsets("moore", 1, dims=2 if self.planar else 3, include_center=True)

        def to_design(g, factor):
            """Derivatives with respect to the design variables (chain rule through the
            interpolation, projection, filter and symmetry)."""
            g = g.reshape(self.shape) * factor
            return sym(filt.transpose(g) if use_filter else g)

        hist = {
            "iteration": [],
            "objective": [],
            "volume": [],
            "change": [],
            "beta": [],
            "solver_iterations": [],
        }
        scale0 = None
        alpha_old = None
        converged = False
        it = 0
        density = x
        tensor = np.full((6, 6), np.nan)
        value = np.nan
        for it in range(1, self.iterations + 1):
            xt = filt(x) if use_filter else x
            density = heaviside(xt, beta) if beta > 0 else xt
            s = eps + (1.0 - eps) * density**p
            ds = (1.0 - eps) * p * np.maximum(density, 1e-12) ** (p - 1)
            before = cell.iterations
            tensor, energies = cell.solve(s)
            value, grad_s, extra = self.objective_value(tensor, cell, energies, it)
            dproj = heaviside_derivative(xt, beta) if beta > 0 else np.ones(self.shape)
            g_x = to_design(grad_s, ds * dproj)
            g_v = to_design(np.full(self.shape, 1.0 / n), dproj)
            volume = float(density.mean())
            if self.method == "oc":
                x_new = self._oc(x, g_x, g_v, filt, beta, sym)
            elif self.method == "mma":
                if scale0 is None:
                    scale0 = 10.0 / max(abs(value), 1e-12)
                cons = [volume / self.volume_fraction - 1.0] + [c for c, _ in extra]
                grads = [g_v.ravel() / self.volume_fraction]
                grads += [to_design(g, ds * dproj).ravel() for _, g in extra]
                x_new = mma.update(
                    x.ravel(), scale0 * value, scale0 * g_x.ravel(), cons, np.array(grads)
                )
                x_new = sym(np.clip(x_new.reshape(self.shape), 0.0, 1.0))
            elif beso:
                current_volume = max(
                    self.volume_fraction, current_volume * (1.0 - self.evolution_rate)
                )
                alpha = filt(-g_x)
                if alpha_old is not None:
                    alpha = 0.5 * (alpha + alpha_old)
                alpha_old = alpha
                x_new = sym(self._beso_threshold(sym(alpha), current_volume))
                x_new = np.where(x_new >= 0.5, 1.0, 1e-3)
            else:
                x_new = self._hca(x, -g_x, hood, sym)
            change = float(np.max(np.abs(x_new - x)))
            x = x_new
            hist["iteration"].append(it)
            hist["objective"].append(float(value))
            hist["volume"].append(volume)
            hist["change"].append(change)
            hist["beta"].append(beta)
            hist["solver_iterations"].append(cell.iterations - before)
            if callback is not None and callback(it, density, value) is True:
                break  # stopped by the caller: the design so far
            if beso:
                done = (
                    current_volume <= self.volume_fraction + 1e-12
                    and len(hist["objective"]) > 10
                    and np.ptp(hist["objective"][-10:]) <= self.tolerance * abs(value)
                )
            else:
                done = change < self.tolerance
            sharp = beta == 0.0 or beta >= self.projection
            if done and sharp:
                converged = True
                break
            if beta > 0 and not sharp and (done or it % 40 == 0):
                beta = min(2.0 * beta, self.projection)
                if mma is not None:
                    mma.reset()
        xt = filt(x) if use_filter else x
        density = heaviside(xt, beta) if beta > 0 else xt
        binary = density > 0.5
        # the black-and-white design, with true voids and a tight tolerance
        bcell = PeriodicCell(
            self.shape,
            (self.voxel_size,) * 3,
            physics=self.physics,
            material=self.material,
            tol=1e-6,
        )
        binary_tensor, _ = bcell.solve(binary.astype(float))
        return CellDesign(
            density=density,
            design=x,
            tensor=tensor,
            binary_tensor=binary_tensor,
            objective=float(value),
            volume_fraction=float(density.mean()),
            binary_fraction=float(binary.mean()),
            iterations=it,
            converged=converged,
            history=hist,
            settings=self.settings(),
            time_s=time.perf_counter() - t0,
        )

    # -- updates -------------------------------------------------------------------------

    def _volume_of(self, x, filt, beta) -> float:
        xt = filt(x)
        return float((heaviside(xt, beta) if beta > 0 else xt).mean())

    def _oc(self, x, g_x, g_v, filt, beta, sym):
        """Optimality criteria with a bisection on the Lagrange multiplier."""
        b0 = np.maximum(-g_x, 1e-30) / np.maximum(g_v, 1e-30)
        lo, hi = 0.0, float(b0.max()) * 1e3 + 1e-30
        x_new = x
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            x_new = np.clip(
                x * np.sqrt(b0 / mid),
                np.maximum(0.0, x - self.move),
                np.minimum(1.0, x + self.move),
            )
            if self._volume_of(x_new, filt, beta) > self.volume_fraction:
                lo = mid
            else:
                hi = mid
            if hi - lo <= 1e-9 * (hi + lo):
                break
        return sym(x_new)

    def _beso_threshold(self, alpha, volume) -> np.ndarray:
        """Solid voxels: the ``volume`` fraction of highest sensitivity numbers."""
        n_solid = round(volume * alpha.size)
        out = np.zeros(alpha.size)
        if n_solid > 0:
            order = np.argsort(-alpha.ravel(), kind="stable")
            out[order[:n_solid]] = 1.0
        return out.reshape(alpha.shape)

    def _hca(self, x, stimulus, hood, sym):
        """Hybrid cellular automaton: proportional control of each voxel's density by the
        mean stimulus of its Moore neighbourhood, around a set point fixed by the volume."""
        local = kernel_sum(np.maximum(stimulus, 0.0), hood) / len(hood)
        local = sym(local)
        lo, hi = 1e-30, float(local.max()) * 1e3 + 1e-30
        x_new = x
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            x_new = np.clip(
                x + self.move * (local / mid - 1.0),
                np.maximum(1e-3, x - self.move),
                np.minimum(1.0, x + self.move),
            )
            if x_new.mean() > self.volume_fraction:
                lo = mid
            else:
                hi = mid
            if hi / lo < 1.0 + 1e-9:
                break
        return sym(x_new)
