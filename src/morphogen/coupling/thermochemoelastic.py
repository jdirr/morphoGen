"""Thermo-chemo-elastic couplings in time: heat, a diffusing species and the strains they
cause in the volume of a microstructure, the fields in three dimensions at each time.

Three fields evolve together in the phases of a sample (a voxel image or a conforming
mesh of linear tetrahedra), each phase with its :class:`ThermoChemoMaterial`:

* the temperature ``T``, by Fourier's law with a volumetric heat capacity ``C`` and the
  heat that the reaction of the species releases,
  ``C dT/dt = div(k grad T) + H k S u``;
* the activity ``u`` of a species (moisture, a solvent, hydrogen, lithium), of
  concentration ``c = S u`` in a phase of solubility ``S`` and continuous across the
  interfaces, as in :mod:`morphogen.coupling.diffusion`; it diffuses with the
  permeability ``P = D S``, is consumed by a first-order reaction of rate ``k``, and is
  drawn by the gradient of the hydrostatic stress ``sigma_h`` when it has a partial molar
  volume ``Omega`` (stress-assisted diffusion, Sofronis & McMeeking 1989):
  ``S du/dt = div(P grad u - P u (Omega / R T) grad sigma_h) - k S u``;
* the displacement, in quasi-static equilibrium ``div sigma = 0`` with ``sigma =
  C_el (eps - eps*)`` and the eigenstrain of the thermal and chemical expansions,
  ``eps* = alpha (T - T_ref) + beta (c - c_ref)`` (coefficients, principal values along
  the axes or tensors, per phase).

The diffusivity and the rate of the reaction follow Arrhenius' law from their values at
the reference temperature ``T_ref``, ``D(T) = D exp(-(Q_D / R)(1/T - 1/T_ref))`` (likewise
``k``); ``T_ref`` is also the temperature without thermal strain, and ``c_ref`` the
concentration without chemical strain (that of the reference activity, by default the
initial one). The fields are coupled both ways when the reaction heats (``H``) or when
the species feels the stress (``Omega``): each step then solves the three problems one
after the other (staggered: temperature, species, displacement) to a fixed point;
otherwise one pass is exact, and the displacement is solved at the output times only.

**Discretizations.** On voxel images (:class:`~morphogen.image.VoxelImage`) the
temperature and the activity are finite volumes at the centres of the voxels: the flux
between two neighbours goes through the harmonic mean of their conductivities (exact for
layers in series; the components of the tensors along the axes), the faces of the box
held at a value hold it half a voxel from the centres of their voxels, the other faces
are insulated or periodic; the drift of the species is the exponential fitting of
Scharfetter & Gummel (1969), ``F_ij = g (B(-w) u_i - B(w) u_j)`` with ``B(x) = x / (e^x -
1)`` and ``w = Omega (sigma_j - sigma_i) / R T``, whose flux vanishes exactly for the
Boltzmann distribution ``u ~ exp(Omega sigma_h / R T)``. The displacement lives at the
corners of the voxels: trilinear hexahedra (:mod:`morphogen.homogenization.elements`),
each loaded by the eigenstrain of its voxel. On conforming meshes
(:class:`~morphogen.mesh.tetra.TetMesh`, linear tetrahedra) every field is linear in each
element: the temperature and the activity at the nodes with lumped capacities, the
concentration and the eigenstrain of an element from the mean of its nodes, the drift
by the gradient of the hydrostatic stress averaged at the nodes (Galerkin); the faces held
at a value hold it at their nodes.

**Time.** Implicit Euler steps (``scheme="euler"``) or TR-BDF2 (``"trbdf2"``, Bank et al.
1985: a trapezoidal stage to ``t + gamma dt`` and a second-order backward difference to
``t + dt``, ``gamma = 2 - sqrt 2``, L-stable, the same matrix in both stages); each
interval between two output times is split into ``steps`` steps, those of the first
interval growing geometrically from about a ten-thousandth of it, for the start in
``sqrt(t)`` of a diffusion from a face.

**Boundary conditions.** ``temperature`` and ``activity`` map faces of the box (``"x-"``,
``"x+"``, ...) to a value or a function of time, the other faces being insulated and
impermeable; ``supports`` map faces to ``"clamped"`` (the displacement vanishes),
``"symmetry"`` (its normal component vanishes) or ``"free"`` (the default). Along the
periodic axes of a voxel image the fields are periodic and the displacement is periodic
plus a mean strain (its components between periodic axes), free by default (the mean
stress vanishes: the cell expands freely) or imposed (``mean_strain``). The rigid motions
that the supports leave are removed by fixing as many components of the displacement of
a few nodes far apart (statically determinate: no reaction to the self-equilibrated
loads of the eigenstrains).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.integrate import trapezoid
from scipy.spatial import cKDTree

from morphogen.homogenization.elements import hex_element
from morphogen.homogenization.materials import Material, isotropic_matrix, lame
from morphogen.homogenization.solvers import DIRECT_LIMIT, LinearSolver
from morphogen.image import VoxelImage
from morphogen.mesh.tetra import TetMesh
from morphogen.mesh.voxel import HEX_CORNERS

#: The gas constant, J/(mol K)
R_GAS = 8.314462618
FACES = ("x-", "x+", "y-", "y+", "z-", "z+")
SUPPORTS = ("clamped", "symmetry", "free")
SCHEMES = ("euler", "trbdf2")
#: The coefficient of the stages of TR-BDF2
GAMMA = 2.0 - math.sqrt(2.0)
#: Voigt components (11, 22, 33, 23, 31, 12) of the pairs of axes
VOIGT_OF = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (2, 1): 3, (0, 2): 4, (2, 0): 4,
            (0, 1): 5, (1, 0): 5}  # fmt: skip
#: The capacity given to the unknowns that no solid touches (pores): their fields stay
#: at their initial values and the systems stay regular
_TINY = 1e-300


def strain_voigt(value, name: str = "strain") -> np.ndarray:
    """A strain per unit of its cause (one coefficient, three principal values along the
    axes, or a symmetric 3 x 3 tensor) as a Voigt vector ``11, 22, 33, 23, 31, 12`` with
    engineering shears."""
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        a = float(a) * np.eye(3)
    elif a.shape == (3,):
        a = np.diag(a)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: one coefficient, three principal values or a 3 x 3 tensor")
    if not np.allclose(a, a.T, rtol=1e-12, atol=1e-15 * max(float(np.abs(a).max()), 1e-300)):
        raise ValueError(f"{name}: a symmetric tensor")
    return np.array([a[0, 0], a[1, 1], a[2, 2], 2.0 * a[1, 2], 2.0 * a[2, 0], 2.0 * a[0, 1]])


def _tensor(value, name: str) -> np.ndarray:
    """A transport coefficient (one value, three principal values along the axes or a
    symmetric 3 x 3 tensor) as a 3 x 3 tensor."""
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        a = float(a) * np.eye(3)
    elif a.shape == (3,):
        a = np.diag(a)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: one value, three principal values or a 3 x 3 tensor")
    a = 0.5 * (a + a.T)
    if np.linalg.eigvalsh(a).min() < -1e-12 * max(float(np.abs(a).max()), 1e-300):
        raise ValueError(f"{name}: a non-negative tensor")
    return a


@dataclass(frozen=True)
class ThermoChemoMaterial:
    """The coefficients of a phase in the coupled problem, in consistent units (SI: Pa,
    W/(m K), J/(m^3 K), m^2/s, 1/K...):

    * elasticity: ``young`` and ``poisson`` (isotropic), or ``stiffness`` (6 x 6, Voigt
      ``11, 22, 33, 23, 31, 12`` with engineering shears);
    * heat: ``conductivity`` (one value, principal values along the axes or a 3 x 3
      tensor) and the volumetric ``heat_capacity`` (``rho c_p``);
    * species: ``diffusivity`` (likewise), ``solubility`` (the concentration at unit
      activity), the rate ``reaction_rate`` of its first-order consumption (1/s) and the
      heat ``reaction_heat`` released per unit of concentration consumed;
    * strains: ``expansion`` (thermal, per kelvin) and ``swelling`` (chemical, per unit of
      concentration), one coefficient, principal values or a tensor;
    * couplings: the partial molar volume ``molar_volume`` of the species (its drift by
      the stress) and the activation energies (J/mol) of the diffusivity and of the
      reaction (Arrhenius' law about the reference temperature).

    ``void``: a pore, which carries no stress, no heat and no species."""

    young: float | None = None
    poisson: float | None = None
    stiffness: tuple | None = None
    conductivity: float | tuple = 0.0
    heat_capacity: float = 1.0
    diffusivity: float | tuple = 0.0
    solubility: float = 1.0
    reaction_rate: float = 0.0
    reaction_heat: float = 0.0
    expansion: float | tuple = 0.0
    swelling: float | tuple = 0.0
    molar_volume: float = 0.0
    diffusion_activation: float = 0.0
    reaction_activation: float = 0.0
    void: bool = False

    def __post_init__(self) -> None:
        if self.void:
            return
        if self.stiffness is None and (self.young is None or self.poisson is None):
            raise ValueError("a solid phase needs young and poisson, or a stiffness matrix")
        if self.stiffness is not None and self.young is not None:
            raise ValueError("give young/poisson or a stiffness matrix, not both")
        if self.young is not None and not (self.young > 0 and -1.0 < self.poisson < 0.5):
            raise ValueError("young: positive; poisson: in (-1, 0.5)")
        if self.stiffness is not None:
            m = np.asarray(self.stiffness, dtype=float)
            if m.shape != (6, 6) or np.linalg.eigvalsh(0.5 * (m + m.T)).min() <= 0:
                raise ValueError("stiffness: a symmetric positive definite 6 x 6 matrix")
        for name in ("heat_capacity", "solubility"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name}: positive")
        for name in ("reaction_rate", "diffusion_activation", "reaction_activation"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name}: non-negative")
        _tensor(self.conductivity, "conductivity")
        _tensor(self.diffusivity, "diffusivity")
        strain_voigt(self.expansion, "expansion")
        strain_voigt(self.swelling, "swelling")

    @classmethod
    def from_material(cls, material: Material, **coefficients) -> ThermoChemoMaterial:
        """The coefficients of a :class:`~morphogen.homogenization.materials.Material`
        (its elasticity, conductivity, diffusivity, thermal expansion and swelling),
        completed or replaced by ``coefficients`` (``heat_capacity``, ``solubility``...)."""
        if material.void:
            return cls(void=True)
        values: dict = {}
        if material.stiffness is not None:
            values["stiffness"] = material.stiffness
        elif material.young is not None:
            values["young"], values["poisson"] = material.young, material.poisson
        if material.conductivities is not None:
            values["conductivity"] = material.conductivities
        elif material.conductivity is not None:
            values["conductivity"] = material.conductivity
        for name in ("diffusivity", "expansion", "swelling"):
            if getattr(material, name) is not None:
                values[name] = getattr(material, name)
        values.update(coefficients)
        return cls(**values)

    def elastic_matrix(self) -> np.ndarray:
        """The elasticity matrix (6 x 6, Voigt, engineering shears)."""
        if self.stiffness is not None:
            m = np.asarray(self.stiffness, dtype=float)
            return 0.5 * (m + m.T)
        return isotropic_matrix(*lame(self.young, self.poisson))

    def thermal_strain(self) -> np.ndarray:
        """The eigenstrain of one kelvin (Voigt, engineering shears)."""
        return strain_voigt(self.expansion, "expansion")

    def chemical_strain(self) -> np.ndarray:
        """The eigenstrain of a unit concentration (Voigt, engineering shears)."""
        return strain_voigt(self.swelling, "swelling")


def arrhenius(activation, temperature, reference: float) -> np.ndarray:
    """``exp(-(Q / R)(1/T - 1/T_ref))``: the factor of a coefficient of activation energy
    ``Q`` (J/mol) at the temperature ``T`` over its value at ``T_ref`` (kelvins)."""
    q = np.asarray(activation, dtype=float)
    t = np.asarray(temperature, dtype=float)
    return np.exp(-q / R_GAS * (1.0 / t - 1.0 / float(reference)))


def bernoulli(x) -> np.ndarray:
    """``B(x) = x / (exp(x) - 1)`` (1 at 0): the weights of the Scharfetter–Gummel fluxes;
    ``B(-x) = e^x B(x)``."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    small = np.abs(x) < 1e-6
    big = ~small
    out[small] = 1.0 - 0.5 * x[small] + x[small] ** 2 / 12.0
    out[big] = x[big] / np.expm1(x[big])
    return out


def _harmonic(a, b):
    s = a + b
    return np.divide(2.0 * a * b, s, out=np.zeros_like(s), where=s > 0)


def _face(name: str) -> tuple[int, int]:
    if name not in FACES:
        raise ValueError(f"faces: among {', '.join(FACES)}, got {name!r}")
    return FACES.index(name) // 2, FACES.index(name) % 2


def _value_at(value, t: float) -> float:
    return float(value(t)) if callable(value) else float(value)


def _tensor_of(e: np.ndarray) -> np.ndarray:
    """The symmetric tensor of a Voigt strain (engineering shears)."""
    return np.array(
        [
            [e[0], 0.5 * e[5], 0.5 * e[4]],
            [0.5 * e[5], e[1], 0.5 * e[3]],
            [0.5 * e[4], 0.5 * e[3], e[2]],
        ]
    )


# ----------------------------------------------------------------------------------------
# Rigid motions


def rigid_modes(coordinates: np.ndarray, periodic_axes=()) -> np.ndarray:
    """The rigid motions (columns; rows ``3 node + component``) of nodes that periodicity
    along ``periodic_axes`` allows: the translations, and the rotation about an axis when
    the two others are not periodic."""
    periodic_axes = set(periodic_axes)
    n = len(coordinates)
    x = coordinates - coordinates.mean(axis=0)
    modes = []
    for a in range(3):
        m = np.zeros((n, 3))
        m[:, a] = 1.0
        modes.append(m.ravel())
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        if b in periodic_axes or c in periodic_axes:
            continue
        m = np.zeros((n, 3))
        m[:, b] = -x[:, c]
        m[:, c] = x[:, b]
        modes.append(m.ravel())
    return np.column_stack(modes)


def supports_of(modes: np.ndarray, constrained: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """The degrees of freedom whose fixing removes the rigid motions that the
    ``constrained`` ones leave, and no more: the combinations of ``modes`` that vanish on
    ``constrained`` (singular value decomposition), fixed at as many ``candidates`` chosen
    by a QR factorization with column pivoting (the best conditioned set)."""
    if constrained.size:
        _, s, vt = np.linalg.svd(modes[constrained], full_matrices=True)
        rank = int(np.sum(s > 1e-9 * max(float(s.max(initial=0.0)), 1.0)))
        survivors = vt[rank:].T
    else:
        survivors = np.eye(modes.shape[1])
    if survivors.shape[1] == 0 or candidates.size == 0:
        return np.zeros(0, dtype=np.int64)
    basis = modes[candidates] @ survivors
    _, _, piv = sla.qr(basis.T, pivoting=True, mode="economic")
    return np.sort(candidates[piv[: survivors.shape[1]]])


def support_candidates(coordinates: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """The degrees of freedom ``allowed`` of the nodes nearest to the corners, the middles
    of the edges and faces and the centre of their bounding box: those that may carry
    the supports of the rigid motions."""
    nodes = np.flatnonzero(allowed.reshape(-1, 3).any(axis=1))
    if nodes.size == 0:
        return np.zeros(0, dtype=np.int64)
    x = coordinates[nodes]
    lo, hi = x.min(axis=0), x.max(axis=0)
    targets = np.array(np.meshgrid(*[(0.0, 0.5, 1.0)] * 3, indexing="ij")).reshape(3, -1).T
    picks = np.unique(nodes[cKDTree(x).query(lo + targets * (hi - lo))[1]])
    dofs = (3 * picks[:, None] + np.arange(3)).ravel()
    return dofs[allowed[dofs]]


def constrained_dofs(supports: dict, face_nodes, used: np.ndarray) -> np.ndarray:
    """The degrees of freedom (a mask, ``3 node + component``) that the ``supports`` of the
    faces fix (``face_nodes(name)``: the nodes of a face), at the ``used`` nodes."""
    fixed = np.zeros(3 * used.size, dtype=bool)
    for name, kind in supports.items():
        if kind not in SUPPORTS:
            raise ValueError(f"supports: {', '.join(SUPPORTS)}; got {kind!r}")
        if kind == "free":
            continue
        a, _ = _face(name)
        nodes = face_nodes(name)
        nodes = nodes[used[nodes]]
        if kind == "clamped":
            fixed[(nodes[:, None] * 3 + np.arange(3)).ravel()] = True
        else:
            fixed[nodes * 3 + a] = True
    return fixed


def _voids(materials, phase: np.ndarray) -> np.ndarray:
    """Whether the material of each cell is a pore."""
    table = np.zeros(max(materials) + 1, dtype=bool)
    for p, m in materials.items():
        table[p] = m.void
    return table[np.asarray(phase)]


def assembled(dofs: np.ndarray, local: np.ndarray, n: int) -> sp.csr_matrix:
    """The sparse matrix (n x n) of the element matrices ``local`` (elements x m x m) summed
    on their degrees of freedom ``dofs`` (elements x m)."""
    rows = np.broadcast_to(dofs[:, :, None], local.shape).ravel()
    cols = np.broadcast_to(dofs[:, None, :], local.shape).ravel()
    out = sp.coo_matrix((local.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    out.sum_duplicates()
    return out


# ----------------------------------------------------------------------------------------
# Voxel images


class VoxelSpace:
    """Fields at the centres of the voxels (finite volumes): the interior faces (between
    two voxels, across the periodic faces too) and the faces of the box, with their
    ratios of area to distance."""

    kind = "voxels"

    def __init__(self, shape, spacing, origin=(0.0, 0.0, 0.0), periodic_axes=()):
        self.shape = tuple(int(s) for s in shape)
        self.spacing = np.asarray(spacing, dtype=float)
        self.origin = np.asarray(origin, dtype=float)
        self.n = int(np.prod(self.shape))
        self.cell_volume = float(np.prod(self.spacing))
        self.periodic_axes = set(periodic_axes)
        index = np.arange(self.n).reshape(self.shape)
        fi, fj, fa, fr = [], [], [], []
        for a in range(3):
            n_a = self.shape[a]
            if n_a == 1:
                continue
            area = float(np.prod([self.spacing[b] for b in range(3) if b != a]))
            i = index
            j = np.roll(index, -1, axis=a)
            if a not in self.periodic_axes:
                sl = [slice(None)] * 3
                sl[a] = slice(0, n_a - 1)
                i, j = i[tuple(sl)], j[tuple(sl)]
            fi.append(i.ravel())
            fj.append(j.ravel())
            fa.append(np.full(i.size, a))
            fr.append(np.full(i.size, area / self.spacing[a]))
        self.face_i = np.concatenate(fi) if fi else np.zeros(0, np.int64)
        self.face_j = np.concatenate(fj) if fj else np.zeros(0, np.int64)
        self.face_axis = np.concatenate(fa) if fa else np.zeros(0, np.int64)
        self.face_ratio = np.concatenate(fr) if fr else np.zeros(0)
        self.box: dict[str, tuple[np.ndarray, int, float]] = {}
        for name in FACES:
            a, side = _face(name)
            if a in self.periodic_axes:
                continue
            sl = [slice(None)] * 3
            sl[a] = 0 if side == 0 else self.shape[a] - 1
            area = float(np.prod([self.spacing[b] for b in range(3) if b != a]))
            self.box[name] = (index[tuple(sl)].ravel(), a, area / (0.5 * self.spacing[a]))

    def lump(self, cell_values) -> np.ndarray:
        return np.asarray(cell_values, dtype=float) * self.cell_volume

    def at_cells(self, values: np.ndarray) -> np.ndarray:
        return values

    def conductances(self, kappa_axes: np.ndarray) -> np.ndarray:
        """The conductances of the interior faces: the harmonic means of the
        coefficients (cells x 3, along each axis) of their two voxels times their ratios."""
        a = self.face_axis
        return _harmonic(kappa_axes[self.face_i, a], kappa_axes[self.face_j, a]) * self.face_ratio

    def laplacian(self, g: np.ndarray) -> sp.csr_matrix:
        """``sum_f g_f (e_i - e_j)(e_i - e_j)^T`` over the interior faces."""
        i, j = self.face_i, self.face_j
        rows = np.concatenate([i, j, i, j])
        cols = np.concatenate([j, i, i, j])
        data = np.concatenate([-g, -g, g, g])
        return sp.coo_matrix((data, (rows, cols)), shape=(self.n, self.n)).tocsr()

    def face_loads(self, kappa_axes: np.ndarray, faces) -> dict[str, np.ndarray]:
        """The conductances (a vector over the cells) of each face of the box held at a
        value, half a voxel from the centres of its voxels: the load of a unit value."""
        loads = {}
        for name in faces:
            if name not in self.box:
                raise ValueError(f"{name}: a face across a periodic axis holds no value")
            cells, axis, ratio = self.box[name]
            vec = np.zeros(self.n)
            np.add.at(vec, cells, kappa_axes[cells, axis] * ratio)
            loads[name] = vec
        return loads

    def operator(self, kappa: np.ndarray, faces) -> tuple[sp.csr_matrix, dict]:
        """The conductance matrix of cells of tensors ``kappa`` (cells x 3 x 3) whose
        faces ``faces`` are held: the laplacian of the interior faces with the
        conductances of the faces held on its diagonal, and their loads."""
        axes = np.einsum("nii->ni", kappa)
        loads = self.face_loads(axes, faces)
        k = self.laplacian(self.conductances(axes))
        if loads:
            k = k + sp.diags(sum(loads.values()))
        return k.tocsr(), loads

    def drift(self, perm, omega, temperature, hydrostatic, faces) -> tuple[sp.csr_matrix, dict]:
        """The fluxes of the species between the cells with their drift by the stress
        (Scharfetter–Gummel), with the faces held as in :meth:`operator`."""
        axes = np.einsum("nii->ni", perm)
        i, j = self.face_i, self.face_j
        g = self.conductances(axes)
        w = (
            0.5
            * (omega[i] + omega[j])
            * (hydrostatic[j] - hydrostatic[i])
            / (R_GAS * 0.5 * (temperature[i] + temperature[j]))
        )
        bp, bm = bernoulli(w), bernoulli(-w)
        rows = np.concatenate([i, i, j, j])
        cols = np.concatenate([i, j, j, i])
        data = np.concatenate([g * bm, -g * bp, g * bp, -g * bm])
        k = sp.coo_matrix((data, (rows, cols)), shape=(self.n, self.n)).tocsr()
        loads = self.face_loads(axes, faces)
        if loads:
            k = k + sp.diags(sum(loads.values()))
        return k.tocsr(), loads

    def reduce(self, matrix, faces):
        """The system to solve: the matrix itself (the faces held load it)."""
        return matrix, None

    def reduced_rhs(self, rhs, structure, values: dict, loads: dict) -> np.ndarray:
        for name, value in values.items():
            rhs = rhs + loads[name] * value
        return rhs

    def expand(self, x, structure, values: dict) -> np.ndarray:
        return x


class _HexMechanics:
    """Trilinear hexahedra on the solid voxels, the nodes at their corners (``n`` of them
    along the periodic axes, whose last plane is the first, ``n + 1`` along the others),
    with the mean strain of the periodic axes as unknowns."""

    def __init__(self, space: VoxelSpace, phase: np.ndarray, materials, supports, mean):
        shape, h = space.shape, space.spacing
        per = space.periodic_axes
        self.space = space
        self.nodes_shape = tuple(shape[a] if a in per else shape[a] + 1 for a in range(3))
        nx, ny, nz = self.nodes_shape
        self.n_nodes = nx * ny * nz
        idx = np.indices(self.nodes_shape).reshape(3, -1).T
        self.coordinates = space.origin + idx * h
        self.solid = np.flatnonzero(~_voids(materials, phase))
        ijk = np.stack(np.unravel_index(self.solid, shape), axis=1)
        corners = (ijk[:, None, :] + HEX_CORNERS[None, :, :]) % np.array(self.nodes_shape)
        self.element_nodes = (corners[..., 0] * ny + corners[..., 1]) * nz + corners[..., 2]
        self.element_dofs = (self.element_nodes[:, :, None] * 3 + np.arange(3)).reshape(-1, 24)
        element = hex_element(tuple(float(v) for v in h))
        self.b_mean = np.asarray(element.b_mean)
        ph = phase[self.solid]
        self.phases = sorted({int(p) for p in ph})
        self.d = {p: materials[p].elastic_matrix() for p in self.phases}
        self.elastic = np.empty((self.solid.size, 6, 6))
        self.load_matrix = np.empty((self.solid.size, 24, 6))
        local = np.empty((self.solid.size, 24, 24))
        for p in self.phases:
            sel = ph == p
            d = self.d[p]
            self.elastic[sel] = d
            self.load_matrix[sel] = space.cell_volume * self.b_mean.T @ d
            local[sel] = element.weight * np.einsum("gai,ab,gbj->ij", element.b_gauss, d,
                                                    element.b_gauss)  # fmt: skip
        n_dofs = 3 * self.n_nodes
        k = assembled(self.element_dofs, local, n_dofs)
        used = np.zeros(self.n_nodes, dtype=bool)
        used[self.element_nodes.ravel()] = True
        self.used = used
        # the mean strain: its components between periodic axes
        self.macro = sorted({VOIGT_OF[(a, b)] for a in per for b in per})
        imposed = {int(c): float(v) for c, v in (mean.get("strain") or {}).items()}
        if any(c not in self.macro for c in imposed):
            raise ValueError("mean_strain: components (Voigt) between periodic axes only")
        self.imposed = imposed
        self.free_macro = [c for c in self.macro if c not in imposed]
        self.mean_stress = (
            np.zeros(6) if mean.get("stress") is None else np.asarray(mean["stress"], float)
        )
        node_idx = np.indices(self.nodes_shape).reshape(3, -1)

        def face_nodes(name):
            a, side = _face(name)
            if a in per:
                raise ValueError(f"{name}: a face across a periodic axis has no support")
            return np.flatnonzero(node_idx[a] == (0 if side == 0 else self.nodes_shape[a] - 1))

        fixed = constrained_dofs(supports, face_nodes, used)
        dof_used = np.repeat(used, 3)
        modes = rigid_modes(self.coordinates, per)
        modes[~dof_used] = 0.0
        candidates = support_candidates(self.coordinates, dof_used & ~fixed)
        self.supported = supports_of(modes, np.flatnonzero(fixed & dof_used), candidates)
        fixed[self.supported] = True
        self.free = np.flatnonzero(dof_used & ~fixed)
        self.matrix = k
        near = modes[self.free]
        near = near[:, np.linalg.norm(near, axis=0) > 0]
        self.solver = LinearSolver(
            k[self.free][:, self.free].tocsr(), method="auto", tol=1e-11,
            near_null=near if near.size else None, dofs=self.free,
        )  # fmt: skip
        self.k_we = self.k_ee = None
        if self.macro:
            m = len(self.macro)
            kwe = np.zeros((n_dofs, m))
            kee = np.zeros((m, m))
            for p in self.phases:
                sel = ph == p
                block = self.load_matrix[sel][:, :, self.macro]  # V b^T D, the macro columns
                np.add.at(kwe, self.element_dofs[sel].ravel(), block.reshape(-1, m))
                kee += (
                    int(sel.sum()) * space.cell_volume * self.d[p][np.ix_(self.macro, self.macro)]
                )
            self.k_we, self.k_ee = kwe, kee
            if self.free_macro:
                cols = [self.macro.index(c) for c in self.free_macro]
                coupling = kwe[self.free][:, cols]
                self._w = np.column_stack(
                    [self.solver.solve(coupling[:, q])[0] for q in range(len(cols))]
                )
                self._schur = np.linalg.inv(kee[np.ix_(cols, cols)] - coupling.T @ self._w)

    def load(self, eigen_solid: np.ndarray) -> np.ndarray:
        """The nodal forces of the eigenstrains of the solid voxels (solid x 6)."""
        f = np.einsum("eij,ej->ei", self.load_matrix, eigen_solid)
        out = np.zeros(3 * self.n_nodes)
        np.add.at(out, self.element_dofs.ravel(), f.ravel())
        return out

    def solve(self, eigen_cells: np.ndarray):
        """The displacements of the nodes (n x 3, the mean strain included; NaN where only
        pores meet), the mean strain (6) and the stresses of the voxels (cells x 6, zero in
        the pores) under the eigenstrains of the voxels (cells x 6)."""
        space = self.space
        eigen = eigen_cells[self.solid]
        f = self.load(eigen)
        e = np.zeros(6)
        for c, v in self.imposed.items():
            e[c] = v
        if self.macro and self.imposed:
            imp = [self.macro.index(c) for c in self.imposed]
            f = f - self.k_we[:, imp] @ np.array(list(self.imposed.values()))
        w = self.solver.solve(f[self.free])[0]
        if self.macro and self.free_macro:
            cols = [self.macro.index(c) for c in self.free_macro]
            f_e = np.einsum("eij,ej->i", self.elastic, eigen) * space.cell_volume
            f_e = f_e[self.macro]
            if self.imposed:
                imp = [self.macro.index(c) for c in self.imposed]
                f_e = f_e - self.k_ee[:, imp] @ np.array(list(self.imposed.values()))
            rhs = (
                f_e[cols]
                + space.cell_volume * space.n * self.mean_stress[self.free_macro]
                - self.k_we[self.free][:, cols].T @ w
            )
            unknown = self._schur @ rhs
            w = w - self._w @ unknown
            for q, c in enumerate(self.free_macro):
                e[c] = unknown[q]
        u = np.zeros(3 * self.n_nodes)
        u[self.free] = w
        u = u.reshape(-1, 3)
        strain = np.einsum("ij,ej->ei", self.b_mean, u[self.element_nodes].reshape(-1, 24))
        sigma = np.zeros((space.n, 6))
        sigma[self.solid] = np.einsum("eij,ej->ei", self.elastic, strain + e - eigen)
        u = u + self.coordinates @ _tensor_of(e).T
        u[~self.used] = np.nan
        return u, e, sigma

    def full_grid(self, u: np.ndarray, e: np.ndarray) -> np.ndarray:
        """The displacements at all the corners of the voxels (``n + 1`` per axis): those of
        the last plane of a periodic axis are those of the first plus the mean strain."""
        out = u.reshape(*self.nodes_shape, 3)
        tensor = _tensor_of(e)
        for a in sorted(self.space.periodic_axes):
            jump = np.zeros(3)
            jump[a] = self.space.shape[a] * self.space.spacing[a]
            out = np.concatenate([out, np.take(out, [0], axis=a) + tensor @ jump], axis=a)
        return out


class _Voxels(VoxelSpace):
    """The fields of a voxel image: finite volumes and trilinear hexahedra."""

    def __init__(self, image: VoxelImage, phases: np.ndarray, materials, periodic_axes,
                 supports, mean):  # fmt: skip
        super().__init__(phases.shape, image.spacing, image.origin, periodic_axes)
        self.image = image
        self.phase = phases.ravel()
        self.cells = self.n
        self.weights = self.cell_volume * ~_voids(materials, self.phase)
        self.total_volume = self.cell_volume * self.n
        self.mechanics = _HexMechanics(self, self.phase, materials, supports, mean)

    def fields_shape(self) -> tuple:
        return self.shape


# ----------------------------------------------------------------------------------------
# Conforming meshes of linear tetrahedra


def p1_gradients(nodes: np.ndarray, elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The volumes of linear tetrahedra and the gradients of their barycentric
    coordinates (elements x 4 x 3)."""
    x = nodes[elements[:, :4]]
    edges = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=1)
    inv = np.linalg.inv(edges)  # the gradients of lambda_1..3 are the columns
    g = np.empty((len(elements), 4, 3))
    g[:, 1:, :] = np.transpose(inv, (0, 2, 1))
    g[:, 0, :] = -g[:, 1:, :].sum(axis=1)
    return np.abs(np.linalg.det(edges)) / 6.0, g


def _strain_matrices(g: np.ndarray) -> np.ndarray:
    """The strain-displacement matrices (elements x 6 x 12) of linear tetrahedra."""
    b = np.zeros((len(g), 6, 12))
    for a in range(4):
        gx, gy, gz = g[:, a, 0], g[:, a, 1], g[:, a, 2]
        c = 3 * a
        b[:, 0, c], b[:, 1, c + 1], b[:, 2, c + 2] = gx, gy, gz
        b[:, 3, c + 1], b[:, 3, c + 2] = gz, gy
        b[:, 4, c], b[:, 4, c + 2] = gz, gx
        b[:, 5, c], b[:, 5, c + 1] = gy, gx
    return b


class _Tetrahedra:
    """Every field linear in each tetrahedron: the temperature, the activity and the
    displacement at the nodes (lumped capacities)."""

    kind = "mesh"

    def __init__(self, mesh: TetMesh, materials, supports: dict):
        if mesh.order != 1:
            mesh = mesh.linear()
        self.mesh = mesh
        self.n = mesh.n_nodes
        self.cells = mesh.n_elements
        self.phase = mesh.phase
        self.periodic_axes: set = set()
        self.volumes, self.grads = p1_gradients(mesh.nodes, mesh.elements)
        void = _voids(materials, self.phase)
        self.solid = np.flatnonzero(~void)
        self.weights = self.lump(np.where(void, 0.0, 1.0))
        self.total_volume = float(np.prod(mesh.upper - mesh.lower))
        self.box = {name: mesh.face_nodes(*_face(name)) for name in FACES}
        self.mechanics = _TetMechanics(self, materials, supports)

    def lump(self, cell_values) -> np.ndarray:
        out = np.zeros(self.n)
        share = 0.25 * self.volumes * np.broadcast_to(np.asarray(cell_values, float), self.cells)
        np.add.at(out, self.mesh.elements.ravel(), np.repeat(share, 4))
        return out

    def at_cells(self, values: np.ndarray) -> np.ndarray:
        return values[self.mesh.elements].mean(axis=1)

    def operator(self, kappa: np.ndarray, faces) -> tuple[sp.csr_matrix, dict]:
        g = self.grads
        local = np.einsum("e,eai,eij,ebj->eab", self.volumes, g, kappa, g)
        return assembled(self.mesh.elements, local, self.n), {}

    def drift(self, perm, omega, temperature, hydrostatic, faces) -> tuple[sp.csr_matrix, dict]:
        """The conductances with the drift of the species by the gradient of the
        hydrostatic stress (at the nodes): ``-int P u (Omega / R T) grad sigma_h . grad
        v`` (Galerkin)."""
        g = self.grads
        e = self.mesh.elements
        grad_s = np.einsum("eai,ea->ei", g, hydrostatic[e])
        w = (omega / (R_GAS * temperature[e].mean(axis=1)))[:, None] * grad_s
        pw = np.einsum("eij,ej->ei", perm, w)
        rows = -np.einsum("ei,eai->ea", pw, g) * (0.25 * self.volumes)[:, None]
        local = np.broadcast_to(rows[:, :, None], (len(e), 4, 4))
        k, loads = self.operator(perm, faces)
        return (k + assembled(e, np.ascontiguousarray(local), self.n)).tocsr(), loads

    def _held(self, faces) -> np.ndarray:
        """For each node, the index of the first face held that holds it (-1: free)."""
        which = np.full(self.n, -1)
        for q, name in reversed(list(enumerate(faces))):
            which[self.box[name]] = q
        return which

    def reduce(self, matrix, faces):
        """The system of the free nodes (the nodes of the faces held at their values: the
        face listed first where two meet) and its coupling with the nodes held."""
        names = list(faces)
        which = self._held(names)
        held = np.flatnonzero(which >= 0)
        if held.size == 0:
            return matrix, None
        free = np.flatnonzero(which < 0)
        a = sp.csr_matrix(matrix)
        structure = (free, held, which[held], names, a[free][:, held].tocsr())
        return a[free][:, free].tocsr(), structure

    def reduced_rhs(self, rhs, structure, values: dict, loads: dict) -> np.ndarray:
        if structure is None:
            return rhs
        free, _, which, names, coupling = structure
        imposed = np.array([values[names[q]] for q in which])
        return rhs[free] - coupling @ imposed

    def expand(self, x, structure, values: dict) -> np.ndarray:
        if structure is None:
            return x
        free, held, which, names, _ = structure
        out = np.empty(self.n)
        out[free] = x
        out[held] = [values[names[q]] for q in which]
        return out

    def fields_shape(self) -> tuple:
        return (self.n,)


class _TetMechanics:
    """Linear tetrahedra (the solid ones) loaded by the eigenstrain of each element."""

    def __init__(self, space: _Tetrahedra, materials, supports: dict):
        mesh = space.mesh
        self.space = space
        self.n_nodes = mesh.n_nodes
        self.coordinates = mesh.nodes
        self.solid = space.solid
        elements = mesh.elements[self.solid]
        self.b = _strain_matrices(space.grads[self.solid])
        ph = space.phase[self.solid]
        self.elastic = np.empty((len(elements), 6, 6))
        for p in np.unique(ph):
            self.elastic[ph == p] = materials[int(p)].elastic_matrix()
        self.vol = space.volumes[self.solid]
        self.element_dofs = (elements[:, :, None] * 3 + np.arange(3)).reshape(-1, 12)
        bt = np.transpose(self.b, (0, 2, 1))
        local = np.einsum("e,eai,eij,ejb->eab", self.vol, bt, self.elastic, self.b)
        n_dofs = 3 * self.n_nodes
        k = assembled(self.element_dofs, local, n_dofs)
        used = np.zeros(self.n_nodes, dtype=bool)
        used[elements.ravel()] = True
        self.used = used
        fixed = constrained_dofs(supports, lambda name: space.box[name], used)
        dof_used = np.repeat(used, 3)
        modes = rigid_modes(self.coordinates)
        modes[~dof_used] = 0.0
        candidates = support_candidates(self.coordinates, dof_used & ~fixed)
        self.supported = supports_of(modes, np.flatnonzero(fixed & dof_used), candidates)
        fixed[self.supported] = True
        self.free = np.flatnonzero(dof_used & ~fixed)
        self.matrix = k
        near = modes[self.free]
        near = near[:, np.linalg.norm(near, axis=0) > 0]
        self.solver = LinearSolver(
            k[self.free][:, self.free].tocsr(), method="auto", tol=1e-11,
            near_null=near if near.size else None, dofs=self.free,
        )  # fmt: skip

    def solve(self, eigen_cells: np.ndarray):
        """The displacements of the nodes (n x 3, NaN at the nodes of pores only), the mean
        strain (zero) and the stresses of the elements (elements x 6, zero in the pores)
        under their eigenstrains (elements x 6)."""
        eigen = eigen_cells[self.solid]
        forces = np.einsum("e,eia,eij,ej->ea", self.vol, self.b, self.elastic, eigen)
        f = np.zeros(3 * self.n_nodes)
        np.add.at(f, self.element_dofs.ravel(), forces.ravel())
        u = np.zeros(3 * self.n_nodes)
        u[self.free] = self.solver.solve(f[self.free])[0]
        strain = np.einsum("eij,ej->ei", self.b, u[self.element_dofs])
        sigma = np.zeros((self.space.cells, 6))
        sigma[self.solid] = np.einsum("eij,ej->ei", self.elastic, strain - eigen)
        u = u.reshape(-1, 3)
        u[~self.used] = np.nan
        return u, np.zeros(6), sigma


# ----------------------------------------------------------------------------------------
# Results


@dataclass
class ThermoChemoElasticResult:
    """The fields at the output ``times``: ``temperature`` and ``activity`` (at the voxels,
    or at the nodes of a mesh), ``concentration`` (``S u``: at the voxels, or at the
    elements of a mesh), ``displacement`` (at all the corners of the voxels, or at the
    nodes), ``stress`` (at the voxels or the elements, Voigt ``11, 22, 33, 23, 31, 12``)
    and the ``mean_strain`` of the periodic axes (zero without them), each with the times
    first; ``history``: the mean temperature of the solid, the mean concentration over the
    box (the uptake), the mean stress over the box and the displacements of the
    ``points`` (of the nodes nearest to them) at the same times; ``nodes``: the
    coordinates of the nodes of the displacements."""

    times: np.ndarray
    temperature: np.ndarray
    activity: np.ndarray
    concentration: np.ndarray
    displacement: np.ndarray
    stress: np.ndarray
    mean_strain: np.ndarray
    history: dict
    nodes: np.ndarray
    kind: str
    sample: object = field(repr=False, default=None)
    info: dict = field(default_factory=dict)

    @property
    def hydrostatic(self) -> np.ndarray:
        """The hydrostatic stress ``tr(sigma) / 3`` at the output times."""
        return self.stress[..., :3].mean(axis=-1)

    @property
    def von_mises(self) -> np.ndarray:
        """The von Mises stress at the output times."""
        s = self.stress
        d = s[..., :3] - s[..., :3].mean(axis=-1, keepdims=True)
        return np.sqrt(1.5 * (np.sum(d * d, axis=-1) + 2.0 * np.sum(s[..., 3:] ** 2, axis=-1)))

    def curvature(self, step: int = -1, *, axis: int = 0, normal: int = 2, span=None) -> float:
        """The curvature of a strip along ``axis`` bending in the plane of ``normal`` at the
        output time ``step``: minus the second derivative of the deflection of the nodes
        of the plane nearest to its mid-plane (least squares ``w = -k x^2 / 2 + a x + b +
        c y``, as :func:`~morphogen.coupling.strip_curvature`), positive when it bends
        towards its lower face; ``span``: the interval of ``x`` fitted (the ends left
        out)."""
        x = self.nodes
        d = self.displacement[step].reshape(-1, 3)
        z = x[:, normal]
        mid = 0.5 * (z.min() + z.max())
        levels = np.unique(z)
        level = levels[np.argmin(np.abs(levels - mid))]
        close = (np.abs(z - level) <= 1e-9 * max(float(np.ptp(z)), 1.0)) & np.isfinite(d).all(1)
        if span is not None:
            close &= (x[:, axis] >= span[0]) & (x[:, axis] <= span[1])
        s = x[close, axis]
        y = x[close, 3 - axis - normal]
        basis = np.column_stack([0.5 * s**2, s, np.ones_like(s), y])
        coef, *_ = np.linalg.lstsq(basis, d[close, normal], rcond=None)
        return -float(coef[0])

    def write_xdmf(self, path) -> Path:
        """An XDMF time series (``path``, read by ParaView) and its raw binary arrays, one
        per field and time: the temperature, activity, concentration, hydrostatic and von
        Mises stresses at the voxels (or at the nodes and elements of a mesh), the
        displacement at the nodes."""
        path = Path(path)
        grids = []

        def raw(name, data, k):
            target = path.parent / f"{path.stem}_{name}_{k}.raw"
            target.write_bytes(np.ascontiguousarray(data, dtype="<f8").tobytes())
            return target.name

        if self.kind == "mesh":
            mesh = self.sample
            n, m = mesh.n_nodes, mesh.n_elements
            (path.parent / f"{path.stem}_tetrahedra.raw").write_bytes(
                np.ascontiguousarray(mesh.elements[:, :4], dtype="<i8").tobytes()
            )
            (path.parent / f"{path.stem}_nodes.raw").write_bytes(
                np.ascontiguousarray(mesh.nodes, dtype="<f8").tobytes()
            )
            topology = (
                f'<Topology TopologyType="Tetrahedron" NumberOfElements="{m}">\n'
                f'          <DataItem Dimensions="{m} 4" NumberType="Int" Precision="8" '
                f'Format="Binary" Endian="Little">{path.stem}_tetrahedra.raw</DataItem>\n'
                "        </Topology>\n"
                '        <Geometry GeometryType="XYZ">\n'
                f'          <DataItem Dimensions="{n} 3" NumberType="Float" Precision="8" '
                f'Format="Binary" Endian="Little">{path.stem}_nodes.raw</DataItem>\n'
                "        </Geometry>"
            )
        else:
            nx, ny, nz = self.temperature.shape[1:]
            h, o = self.info["spacing"], self.info["origin"]
            cell_dims, node_dims = f"{nz} {ny} {nx}", f"{nz + 1} {ny + 1} {nx + 1}"
            topology = (
                f'<Topology TopologyType="3DCoRectMesh" Dimensions="{node_dims}"/>\n'
                '        <Geometry GeometryType="ORIGIN_DXDYDZ">\n'
                '          <DataItem Dimensions="3" NumberType="Float" Precision="8" '
                f'Format="XML">{o[2]!r} {o[1]!r} {o[0]!r}</DataItem>\n'
                '          <DataItem Dimensions="3" NumberType="Float" Precision="8" '
                f'Format="XML">{h[2]!r} {h[1]!r} {h[0]!r}</DataItem>\n'
                "        </Geometry>"
            )
        for k, t in enumerate(self.times):
            if self.kind == "mesh":
                arrays = [
                    ("temperature", "Scalar", "Node", f"{n}", self.temperature[k]),
                    ("activity", "Scalar", "Node", f"{n}", self.activity[k]),
                    ("concentration", "Scalar", "Cell", f"{m}", self.concentration[k]),
                    ("hydrostatic", "Scalar", "Cell", f"{m}", self.hydrostatic[k]),
                    ("von_mises", "Scalar", "Cell", f"{m}", self.von_mises[k]),
                    ("displacement", "Vector", "Node", f"{n} 3",
                     np.nan_to_num(self.displacement[k])),
                ]  # fmt: skip
            else:

                def cells(a):
                    return np.transpose(a, (2, 1, 0))

                arrays = [
                    ("temperature", "Scalar", "Cell", cell_dims, cells(self.temperature[k])),
                    ("activity", "Scalar", "Cell", cell_dims, cells(self.activity[k])),
                    ("concentration", "Scalar", "Cell", cell_dims, cells(self.concentration[k])),
                    ("hydrostatic", "Scalar", "Cell", cell_dims, cells(self.hydrostatic[k])),
                    ("von_mises", "Scalar", "Cell", cell_dims, cells(self.von_mises[k])),
                    ("displacement", "Vector", "Node", node_dims + " 3",
                     np.nan_to_num(self.displacement[k]).transpose(2, 1, 0, 3)),
                ]  # fmt: skip
            attrs = "\n".join(
                f'        <Attribute Name="{name}" AttributeType="{kind}" Center="{center}">\n'
                f'          <DataItem Dimensions="{dims}" NumberType="Float" Precision="8" '
                f'Format="Binary" Endian="Little">{raw(name, data, k)}</DataItem>\n'
                "        </Attribute>"
                for name, kind, center, dims, data in arrays
            )
            grids.append(
                f'      <Grid Name="t{k}" GridType="Uniform">\n'
                f'        <Time Value="{float(t)!r}"/>\n'
                f"        {topology}\n{attrs}\n"
                "      </Grid>"
            )
        path.write_text(
            '<?xml version="1.0" ?>\n<Xdmf Version="3.0">\n  <Domain>\n'
            '    <Grid Name="fields" GridType="Collection" CollectionType="Temporal">\n'
            + "\n".join(grids)
            + "\n    </Grid>\n  </Domain>\n</Xdmf>\n",
            encoding="ascii",
        )
        return path


def time_steps(times, steps: int) -> list[tuple[float, bool]]:
    """The steps of the output ``times`` (their lengths, and whether an output time ends
    them): ``steps`` per interval, those of the interval from 0 growing geometrically
    (ratio 1.25) from about a ten-thousandth of it; an output at 0 is a step of length 0."""
    out = []
    t = 0.0
    for t_out in np.asarray(times, dtype=float):
        span = float(t_out) - t
        if span <= 0:
            out.append((0.0, True))
            continue
        if t == 0.0:
            r = 1.25
            dts = r ** np.arange(steps)
            dts *= span / dts.sum()
        else:
            dts = np.full(steps, span / steps)
        out.extend((float(dt), q == steps - 1) for q, dt in enumerate(dts))
        t = float(t_out)
    return out


# ----------------------------------------------------------------------------------------
# The problem


class ThermoChemoElastic:
    """The coupled problem of a sample: a voxel image (of phases) or a mesh of linear
    tetrahedra (their phases), the :class:`ThermoChemoMaterial` of each phase (or a
    :class:`~morphogen.homogenization.materials.Material`, or a mapping of coefficients),
    the reference temperature (no thermal strain; the coefficients given there), the
    initial state (a uniform temperature and activity; the chemical strains count from
    the concentration of ``reference_activity``, by default the initial activity), the
    faces held at a temperature or an activity (a value or a function of time), the
    supports of the faces (``clamped``, ``symmetry``, ``free``), the periodic axes of a
    voxel image (by default those of a periodic image that no face held or supported
    crosses) and the mean strain (Voigt components, between periodic axes) or the mean
    stress (6 values, zero by default) of the periodic axes."""

    def __init__(
        self,
        sample,
        materials: dict,
        *,
        reference_temperature: float = 293.15,
        initial_temperature: float | None = None,
        initial_activity: float = 0.0,
        reference_activity: float | None = None,
        temperature: dict | None = None,
        activity: dict | None = None,
        supports: dict | None = None,
        periodic=None,
        mean_strain: dict | None = None,
        mean_stress=None,
    ):
        self.materials = {}
        for p, m in materials.items():
            if isinstance(m, Material):
                m = ThermoChemoMaterial.from_material(m)
            elif isinstance(m, dict):
                m = ThermoChemoMaterial(**m)
            self.materials[int(p)] = m
        self.reference_temperature = float(reference_temperature)
        self.initial_temperature = float(
            reference_temperature if initial_temperature is None else initial_temperature
        )
        if self.reference_temperature <= 0 or self.initial_temperature <= 0:
            raise ValueError("temperatures: in kelvins, positive")
        self.initial_activity = float(initial_activity)
        self.reference_activity = float(
            initial_activity if reference_activity is None else reference_activity
        )
        self.temperature_faces = dict(temperature or {})
        self.activity_faces = dict(activity or {})
        supports = dict(supports or {})
        for name in (*self.temperature_faces, *self.activity_faces, *supports):
            _face(name)
        if isinstance(sample, TetMesh):
            self._check_phases(np.unique(sample.phase))
            if periodic:
                raise ValueError("periodic: voxel images only (meshes: faces held or free)")
            if mean_strain or mean_stress is not None:
                raise ValueError("mean_strain, mean_stress: periodic voxel images only")
            self.space = _Tetrahedra(sample, self.materials, supports)
        else:
            if not isinstance(sample, VoxelImage):
                sample = VoxelImage(np.asarray(sample))
            labels = np.asarray(sample.data).astype(np.int64)
            if labels.ndim != 3:
                raise ValueError("sample: a 3D voxel image of phases, or a TetMesh")
            self._check_phases(np.unique(labels))
            held = {_face(f)[0] for f in (*self.temperature_faces, *self.activity_faces)}
            held |= {_face(f)[0] for f, kind in supports.items() if kind != "free"}
            if periodic is None or isinstance(periodic, bool):
                wanted = sample.periodic if periodic is None else periodic
                axes = {a for a in range(3) if wanted and a not in held}
            else:
                axes = {int(a) for a in periodic}
                if axes & held:
                    raise ValueError(
                        f"periodic: the axes {sorted(axes & held)} have faces held or supported"
                    )
            mean = {"strain": dict(mean_strain or {}), "stress": mean_stress}
            self.space = _Voxels(sample, labels, self.materials, axes, supports, mean)
        self.sample = sample
        self._coefficients()

    def _check_phases(self, phases) -> None:
        missing = [int(p) for p in phases if int(p) not in self.materials]
        if missing:
            raise ValueError(f"materials: no material for the phases {missing}")

    def _coefficients(self) -> None:
        """The coefficients of the cells (voxels or elements)."""
        space, mats = self.space, self.materials
        ph = space.phase

        def per_cell(fn, shape=()):
            table = np.zeros((max(mats) + 1, *shape))
            for p, m in mats.items():
                table[p] = 0.0 if m.void else fn(m)
            return table[ph]

        self.void = _voids(mats, ph)
        self.conductivity = per_cell(lambda m: _tensor(m.conductivity, "conductivity"), (3, 3))
        self.diffusivity = per_cell(lambda m: _tensor(m.diffusivity, "diffusivity"), (3, 3))
        self.heat_capacity = per_cell(lambda m: m.heat_capacity)
        self.solubility = per_cell(lambda m: m.solubility)
        self.rate = per_cell(lambda m: m.reaction_rate)
        self.heat = per_cell(lambda m: m.reaction_heat)
        self.omega = per_cell(lambda m: m.molar_volume)
        self.q_diffusion = per_cell(lambda m: m.diffusion_activation)
        self.q_reaction = per_cell(lambda m: m.reaction_activation)
        self.alpha = per_cell(lambda m: m.thermal_strain(), (6,))
        self.beta = per_cell(lambda m: m.chemical_strain(), (6,))
        self.cap_t = np.maximum(space.lump(self.heat_capacity), _TINY)
        self.cap_u = np.maximum(space.lump(self.solubility), _TINY)
        self.arrhenius = bool(np.any(self.q_diffusion) or np.any(self.q_reaction))
        self.reacts = bool(np.any(self.rate > 0))
        self.heats = bool(np.any(self.heat * self.rate != 0))
        self.drifts = bool(np.any(self.omega != 0))
        self.coupled = self.heats or self.drifts
        # the fields that nothing moves from their uniform initial values (no face held,
        # no source, no sink, no drift) are not solved
        self.thermal = bool(self.temperature_faces) or self.heats
        self.chemical = bool(self.activity_faces) or self.reacts or self.drifts

    # -- parts of the systems ---------------------------------------------------------------

    def _factors(self, temperature):
        """The Arrhenius factors of the diffusivity and of the rate at the cells."""
        if not self.arrhenius:
            return 1.0, 1.0
        t = self.space.at_cells(temperature)
        ref = self.reference_temperature
        return arrhenius(self.q_diffusion, t, ref), arrhenius(self.q_reaction, t, ref)

    def species_operator(self, temperature, hydrostatic=None):
        """The matrix of the species (its conductances, with the drift by the stress and
        the consumption) at a temperature and a hydrostatic stress (at the unknowns), and
        the loads of its faces held."""
        space = self.space
        f_d, f_k = self._factors(temperature)
        perm = self.diffusivity * (self.solubility * f_d)[:, None, None]
        if self.drifts and hydrostatic is not None:
            k, loads = space.drift(perm, self.omega, temperature, hydrostatic, self.activity_faces)
        else:
            k, loads = space.operator(perm, self.activity_faces)
        if self.reacts:
            k = k + sp.diags(space.lump(self.rate * f_k * self.solubility))
        return k.tocsr(), loads

    def heat_operator(self):
        """The matrix of the temperature (its conductances) and the loads of its faces held."""
        return self.space.operator(self.conductivity, self.temperature_faces)

    def source(self, temperature, activity):
        """The heat released by the reaction at the unknowns, ``H k(T) S u`` lumped."""
        if not self.heats:
            return 0.0
        space = self.space
        _, f_k = self._factors(temperature)
        rate = self.heat * self.rate * f_k * self.solubility * space.at_cells(activity)
        return space.lump(rate)

    def eigenstrain(self, temperature, activity) -> np.ndarray:
        """The eigenstrains of the cells (Voigt, engineering shears)."""
        space = self.space
        dt = space.at_cells(temperature) - self.reference_temperature
        dc = self.solubility * (space.at_cells(activity) - self.reference_activity)
        return self.alpha * dt[:, None] + self.beta * dc[:, None]

    def hydrostatic_at_unknowns(self, sigma) -> np.ndarray:
        """The hydrostatic stress at the unknowns of the species: that of the voxels, or the
        volume-weighted mean of the solid elements at the nodes."""
        space = self.space
        s = sigma[:, :3].mean(axis=1)
        if space.kind == "voxels":
            return s
        w = space.lump(np.where(self.void, 0.0, 1.0))
        total = space.lump(np.where(self.void, 0.0, s))
        return np.divide(total, w, out=np.zeros_like(total), where=w > 0)

    def _solve(self, matrix, rhs, faces, loads, t, x0, *, symmetric=True, cache=None):
        """The solution of a system whose faces ``faces`` are held at their values at the
        time ``t`` (the factorization kept in ``cache`` for a constant matrix), solved for
        its change from ``x0`` (the last iterate): with insulated faces and steps long
        before the time of diffusion, ``C / dt`` alone fixes the uniform field, and the
        round-off of a solve for the field itself would drift it at every step."""
        space = self.space
        values = {name: _value_at(v, t) for name, v in faces.items()}
        if cache is not None and "solver" in cache:
            solver, structure = cache["solver"], cache["structure"]
        else:
            reduced, structure = space.reduce(matrix, faces)
            n = reduced.shape[0]
            method = "direct" if n <= DIRECT_LIMIT else "auto"
            solver = LinearSolver(reduced, method=method, tol=1e-12, symmetric=symmetric)
            if cache is not None:
                cache["solver"], cache["structure"] = solver, structure
        b = space.reduced_rhs(rhs, structure, values, loads)
        start = x0[structure[0]] if structure is not None else x0
        x = start + solver.solve(b - solver.matrix @ start)[0]
        return space.expand(x, structure, values)

    @staticmethod
    def _rate(k_full, theta, faces, loads, t):
        """``-K theta`` and the loads of the faces held at the time ``t``: the rate of the
        content without the capacity (the explicit part of the trapezoidal stage)."""
        out = -(k_full @ theta)
        for name, value in faces.items():
            if name in loads:
                out = out + loads[name] * _value_at(value, t)
        return out

    # -- time ------------------------------------------------------------------------------

    def run(
        self,
        times,
        *,
        steps: int = 20,
        scheme: str = "euler",
        points=None,
        tol: float = 1e-10,
        max_iterations: int = 100,
    ) -> ThermoChemoElasticResult:
        """The fields at the output ``times`` (increasing, from 0): ``steps`` steps per
        interval, ``scheme`` ``"euler"`` or ``"trbdf2"``; ``points``: coordinates (rows)
        whose displacements (those of the nodes nearest to them) the history follows;
        ``tol``: the relative change of the temperature and the activity at which the
        iterations of a step of a problem coupled both ways stop."""
        if scheme not in SCHEMES:
            raise ValueError(f"scheme: {', '.join(SCHEMES)}")
        times = np.atleast_1d(np.asarray(times, dtype=float))
        if times.size == 0 or np.any(np.diff(times) <= 0) or times[0] < 0:
            raise ValueError("times: increasing and non-negative")
        space, mech = self.space, self.space.mechanics
        temp = np.full(space.n, self.initial_temperature)
        act = np.full(space.n, self.initial_activity)
        k_t, loads_t = self.heat_operator()
        hyd = None
        if self.drifts:
            hyd = self.hydrostatic_at_unknowns(mech.solve(self.eigenstrain(temp, act))[2])
        constant_u = not (self.arrhenius or self.drifts)
        species = self.species_operator(temp) if constant_u else None
        caches_t: dict = {}
        caches_u: dict = {}
        iterations: list[int] = []

        def species_parts(temp_now):
            return species if constant_u else self.species_operator(temp_now, hyd)

        def hydrostatic(temp_now, act_now):
            return self.hydrostatic_at_unknowns(mech.solve(self.eigenstrain(temp_now, act_now))[2])

        def stage(tau, base_t, base_u, t_new, temp0, act0):
            """One implicit stage, ``(C / tau + K) theta = C base / tau + loads + sources`` at
            ``t_new``. Coupled both ways, the staggered pass (the temperature with the
            source of the last iterate, then the species at that temperature and in the
            stress of the last iterate) is a map ``G`` whose fixed point is the stage,
            reached by Aitken's dynamic relaxation ``z <- z + w (G(z) - z)`` (Irons & Tuck
            1969; Küttler & Wall 2008), ``w`` from the last two residuals."""
            nonlocal hyd
            temp_z, act_z = temp0, act0
            weight, previous = 1.0, None
            count = 0
            for count in range(1, max_iterations + 1):
                if self.drifts and count > 1:
                    hyd = hydrostatic(temp_z, act_z)
                temp_g = temp_z
                if self.thermal:
                    a_t = (sp.diags(self.cap_t / tau) + k_t).tocsr()
                    temp_g = self._solve(
                        a_t, self.cap_t / tau * base_t + self.source(temp_z, act_z),
                        self.temperature_faces, loads_t, t_new, temp_z,
                        cache=caches_t.setdefault(tau, {}),
                    )  # fmt: skip
                act_g = act_z
                if self.chemical:
                    k_u, loads_u = species_parts(temp_g)
                    a_u = (sp.diags(self.cap_u / tau) + k_u).tocsr()
                    act_g = self._solve(
                        a_u, self.cap_u / tau * base_u, self.activity_faces, loads_u, t_new,
                        act_z, symmetric=not self.drifts,
                        cache=caches_u.setdefault(tau, {}) if constant_u else None,
                    )  # fmt: skip
                if not self.coupled:
                    temp_z, act_z = temp_g, act_g
                    break
                scale_t = max(float(np.max(np.abs(temp_g))), 1e-300)
                scale_u = max(float(np.max(np.abs(act_g))), 1e-300)
                residual = np.concatenate([(temp_g - temp_z) / scale_t, (act_g - act_z) / scale_u])
                change = float(np.max(np.abs(residual)))
                if change <= tol:
                    temp_z, act_z = temp_g, act_g
                    break
                if previous is not None:
                    diff = residual - previous
                    weight = -weight * float(previous @ diff) / max(float(diff @ diff), 1e-300)
                temp_z = temp_z + weight * (temp_g - temp_z)
                act_z = act_z + weight * (act_g - act_z)
                previous = residual
            if self.drifts:
                hyd = hydrostatic(temp_z, act_z)
            iterations.append(count)
            return temp_z, act_z

        records: dict[str, list] = {
            k: [] for k in ("temperature", "activity", "concentration", "displacement",
                            "stress", "mean_strain")
        }  # fmt: skip
        t = 0.0
        for dt, output in time_steps(times, steps):
            if dt > 0:
                if scheme == "euler":
                    temp, act = stage(dt, temp, act, t + dt, temp, act)
                else:
                    g = GAMMA
                    k_u, loads_u = species_parts(temp)
                    f_t = self._rate(k_t, temp, self.temperature_faces, loads_t, t)
                    f_t = f_t + self.source(temp, act)
                    f_u = self._rate(k_u, act, self.activity_faces, loads_u, t)
                    tau = 0.5 * g * dt
                    temp_g, act_g = stage(
                        tau, temp + tau * f_t / self.cap_t, act + tau * f_u / self.cap_u,
                        t + g * dt, temp, act,
                    )  # fmt: skip
                    c1, c0 = 1.0 / (g * (2.0 - g)), (1.0 - g) ** 2 / (g * (2.0 - g))
                    temp, act = stage(
                        (1.0 - g) / (2.0 - g) * dt, c1 * temp_g - c0 * temp,
                        c1 * act_g - c0 * act, t + dt, temp_g, act_g,
                    )  # fmt: skip
                t += dt
            if output:
                u, e, sigma = mech.solve(self.eigenstrain(temp, act))
                records["temperature"].append(temp.reshape(space.fields_shape()).copy())
                records["activity"].append(act.reshape(space.fields_shape()).copy())
                conc = self.solubility * space.at_cells(act)
                if space.kind == "voxels":
                    records["concentration"].append(conc.reshape(space.shape))
                    records["displacement"].append(mech.full_grid(u, e))
                    records["stress"].append(sigma.reshape(*space.shape, 6))
                else:
                    records["concentration"].append(conc)
                    records["displacement"].append(u)
                    records["stress"].append(sigma)
                records["mean_strain"].append(e)
        out = {k: np.asarray(v) for k, v in records.items()}
        cells = space.cell_volume * np.ones(space.n) if space.kind == "voxels" else space.volumes
        box = space.total_volume
        history = {
            "mean_temperature": np.array(
                [np.sum(space.weights * x.ravel()) / space.weights.sum() for x in
                 out["temperature"]]
            ),
            "uptake": np.array([float(cells @ c.ravel()) / box for c in out["concentration"]]),
            "mean_stress": np.array([cells @ s.reshape(-1, 6) / box for s in out["stress"]]),
        }  # fmt: skip
        if space.kind == "voxels":
            nodes = space.origin + np.indices(np.asarray(space.shape) + 1).reshape(3, -1).T * (
                space.spacing
            )
            info = {"spacing": tuple(map(float, space.spacing)),
                    "origin": tuple(map(float, space.origin))}  # fmt: skip
        else:
            nodes = space.mesh.nodes
            info = {}
        if points is not None:
            nearest = cKDTree(nodes).query(np.atleast_2d(np.asarray(points, dtype=float)))[1]
            history["points"] = np.asarray([d.reshape(-1, 3)[nearest] for d in out["displacement"]])
        info.update(scheme=scheme, steps=int(steps), iterations=int(np.sum(iterations)),
                    max_iterations=int(max(iterations, default=0)))  # fmt: skip
        return ThermoChemoElasticResult(
            times=times, temperature=out["temperature"], activity=out["activity"],
            concentration=out["concentration"], displacement=out["displacement"],
            stress=out["stress"], mean_strain=out["mean_strain"], history=history, nodes=nodes,
            kind=space.kind, sample=space.mesh if space.kind == "mesh" else self.sample,
            info=info,
        )  # fmt: skip


def thermochemoelastic(sample, materials: dict, times, **options) -> ThermoChemoElasticResult:
    """The fields of the coupled problem of ``sample`` at the output ``times``: the options
    of :class:`ThermoChemoElastic` (the boundary conditions, the reference and initial
    states, the periodic axes) and of :meth:`ThermoChemoElastic.run` (``steps``,
    ``scheme``, ``points``, ``tol``, ``max_iterations``)."""
    names = ("steps", "scheme", "points", "tol", "max_iterations")
    run_options = {k: options.pop(k) for k in names if k in options}
    return ThermoChemoElastic(sample, materials, **options).run(times, **run_options)


def beam_curvature(z, modulus, eigenstrain) -> float:
    """The curvature of a free beam whose fibres at the heights ``z`` (increasing, the
    points of a trapezoidal rule through its thickness) have the Young's moduli
    ``modulus`` and the eigenstrains ``eigenstrain`` along its axis (uniaxial stress):
    ``k = (A M* - B N*) / (A D - B^2)`` with ``A, B, D = int E (1, z, z^2) dz`` and
    ``N*, M* = int E eps* (1, z) dz``; positive when it bends towards its lower face."""
    z = np.asarray(z, dtype=float)
    e = np.broadcast_to(np.asarray(modulus, dtype=float), z.shape)
    s = np.broadcast_to(np.asarray(eigenstrain, dtype=float), z.shape)
    a, b, d = (trapezoid(e * z**p, z) for p in (0, 1, 2))
    n, m = trapezoid(e * s, z), trapezoid(e * s * z, z)
    return float((a * m - b * n) / (a * d - b * b))
