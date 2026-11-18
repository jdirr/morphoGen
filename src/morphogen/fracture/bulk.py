"""The bulk of the crack problems by the behaviours of morphoGen's points: von Mises
elasto-(visco)plasticity (logarithmic strains in finite strains), porous plasticity (GTN,
Rousselier, elliptic: :mod:`morphogen.homogenization.porous`, the ductile damage of metals
by the growth of their voids), crystal plasticity (multiplicative in finite strains, each
grain turned by its orientation) and MFront laws through MGIS, in small and finite
strains, with their internal variables.

The points are those of the nonlinear finite elements of the homogenization
(:mod:`morphogen.homogenization.nonlinear`, :mod:`morphogen.homogenization.finite`): one
per linear tetrahedron, integrated implicitly over each step from the internal variables
committed at the end of the previous converged step, with their consistent tangents,
assembled into the nodal forces ``∫ Bᵀ σ dV`` (``∫ P · ∇N dV_0`` in finite strains) and the
tangent stiffness of the Newton solver of :class:`~morphogen.fracture.cohesive.
CohesiveProblem`.

Energies (per unit reference volume, at the committed state):

* the *work* density ``w = ∫ P : dF`` (``∫ σ : dε``), accumulated over the converged steps
  by the trapezoidal rule: the stored and dissipated energies together, the energy
  density of the deformation theory of plasticity; summed over the body it is the work
  of the loads;
* the *free* energy ``ψ``: elastic, ``1/2 σ : C⁻¹ : σ`` in small strains, ``1/2 T : (ε - εᵖ)``
  with the stress ``T`` conjugate to the logarithmic strain ``ε`` for von Mises points in
  finite strains, ``1/2 Sₑ : Eₑ`` of the elastic Green-Lagrange strain of ``Fₑ = F Fₚ⁻¹`` for
  crystals; and the energy stored by hardening: ``1/2 H p² + Σ Q (p - (1 - e^(-b p)) / b)``
  and ``Σ 3/(4 C) X : X`` for von Mises points (``∂ψ/∂p = R(p)``, ``∂ψ/∂X = 3 X / 2C``),
  ``Σ C α²/2`` of the kinematic hardening of the slip systems of crystals (their isotropic
  hardening is counted as dissipated, and so is the hardening of the matrix of porous
  points, isotropic and kinematic, whose plastic strain enters the elastic energy as that
  of von Mises points).
  MFront laws have none (NaN).

The configurational forces of the internal variables (:meth:`PointBulk.configurational`):
``Div (ψ I - Hᵀ P) = ∂ψ/∂α · ∇α`` in a homogeneous body (``P = ∂ψ/∂F`` at fixed internal
variables ``α``), with ``∂ψ/∂εᵖ = -σ`` (``-T``), ``∂ψ/∂Fₚ⁻¹ = Fᵀ Fₑ Sₑ`` for crystals, and the
derivatives of the stored energies above; the gradients of the internal variables (one
value per element) come from their projection on the nodes (volume-weighted means of
the elements around each node, the two lips of a crack apart, and on the crack meshes of a
morphology its materials apart: a node of their interfaces has a value for each).
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from morphogen.homogenization import finite as fin
from morphogen.homogenization.behaviours import (
    _KIN,
    CRYSTAL,
    HILL,
    MAX_ISOTROPIC,
    MAX_KINEMATIC,
    MFRONT,
    PLASTIC,
    STATE_P,
    STATE_X,
)
from morphogen.homogenization.crystal import CP_ALPHA
from morphogen.homogenization.fe import GAUSS, _expand_columns, element_colours, node_pattern
from morphogen.homogenization.materials import material_table
from morphogen.homogenization.nonlinear import (
    _fill_tangent,
    _internal_forces,
    _point_strains,
    _update,
    behaviour_table,
    external_points,
)
from morphogen.mesh.tetra import TetMesh

#: Weights of the Voigt components (tensor shears) in the double contraction of symmetric
#: tensors
_TWICE = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])


def nonlinear_materials(materials, phases) -> bool:
    """Whether a phase among ``phases`` has an inelastic behaviour (von Mises, porous or
    crystal plasticity, MFront, a learned law)."""
    for p in np.unique(phases):
        m = materials.get(int(p))
        if m is None or m.void:
            continue
        if m.nonlinear:
            return True
    return False


def _tensor(voigt) -> np.ndarray:
    """Symmetric 3 x 3 tensors (..., 3, 3) of Voigt components with tensor shears."""
    v = np.asarray(voigt, dtype=float)
    out = np.empty((*v.shape[:-1], 3, 3))
    for k, (a, b) in enumerate(((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))):
        out[..., a, b] = v[..., k]
        out[..., b, a] = v[..., k]
    return out


def _tet_gradients(nodes, elements):
    """Gradients of the shape functions (elements x 4 x 3) and volumes of linear
    tetrahedra."""
    x = nodes[elements]
    dm = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=2)
    inv = np.linalg.inv(dm)
    grads = np.concatenate([-inv.sum(axis=1, keepdims=True), inv], axis=1)
    return grads, np.abs(np.linalg.det(dm)) / 6.0


class PointBulk:
    """The bulk of a crack problem (see the module): a mesh of linear tetrahedra, the
    materials of its phases (an orientation per grain of ``mesh.cell`` from
    ``mesh.orientations``) and the kinematics; :meth:`evaluate` integrates the points for a
    displacement, :meth:`commit` keeps their state at the end of a converged step."""

    def __init__(self, mesh: TetMesh, materials, finite: bool):
        if mesh.order != 1:
            raise ValueError("PointBulk: linear tetrahedra")
        self.mesh, self.finite = mesh, bool(finite)
        self.materials = materials
        self.table = material_table(mesh.phase, materials, "elastic", grains=mesh.cell,
                                    orientations=mesh.orientations)  # fmt: skip
        if np.any(self.table.mat < 0):
            raise ValueError("PointBulk: void phases (remove their elements from the mesh)")
        beh = behaviour_table(mesh.phase, materials, self.table, finite=self.finite)
        self.behaviours = beh
        self.gp, self.gw = GAUSS[1]  # one point per element
        self.kind = np.ascontiguousarray(beh.kind)
        self.pmat = np.ascontiguousarray(beh.pmat)
        self.mat = np.ascontiguousarray(self.table.mat)
        n, nc = mesh.n_elements, 9 if self.finite else 6
        self.grads = np.zeros((n, nc))
        self.wts = np.zeros(n)
        self.stress = np.zeros((n, nc))
        self.tangent = np.zeros((n, nc, nc))
        self.old = np.zeros((n, beh.width))
        self.new = np.zeros((n, beh.width))
        self.work = np.zeros(n)
        self.committed_stress = np.zeros((n, nc))
        self.committed_grads = np.zeros((n, nc))
        self.indptr_n, self.indices_n = node_pattern(mesh.elements, mesh.n_nodes)
        self.by_colour, self.colours = element_colours(mesh.elements, mesh.n_nodes)
        self.data = np.zeros(9 * int(self.indptr_n[-1]))
        self.indices = _expand_columns(self.indptr_n, self.indices_n, 3)
        lengths = np.repeat(np.diff(self.indptr_n) * 3, 3)
        self.indptr = np.zeros(3 * mesh.n_nodes + 1, dtype=np.int64)
        np.cumsum(lengths, out=self.indptr[1:])
        self.nodal = np.zeros(3 * mesh.n_nodes)
        # MFront behaviours (MGIS) and learned laws (their models)
        self._mfront = external_points(beh.mfront, self.kind, self.pmat, self.mat,
                                       self.table.rotations, self.finite)  # fmt: skip

    @property
    def n_dofs(self) -> int:
        return 3 * self.mesh.n_nodes

    def reset(self) -> None:
        """The natural state: no strain, no internal variable, no work."""
        for a in (self.old, self.new, self.work, self.committed_stress, self.committed_grads):
            a[...] = 0.0

    def evaluate(self, u, dt: float, tangent: bool):
        """Nodal forces (all dofs) and, with ``tangent``, the tangent stiffness (CSR) of
        the bulk for the displacement ``u`` (flat) at the end of a step ``dt`` from the
        committed state; None if the integration of a point fails."""
        m = self.mesh
        kernel = fin.point_gradients if self.finite else _point_strains
        kernel(m.nodes, m.elements, 1, np.ascontiguousarray(u, dtype=float), self.gp, self.gw,
               self.grads, self.wts)  # fmt: skip
        failed = _update(self.finite, self.kind, self.mat, self.pmat, self.behaviours.data,
                         self.grads, 0.0, float(dt), self.old, self.new, self.stress,
                         self.tangent, 0, 1.0)  # fmt: skip
        if self._mfront is not None:
            failed += self._mfront.integrate(self.grads, 0.0, float(dt), self.old, self.new,
                                             self.stress, self.tangent)  # fmt: skip
        if failed:
            return None
        forces = fin.internal_forces if self.finite else _internal_forces
        forces(m.nodes, m.elements, 1, self.gp, self.stress, self.wts, self.by_colour,
               self.colours, self.nodal)  # fmt: skip
        k = None
        if tangent:
            fill = fin.fill_tangent if self.finite else _fill_tangent
            fill(m.nodes, m.elements, 1, self.gp, self.tangent, self.wts, self.indptr_n,
                 self.indices_n, self.by_colour, self.colours, self.data)  # fmt: skip
            n = self.n_dofs
            k = sp.csr_matrix((self.data.copy(), self.indices, self.indptr), shape=(n, n))
        return self.nodal.copy(), k

    def commit(self) -> None:
        """Keep the state of the last evaluation (the end of a converged step) and add the
        work of the step (trapezoidal rule)."""
        increment = self.grads - self.committed_grads
        self.work += 0.5 * np.einsum("ij,ij->i", self.committed_stress + self.stress, increment)
        self.committed_stress[...] = self.stress
        self.committed_grads[...] = self.grads
        self.old, self.new = self.new, self.old

    # -- energies ------------------------------------------------------------------------

    def stress_tensors(self) -> np.ndarray:
        """The committed stresses as 3 x 3 tensors (first Piola-Kirchhoff in finite
        strains, Cauchy in small strains)."""
        if self.finite:
            return self.committed_stress.reshape(-1, 3, 3).copy()
        return _tensor(self.committed_stress)

    def _elastic(self):
        """The elastic free energy of every point, and the conjugate of the plastic strain
        (``-σ`` or ``-T``, tensor components) or of ``Fₚ⁻¹`` (9) of the crystals."""
        matrices = self.behaviours.data[0]
        state = self.old
        n = self.mesh.n_elements
        psi = np.zeros(n)
        conjugate = np.zeros((n, 9))
        crystal = self.kind == CRYSTAL
        if not self.finite:
            compliance = np.linalg.inv(matrices)
            sig = self.committed_stress
            strain = np.einsum("eij,ej->ei", compliance[self.mat], sig)  # engineering
            psi = 0.5 * np.einsum("ei,ei->e", sig, strain)
            conjugate[:, :6] = -sig
            return psi, conjugate
        others = ~crystal
        if np.any(others):
            log = fin.log_strain(self.committed_grads[others])  # engineering
            plastic = state[others, :6] * _TWICE
            elastic = log - plastic
            t = np.einsum("eij,ej->ei", matrices[self.mat[others]], elastic)
            psi[others] = 0.5 * np.einsum("ei,ei->e", t, elastic)
            conjugate[others, :6] = -t
        if np.any(crystal):
            ns = self.behaviours.data[7][self.pmat[crystal]]
            off = CP_ALPHA + 2 * ns
            rows = np.flatnonzero(crystal)
            g = np.stack([state[i, o : o + 9] for i, o in zip(rows, off, strict=True)])
            g = g.reshape(-1, 3, 3) + np.eye(3)
            f = self.committed_grads[crystal].reshape(-1, 3, 3) + np.eye(3)
            fe = f @ g
            ee = 0.5 * (np.swapaxes(fe, 1, 2) @ fe - np.eye(3))
            e6 = fin._voigt(ee) * _TWICE  # engineering
            s6 = np.einsum("eij,ej->ei", matrices[self.mat[crystal]], e6)
            psi[crystal] = 0.5 * np.einsum("ei,ei->e", s6, e6)
            conjugate[crystal] = (np.swapaxes(f, 1, 2) @ fe @ _tensor(s6)).reshape(-1, 9)
        return psi, conjugate

    def _hardening(self):
        """The energy stored by hardening of every point and the list of the hardening
        variables with their conjugates (``∂ψ/∂α``): (values, conjugates) per point."""
        data = self.behaviours.data
        params, nk, cp_par, cp_n = data[2], data[3], data[5], data[7]
        state = self.old
        n = self.mesh.n_elements
        psi = np.zeros(n)
        variables = []
        plastic = (self.kind == PLASTIC) | (self.kind == HILL)  # the same hardenings
        if np.any(plastic):
            par = params[self.pmat[plastic]]
            p = state[plastic, STATE_P]
            h = par[:, 3]
            stored = 0.5 * h * p * p
            force = h * p
            for i in range(MAX_ISOTROPIC):
                q, b = par[:, 6 + 2 * i], par[:, 7 + 2 * i]
                safe = np.where(b > 0, b, 1.0)
                decay = np.exp(-safe * p)
                stored += np.where(q != 0, q * (p - (1.0 - decay) / safe), 0.0)
                force += np.where(q != 0, q * (1.0 - decay), 0.0)
            values = np.zeros((n, 1 + 6 * MAX_KINEMATIC))
            forces = np.zeros_like(values)
            values[plastic, 0] = p
            forces[plastic, 0] = force
            for k in range(min(nk, MAX_KINEMATIC)):
                c = par[:, _KIN + 2 * k]
                x = state[plastic, STATE_X + 6 * k : STATE_X + 6 * k + 6]
                inverse = np.where(c > 0, 1.0 / np.where(c > 0, c, 1.0), 0.0)
                stored += 0.75 * inverse * np.einsum("ei,ei->e", x * _TWICE, x)
                values[plastic, 1 + 6 * k : 7 + 6 * k] = x
                forces[plastic, 1 + 6 * k : 7 + 6 * k] = 1.5 * inverse[:, None] * x * _TWICE
            psi[plastic] = stored
            variables.append((values, forces))
        crystal = self.kind == CRYSTAL
        if np.any(crystal):
            n_max = cp_par.shape[1]
            values = np.zeros((n, n_max))
            forces = np.zeros_like(values)
            for i in np.flatnonzero(crystal):
                c = self.pmat[i]
                ns = int(cp_n[c])
                alpha = state[i, CP_ALPHA : CP_ALPHA + ns]
                stiff = cp_par[c, :ns, 3]
                psi[i] = 0.5 * float(stiff @ (alpha * alpha))
                values[i, :ns] = alpha
                forces[i, :ns] = stiff * alpha
            variables.append((values, forces))
        return psi, variables

    def free_energy(self) -> np.ndarray:
        """The free energy density of every point (see the module; NaN for MFront)."""
        elastic, _ = self._elastic()
        stored, _ = self._hardening()
        psi = elastic + stored
        psi[self.kind == MFRONT] = math.nan
        return psi

    def configurational(self) -> np.ndarray:
        """``Div (ψ I - Hᵀ P) = ∂ψ/∂α · ∇α`` of every element (elements x 3; see the
        module)."""
        if np.any(self.kind == MFRONT):
            raise ValueError("configurational forces: no free energy of MFront laws")
        m = self.mesh
        grads, volumes = _tet_gradients(m.nodes, m.elements)
        _, conjugate = self._elastic()
        state = self.old
        plastic_strain = np.zeros((m.n_elements, 9))
        crystal = self.kind == CRYSTAL
        if self.finite and np.any(crystal):
            ns = self.behaviours.data[7][self.pmat[crystal]]
            off = CP_ALPHA + 2 * ns
            rows = np.flatnonzero(crystal)
            plastic_strain[crystal] = np.stack(
                [state[i, o : o + 9] for i, o in zip(rows, off, strict=True)]
            )
        others = ~crystal if self.finite else np.ones(m.n_elements, dtype=bool)
        plastic_strain[others, :6] = state[others, :6] * _TWICE  # tensor contraction weights
        pairs = [(plastic_strain, conjugate)]
        _, variables = self._hardening()
        pairs += variables
        out = np.zeros((m.n_elements, 3))
        if m.side is None:
            groups = np.zeros(m.n_elements, dtype=np.int64)
        else:  # a crack mesh of a morphology: the materials apart
            from morphogen.fracture.mixed import material_groups

            groups = material_groups(m.phase, self.materials)
        for g in np.unique(groups):
            sel = np.flatnonzero(groups == g)
            corners = m.elements[sel]
            weight = np.bincount(corners.ravel(), np.repeat(volumes[sel], 4), minlength=m.n_nodes)
            weight = np.where(weight > 0, weight, 1.0)
            for values, forces in pairs:
                nodal = np.zeros((m.n_nodes, values.shape[1]))
                for a in range(4):
                    np.add.at(nodal, corners[:, a], volumes[sel, None] * values[sel])
                nodal /= weight[:, None]
                gradient = np.einsum("eac,eak->eck", nodal[corners], grads[sel])
                out[sel] += np.einsum("ec,eck->ek", forces[sel], gradient)
        return out
