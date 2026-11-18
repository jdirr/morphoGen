"""Post-bifurcated patterns of periodic architectured materials by group theory, after
Azulay & Combescure (2024) and the thesis of R. Azulay (2023), for frames of beams in the plane
and in space (lattices of struts) and for continua (tetrahedral meshes and voxel images).

The symmetry group ``G`` of a cell of several primitive cells (its point group and its
inner translations, :mod:`morphogen.symmetry`), restricted to the symmetry of the
macroscopic loading (``R E R^T = E``), acts on the periodic fluctuations by
``T(g)``: the nodes permuted, their displacements turned by ``R`` (and the rotations of
beams by ``det R``). Its real irreducible representations split the fluctuations into
isotypic subspaces ``V_μ`` (projectors ``P_μ``), in which the stiffness matrices are
block-diagonal. For each irrep, the isotropy subgroups ``H`` with one-dimensional
fixed-point subspaces (the equivariant branching lemma) are the symmetry groups of the
bifurcated branches; the pattern of ``H`` is the displacement of ``Fix(H) ∩ V_μ`` (the
projector ``P_H = (1/|H|) Σ_h T(h)`` times ``P_μ``) that minimizes the elastic energy
``u^T K_e u`` at unit norm: the eigenvector of the smallest eigenvalue of the reduced
elastic stiffness. The order of the bifurcation points along the principal path comes
from the linear buckling ``(K_e + λ K_g) φ = 0`` of the loading, solved in each isotypic
subspace (its lowest ``λ`` and multiplicity); the critical modes of a small cell are
also identified by their projections on the isotypic subspaces. Translations are left
out of every subspace.

The isotypic subspaces are found without dense projectors: their dimension is the trace
``Σ_g w(g) tr T(g)`` and their basis the range of ``P_μ`` applied to random vectors, so
that meshes of thousands of nodes stay tractable. In a continuum, the geometric stiffness
is the initial-stress matrix ``δ_ij ∫ ∇N_a · σ⁰ ∇N_b dV`` of the stress of the periodic
linear solution under the macroscopic strain, and the group of a mesh is detected: the
operations of the cube about the centre of the box combined with translations by
multiples of half a primitive cell that map the elements onto elements of the same
phase. A mesh must have them exactly: a conforming mesh of the level set of a symmetric
cell, meshed in its irreducible part and mirrored (:func:`morphogen.mesh.mirrored_mesh`,
whose holes and inclusions are smooth), or a voxel image, cut into six tetrahedra per
voxel by the Kuhn triangulation of its grid, invariant under the operations of the cube
and under translations by even numbers of voxels (but whose staircase interfaces bias
thin ligaments); an unstructured mesh of the whole box has no symmetry.

A cell of two primitive cells along some axes is first split by its translations
(``bloch``, by default beyond ``DENSE_LIMIT`` dofs): their characters ``±1`` are those of
the wave vectors ``k`` of components 0 or ``π/a`` (Γ, X, M and R of a cubic cell of edge
``a``), and the fluctuations of ``k``, periodic or antiperiodic on the primitive cell along
each axis, have the sparse orthonormal basis of the orbit sums
``Σ_t χ_k(t) e_{t(i)} / sqrt|T|`` over the nodes ``i`` of a primitive cell. The stiffness
matrices are block-diagonal in these subspaces, which the point operations fixing ``k``
(its little co-group ``P_k``, found by conjugating the translations in the group table)
split further. By Clifford's theory, each irrep of the cell's group is induced by a star
of wave vectors and an irrep ``σ`` of ``P_k`` (identified by Frobenius reciprocity on the
characters): its dimension is that of ``σ`` times the arms of the star, its multiplicity
that of ``σ`` among the fluctuations of ``k`` and its spectrum that of the block of ``σ``
there, repeated on each arm. No basis of the whole cell is formed: the blocks have the
size of the fluctuations of a primitive cell divided among the irreps of ``P_k``. The
patterns of an irrep are found in the coordinates of the blocks of the arms of its star,
where the group acts by the matrices of the induced representation (``P_k`` on the block
and the characters of ``k``), and lifted to the cell; by default those of the irreps of
the first bifurcation only. :func:`wave_vectors` tells which wave vectors an irrep
carries: its patterns double the period along the axes where ``k_i = π/a``.

Along a nonlinear loading path (:func:`path_bifurcation`), in finite strains and with
nonlinear behaviours, the linear buckling gives way to the stability of the principal
path itself: the periodic element of
:class:`~morphogen.homogenization.nonlinear.NonlinearFEProblem` follows the path, and at
each converged step its consistent tangent (``∫ ∇N · ∂P/∂F · ∇N dV`` in finite strains,
the initial stress of the current state included; for plasticity the tangent of the
loading branch, Hill's comparison solid as the steps shrink), reduced to the master nodes
with the macroscopic components of prescribed stress condensed (``K_ww - K_wE K_EE^-1
K_Ew``), is projected on the isotypic subspaces of the group restricted to the path (its
prescribed strains and stresses, and which components are which). On the symmetric
principal path the tangent commutes with the group, each block loses its positive
definiteness on its own, and the first block whose lowest eigenvalue changes sign marks
the first bifurcation (at the root of the parabola through its last three values; its
multiplicity: the eigenvalues that cross in the step); the patterns are the softest modes
of ``Fix(H) ∩ V_μ`` there. The mechanisms of the natural state are left out from the
start. A switch of branch perturbs the nodes along the critical pattern (those of the
faces of the box along these faces only: an equivariant change, which keeps the cell
periodic and the imperfection in the subspace of the pattern) and follows the path again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations, product

import numpy as np
import scipy.sparse as sp
from numba import njit
from scipy.linalg import eigh

from morphogen.homogenization.fe import (
    GAUSS,
    _corner_gradients,
    _product,
    _strain,
    periodic_map,
)
from morphogen.homogenization.frames import Frame, SpaceFrame, _buckling, positive_subspace
from morphogen.image import VoxelImage
from morphogen.mesh.tetra import TetMesh
from morphogen.symmetry import Group, Irrep, axial_subgroups, representation

#: Cells larger than this (dofs) skip the dense solve of the whole cell's buckling.
DENSE_LIMIT = 1500


@dataclass
class Pattern:
    """A post-bifurcated pattern: its irrep (index into :attr:`Bifurcation.irreps`), the
    class of conjugate isotropy subgroups (the first one fixes ``mode``) and its
    description, the dimension of ``Fix(H) ∩ V_μ``, the unit displacement (dofs of the
    cell: nodes of a frame, master nodes of a mesh) that minimizes the elastic energy
    there and that energy, and the lowest critical load factor and buckling mode of the
    loading in that subspace (NaN and None without loading)."""

    irrep: int
    subgroups: list
    description: dict
    dimension: int
    mode: np.ndarray
    energy: float
    load: float
    buckling_mode: np.ndarray | None


@dataclass
class Bifurcation:
    """The group-theoretic analysis of a cell: its symmetry group (restricted to the
    loading), irreps, the multiplicity of each among the fluctuations, the patterns, the
    lowest critical load factor of each irrep (NaN without loading), the first critical
    load, its irrep and multiplicity, and the norms of the projections of the first
    critical modes of the whole cell (unit norm) on the isotypic subspaces (rows: irreps;
    small cells only): 1 in the irrep of the bifurcation, 0 elsewhere (several irreps when
    their loads coincide, the multiplicity then counting all their modes). For a
    continuum, ``mesh`` is the analysed mesh and ``lift`` maps the dofs of its master
    nodes to all its nodes."""

    group: Group
    irreps: list[Irrep]
    multiplicities: np.ndarray
    patterns: list[Pattern]
    loads: np.ndarray
    first: tuple | None
    projections: np.ndarray | None
    mesh: TetMesh | None = None
    lift: sp.spmatrix | None = field(default=None, repr=False)

    def displacement(self, mode) -> np.ndarray:
        """The displacement of every node of the mesh (n x 3) of a mode of the cell."""
        if self.lift is None:
            raise ValueError("displacement: for the patterns of a continuum")
        return np.asarray(self.lift @ np.asarray(mode, dtype=float)).reshape(-1, 3)

    def write(self, path, irreps=None):
        """A VTK unstructured grid of the mesh with the pattern of each axial class (of all
        the irreps, or of the indices ``irreps``) as point data ``pattern_<irrep>_<k>``."""
        from morphogen.io.meshes import write_vtu_tet

        data, counts = {}, {}
        for p in self.patterns:
            if irreps is not None and p.irrep not in irreps:
                continue
            counts[p.irrep] = counts.get(p.irrep, 0) + 1
            data[f"pattern_{p.irrep + 1}_{counts[p.irrep]}"] = self.displacement(p.mode)
        return write_vtu_tet(path, self.mesh, point_data=data)

    def figure(self, frame, *, tiles=None, scale: float = 1.0, irreps=None):
        """The patterns of a frame drawn on ``tiles`` copies of the cell (3 x 3 in the
        plane, 1 x 1 x 1 in space), one row per irrep (all, or the indices ``irreps``), as
        the tables of Azulay & Combescure; returns the figure (``fig.savefig``)."""
        from morphogen.viz.figures import _figure

        space = isinstance(frame, SpaceFrame)
        tiles = tiles or ((1, 1, 1) if space else (3, 3))
        rows = [i for i in range(len(self.irreps)) if irreps is None or i in irreps]
        rows = [i for i in rows if any(p.irrep == i for p in self.patterns)]
        cols = max((sum(p.irrep == i for p in self.patterns) for i in rows), default=1)
        fig = _figure(2.4 * cols, 2.4 * max(len(rows), 1))
        n_rows = max(len(rows), 1)
        axes = np.empty((n_rows, cols), dtype=object)
        for k in range(n_rows * cols):
            options = {"projection": "3d"} if space else {}
            axes[k // cols, k % cols] = fig.add_subplot(n_rows, cols, k + 1, **options)
        for ax in axes.ravel():
            ax.set_axis_off()
        for r, i in enumerate(rows):
            for c, p in enumerate([p for p in self.patterns if p.irrep == i]):
                ax = axes[r, c]
                frame.plot(ax, p.mode, scale=scale, tiles=tiles, lw=0.6)
                d = p.description
                ax.set_title(
                    f"{i + 1}: {d['point_group']}, |H| = {d['order']} ({len(p.subgroups)})",
                    fontsize=7,
                )
        return fig

    def wave_vectors(self, irrep: int) -> list[tuple] | None:
        """The wave vectors of an irrep (index), :func:`wave_vectors`: ``(0, 0, 0)`` when
        its patterns repeat in every primitive cell."""
        return wave_vectors(self.group, self.irreps[irrep])

    def summary(self) -> list[dict]:
        """One row per irrep: dimension, multiplicity, critical load, wave vectors, and its
        patterns (point group, order, translations, number of conjugates, energy, load)."""
        rows = []
        for i, irrep in enumerate(self.irreps):
            rows.append(
                {
                    "irrep": i + 1,
                    "dim": irrep.dim,
                    "multiplicity": int(self.multiplicities[i]),
                    "load": float(self.loads[i]),
                    "wave_vectors": self.wave_vectors(i),
                    "patterns": [
                        {
                            **p.description,
                            "conjugates": len(p.subgroups),
                            "energy": p.energy,
                            "load": p.load,
                        }
                        for p in self.patterns
                        if p.irrep == i
                    ],
                }
            )
        return rows


def frame_representation(frame, group: Group) -> list:
    """The matrices ``T(g)`` (sparse) of a group on the dofs of a frame: ``(u, v, θ)`` in
    the plane, ``(u, v, w, θx, θy, θz)`` in space."""
    perm = group.action(frame.nodes)
    return representation(group, perm, [frame.block(r) for r in group.rotations])


@njit(cache=True)
def _initial_stress(nodes, elements, order, mat, matrices, u, gp, gw):
    """``∫ ∇N_a · σ ∇N_b dV`` of each element (n_el x nn x nn) for the stress ``σ = D ε(u)``
    at the Gauss points."""
    n_el, nn = elements.shape
    out = np.zeros((n_el, nn, nn))
    x = np.empty((4, 3))
    gl = np.empty((4, 3))
    grad = np.empty((nn, 3))
    eps = np.empty(6)
    sig = np.empty(6)
    s = np.empty((3, 3))
    for e in range(n_el):
        for a in range(4):
            for d in range(3):
                x[a, d] = nodes[elements[e, a], d]
        vol = abs(_corner_gradients(x, gl))
        dm = matrices[mat[e]]
        for g in range(gp.shape[0]):
            _strain(order, gl, gp[g], elements, e, u, grad, eps)
            _product(dm, eps, sig)
            s[0, 0], s[1, 1], s[2, 2] = sig[0], sig[1], sig[2]
            s[1, 2] = s[2, 1] = sig[3]
            s[2, 0] = s[0, 2] = sig[4]
            s[0, 1] = s[1, 0] = sig[5]
            w = gw[g] * vol
            for a in range(nn):
                for b in range(nn):
                    acc = 0.0
                    for i in range(3):
                        for j in range(3):
                            acc += grad[a, i] * s[i, j] * grad[b, j]
                    out[e, a, b] += w * acc
    return out


def initial_stress_matrix(problem, u) -> sp.csr_matrix:
    """The geometric (initial-stress) stiffness ``δ_ij ∫ ∇N_a · σ ∇N_b dV`` (all dofs) of
    the mesh of an elastic finite element problem for the displacement ``u`` (all dofs)."""
    m = problem.mesh
    gp, gw = GAUSS[m.order]
    blocks = _initial_stress(
        m.nodes,
        m.elements,
        m.order,
        np.ascontiguousarray(problem.mat, dtype=np.int64),
        np.ascontiguousarray(problem.table.matrices, dtype=float),
        np.ascontiguousarray(u, dtype=float),
        gp,
        gw,
    )
    e = m.elements
    nn = e.shape[1]
    rows = np.repeat(e, nn, axis=1).ravel()
    cols = np.tile(e, (1, nn)).ravel()
    scalar = sp.csr_matrix((blocks.ravel(), (rows, cols)), shape=(m.n_nodes, m.n_nodes))
    return sp.kron(scalar, sp.identity(3), format="csr")


def _box_operations(size) -> list[np.ndarray]:
    """The signed permutations of the axes that map a box of edges ``size`` onto itself."""
    size = np.asarray(size, dtype=float)
    out = []
    for perm in permutations(range(3)):
        if not np.allclose(size[list(perm)], size, rtol=1e-9):
            continue
        for signs in product((1.0, -1.0), repeat=3):
            r = np.zeros((3, 3))
            r[np.arange(3), perm] = signs
            out.append(r)
    return out


def mesh_symmetries(mesh: TetMesh, cells=(1, 1, 1), tol: float = 1e-6) -> Group:
    """The symmetry group of a periodic mesh made of ``cells`` primitive cells: the
    operations of the cube about the centre of its box (those that map the box onto
    itself) combined with translations by multiples of half a primitive cell, kept when
    they map the elements onto elements of the same phase."""
    from scipy.spatial import cKDTree

    size, lower = np.asarray(mesh.size, float), np.asarray(mesh.lower, float)
    centre = lower + 0.5 * size
    centroids = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)

    def unit(y):
        f = (y - lower) / size
        f = f - np.floor(f)
        return np.where(f >= 1.0, 0.0, f)

    tree = cKDTree(unit(centroids), boxsize=1.0)
    rng = np.random.default_rng(0)
    probe = rng.choice(len(centroids), size=min(64, len(centroids)), replace=False)
    steps = [np.arange(2 * int(n)) * s / (2 * int(n)) for n, s in zip(cells, size, strict=True)]
    accepted = []
    for r in _box_operations(size):
        for t in product(*steps):
            t = np.asarray(t)
            ok = True
            for idx in (probe, slice(None)):
                y = (centroids[idx] - centre) @ r.T + centre + t
                dist, near = tree.query(unit(y))
                if np.any(dist > tol / float(size.min())) or np.any(
                    mesh.phase[near] != mesh.phase[idx]
                ):
                    ok = False
                    break
            if ok:
                accepted.append((r, t))
    return Group.generate(accepted, np.diag(size), centre)


@dataclass
class _Cell:
    """The fluctuations of a cell: master node positions, the action of a rotation on the
    dofs of a node, the elastic and geometric stiffness (sparse) of the dofs, the
    translations (orthonormal columns) and the map to all the nodes of a mesh."""

    positions: np.ndarray
    lattice: np.ndarray
    centre: np.ndarray
    block: object
    ke: sp.spmatrix
    kg: sp.spmatrix | None
    translations: np.ndarray
    mesh: TetMesh | None = None
    lift: sp.spmatrix | None = None


def _frame_cell(frame, strain) -> _Cell:
    kg = None
    if strain is not None:
        forces, _ = frame.axial_forces(strain)
        kg = sp.csr_matrix(frame._assemble(forces)[1])
    return _Cell(
        frame.nodes,
        frame.lattice,
        frame.centre,
        frame.block,
        sp.csr_matrix(frame.stiffness()),
        kg,
        frame.translations(),
    )


def _continuum_cell(mesh: TetMesh, materials, strain) -> _Cell:
    from morphogen.homogenization.elements import VOIGT
    from morphogen.homogenization.fe import FEElasticProblem

    problem = FEElasticProblem(mesh, materials, bc="periodic")
    if problem.empty:
        raise ValueError("post_bifurcation: no load-carrying material in the cell")
    m = problem.mesh
    t, masters, _ = periodic_map(m)
    t3 = sp.kron(t, sp.identity(3), format="csr")
    ke = (t3.T @ problem._stiffness() @ t3).tocsr()
    kg = None
    if strain is not None:
        e = np.asarray(strain, dtype=float)
        x = np.array([e[a, b] * (1.0 if a == b else 2.0) for a, b in VOIGT])
        u, _ = problem._field(x, 0.0)
        kg = (t3.T @ initial_stress_matrix(problem, u) @ t3).tocsr()
    n = len(masters)
    translations = np.zeros((3 * n, 3))
    for c in range(3):
        translations[c::3, c] = 1.0 / np.sqrt(n)
    return _Cell(
        m.nodes[masters],
        np.diag(m.size),
        m.lower + 0.5 * m.size,
        lambda r: r,
        ke,
        kg,
        translations,
        m,
        t3,
    )


def _isotypic_basis(mats, weights, translations, rng, traces) -> np.ndarray:
    """Orthonormal basis of the isotypic subspace (translations removed): the range of
    ``P = Σ_g w(g) T(g)`` applied to random vectors, its dimension the trace of ``P``."""
    rank = round(float(weights @ traces))
    if rank <= 0:
        return np.zeros((translations.shape[0], 0))
    s = rng.normal(size=(translations.shape[0], rank + 8))
    y = sum(w * (mats[g] @ s) for g, w in enumerate(weights) if w != 0.0)
    y = y - translations @ (translations.T @ y)
    values, vectors = eigh(y.T @ y)  # the range by the Gram matrix of a random sample
    keep = values > 1e-12 * max(float(values[-1]), 1e-300)
    basis = y @ (vectors[:, keep] / np.sqrt(values[keep]))
    return basis[:, ::-1][:, :rank]


def _axial_patterns(group: Group, i: int, irrep: Irrep, kq, gq, fixed, lift, floor) -> list:
    """The patterns of the axial classes of an irrep (see the module) in coordinates of its
    isotypic subspace: the stiffness ``kq`` and geometric stiffness ``gq`` there (dense or
    sparse; None without loading), ``fixed(H)`` the matrix of the projector
    ``(1/|H|) Σ_h T(h)`` and ``lift`` the map of coordinates to the dofs of the cell."""
    out = []
    for cls in axial_subgroups(group, irrep):
        p = fixed(sorted(cls.subgroups[0]))
        p += p.T
        p *= 0.5
        # Fix(H) in the isotypic subspace: the eigenvectors of eigenvalue 1
        _, c = eigh(p, subset_by_value=(0.5, np.inf), overwrite_a=True)
        if c.shape[1] == 0:
            continue
        c = c @ positive_subspace(c.T @ (kq @ c))  # mechanisms are not patterns
        if c.shape[1] == 0:
            continue
        kc = c.T @ (kq @ c)
        energies, modes = eigh(0.5 * (kc + kc.T))
        load, buckled = np.nan, None
        if gq is not None:
            lam, phi = _buckling(kc, c.T @ (gq @ c), c, 1, floor)
            if lam.size:
                load, buckled = float(lam[0]), lift(phi[:, 0])
        out.append(
            Pattern(i, cls.subgroups, cls.description, c.shape[1], lift(c @ modes[:, 0]),
                    float(energies[0]), load, buckled)
        )  # fmt: skip
    return out


def _wave_characters(group: Group):
    """The pure translations (element indices) of the group of a cell of one or two
    primitive cells along each axis and the characters of its wave vectors on them: a
    mapping of the wave vectors (components 0 or 1 in units of ``π`` over the primitive
    edges; the first of those of the same character) to their signs; None for another
    group."""
    dim = group.rotations.shape[1]
    pure = np.flatnonzero(np.abs(group.rotations - np.eye(dim)).max(axis=(1, 2)) < 1e-9)
    halves = 2.0 * group.translations[pure] @ np.linalg.inv(group.lattice).T
    m = np.round(halves)
    if np.abs(halves - m).max() > 1e-6 or np.any((m != 0) & (m != 1)):
        return None
    chars: dict[tuple, np.ndarray] = {}
    for k in product(*[(0, 1) if a else (0,) for a in m.any(axis=0)]):
        c = (-1.0) ** (m @ np.array(k))
        if not any(np.array_equal(c, other) for other in chars.values()):
            chars[tuple(int(v) for v in k)] = c
    return pure, chars


def wave_vectors(group: Group, irrep: Irrep) -> list[tuple] | None:
    """The wave vectors (the arms of the star) that an irrep of the group of a cell of one
    or two primitive cells along each axis carries, components 0 or 1 in units of ``π``
    over the primitive edges: ``k_i = 1`` where its patterns change sign from a primitive
    cell to the next along axis ``i`` (the period doubled), ``(0, 0, 0)`` for patterns
    repeated in every primitive cell; None for another group."""
    found = _wave_characters(group)
    if found is None:
        return None
    pure, chars = found
    return [k for k, c in chars.items() if irrep.character[pure] @ c > 0.5 * len(pure)]


@dataclass
class _Star:
    """A star of wave vectors of a cell split by its translations (see the module): its
    first wave vector, the point operations (element indices) that carry it to each arm,
    the arm to which each element of the group carries it, the sparse orthonormal basis of
    its fluctuations (dofs of the cell x those of a primitive cell), its little co-group
    (element index -> position) with their matrices there and their traces, the
    translations of the cell there, its character on the translations, and the irreps of
    the little co-group with the irreps (indices) of the cell's group that they induce."""

    wave: tuple
    arms: list
    arm_of: np.ndarray
    space: sp.csr_matrix
    little: dict
    mats: list
    traces: np.ndarray
    moving: np.ndarray
    signs: np.ndarray
    induced: list

    def basis(self, j: int, seed) -> np.ndarray:
        """Orthonormal basis of the isotypic subspace of the ``j``-th irrep of the little
        co-group among the fluctuations of the wave vector."""
        weights = self.induced[j][0].projector_weights()
        rng = np.random.default_rng(seed)
        return _isotypic_basis(self.mats, weights, self.moving, rng, self.traces)


def _bloch_stars(group: Group, cell: _Cell):
    """The stars of a cell split by its translations (see the module), the point part ``p``
    of each element ``g = t p`` (the element of the same point operation without
    translation), the position of its translation ``t`` among the translations, and the
    operators of the point operations on the dofs of the cell; None when the group is not
    the product of halving translations and point operations about the centre."""
    found = _wave_characters(group)
    if found is None:
        return None
    pure, chars = found
    scale = float(np.abs(group.lattice).max())
    zero = np.flatnonzero(np.abs(group.translations).max(axis=1) <= 1e-9 * scale)
    if len(zero) * len(pure) != group.order:
        return None
    table, inverse = group.table, group.inverse
    by_rotation = {np.round(group.rotations[p], 6).tobytes(): int(p) for p in zero}
    point = np.array([by_rotation.get(np.round(r, 6).tobytes(), -1) for r in group.rotations])
    where = np.full(group.order, -1)
    where[pure] = np.arange(len(pure))
    shift = where[table[np.arange(group.order), inverse[np.maximum(point, 0)]]]
    if np.any(point < 0) or np.any(shift < 0):
        return None
    key = {(c > 0).tobytes(): k for k, c in chars.items()}

    def image(p, k):  # the wave vector p k, of character t -> χ_k(p^-1 t p)
        moved = where[table[table[inverse[p], pure], p]]
        return key[(chars[k][moved] > 0).tobytes()]

    positions = np.asarray(cell.positions, dtype=float)
    perm = group.subgroup(pure).action(positions)
    reps = np.unique(perm.min(axis=0))  # the orbits of the translations, by their lowest node
    images = perm[:, reps]
    free = np.all(np.diff(np.sort(images, axis=0), axis=0) > 0)
    if len(reps) * len(pure) != len(positions) or not free:
        return None
    width = cell.ke.shape[0] // len(positions)
    p0 = group.subgroup(zero)
    ops = representation(p0, p0.action(positions), [cell.block(r) for r in p0.rotations])
    operators = dict(zip(zero.tolist(), ops, strict=True))

    def space(k):  # the orbit sums of the character of k
        rows = (width * images)[:, :, None] + np.arange(width)
        cols = (width * np.arange(len(reps)))[:, None] + np.arange(width)
        cols = np.broadcast_to(cols, rows.shape)
        vals = np.broadcast_to((chars[k] / np.sqrt(len(pure)))[:, None, None], rows.shape)
        shape = (cell.ke.shape[0], width * len(reps))
        return sp.csr_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=shape)

    irreps = group.irreps()
    characters = np.array([r.character for r in irreps])
    stars, seen = [], set()
    for k0 in chars:
        if k0 in seen:
            continue
        arms: dict[tuple, int] = {}
        for p in zero:
            arms.setdefault(image(int(p), k0), int(p))
        seen.update(arms)
        index = {k: a for a, k in enumerate(arms)}
        arm_zero = {int(p): index[image(int(p), k0)] for p in zero}
        arm_of = np.array([arm_zero[int(p)] for p in point])
        little = [int(p) for p in zero if arm_of[p] == 0]
        b = space(k0)
        mats = [(b.T @ operators[p] @ b).tocsr() for p in little]
        traces = np.array([m.diagonal().sum() for m in mats])
        moving = np.zeros((b.shape[1], 0))
        if not any(k0) and cell.translations.shape[1]:  # the translations of the cell
            moving = np.linalg.qr(b.T @ cell.translations)[0]
        pairs = table[np.ix_(pure, little)]  # the elements t p of the little group
        induced = []
        for sigma in group.subgroup(little).irreps():
            # the irrep induced by (k, σ): Frobenius reciprocity on the little group
            overlap = np.einsum("t,mtp,p->m", chars[k0], characters[:, pairs], sigma.character)
            mu = int(np.argmax(overlap))
            if irreps[mu].dim != len(arms) * sigma.dim:
                return None
            induced.append((sigma, mu))
        stars.append(
            _Star(k0, list(arms.values()), arm_of, b, {p: j for j, p in enumerate(little)},
                  mats, traces, moving, chars[k0], induced)
        )  # fmt: skip
    if sorted(mu for star in stars for _, mu in star.induced) != list(range(len(irreps))):
        return None  # each irrep induced once
    return stars, point, shift, operators


def _star_patterns(group: Group, i: int, star: _Star, q, reduced, point, shift, operators,
                   floor) -> list:  # fmt: skip
    """The patterns of the irrep ``i`` of a cell split by its translations (see the module),
    in the coordinates of its block ``q`` on each arm of its star: an element ``h`` carries
    the block of arm ``a`` to that of arm ``b`` by the matrix of ``g_b^-1 h g_a = t p``
    (``p`` in the little co-group), ``χ_k(t) q^T M_p q``; ``reduced``: the stiffness
    matrices of the fluctuations of the wave vector."""
    s, d = len(star.arms), q.shape[1]
    table, inverse = group.table, group.inverse

    def fixed(members):
        sums = {}
        for h in members:
            for a, ga in enumerate(star.arms):
                x = table[h, ga]
                b = int(star.arm_of[x])
                lt = table[inverse[star.arms[b]], x]
                term = star.signs[shift[lt]] * star.mats[star.little[int(point[lt])]]
                sums[b, a] = sums[b, a] + term if (b, a) in sums else term
        out = np.zeros((s * d, s * d))
        for (b, a), m in sums.items():
            out[b * d : (b + 1) * d, a * d : (a + 1) * d] = q.T @ (m @ q)
        return out / len(members)

    def lift(x):
        local = star.space @ (q @ np.reshape(x, (s, d)).T)
        return sum(operators[ga] @ local[:, a] for a, ga in enumerate(star.arms))

    ke, kg = reduced
    kq = q.T @ (ke @ q)
    gq = None if kg is None else q.T @ (kg @ q)
    if s > 1:  # the same block on every arm
        kq = sp.block_diag([sp.csr_matrix(kq)] * s, format="csr")
        gq = None if gq is None else sp.block_diag([sp.csr_matrix(gq)] * s, format="csr")
    return _axial_patterns(group, i, group.irreps()[i], kq, gq, fixed, lift, floor)


def _bloch_bifurcation(cell: _Cell, group: Group, decomposition, rtol, which, seed):
    """The analysis of a cell split by its translations (see the module): the loads and
    multiplicities of the irreps from their blocks on the first arm of their stars (one at
    a time), and the patterns of the irreps of the first bifurcation (``which``
    ``"critical"``) or of all those present (``"all"``)."""
    stars, point, shift, operators = decomposition
    irreps = group.irreps()
    multiplicities = np.zeros(len(irreps), dtype=np.int64)
    loads = np.full(len(irreps), np.nan)
    floors = np.full(len(irreps), np.inf)
    spectra, of, reduced = {}, {}, []
    for s, star in enumerate(stars):
        b = star.space
        kg = None if cell.kg is None else (b.T @ cell.kg @ b).tocsr()
        reduced.append(((b.T @ cell.ke @ b).tocsr(), kg))
        for j, (sigma, mu) in enumerate(star.induced):
            q = star.basis(j, [seed, s, j])
            if q.shape[1] == 0:
                continue
            multiplicities[mu] = round(q.shape[1] / sigma.dim)
            of[mu] = (s, j)
            if kg is None:
                continue
            ke = reduced[s][0]
            lam, _ = _buckling(q.T @ (ke @ q), q.T @ (kg @ q), np.eye(q.shape[1]),
                               2 * sigma.dim + 2)  # fmt: skip
            spectra[mu] = lam
            if lam.size:
                loads[mu], floors[mu] = lam[0], 1e-10 / lam[0]
    first, chosen = None, sorted(of) if which == "all" else []
    if np.isfinite(loads).any():
        critical = float(np.nanmin(loads))
        count = sum(
            int(np.sum(np.abs(lam - critical) <= rtol * critical)) * len(stars[of[i][0]].arms)
            for i, lam in spectra.items()
        )  # each eigenvalue of a block on every arm of its star
        first = (critical, int(np.nanargmin(loads)), count)
        if which == "critical":
            chosen = np.flatnonzero(np.abs(loads - critical) <= rtol * critical).tolist()
    found: list[Pattern] = []
    for i in chosen:
        s, j = of[i]  # the basis again, with the same seed
        q = stars[s].basis(j, [seed, s, j])
        found += _star_patterns(group, i, stars[s], q, reduced[s], point, shift, operators,
                                floors[i])  # fmt: skip
    return Bifurcation(group, irreps, multiplicities, found, loads, first, None, cell.mesh,
                       cell.lift)  # fmt: skip


def post_bifurcation(
    sample,
    strain=None,
    *,
    materials=None,
    group: Group | None = None,
    cells=(1, 1, 1),
    order: int = 1,
    phases=None,
    rtol: float = 1e-6,
    seed: int = 0,
    bloch: bool | None = None,
    patterns: str | None = None,
) -> Bifurcation:
    """The post-bifurcated patterns of a periodic cell (see the module): a
    :class:`~morphogen.homogenization.frames.Frame` (plane) or
    :class:`~morphogen.homogenization.frames.SpaceFrame`, or a continuum given by its
    tetrahedral mesh or voxel image (six tetrahedra per voxel, keeping ``phases``; elements
    of ``order`` 1 or 2) and ``materials``. ``strain`` is the macroscopic loading (2 x 2
    or 3 x 3) whose symmetry restricts the group and whose linear buckling orders the
    bifurcations; ``group`` by default that of the frame's generators, or that detected
    on the mesh of ``cells`` primitive cells (:func:`mesh_symmetries`). ``bloch``: split
    a cell of two primitive cells along some axes by its translations first (by default
    beyond ``DENSE_LIMIT`` dofs); ``patterns``: those of all the irreps present
    (``"all"``, the default without ``bloch``) or of the irreps of the first bifurcation
    (``"critical"``, the default with it)."""
    if patterns not in (None, "all", "critical"):
        raise ValueError("post_bifurcation: patterns 'all' or 'critical'")
    if isinstance(sample, (Frame, SpaceFrame)):
        cell = _frame_cell(sample, strain)
        if group is None:
            group = Group.generate(sample.generators, sample.lattice, sample.centre)
    else:
        if materials is None:
            raise ValueError("post_bifurcation: the materials of a continuum")
        if isinstance(sample, VoxelImage):
            from morphogen.homogenization.bloch import voxel_tet_mesh

            sample = voxel_tet_mesh(sample, phases)
        # straight edges: the symmetries of the mesh are those of its corners
        mesh = sample.quadratic().straight() if int(order) == 2 else sample.linear()
        cell = _continuum_cell(mesh, materials, strain)
        if group is None:
            group = mesh_symmetries(cell.mesh, cells)
    if strain is not None:
        group = group.subgroup(group.stabilizer_of(strain))
    n = cell.ke.shape[0]
    asked = bloch
    if bloch is None:  # by default for the large cells of several primitive cells
        waves = _wave_characters(group)
        bloch = n > DENSE_LIMIT and waves is not None and len(waves[0]) > 1
    if bloch:
        decomposition = _bloch_stars(group, cell)
        if decomposition is not None:
            which = patterns or "critical"
            return _bloch_bifurcation(cell, group, decomposition, rtol, which, seed)
        if asked:
            raise ValueError(
                "post_bifurcation: bloch for the cells of one or two primitive cells along"
                " each axis whose group is the product of its translations and point operations"
            )
    which = patterns or "all"
    perm = group.action(cell.positions)
    mats = representation(group, perm, [cell.block(r) for r in group.rotations])
    traces = np.array([m.diagonal().sum() for m in mats])
    irreps = group.irreps()
    rng = np.random.default_rng(seed)
    multiplicities = np.zeros(len(irreps), dtype=np.int64)
    loads = np.full(len(irreps), np.nan)
    bases, spectra, floors = [], [], []
    floor, whole = None, None
    if cell.kg is not None and n <= DENSE_LIMIT:  # the whole cell, for the projections
        free = cell.translations.shape[1]
        z = np.linalg.qr(np.hstack([cell.translations, np.eye(n)]))[0][:, free:]
        full_ke, full_kg = z.T @ (cell.ke @ z), z.T @ (cell.kg @ z)
        whole = _buckling(full_ke, full_kg, z, 3 * max(r.dim for r in irreps))
        floor = 1e-8 / whole[0][0] if whole[0].size else np.inf
    for i, irrep in enumerate(irreps):
        weights = irrep.projector_weights()
        q = _isotypic_basis(mats, weights, cell.translations, rng, traces)
        bases.append(q)
        multiplicities[i] = round(q.shape[1] / irrep.dim)
        spectra.append(np.zeros(0))
        floors.append(floor)
        if q.shape[1] == 0 or cell.kg is None:
            continue
        kq, gq = q.T @ (cell.ke @ q), q.T @ (cell.kg @ q)
        if floor is None:  # no whole solve: the scale of this block
            lam0, _ = _buckling(kq, gq, np.eye(q.shape[1]), 1)
            floors[i] = 1e-10 / lam0[0] if lam0.size else np.inf
        lam, _ = _buckling(kq, gq, np.eye(q.shape[1]), 2 * irrep.dim + 2, floors[i])
        spectra[i] = lam
        if lam.size:
            loads[i] = lam[0]
    first = projections = None
    chosen = [i for i, q in enumerate(bases) if q.shape[1]] if which == "all" else []
    if cell.kg is not None and np.isfinite(loads).any():
        critical = float(np.nanmin(loads))
        count = int(sum(np.sum(np.abs(s - critical) <= rtol * critical) for s in spectra))
        first = (critical, int(np.nanargmin(loads)), count)
        if which == "critical":
            chosen = np.flatnonzero(np.abs(loads - critical) <= rtol * critical).tolist()
        if whole is not None and whole[0].size:
            lam_full, phi_full = whole
            k = int(np.sum(np.abs(lam_full - critical) <= rtol * critical))
            modes = phi_full[:, :k] / np.linalg.norm(phi_full[:, :k], axis=0)
            projections = np.array([np.linalg.norm(q.T @ modes, axis=0) for q in bases])
    found: list[Pattern] = []
    for i in chosen:
        q = bases[i]
        kq = q.T @ (cell.ke @ q)
        gq = None if cell.kg is None else q.T @ (cell.kg @ q)

        def fixed(members, q=q):
            return q.T @ (sum(mats[h] for h in members) @ q) / len(members)

        found += _axial_patterns(group, i, irreps[i], kq, gq, fixed, lambda x, q=q: q @ x,
                                 floors[i])  # fmt: skip
    return Bifurcation(
        group, irreps, multiplicities, found, loads, first, projections, cell.mesh, cell.lift
    )


# -- along nonlinear paths ----------------------------------------------------------------------


@dataclass
class PathBifurcation(Bifurcation):
    """The analysis of a cell along a nonlinear loading path (:func:`path_bifurcation`).
    Besides the fields of :class:`Bifurcation` (``loads``: the time at which the tangent of
    each irrep loses its positive definiteness, NaN when it does not; ``first``: the
    earliest, its irrep and the number of modes that cross within that step; the patterns:
    the softest modes of the axial classes at that point): the times of the converged
    steps (from the natural state) and their macroscopic strains (the six components of the
    path), the lowest eigenvalues of the tangent in each isotypic subspace (steps x irreps
    x k, relative to the largest diagonal entry of its block in the natural state; NaN for
    the irreps absent from the fluctuations), the macroscopic strain at the first
    bifurcation, the response along the principal path
    (:class:`~morphogen.homogenization.paths.PathResult`) and, after a switch of branch,
    the response of the cell perturbed along the critical pattern, with the times of its
    steps and the amplitude of that pattern in its fluctuation."""

    times: np.ndarray | None = None
    strains: np.ndarray | None = None
    spectra: np.ndarray | None = None
    critical_strain: np.ndarray | None = None
    principal: object = field(default=None, repr=False)
    branch: object = field(default=None, repr=False)
    branch_times: np.ndarray | None = None
    amplitudes: np.ndarray | None = None


def _symmetric(v) -> np.ndarray:
    """The symmetric 3 x 3 tensor of six components (11, 22, 33, 23, 31, 12)."""
    from morphogen.homogenization.elements import VOIGT

    t = np.zeros((3, 3))
    for k, (a, b) in enumerate(VOIGT):
        t[a, b] = t[b, a] = v[k]
    return t


def _loading_group(group: Group, path) -> Group:
    """The subgroup that leaves a loading path invariant: its prescribed strains and its
    prescribed stresses at every time, and which components are strains or stresses (the
    point operations of mesh groups are signed permutations of the axes)."""
    controls = path.strain_controlled
    keep = np.ones(group.order, dtype=bool)
    for row in np.asarray(path.values):
        for mask in (controls, ~controls):
            keep &= group.stabilizer_of(_symmetric(np.where(mask, row, 0.0)))
    flags = _symmetric(np.where(controls, 1.0, 2.0))
    for g, r in enumerate(group.rotations):
        a = np.abs(r)
        keep[g] &= bool(np.abs(a @ flags @ a.T - flags).max() < 1e-9)
    return group.subgroup(keep)


def _crossing(t, lam, k) -> float:
    """The time between the steps ``k - 1`` and ``k`` at which the eigenvalue ``lam``
    changes sign: the root of the parabola through the values of the steps ``k - 2`` to
    ``k`` (of the line through the last two at the first step)."""
    t0, t1 = float(t[k - 1]), float(t[k])
    if k >= 2:
        roots = np.roots(np.polyfit(t[k - 2 : k + 1] - t0, lam[k - 2 : k + 1], 2)) + t0
        roots = roots[np.abs(roots.imag) <= 1e-12 * max(abs(t1), 1.0)].real
        inside = roots[(roots >= t0 - 1e-12 * abs(t1)) & (roots <= t1 * (1 + 1e-12))]
        if inside.size:
            return float(inside.min())
    return t0 + (t1 - t0) * float(lam[k - 1] / (lam[k - 1] - lam[k]))


def path_bifurcation(
    sample,
    path,
    *,
    materials,
    kinematics: str = "finite",
    group: Group | None = None,
    cells=(1, 1, 1),
    order: int = 1,
    phases=None,
    eigenvalues: int = 3,
    stop: bool = True,
    branch: float | None = None,
    rtol: float = 1e-3,
    seed: int = 0,
    **options,
) -> PathBifurcation:
    """The bifurcations of a periodic cell along a nonlinear loading path (see the
    module): a tetrahedral mesh or a voxel image (six tetrahedra per voxel, keeping
    ``phases``; elements of ``order`` 1 or 2) of ``cells`` primitive cells, its
    ``materials`` (elastic, von Mises elasto-(visco)plastic, crystal or MFront behaviours),
    the ``path`` (:class:`~morphogen.homogenization.paths.LoadingPath` or its mapping)
    followed by :class:`~morphogen.homogenization.nonlinear.NonlinearFEProblem` with
    periodic conditions in ``kinematics`` (``options``: its options; the tangents of the
    points symmetrized unless they say otherwise, ``tangent="symmetric"``: the analysis is
    that of the symmetric part). The tangent of each converged step is block-diagonalized
    in the isotypic subspaces of the group of the cell restricted to the path; the
    ``eigenvalues`` lowest eigenvalues of each block (times the dimension of its irrep)
    are followed, and the first that changes sign marks a bifurcation. ``stop``: the
    principal path ends at the first bifurcation. ``branch``: the amplitude (a fraction
    of the smallest edge of the box; about 0.01 to 0.02, a smaller one making the
    transition so abrupt that Newton's method may fall back onto the principal branch) of
    an imperfection of the nodes along the critical pattern, with which the path is
    followed again onto the bifurcated branch."""
    from morphogen.homogenization.nonlinear import NonlinearFEProblem
    from morphogen.homogenization.paths import LoadingPath

    path = LoadingPath.from_spec(path)
    if isinstance(sample, (Frame, SpaceFrame)):
        raise ValueError("path_bifurcation: continua (frames: post_bifurcation)")
    if isinstance(sample, VoxelImage):
        from morphogen.homogenization.bloch import voxel_tet_mesh

        sample = voxel_tet_mesh(sample, phases)
    mesh = sample.quadratic().straight() if int(order) == 2 else sample.linear()
    # the blocks below are those of the symmetric part of the tangent
    options = {"tangent": "symmetric", **options}
    problem = NonlinearFEProblem(mesh, materials, "periodic", kinematics=kinematics, **options)
    if problem.empty:
        raise ValueError("path_bifurcation: no load-carrying material in the cell")
    m = problem.mesh
    t_nodes, masters, _ = periodic_map(m)
    lift = sp.kron(t_nodes, sp.identity(3), format="csr")
    group = _loading_group(mesh_symmetries(m, cells) if group is None else group, path)
    perm = group.action(m.nodes[masters])
    mats = representation(group, perm, list(group.rotations))
    traces = np.array([t.diagonal().sum() for t in mats])
    irreps = group.irreps()
    n = 3 * len(masters)
    translations = np.zeros((n, 3))
    for c in range(3):
        translations[c::3, c] = 1.0 / np.sqrt(len(masters))
    rng = np.random.default_rng(seed)
    bases = [
        _isotypic_basis(mats, r.projector_weights(), translations, rng, traces) for r in irreps
    ]
    multiplicities = np.array(
        [round(q.shape[1] / r.dim) for q, r in zip(bases, irreps, strict=True)]
    )
    elastic = (lift.T @ problem._stiffness() @ lift).tocsr()  # the natural state
    # the mechanisms of the natural state (rotations of parts free of the faces...) left out
    bases = [q if q.shape[1] == 0 else q @ positive_subspace(q.T @ (elastic @ q)) for q in bases]
    counts = [min(q.shape[1], int(eigenvalues) * r.dim) for q, r in zip(bases, irreps, strict=True)]
    width = max(max(counts), 1)
    scales = [1.0] * len(irreps)

    def blocks_of(k, columns=None, kee=None, reduced=False):
        """The tangent in each isotypic subspace, the unknown macroscopic components
        condensed (their stresses prescribed)."""
        k = k if reduced else (lift.T @ k @ lift).tocsr()
        g = None if columns is None else lift.T @ columns
        out = []
        for q in bases:
            if q.shape[1] == 0:
                out.append(None)
                continue
            b = q.T @ (k @ q)
            if g is not None:
                gq = q.T @ g
                b = b - gq @ np.linalg.solve(kee, gq.T)
            out.append(0.5 * (b + b.T))
        return out

    def spectrum(blocks):
        row = np.full((len(irreps), width), np.nan)
        for i, b in enumerate(blocks):
            if b is not None and counts[i]:
                values = eigh(b, eigvals_only=True, subset_by_index=[0, counts[i] - 1])
                row[i, : values.size] = values / scales[i]
        return row

    natural = blocks_of(elastic, reduced=True)  # the elastic tangent of the natural state
    scales = [1.0 if b is None else float(np.abs(np.diag(b)).max()) for b in natural]
    record = {"t": [float(path.times[0])], "e": [np.zeros(6)], "rows": [spectrum(natural)]}
    record.update({"blocks": [natural], "critical": None})

    def monitor(state):
        columns = kee = None
        if state.unknown.size:
            columns, kee = state.disc.macro_columns(state.tangent, state.wts)
            columns, kee = columns[:, state.unknown], kee[np.ix_(state.unknown, state.unknown)]
        blocks = blocks_of(state.disc.full_tangent(state.tangent, state.wts), columns, kee)
        row = spectrum(blocks)
        record["t"].append(float(state.time))
        record["e"].append(np.asarray(state.strain, dtype=float).copy())
        record["rows"].append(row)
        crossed = bool(np.any(row[:, 0] < 0))
        if crossed and record["critical"] is None:
            record["critical"] = (record["blocks"][-1], blocks)
        record["blocks"] = [blocks]
        return stop and crossed

    principal = problem.run(path, name="principal", monitor=monitor)
    times, strains = np.array(record["t"]), np.array(record["e"])
    spectra = np.array(record["rows"])
    loads = np.full(len(irreps), np.nan)
    steps = np.full(len(irreps), -1)
    for i in range(len(irreps)):
        negative = np.flatnonzero(spectra[:, i, 0] < 0)
        if negative.size and negative[0] > 0:
            steps[i] = negative[0]
            loads[i] = _crossing(times, spectra[:, i, 0], negative[0])
    first = critical_strain = None
    blocks = record["blocks"][-1]
    if np.isfinite(loads).any():
        critical = float(np.nanmin(loads))
        irrep = int(np.nanargmin(loads))
        together = np.flatnonzero(np.abs(loads - critical) <= rtol * max(abs(critical), 1e-300))
        count = int(sum(np.sum(spectra[steps[i], i] < 0) for i in together))
        first = (critical, irrep, count)
        k = steps[irrep]
        weight = (critical - times[k - 1]) / (times[k] - times[k - 1])
        critical_strain = (1 - weight) * strains[k - 1] + weight * strains[k]
        before, after = record["critical"]
        blocks = before if weight < 0.5 else after  # the step nearer to the bifurcation
    patterns: list[Pattern] = []
    for i, irrep in enumerate(irreps):
        q, b = bases[i], blocks[i]
        if b is None:
            continue
        for cls in axial_subgroups(group, irrep):
            members = sorted(cls.subgroups[0])
            y = sum(mats[h] @ q for h in members) / len(members)
            values, vectors = eigh(0.5 * ((q.T @ y) + (q.T @ y).T))
            c = vectors[:, values > 0.5]  # Fix(H) in the isotypic subspace
            if c.shape[1] == 0:
                continue
            energies, modes = eigh(0.5 * (c.T @ b @ c + (c.T @ b @ c).T))
            mode = q @ (c @ modes[:, 0])
            energy = float(energies[0] / scales[i])
            patterns.append(
                Pattern(i, cls.subgroups, cls.description, c.shape[1], mode, energy,
                        float(loads[i]), None)
            )  # fmt: skip
    out = PathBifurcation(group, irreps, multiplicities, patterns, loads, first, None, m, lift,
                          times, strains, spectra, critical_strain, principal)  # fmt: skip
    if branch and first is not None:
        candidates = [p for p in patterns if p.irrep == first[1]]
        if candidates:  # the softest pattern of an axial class of the critical irrep
            mode = min(candidates, key=lambda p: p.energy).mode
        else:
            _, v = eigh(blocks[first[1]], subset_by_index=[0, 0])
            mode = bases[first[1]] @ v[:, 0]
        _switch_branch(out, mode, path, materials, kinematics, float(branch), masters, options)
    return out


def _switch_branch(result, mode, path, materials, kinematics, amplitude, masters, options):
    """Follow the path again with the nodes perturbed along the critical ``mode``: the
    bifurcated branch, and the amplitude of the mode in the fluctuation at each step (into
    ``result``)."""
    from dataclasses import replace

    from morphogen.homogenization.nonlinear import NonlinearFEProblem

    m = result.mesh
    span = float(np.max(m.size))
    # the nodes of the faces of the box move along them only: the faces stay plane and
    # periodic (an equivariant change, which keeps the pattern in its subspace)
    along = (np.abs(m.nodes - m.lower) > 1e-9 * span) & (np.abs(m.nodes - m.upper) > 1e-9 * span)
    u = result.displacement(mode) * along
    scale = amplitude * float(np.min(m.size)) / max(float(np.abs(u).max()), 1e-300)
    perturbed = replace(m, nodes=m.nodes + scale * u)
    again = NonlinearFEProblem(perturbed, materials, "periodic", kinematics=kinematics, **options)
    record = {"t": [], "a": []}

    def monitor(state):
        w = (state.disc.t_mat @ state.q).reshape(-1, 3)[masters].ravel()
        record["t"].append(float(state.time))
        record["a"].append(float(w @ mode))
        return False

    result.branch = again.run(path, name="branch", fields=True, monitor=monitor)
    result.branch_times = np.array(record["t"])
    result.amplitudes = np.array(record["a"])
