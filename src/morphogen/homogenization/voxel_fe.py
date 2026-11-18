"""The trilinear voxel elements of the FFT solvers assembled on the voxels of the materials,
and solved by conjugate gradients preconditioned by a sparse factorization.

The FFT solvers (:mod:`morphogen.homogenization.fft`) discretize the periodic cell by
trilinear hexahedra on the voxels and solve the equilibrium ``K w = f`` of the fluctuation
without assembling ``K``, by conjugate gradients preconditioned by the inverse of the
stiffness ``K0`` of a homogeneous reference medium (Schneider, Merkert & Kabel 2017). On a
sparse phase surrounded by void near its rigidity threshold the preconditioner is a poor
one: the energy of ``K0`` spreads a displacement of the phase over the void around it,
while the clusters of voxels pinned together at their corners and edges turn about them at
an energy that vanishes with their rigidity, so that the iterations grow as about ``45 L``
on planar images of ``L`` pixels (2,885 on 64²) and do not converge within ``50 L + 1000``
in 3-D (12³ to 16³ voxels of sites of the cubic lattice stop at relative residuals of
``1e-6`` to ``2e-4``, with moduli off by 5 to 16 %).

These problems solve the same discrete equations (the element matrices, loads, averages
and fields of the FFT problems, which they extend) with the matrix assembled: the element
matrices of the voxels that carry a material are summed into the sparse matrix of the
degrees of freedom of their nodes (the void voxels, and the nodes that only they touch, are
left out: their rows of ``K`` vanish), and ``K w = f`` is solved by conjugate gradients
preconditioned by the sparse LU factorization (SuperLU, minimum-degree ordering of ``A^T +
A``, the pivots on the diagonal) of ``K + delta I``, ``delta`` being ``shift`` (``1e-12``)
times the mean of the diagonal. ``K`` itself is singular: the clusters that do not wrap
around the periodic box float, and the floppy clusters, hinged at their corners and edges,
have mechanisms (zero energy, zero strain at every Gauss point; at least 400 on a box of
24³ voxels of sites of the cubic lattice at its rigidity threshold), but the loads are
orthogonal to them and the shifted factorization, which keeps the modes of ``K`` and maps
its range onto itself, only preconditions: the iterates stay in the range of ``K``, the
moduli are those of ``K`` to the tolerance of the iterations, and the round-off that the
factorization leaves in the mechanisms strains no voxel. The modes softer than ``delta``
take a few more iterations, the iterations restart from the true residual every
:data:`CYCLE` (the recursion accumulates the round-off of the mechanisms, which the
preconditioner amplifies by ``1 / delta``), and one to three iterations suffice on the
planar images and the boxes of up to 48³ voxels of the sites at their rigidity threshold.
A larger shift is not safer: with ``1e-10`` one image of 24³ voxels in 558 stalled at a
relative residual of ``5e-7`` after 2,200 iterations (its modulus off by 0.1 %), with
``1e-8`` at ``2e-5``, where ``1e-12`` takes 3 iterations to ``1e-8`` and 10 to ``1e-10``.
The factorization is made once per problem, at the first solution (its time is the
``setup_s`` of the solutions), and serves all the loadings. The floppy images whose
mechanisms take large displacements for a unit strain (``1e3`` to ``7e5`` on planar boxes
of 96² to 256² pixels at their rigidity threshold) are not resolved by their stress
averages: the round-off of the element matrices, the same in every voxel, acts on these
displacements, and the stress average, linear in the error of the solution, strays to
``-1e-10``, ``-1e-7`` and ``-6e-7`` where it should vanish, at any tolerance
(``-1.30e-7`` at ``1e-10`` as at ``5e-12``, the floor of the residual, on 128² pixels);
the energy, the mean of ``sigma : epsilon`` from the strains of the voxels, stationary, is
an upper bound of the modulus whose error is of second order (``3.6e-9`` there, ``1e-20``
to ``1e-11`` for most; :func:`morphogen.analysis.percolation.rigidity_threshold` decides
by it), and perturbing the element matrices at their round-off scatters it by 17 to 50 %
for the floppy images, by 1.4e-3 at most for those that bear, whose stress averages agree
with their energies within 30 % (``3e-4`` from ``1e-7`` up). :func:`load_bearing` keeps
the voxels that bear alone, exactly: at their rigidity thresholds 79 to 82 % of the sites
of the cubic lattice on 24³ to 64³ voxels (75 to 79 % of the degrees of freedom), 70 to
94 % of those of the square lattice on 32² and 48² pixels.

Its cost is that of the fill-in, which the FFT solvers do not have, and the fill grows
steeply with the fraction of the voxels. On one process (the laptop on mains, otherwise
idle), at the rigidity threshold of the sites of the cubic lattice (0.1307), the
factorization takes 0.38 s on 32³ voxels (2.9 million entries in the factors, 0.03 GiB),
0.79 s on 40³, 1.5 s on 48³ (0.12 GiB), 2.6 s on 56³ and 4.1 s on 64³ (25 million, 0.28
GiB), the assembly 0.14 to 1.2 s and a solution with the factors 0.02 to 0.15 s; the
voxels that bear alone take 0.30 s on 32³ and 3.5 s on 64³ (0.24 GiB), and finding them
(:func:`load_bearing`) 0.1 s on 64³ and 0.7 s on 96³ (it took 0.5 s on 48³ and 2.5 s on
64³ while the union of its parts, a plain forest, grew chains as long as the parts).
SuperLU commits its first guess of the fill, 20 times the entries of the matrix in four
arrays (37 bytes per entry: 9.1 GiB on 64³ voxels at the threshold, 31 GiB on 96³), of
which the factors fill 1.5 to 1.8 times the entries in 3-D and 2.1 to 2.4 in 2-D; Windows
charges the untouched pages against the commit of the machine (its memory and page file),
and a factorization that fails for want of memory does not free its arrays (SciPy 1.18):
processes running side by side on large boxes want the commit of each factorization
limited, between the guesses of 2.5 and 5 times the entries, which SuperLU meets by halving
its guess (the same factors; 2.4 GiB on 64³). Above the
threshold the factors fill in: on 48³ voxels 2.8 s at a fraction of 0.15, 12.8 s at 0.17
and 122 s (1.28 GiB) at 0.2096, the upper bound of the bisections of the first
measurements, whose 7 s on 40³ voxels and 30 s on 48³ per solution were those of these
images (the bisection of the rigidity threshold now tries its bounds last). The ordering
is that of minimum degree on ``A^T + A``, the matrix symmetric (``SymmetricMode``,
diagonal pivots): on 24³ voxels at the threshold it fills 1.2 million entries in 0.45 s,
where the column ordering COLAMD fills 3.8 million in 1.2 s, minimum degree on ``A^T A``
1.8 million in 1.0 s and a nested dissection by planes of nodes (two per periodic axis,
then the natural order) 13.7 million in 12.8 s, the planes cutting through a sparse random
structure that minimum degree follows; SuperLU's relaxed supernodes and panels (``relax``
3, ``panel_size`` 16) change nothing. The factors of one image do not precondition the
next of a bisection, a few voxels apart (48³ voxels near the threshold, 345 to 727 voxels
added or removed: 500 to 2,500 iterations without converging, 34 to 123 s, against 0.7 to
3.7 s for new factors and one iteration): the voxels added or removed change the
mechanisms, which the shifted factors amplify by the inverse of the shift. Boolean discs
of 4 pixels of radius at 0.61 take 4.8 s on 256² pixels and 22 s on 512² (0.47 GiB),
spheres of 3 voxels of radius at 0.21 7.9 s on 48³ voxels (0.32 GiB; the machine
loaded). In 2-D, at the rigidity threshold of the sites of the square lattice, the
factorization and its solutions take 0.08 to 0.15 s on 64² pixels (against 2.8 to 7.8 s
for 1,310 to 2,665 iterations of the FFT solver, CPU and GPU), 0.6 s on 128² (16 to 58 s
for 7,000 to 7,400 iterations, most at the bound with a residual of ``3e-5``) and 4 s on
256², on the CPU, which the problems use. On a dense phase the FFT solvers converge in a
few dozen iterations where the factors fill in (sites at 0.5: 0.14 s against 12.8 s on
16³ voxels, 0.8 s, 0.27 s on the GPU, against more than 8 min and 3.6 GB on 32³), and the
voxels of the other phases at a contrast instead of void are all assembled (more than 6
min on 24³ voxels): these problems are for sparse phases surrounded by void, near their
rigidity threshold.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit

from morphogen.homogenization import kernels
from morphogen.homogenization.fft import FFTElasticProblem, FFTThermalProblem
from morphogen.homogenization.solvers import NEGLIGIBLE, SolveInfo, conjugate_gradient, norm

log = logging.getLogger("morphogen")

#: The shift of the diagonal of the matrix factorized for the preconditioner, over the mean
#: of its diagonal
SHIFT = 1e-12
#: The iterations of a cycle of the conjugate gradients, which then restart from the residual
#: of their iterate
CYCLE = 100


def _folded(shape, matrices: np.ndarray, nd: int) -> tuple[np.ndarray, np.ndarray]:
    """The corners of the voxel element that stay distinct on the periodic grid ``shape``
    (its two nodes along an axis one voxel thick are one) and the element matrices folded
    onto them: ``F^T K F``, ``F`` copying the values of the corners kept to the others."""
    corners = kernels.CORNERS.copy()
    corners[:, np.asarray(shape) == 1] = 0
    rep = kernels.CORNER_INDEX[corners[:, 0], corners[:, 1], corners[:, 2]]
    kept = np.unique(rep)
    fold = np.zeros((8 * nd, kept.size * nd))
    for q, k in enumerate(np.searchsorted(kept, rep).tolist()):
        fold[q * nd : (q + 1) * nd, k * nd : (k + 1) * nd] = np.eye(nd)
    return kept, np.einsum("ai,mab,bj->mij", fold, matrices, fold, optimize=True)


def voxel_stiffness(shape, mat, matrices, nd: int) -> tuple[sp.csr_matrix, np.ndarray]:
    """The matrix of the voxel elements of the periodic grid ``shape`` whose material
    ``mat`` (one per voxel, C order) is not void (-1), ``matrices[m]`` being the element
    matrix (8 nd x 8 nd, the corners in the order of
    :data:`morphogen.homogenization.kernels.CORNERS`) of material ``m``, summed on the
    degrees of freedom of their nodes alone; with these degrees of freedom, ``node * nd +
    component`` on the grid (increasing): the rows and columns of the matrix."""
    nx, ny, nz = (int(s) for s in shape)
    mat = np.asarray(mat).ravel()
    elements = np.flatnonzero(mat >= 0)
    kept, folded = _folded((nx, ny, nz), np.asarray(matrices, dtype=float), nd)
    i, j, k = np.unravel_index(elements, (nx, ny, nz))
    c = kernels.CORNERS[kept]
    nodes = ((i[:, None] + c[:, 0]) % nx * ny + (j[:, None] + c[:, 1]) % ny) * nz + (
        k[:, None] + c[:, 2]
    ) % nz
    used, local = np.unique(nodes, return_inverse=True)
    m = kept.size * nd
    dofs = (local.reshape(nodes.shape)[:, :, None] * nd + np.arange(nd)).reshape(-1, m)
    rows = np.broadcast_to(dofs[:, :, None], (elements.size, m, m)).ravel()
    cols = np.broadcast_to(dofs[:, None, :], (elements.size, m, m)).ravel()
    n = used.size * nd
    matrix = sp.coo_matrix((folded[mat[elements]].ravel(), (rows, cols)), shape=(n, n)).tocsr()
    matrix.sum_duplicates()
    return matrix, (used[:, None] * nd + np.arange(nd)).ravel()


@njit(cache=True)
def _edge_blocks(indptr, dst, eid, n_edges):
    """The biconnected components of an undirected graph (Hopcroft & Tarjan 1973, by an
    iterative depth-first search): the block of each edge, the edges numbered ``eid`` in the
    adjacency lists ``dst[indptr[v]:indptr[v + 1]]`` of the vertices (each edge in the lists
    of its two ends), and the number of blocks."""
    n = indptr.size - 1
    disc = np.full(n, -1, np.int64)
    low = np.zeros(n, np.int64)
    parent = np.full(n, -1, np.int64)  # the edge of the search that reached the vertex
    nxt = indptr[:-1].copy()
    block = np.full(n_edges, -1, np.int64)
    path = np.empty(n, np.int64)
    edges = np.empty(n_edges, np.int64)
    t = nb = top_e = 0
    for root in range(n):
        if disc[root] >= 0:
            continue
        disc[root] = low[root] = t
        t += 1
        top = 0
        path[0] = root
        while top >= 0:
            v = path[top]
            if nxt[v] < indptr[v + 1]:
                k = nxt[v]
                nxt[v] += 1
                w, e = dst[k], eid[k]
                if e == parent[v]:
                    continue
                if disc[w] < 0:
                    edges[top_e] = e
                    top_e += 1
                    parent[w] = e
                    disc[w] = low[w] = t
                    t += 1
                    top += 1
                    path[top] = w
                elif disc[w] < disc[v]:  # a back edge, to an ancestor
                    edges[top_e] = e
                    top_e += 1
                    low[v] = min(low[v], disc[w])
            else:
                top -= 1
                if top >= 0:
                    u = path[top]
                    low[u] = min(low[u], low[v])
                    if low[v] >= disc[u]:  # u cuts the edges stacked since parent[v] off
                        while True:
                            top_e -= 1
                            block[edges[top_e]] = nb
                            if edges[top_e] == parent[v]:
                                break
                        nb += 1
    return block, nb


@njit(cache=True, inline="always")
def _find(root, a):
    """The root of ``a`` in the forest ``root``, halving the path to it."""
    while root[a] != a:
        root[a] = root[root[a]]
        a = root[a]
    return a


@njit(cache=True)
def _parts(block, nb, nc):
    """The blocks of the incidences of the voxels to their ``nc`` nodes merged where they
    share a voxel (union-find, by size with the paths halved, in near-linear time where the
    plain forest grew chains as long as the parts: 2.5 s on the 34,000 voxels of the sites of
    a box of 64³ at the rigidity threshold, against 0.02 s): the part of each voxel,
    numbered from 0 in the order of their first voxels, and their number."""
    root = np.arange(nb)
    size = np.ones(nb, np.int64)
    nv = block.size // nc
    for v in range(nv):
        a = _find(root, block[v * nc])
        for q in range(1, nc):
            b = _find(root, block[v * nc + q])
            if b != a:
                if size[a] < size[b]:
                    a, b = b, a
                root[b] = a
                size[a] += size[b]
    label = np.full(nb, -1, np.int64)
    part = np.empty(nv, np.int64)
    count = 0
    for v in range(nv):
        a = _find(root, block[v * nc])
        if label[a] < 0:
            label[a] = count
            count += 1
        part[v] = label[a]
    return part, count


@njit(cache=True)
def _windings(local, node_ptr, node_inc, corners, part, n_parts):
    """Whether a cycle of each part winds around the periodic grid along each axis: the
    voxels of a part reached through the nodes they share (a voxel at the lift ``P`` and a
    corner ``c`` reach a voxel sharing that node at its corner ``d`` at ``P + c - d``), a
    voxel reached at two lifts closing a cycle whose winding is their difference."""
    nv, nc = local.shape
    lift = np.zeros((nv, 3), np.int64)
    seen = np.zeros(nv, np.bool_)
    queue = np.empty(nv, np.int64)
    winds = np.zeros((n_parts, 3), np.bool_)
    for start in range(nv):
        if seen[start]:
            continue
        seen[start] = True
        queue[0] = start
        head, tail = 0, 1
        while head < tail:
            v = queue[head]
            head += 1
            for q in range(nc):
                n = local[v, q]
                for k in range(node_ptr[n], node_ptr[n + 1]):
                    w, r = node_inc[k] // nc, node_inc[k] % nc
                    if part[w] != part[v]:
                        continue
                    if not seen[w]:
                        seen[w] = True
                        for a in range(3):
                            lift[w, a] = lift[v, a] + corners[q, a] - corners[r, a]
                        queue[tail] = w
                        tail += 1
                    else:
                        for a in range(3):
                            if lift[w, a] != lift[v, a] + corners[q, a] - corners[r, a]:
                                winds[part[v], a] = True
    return winds


def load_bearing(mask, axis: int) -> np.ndarray:
    """The voxels of ``mask`` (a periodic image, ``True`` where the material is, the others
    void) that bear a uniaxial strain, or conduct a gradient, along the axis number
    ``axis``: the others carry no load in the exact solution, and the moduli and
    conductivities of the image along the axis are those of these voxels alone.

    The trilinear voxels meet at the nodes they share. Cut at the nodes that join them
    alone (the articulation points of the graph of the voxels and their nodes), the voxels
    fall into parts, the biconnected components of that graph merged where they share a
    voxel, which meet one another at single nodes and form a tree: any part can translate
    (its temperature shift) as a whole to follow the node by which it hangs from the rest
    of its branch, at no energy, so that the parts bear independently. A part through
    which no cycle winds around the box along the axis takes the macroscopic strain
    (gradient) as a rigid translation of its lift, single-valued, at zero energy: it
    bears nothing, whether it dangles from a single corner, floats or lies between parts
    that bear, and is left out; those through which a cycle winds along the axis are kept.
    This is exact: the clusters hinged at two nodes (in 3-D an edge, a hinge), which may
    bear as bars between them, are kept with the part they belong to. An axis one voxel
    thick, whose two nodes the periodic grid folds into one, keeps every voxel."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 2:
        mask = mask[:, :, None]
    shape = np.asarray(mask.shape)
    if shape[axis] == 1 or not mask.any():
        return mask.copy()
    nx, ny, nz = (int(s) for s in shape)
    corners = kernels.CORNERS.copy()
    corners[:, shape == 1] = 0
    kept = np.unique(kernels.CORNER_INDEX[corners[:, 0], corners[:, 1], corners[:, 2]])
    c = corners[kept]
    nc = kept.size
    voxels = np.flatnonzero(mask.ravel())
    i, j, k = np.unravel_index(voxels, mask.shape)
    nodes = ((i[:, None] + c[:, 0]) % nx * ny + (j[:, None] + c[:, 1]) % ny) * nz + (
        k[:, None] + c[:, 2]
    ) % nz
    _, local = np.unique(nodes, return_inverse=True)
    local = local.reshape(nodes.shape).astype(np.int64)
    nv, nn = voxels.size, int(local.max()) + 1
    owner = np.repeat(np.arange(nv, dtype=np.int64), nc)
    src = np.concatenate([owner, nv + local.ravel()])
    dst = np.concatenate([nv + local.ravel(), owner])
    eid = np.tile(np.arange(nv * nc, dtype=np.int64), 2)
    order = np.argsort(src, kind="stable")
    indptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=nv + nn))])
    block, nb = _edge_blocks(indptr.astype(np.int64), dst[order], eid[order], nv * nc)
    part, n_parts = _parts(block, nb, nc)
    by_node = np.argsort(local.ravel(), kind="stable").astype(np.int64)
    node_ptr = np.concatenate([[0], np.cumsum(np.bincount(local.ravel(), minlength=nn))])
    winds = _windings(local, node_ptr.astype(np.int64), by_node, c.astype(np.int64), part,
                      n_parts)  # fmt: skip
    out = np.zeros(mask.size, dtype=bool)
    out[voxels[winds[part, axis]]] = True
    return out.reshape(mask.shape)


def _restarted(apply, precondition, rhs, tol: float, maxiter: int, cycle: int = CYCLE):
    """Conjugate gradients in cycles of at most ``cycle`` iterations, each from the residual
    ``rhs - apply(x)`` of the iterate of the last, until it falls below ``tol`` times the
    norm of ``rhs``, after ``maxiter`` iterations in all, or after a cycle that leaves it no
    smaller (the iterate before it kept): (x, iterations, relative residual). A long run of
    the recursion accumulates the round-off of the preconditioner of a singular matrix
    shifted, which amplifies its null space by the inverse of the shift; the cycles start
    again from the true residual."""
    norm_b = norm(rhs)
    x, r, res, total = np.zeros_like(rhs), rhs, 1.0, 0
    while total < maxiter and res > tol:
        dx, it, _ = conjugate_gradient(apply, precondition, r, tol / res,
                                       min(cycle, maxiter - total))  # fmt: skip
        total += it
        trial = x + dx
        r_trial = rhs - apply(trial)
        res_trial = norm(r_trial) / norm_b
        if it == 0 or res_trial >= res:
            break
        x, r, res = trial, r_trial, res_trial
    return x, total, res


def _factorize(shifted: sp.csr_matrix):
    """The sparse LU factors of a shifted matrix (positive definite): minimum-degree ordering
    of ``A^T + A``, the pivots on the diagonal (those of its Cholesky factorization, stable
    without pivoting), or partial pivoting if one vanishes."""
    matrix = shifted.tocsc()
    try:
        return spla.splu(matrix, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                         options={"SymmetricMode": True})  # fmt: skip
    except RuntimeError:  # pragma: no cover - a pivot cancelled to zero
        return spla.splu(matrix, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})


class _AssembledProblem:
    """The linear solutions of an FFT problem by its assembled matrix (see the module)."""

    def __init__(
        self,
        image,
        materials,
        bc: str = "periodic",
        *,
        tol: float = 1e-8,
        maxiter: int | None = None,
        grains=None,
        orientations=None,
        shift: float = SHIFT,
    ):
        if not (np.isfinite(shift) and 0.0 < shift < 1.0):
            raise ValueError(f"shift: expected 0 < shift < 1, got {shift!r}")
        super().__init__(image, materials, bc, tol=tol, maxiter=maxiter, grains=grains,
                         orientations=orientations, device="cpu")  # fmt: skip
        self.shift = float(shift)
        self._factors = None

    def _factorized(self):
        """The matrix, its degrees of freedom, the factors of the shifted matrix and the time
        taken (made at the first solution)."""
        if self._factors is None:
            t0 = time.perf_counter()
            matrix, dofs = voxel_stiffness(self.shape, self.mat, self.element_matrices, self.nd)
            diagonal = matrix.diagonal()
            shifted = matrix + sp.diags(np.full(dofs.size, self.shift * float(diagonal.mean())))
            factors = _factorize(shifted)
            self._factors = (matrix, dofs, factors, time.perf_counter() - t0)
        return self._factors

    def _solve(self, rhs, scale: float):
        if norm(rhs) <= NEGLIGIBLE * scale:
            return np.zeros_like(rhs), SolveInfo("fe-voxels", 0, 0, 0.0, 0.0, 0.0)
        matrix, dofs, factors, setup = self._factorized()
        t0 = time.perf_counter()
        maxiter = self.maxiter or 20 * int(max(self.shape)) + 1000
        x, it, res = _restarted(matrix.dot, factors.solve, np.ascontiguousarray(rhs[dofs]),
                                self.tol, maxiter)  # fmt: skip
        if res > self.tol:
            log.warning("voxel FE solver stopped at relative residual %.2e after %d iterations",
                        res, it)  # fmt: skip
        out = np.zeros_like(rhs)
        out[dofs] = x
        return out, SolveInfo("fe-voxels", dofs.size, it, res, setup, time.perf_counter() - t0)


class VoxelFEElasticProblem(_AssembledProblem, FFTElasticProblem):
    """Periodic linear elasticity of a voxel image, the discrete problem of
    :class:`~morphogen.homogenization.fft.FFTElasticProblem` (trilinear voxels), assembled on
    the voxels of the materials and solved by conjugate gradients preconditioned by the
    sparse factorization of its matrix shifted by ``shift`` times its mean diagonal, on the
    CPU (see :mod:`morphogen.homogenization.voxel_fe`): the same solutions, to the tolerance
    ``tol``, in a few iterations where the FFT solver needs thousands (sparse phases near
    their rigidity threshold, the other phases void)."""


class VoxelFEThermalProblem(_AssembledProblem, FFTThermalProblem):
    """Periodic steady heat conduction of a voxel image, the discrete problem of
    :class:`~morphogen.homogenization.fft.FFTThermalProblem`, assembled on the voxels of the
    materials and solved as :class:`VoxelFEElasticProblem`."""
