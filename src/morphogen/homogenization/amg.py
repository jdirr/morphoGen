"""Smoothed-aggregation algebraic multigrid (Vanek, Mandel & Brezina 1996).

A preconditioner of symmetric positive definite systems such as the stiffness matrices
of the finite element problems. The hierarchy follows the classical recipe (also that
of pyamg's ``smoothed_aggregation_solver``):

* aggregates of unknowns from the graph of the matrix: an unknown with its neighbours
  (pass 1), then the remaining unknowns joined to a neighbouring aggregate (pass 2) or
  grouped together (pass 3);
* tentative prolongator: on each aggregate, an orthonormal basis of the near-null modes
  (rigid-body motions in elasticity, constants in conduction), whose coefficients are
  the near-null modes of the coarse level;
* prolongator smoothed by one damped Jacobi step, ``P = (I - 4/3 D^-1 A / rho) T``;
* Galerkin coarse matrices ``P^T A P``, down to a few hundred unknowns, solved by a
  pseudo-inverse.

One V-cycle with a forward Gauss-Seidel sweep before the coarse correction and a
backward sweep after it is a symmetric preconditioner for conjugate gradients.

On large matrices the sweeps and the products run on several threads
(:class:`Smoother`, :func:`product`): the unknowns are coloured so that no two of a
colour are coupled, and those of a colour are relaxed together — a Gauss-Seidel sweep
in the order of the colours, the same for any number of threads, on a copy of the
matrix whose rows are sorted by colour (the rows of a colour scattered through the
matrix would be read no faster by four threads than all of them in their order by
one); the rows of a product are shared between the threads, each summed as SciPy sums
it. Where the unknowns are components of the displacements of the nodes of a mesh,
the colours are those of the nodes and the copy holds the 3 x 3 blocks of their
couplings: one column index for nine values, and the three components of a node read
together.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit, prange

#: Matrices with fewer stored values than this are multiplied on one thread: starting
#: the threads of the compiled loop would cost more than they bring
THREADED = 50_000
#: Matrices with fewer stored values than this are relaxed in the order of their
#: unknowns on one thread (a sweep in the order of the colours starts the threads once
#: per colour)
COLOURED = 200_000


@njit(cache=True)
def gauss_seidel(indptr, indices, data, x, b, forward):
    """One Gauss-Seidel sweep on ``A x = b`` (CSR), in place, forward or backward."""
    n = b.size
    for step in range(n):
        i = step if forward else n - 1 - step
        diag = 0.0
        s = b[i]
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            if j == i:
                diag += data[k]
            else:
                s -= data[k] * x[j]
        if diag != 0.0:
            x[i] = s / diag


@njit(parallel=True, cache=True)
def _one_sided(indptr, indices):
    """The number of stored values ``(i, j)`` of a CSR pattern with sorted rows whose
    transposed place ``(j, i)`` is not stored: zero for a symmetric pattern."""
    count = 0
    for i in prange(indptr.size - 1):
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            lo, hi = indptr[j], indptr[j + 1]
            while lo < hi:
                mid = (lo + hi) // 2
                if indices[mid] < i:
                    lo = mid + 1
                else:
                    hi = mid
            if lo == indptr[j + 1] or indices[lo] != i:
                count += 1
    return count


def _graph(matrix) -> tuple[np.ndarray, np.ndarray]:
    """The CSR pattern of the couplings of a square matrix (CSR, or BSR: the couplings
    of its blocks): its own when it is symmetric, that of ``A + A^T`` otherwise (a
    product of sparse matrices drops the values that are exactly zero, on one side of
    the diagonal only when round-off leaves something on the other; a matrix with rows
    that are not sorted is taken as such a one)."""
    indptr, indices = matrix.indptr, matrix.indices
    if matrix.has_sorted_indices and not _one_sided(indptr, indices):
        return indptr, indices
    n = indptr.size - 1
    ones = sp.csr_matrix((np.ones(indices.size), indices, indptr), shape=(n, n))
    both = (ones + ones.T).tocsr()
    return both.indptr, both.indices


@njit(cache=True)
def _colours(indptr, indices, n):
    """Greedy colouring of the graph of a matrix whose pattern is symmetric: each unknown
    takes the smallest colour that those coupled with it, among the ones before it, do
    not have. Returns the unknowns sorted by colour (in their order within a colour)
    and the offsets of the colours in that list."""
    colour = np.full(n, -1, dtype=np.int64)
    widest = 0
    for i in range(n):
        widest = max(widest, indptr[i + 1] - indptr[i])
    taken = np.full(widest + 2, -1, dtype=np.int64)  # the unknown that last met a colour
    count = 0
    for i in range(n):
        for k in range(indptr[i], indptr[i + 1]):
            c = colour[indices[k]]
            if c >= 0:
                taken[c] = i
        c = 0
        while taken[c] == i:
            c += 1
        colour[i] = c
        count = max(count, c + 1)
    starts = np.zeros(count + 1, dtype=np.int64)
    for i in range(n):
        starts[colour[i] + 1] += 1
    for c in range(count):
        starts[c + 1] += starts[c]
    order = np.empty(n, dtype=np.int64)
    place = starts[:count].copy()
    for i in range(n):
        order[place[colour[i]]] = i
        place[colour[i]] += 1
    return order, starts


@njit(cache=True)
def _sorted_by_colour(indptr, indices, data, order):
    """A matrix whose unknowns are taken in the order ``order`` (rows and columns: the
    new unknown ``k`` is the old one ``order[k]``), split into its diagonal and the CSR
    arrays of its strictly lower and upper parts (the stored values of each row in the
    order that they had), with 32-bit column indices and values in single precision:
    the copy that the sweeps read. A preconditioner needs no more, and a sweep is bound
    by the memory that it reads: a forward sweep from zero reads the lower part, the
    residual that follows it the upper one."""
    n = order.size
    place = np.empty(n, dtype=np.int32)
    for k in range(n):
        place[order[k]] = k
    lower = 0
    for k in range(n):
        i = order[k]
        for q in range(indptr[i], indptr[i + 1]):
            if place[indices[q]] < k:
                lower += 1
    upper = indices.size - lower  # (with the diagonal: an upper bound)
    l_indptr = np.empty(n + 1, dtype=np.int64)
    u_indptr = np.empty(n + 1, dtype=np.int64)
    l_indices = np.empty(lower, dtype=np.int32)
    u_indices = np.empty(upper, dtype=np.int32)
    l_data = np.empty(lower, dtype=np.float32)
    u_data = np.empty(upper, dtype=np.float32)
    diag = np.zeros(n, dtype=np.float32)
    nl = 0
    nu = 0
    for k in range(n):
        i = order[k]
        l_indptr[k] = nl
        u_indptr[k] = nu
        for q in range(indptr[i], indptr[i + 1]):
            j = place[indices[q]]
            if j < k:
                l_indices[nl] = j
                l_data[nl] = data[q]
                nl += 1
            elif j > k:
                u_indices[nu] = j
                u_data[nu] = data[q]
                nu += 1
            else:
                diag[k] += data[q]
    l_indptr[n] = nl
    u_indptr[n] = nu
    return l_indptr, l_indices, l_data, u_indptr, u_indices[:nu], u_data[:nu], diag


@njit(cache=True)
def _sorted_blocks(indptr, indices, data, order):
    """:func:`_sorted_by_colour` for the 3 x 3 blocks of a matrix (block CSR: ``data``
    holds the nine values of each stored block, row by row; ``order``: the nodes sorted
    by colour): the blocks below and above the diagonal, the diagonal blocks, and the
    place of each stored block among them (those below, then those above, then the
    diagonal ones: :func:`_refresh_blocks` writes other values there)."""
    n = order.size
    place = np.empty(n, dtype=np.int32)
    for k in range(n):
        place[order[k]] = k
    lower = 0
    for k in range(n):
        i = order[k]
        for q in range(indptr[i], indptr[i + 1]):
            if place[indices[q]] < k:
                lower += 1
    upper = indices.size - lower  # (with the diagonal: an upper bound)
    l_indptr = np.empty(n + 1, dtype=np.int64)
    u_indptr = np.empty(n + 1, dtype=np.int64)
    l_indices = np.empty(lower, dtype=np.int32)
    u_indices = np.empty(upper, dtype=np.int32)
    l_data = np.empty((lower, 9), dtype=np.float32)
    u_data = np.empty((upper, 9), dtype=np.float32)
    diag = np.zeros((n, 9), dtype=np.float32)
    dest = np.empty(indices.size, dtype=np.int64)
    nl = 0
    nu = 0
    for k in range(n):
        i = order[k]
        l_indptr[k] = nl
        u_indptr[k] = nu
        for q in range(indptr[i], indptr[i + 1]):
            j = place[indices[q]]
            if j < k:
                l_indices[nl] = j
                for r in range(9):
                    l_data[nl, r] = data[q, r]
                dest[q] = nl
                nl += 1
            elif j > k:
                u_indices[nu] = j
                for r in range(9):
                    u_data[nu, r] = data[q, r]
                dest[q] = -1 - nu  # (shifted below, once the blocks below are counted)
                nu += 1
            else:
                for r in range(9):
                    diag[k, r] += data[q, r]
                dest[q] = -1 - upper - k
    l_indptr[n] = nl
    u_indptr[n] = nu
    for q in range(dest.size):
        if dest[q] < -upper:
            dest[q] = nl + nu + (-1 - upper - dest[q])
        elif dest[q] < 0:
            dest[q] = nl + (-1 - dest[q])
    return l_indptr, l_indices, l_data, u_indptr, u_indices[:nu], u_data[:nu], diag, dest


@njit(parallel=True, cache=True)
def _refresh_blocks(data, dest, l_data, u_data, diag):
    """Other values of the blocks of a matrix (``data``: nine per stored block) at their
    places in the copy of the sweeps (``dest``: :func:`_sorted_blocks`), in single
    precision."""
    nl = l_data.shape[0]
    nu = u_data.shape[0]
    for q in prange(dest.size):
        d = dest[q]
        if d < nl:
            for r in range(9):
                l_data[d, r] = data[q, r]
        elif d < nl + nu:
            for r in range(9):
                u_data[d - nl, r] = data[q, r]
        else:
            for r in range(9):
                diag[d - nl - nu, r] = data[q, r]


def _node_blocks(matrix: sp.csr_matrix, dofs: np.ndarray):
    """A matrix whose unknowns are components of the nodes of a mesh, by 3 x 3 blocks of
    nodes: ``dofs`` gives the degree of freedom ``3 node + component`` of each unknown,
    in increasing order. Returns the blocks (BSR, over the nodes that have an unknown:
    a component that is not one — a held one — is an empty row and column of its block)
    and the place ``3 rank + component`` of each unknown among them; none for unknowns
    in another order."""
    n = matrix.shape[0]
    if dofs.shape != (n,) or n == 0 or np.any(np.diff(dofs) <= 0):
        return None
    rank = np.concatenate(([0], np.cumsum(np.diff(dofs // 3) > 0)))  # the node of each unknown
    slot = 3 * rank + dofs % 3
    size = 3 * (int(rank[-1]) + 1)
    if size != n:
        counts = np.zeros(size, dtype=np.int64)
        counts[slot] = np.diff(matrix.indptr)
        indptr = np.zeros(size + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        columns = slot.astype(np.int32)[matrix.indices]
        matrix = sp.csr_matrix((matrix.data, columns, indptr), shape=(size, size))
    blocks = matrix.tobsr(blocksize=(3, 3))
    blocks.sort_indices()
    return blocks, slot


def _cuts(indptr: np.ndarray, chunks: int = 64) -> np.ndarray:
    """Rows that cut a CSR pattern into chunks of equal numbers of stored values: the
    rows of the part above the diagonal of a matrix sorted by colour go from full (the
    first colour) to empty (the last one), and equal shares of the rows would leave most
    of the work to one thread."""
    targets = (np.arange(chunks + 1) * int(indptr[-1])) // chunks
    cuts = np.searchsorted(indptr, targets, side="left")
    cuts[0], cuts[-1] = 0, indptr.size - 1
    return np.unique(cuts).astype(np.int64)


@njit(parallel=True, cache=True)
def _forward(l_indptr, l_indices, l_data, diag, starts, b, x):
    """``x = (D + L)^-1 b`` for a matrix sorted by colour: the forward Gauss-Seidel sweep
    from zero, colour by colour (``starts``: the offsets of the colours) — the unknowns
    of a colour, which no stored value couples, are relaxed in parallel."""
    for c in range(starts.size - 1):
        for i in prange(starts[c], starts[c + 1]):
            s = b[i]
            for k in range(l_indptr[i], l_indptr[i + 1]):
                s -= l_data[k] * x[l_indices[k]]
            x[i] = s / diag[i] if diag[i] != 0.0 else 0.0


@njit(parallel=True, cache=True)
def _upper(u_indptr, u_indices, u_data, cuts, x, out):
    """``out = -U x``: the residual ``b - A x`` of the ``x`` of a forward sweep from
    zero, for which ``b - (D + L) x`` is zero (``cuts``: the rows that share the stored
    values equally between the threads, :func:`_cuts`)."""
    for t in prange(cuts.size - 1):
        for i in range(cuts[t], cuts[t + 1]):
            s = 0.0
            for k in range(u_indptr[i], u_indptr[i + 1]):
                s -= u_data[k] * x[u_indices[k]]
            out[i] = s


@njit(parallel=True, cache=True)
def _relax(l_indptr, l_indices, l_data, u_indptr, u_indices, u_data, diag, starts, x, b,
           forward):  # fmt: skip
    """One Gauss-Seidel sweep on ``A x = b``, in place, for a matrix sorted by colour: in
    the order of the colours or the reverse one, the unknowns of a colour in parallel."""
    count = starts.size - 1
    for step in range(count):
        c = step if forward else count - 1 - step
        for i in prange(starts[c], starts[c + 1]):
            s = b[i]
            for k in range(l_indptr[i], l_indptr[i + 1]):
                s -= l_data[k] * x[l_indices[k]]
            for k in range(u_indptr[i], u_indptr[i + 1]):
                s -= u_data[k] * x[u_indices[k]]
            if diag[i] != 0.0:
                x[i] = s / diag[i]


@njit(parallel=True, cache=True)
def _forward3(l_indptr, l_indices, l_data, diag, starts, b, x):
    """:func:`_forward` by 3 x 3 blocks: the nodes of a colour in parallel, the three
    components of a node one after the other."""
    for c in range(starts.size - 1):
        for i in prange(starts[c], starts[c + 1]):
            s0 = b[3 * i]
            s1 = b[3 * i + 1]
            s2 = b[3 * i + 2]
            for k in range(l_indptr[i], l_indptr[i + 1]):
                j = 3 * l_indices[k]
                x0 = x[j]
                x1 = x[j + 1]
                x2 = x[j + 2]
                s0 -= l_data[k, 0] * x0 + l_data[k, 1] * x1 + l_data[k, 2] * x2
                s1 -= l_data[k, 3] * x0 + l_data[k, 4] * x1 + l_data[k, 5] * x2
                s2 -= l_data[k, 6] * x0 + l_data[k, 7] * x1 + l_data[k, 8] * x2
            d = diag[i]
            y0 = s0 / d[0] if d[0] != 0.0 else 0.0
            s1 -= d[3] * y0
            y1 = s1 / d[4] if d[4] != 0.0 else 0.0
            s2 -= d[6] * y0 + d[7] * y1
            y2 = s2 / d[8] if d[8] != 0.0 else 0.0
            x[3 * i] = y0
            x[3 * i + 1] = y1
            x[3 * i + 2] = y2


@njit(parallel=True, cache=True)
def _upper3(u_indptr, u_indices, u_data, diag, cuts, x, out):
    """:func:`_upper` by 3 x 3 blocks: the blocks above the diagonal and the part of the
    diagonal blocks above the diagonal."""
    for t in prange(cuts.size - 1):
        for i in range(cuts[t], cuts[t + 1]):
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            for k in range(u_indptr[i], u_indptr[i + 1]):
                j = 3 * u_indices[k]
                x0 = x[j]
                x1 = x[j + 1]
                x2 = x[j + 2]
                s0 -= u_data[k, 0] * x0 + u_data[k, 1] * x1 + u_data[k, 2] * x2
                s1 -= u_data[k, 3] * x0 + u_data[k, 4] * x1 + u_data[k, 5] * x2
                s2 -= u_data[k, 6] * x0 + u_data[k, 7] * x1 + u_data[k, 8] * x2
            d = diag[i]
            out[3 * i] = s0 - d[1] * x[3 * i + 1] - d[2] * x[3 * i + 2]
            out[3 * i + 1] = s1 - d[5] * x[3 * i + 2]
            out[3 * i + 2] = s2


@njit(parallel=True, cache=True)
def _relax3(l_indptr, l_indices, l_data, u_indptr, u_indices, u_data, diag, starts, x, b,
            forward):  # fmt: skip
    """:func:`_relax` by 3 x 3 blocks: the colours of the nodes in their order or the
    reverse one, the three components of a node in theirs or the reverse one."""
    count = starts.size - 1
    for step in range(count):
        c = step if forward else count - 1 - step
        for i in prange(starts[c], starts[c + 1]):
            s0 = b[3 * i]
            s1 = b[3 * i + 1]
            s2 = b[3 * i + 2]
            for k in range(l_indptr[i], l_indptr[i + 1]):
                j = 3 * l_indices[k]
                x0 = x[j]
                x1 = x[j + 1]
                x2 = x[j + 2]
                s0 -= l_data[k, 0] * x0 + l_data[k, 1] * x1 + l_data[k, 2] * x2
                s1 -= l_data[k, 3] * x0 + l_data[k, 4] * x1 + l_data[k, 5] * x2
                s2 -= l_data[k, 6] * x0 + l_data[k, 7] * x1 + l_data[k, 8] * x2
            for k in range(u_indptr[i], u_indptr[i + 1]):
                j = 3 * u_indices[k]
                x0 = x[j]
                x1 = x[j + 1]
                x2 = x[j + 2]
                s0 -= u_data[k, 0] * x0 + u_data[k, 1] * x1 + u_data[k, 2] * x2
                s1 -= u_data[k, 3] * x0 + u_data[k, 4] * x1 + u_data[k, 5] * x2
                s2 -= u_data[k, 6] * x0 + u_data[k, 7] * x1 + u_data[k, 8] * x2
            d = diag[i]
            y0 = x[3 * i]
            y1 = x[3 * i + 1]
            y2 = x[3 * i + 2]
            if forward:
                if d[0] != 0.0:
                    y0 = (s0 - d[1] * y1 - d[2] * y2) / d[0]
                if d[4] != 0.0:
                    y1 = (s1 - d[3] * y0 - d[5] * y2) / d[4]
                if d[8] != 0.0:
                    y2 = (s2 - d[6] * y0 - d[7] * y1) / d[8]
            else:
                if d[8] != 0.0:
                    y2 = (s2 - d[6] * y0 - d[7] * y1) / d[8]
                if d[4] != 0.0:
                    y1 = (s1 - d[3] * y0 - d[5] * y2) / d[4]
                if d[0] != 0.0:
                    y0 = (s0 - d[1] * y1 - d[2] * y2) / d[0]
            x[3 * i] = y0
            x[3 * i + 1] = y1
            x[3 * i + 2] = y2


@njit(parallel=True, cache=True)
def _matvec(indptr, indices, data, x, out):
    """``out = A x`` (CSR), the rows shared between the threads; each row is summed in
    the order of its stored values, as SciPy sums it: the same product to the bit."""
    for i in prange(out.size):
        s = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            s += data[k] * x[indices[k]]
        out[i] = s


@njit(parallel=True, cache=True)
def _dense(matrix, x, out):
    """``out = M x`` for a dense matrix, its rows shared between the threads."""
    for i in prange(out.size):
        row = matrix[i]
        s = 0.0
        for j in range(x.size):
            s += row[j] * x[j]
        out[i] = s


def dense_product(matrix: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``M x`` of a dense matrix and a vector on the threads of the compiled kernels. The
    product of BLAS (``M @ x``) runs on its own threads, which then stay busy for a tenth
    of a second (OpenBLAS: they wait for the next product) and take the cores of the
    kernels that follow: a sweep or a product several times slower, at each iteration
    where the coarsest level of a multigrid cycle is such a product."""
    out = np.empty(matrix.shape[0])
    _dense(np.ascontiguousarray(matrix), np.ascontiguousarray(x, dtype=np.float64), out)
    return out


def product(matrix: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
    """``A x`` of a CSR matrix and a vector: on several threads for a large matrix (the
    product of SciPy runs on one), with the same result to the bit."""
    if (
        matrix.nnz < THREADED
        or matrix.format != "csr"
        or matrix.data.dtype != np.float64
        or x.ndim != 1
        or x.dtype != np.float64
    ):
        return matrix @ x
    out = np.empty(matrix.shape[0])
    _matvec(matrix.indptr, matrix.indices, matrix.data, np.ascontiguousarray(x), out)
    return out


@njit(parallel=True, cache=True)
def _block_matvec(indptr, indices, data, cuts, x, out):
    """``out = A x`` for a matrix stored by 3 x 3 blocks of nodes (block CSR, nine values
    per block, row by row), the rows of the nodes shared between the threads by equal
    numbers of blocks (``cuts``). The three components of a row are summed term by term
    in the order of the columns, as the product of the stored values of the matrix sums
    them (a value that is not stored adds a zero): the same product to the bit."""
    for t in prange(cuts.size - 1):
        for i in range(cuts[t], cuts[t + 1]):
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            for k in range(indptr[i], indptr[i + 1]):
                j = 3 * indices[k]
                x0 = x[j]
                x1 = x[j + 1]
                x2 = x[j + 2]
                s0 += data[k, 0] * x0
                s0 += data[k, 1] * x1
                s0 += data[k, 2] * x2
                s1 += data[k, 3] * x0
                s1 += data[k, 4] * x1
                s1 += data[k, 5] * x2
                s2 += data[k, 6] * x0
                s2 += data[k, 7] * x1
                s2 += data[k, 8] * x2
            out[3 * i] = s0
            out[3 * i + 1] = s1
            out[3 * i + 2] = s2


@njit(parallel=True, cache=True)
def _block_places(indptr, indices, slot, b_indptr, b_indices, place):
    """The place of each stored value of a CSR matrix among the values of its blocks
    (nine per block of the block pattern ``b_indptr``, ``b_indices``, its rows sorted;
    ``slot``: the place ``3 node + component`` of each unknown). Returns the number of
    stored values without a block."""
    missing = 0
    for i in prange(indptr.size - 1):
        node = slot[i] // 3
        row = slot[i] - 3 * node
        for k in range(indptr[i], indptr[i + 1]):
            other = slot[indices[k]] // 3
            lo, hi = b_indptr[node], b_indptr[node + 1]
            while lo < hi:
                mid = (lo + hi) // 2
                if b_indices[mid] < other:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < b_indptr[node + 1] and b_indices[lo] == other:
                place[k] = 9 * lo + 3 * row + slot[indices[k]] - 3 * other
            else:
                place[k] = 0
                missing += 1
    return missing


@njit(parallel=True, cache=True)
def _fill_blocks(indptr, place, values, data):
    """The stored values of a CSR matrix at their places in the values of its blocks
    (``place``: :func:`_block_places`; two rows of the matrix never write at one place)."""
    for p in prange(data.size):
        data[p] = 0.0
    for i in prange(indptr.size - 1):
        for k in range(indptr[i], indptr[i + 1]):
            data[place[k]] += values[k]


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two arrays are one, or views of the same values in memory (SciPy gives
    each matrix its own view of the arrays of a pattern that it is built on)."""
    return a is b or (
        a.shape == b.shape
        and a.dtype == b.dtype
        and a.strides == b.strides
        and a.__array_interface__["data"][0] == b.__array_interface__["data"][0]
    )


class NodeBlocks:
    """A square matrix whose unknowns are components of the displacements of the nodes of
    a mesh, by the 3 x 3 blocks of the couplings of its nodes, in double precision (block
    CSR over the nodes that have an unknown: :attr:`indptr`, :attr:`indices`, sorted;
    :attr:`data`, nine values per block, row by row). It is the form that the products
    of the Krylov iterations read (:meth:`product`: one column index for nine values, the
    three components of a neighbour read together; the same product to the bit as that
    of the stored values taken in the order of their columns), and from which the sweeps
    take their copy (:class:`Smoother`).

    ``dofs``: the degree of freedom ``3 node + component`` of each unknown, in increasing
    order. A component that is not an unknown (one that a condition holds) is an empty
    row and column of its block: the blocks have :attr:`size` rows, the unknown ``i`` at
    the row ``slot[i]``.

    :meth:`refresh` takes the values of another matrix of the same pattern — the tangent
    of the next Newton iteration of a loading path — at their places in the blocks, found
    once."""

    #: (as a SciPy matrix: the rows of the blocks are sorted)
    has_sorted_indices = True

    def __init__(self, matrix: sp.csr_matrix, dofs: np.ndarray):
        by_nodes = _node_blocks(matrix, np.asarray(dofs))
        if by_nodes is None:
            raise ValueError("unknowns that are not in the order of their degrees of freedom")
        blocks, self.slot = by_nodes
        self.n = matrix.shape[0]
        self.size = blocks.shape[0]
        self.indptr, self.indices = blocks.indptr, blocks.indices
        self.data = np.ascontiguousarray(blocks.data, dtype=np.float64).reshape(-1, 9)
        self.cuts = _cuts(self.indptr)
        self._pattern = None  # the CSR pattern of the places of :meth:`refresh`
        self._place = None

    @classmethod
    def of(cls, matrix, dofs) -> NodeBlocks | None:
        """The blocks of a matrix, or none where they would not serve: a small matrix
        (:data:`THREADED`), another format or precision, unknowns that are not in the
        order of their degrees of freedom."""
        if (
            dofs is None
            or matrix.format != "csr"
            or matrix.nnz < THREADED
            or matrix.data.dtype != np.float64
            or matrix.shape[0] >= 2**31 - 1
        ):
            return None
        dofs = np.asarray(dofs)
        if dofs.shape != (matrix.shape[0],) or np.any(np.diff(dofs) <= 0):
            return None
        return cls(matrix, dofs)

    def product(self, x: np.ndarray) -> np.ndarray:
        """``A x``, on several threads."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        if self.size == self.n:
            out = np.empty(self.n)
            _block_matvec(self.indptr, self.indices, self.data, self.cuts, x, out)
            return out
        wide = np.zeros(self.size)
        wide[self.slot] = x
        out = np.empty(self.size)
        _block_matvec(self.indptr, self.indices, self.data, self.cuts, wide, out)
        return out[self.slot]

    def refresh(self, matrix: sp.csr_matrix) -> bool:
        """Takes the values of another matrix of the same unknowns, whose stored values
        all have a block here; returns false, and leaves the blocks as they are, for a
        matrix with other stored values. The places of the stored values are found once
        for the matrices that share the arrays of their pattern (which are kept, and
        must not be changed in place)."""
        if matrix.format != "csr" or matrix.shape[0] != self.n or matrix.data.dtype != np.float64:
            return False
        pattern = self._pattern
        if (
            pattern is None
            or not _same(pattern[0], matrix.indptr)
            or not _same(pattern[1], matrix.indices)
        ):
            kind = np.int32 if 9 * self.indices.size < 2**31 - 1 else np.int64
            place = np.empty(matrix.indices.size, dtype=kind)
            if _block_places(matrix.indptr, matrix.indices, self.slot, self.indptr, self.indices,
                             place):  # fmt: skip
                return False
            self._pattern, self._place = (matrix.indptr, matrix.indices), place
        _fill_blocks(matrix.indptr, self._place, matrix.data, self.data.reshape(-1))
        return True


class Smoother:
    """Gauss-Seidel sweeps on a CSR matrix: in the order of the unknowns on one thread
    for a small matrix; for a large one (:data:`COLOURED`, fewer than 2^31 unknowns),
    in the order of the colours of its unknowns (those of the graph of its couplings,
    :func:`_graph`: no stored value couples two unknowns of a colour, on either side
    of the diagonal), those of a colour relaxed in parallel, on a copy of the matrix
    sorted by colour (:attr:`coloured`). The forward and the backward sweeps are the
    adjoints of each other in both cases, and the result does not depend on the number
    of threads.

    With colours, :meth:`sorted`, :meth:`start`, :meth:`residual`, :meth:`relax` and
    :meth:`unsorted` work on vectors in the order of the colours (a multigrid cycle
    then sorts its right-hand side once, and its solution back once): :attr:`size`
    values, the unknown ``i`` at the place ``place[i]``.

    ``dofs``: the degree of freedom ``3 node + component`` of each unknown, for unknowns
    that are components of the displacements of the nodes of a mesh, in increasing
    order. The colours are then those of the nodes (a third of the colours: no stored
    value couples two nodes of a colour), and the copy holds the 3 x 3 blocks of the
    couplings of the nodes (:attr:`block` is 3): the sweep is the Gauss-Seidel sweep in
    the order of the colours of the nodes, the three components of a node one after
    the other. The vectors in the order of the colours have three places per node; the
    place of a component that is not an unknown (a held one) stays zero. ``blocks``:
    the blocks of the matrix, where they have been made already (:class:`NodeBlocks`);
    :meth:`refresh` then gives the sweeps the values that these blocks have taken since
    (the tangent of another Newton iteration: the same colours, the same places)."""

    def __init__(self, matrix: sp.csr_matrix, dofs: np.ndarray | None = None,
                 blocks: NodeBlocks | None = None):  # fmt: skip
        self.matrix = matrix
        self.coloured = False
        self.block = 1
        self._source = None  # the blocks that the copy of the sweeps was taken from
        n = matrix.shape[0]
        if not (COLOURED <= matrix.nnz and n < 2**31 - 1 and matrix.data.dtype == np.float64):
            return
        if blocks is None:
            blocks = NodeBlocks.of(matrix, dofs)
        if blocks is not None:
            self.size = blocks.size
            nodes, self.starts = _colours(*_graph(blocks), self.size // 3)
            *arrays, self._dest = _sorted_blocks(blocks.indptr, blocks.indices, blocks.data, nodes)
            rank = np.empty(nodes.size, dtype=np.int64)
            rank[nodes] = np.arange(nodes.size)
            self.place = 3 * rank[blocks.slot // 3] + blocks.slot % 3
            self.block = 3
            self._source = blocks
        else:
            order, self.starts = _colours(*_graph(matrix), n)
            arrays = _sorted_by_colour(matrix.indptr, matrix.indices, matrix.data, order)
            self.size = n
            self.place = np.empty(n, dtype=np.int64)
            self.place[order] = np.arange(n)
        self.lower, self.upper, self.diag = arrays[:3], arrays[3:6], arrays[6]
        self.cuts = _cuts(self.upper[0])
        self.coloured = True

    def refresh(self, blocks: NodeBlocks) -> bool:
        """The sweeps of the values that the blocks of the matrix have now (those that
        the copy of the sweeps was taken from: false otherwise, and for sweeps that do
        not work by nodes)."""
        if self._source is None or blocks is not self._source:
            return False
        _refresh_blocks(blocks.data, self._dest, self.lower[2], self.upper[2], self.diag)
        return True

    def sorted(self, v: np.ndarray) -> np.ndarray:
        """A vector in the order of the colours."""
        out = np.zeros(self.size) if self.size != v.size else np.empty(self.size)
        out[self.place] = v
        return out

    def unsorted(self, v: np.ndarray) -> np.ndarray:
        """A vector of the order of the colours, in the order of the unknowns."""
        return v[self.place]

    def start(self, b: np.ndarray) -> np.ndarray:
        """The forward sweep from zero (vectors in the order of the colours)."""
        x = np.empty_like(b)
        forward = _forward3 if self.block == 3 else _forward
        forward(*self.lower, self.diag, self.starts, b, x)
        return x

    def residual(self, x: np.ndarray) -> np.ndarray:
        """``b - A x`` for the ``x`` that :meth:`start` returns (in the order of the
        colours): the part of the matrix above the diagonal only."""
        out = np.empty_like(x)
        if self.block == 3:
            _upper3(*self.upper, self.diag, self.cuts, x, out)
        else:
            _upper(*self.upper, self.cuts, x, out)
        return out

    def relax(self, x: np.ndarray, b: np.ndarray, forward: bool) -> None:
        """A sweep on vectors in the order of the colours, in place."""
        relax = _relax3 if self.block == 3 else _relax
        relax(*self.lower, *self.upper, self.diag, self.starts, x, b, forward)

    def sweep(self, x: np.ndarray, b: np.ndarray, forward: bool) -> None:
        """A sweep on vectors in the order of the unknowns, in place."""
        if not self.coloured:
            a = self.matrix
            gauss_seidel(a.indptr, a.indices, a.data, x, b, forward)
            return
        xs = self.sorted(x)
        self.relax(xs, self.sorted(b), forward)
        x[:] = xs[self.place]


@njit(cache=True)
def _aggregate(indptr, indices, n):
    """Standard aggregation of the graph of a CSR matrix: aggregate index per node."""
    agg = np.full(n, -1, dtype=np.int64)
    count = 0
    for i in range(n):  # pass 1: nodes whose neighbours are all free, with them
        if agg[i] >= 0:
            continue
        free = True
        for k in range(indptr[i], indptr[i + 1]):
            if agg[indices[k]] >= 0:
                free = False
                break
        if free:
            agg[i] = count
            for k in range(indptr[i], indptr[i + 1]):
                agg[indices[k]] = count
            count += 1
    joined = agg.copy()
    for i in range(n):  # pass 2: join a neighbouring aggregate of pass 1
        if agg[i] >= 0:
            continue
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            if agg[j] >= 0:
                joined[i] = agg[j]
                break
    agg = joined
    for i in range(n):  # pass 3: the rest, with their free neighbours
        if agg[i] >= 0:
            continue
        agg[i] = count
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            if agg[j] < 0:
                agg[j] = count
        count += 1
    return agg, count


@njit(cache=True)
def _fit_candidates(order, start, candidates):
    """Orthonormal bases of the candidates on each aggregate (modified Gram-Schmidt).

    ``order`` lists the nodes aggregate by aggregate (``start`` pointers). Returns the
    column of each node's basis vectors (``q``: one row per node, ``k`` columns, the
    ``kept`` first are used), the number of basis vectors per aggregate, and the
    coarse candidates (rows of R, ``k`` per aggregate).
    """
    n, k = candidates.shape
    n_agg = start.size - 1
    q = np.zeros((n, k))
    kept = np.zeros(n_agg, dtype=np.int64)
    coarse = np.zeros((n_agg * k, k))
    for a in range(n_agg):
        lo, hi = start[a], start[a + 1]
        m = hi - lo
        basis = np.zeros((m, k))
        r = np.zeros((k, k))
        nk = 0
        for c in range(k):
            v = np.empty(m)
            norm0 = 0.0
            for t in range(m):
                v[t] = candidates[order[lo + t], c]
                norm0 += v[t] * v[t]
            norm0 = np.sqrt(norm0)
            for p in range(nk):  # project out the kept vectors (twice, for stability)
                for _ in range(2):
                    dot = 0.0
                    for t in range(m):
                        dot += basis[t, p] * v[t]
                    r[p, c] += dot
                    for t in range(m):
                        v[t] -= dot * basis[t, p]
            norm = 0.0
            for t in range(m):
                norm += v[t] * v[t]
            norm = np.sqrt(norm)
            if norm > 1e-10 * norm0 and norm > 1e-300 and nk < m:
                for t in range(m):
                    basis[t, nk] = v[t] / norm
                r[nk, c] = norm
                nk += 1
        kept[a] = nk
        for t in range(m):
            for p in range(nk):
                q[order[lo + t], p] = basis[t, p]
        for p in range(nk):
            for c in range(k):
                coarse[a * k + p, c] = r[p, c]
    return q, kept, coarse


def _tentative(matrix: sp.csr_matrix, candidates: np.ndarray):
    """Tentative prolongator and coarse candidates of a matrix."""
    n = matrix.shape[0]
    agg, n_agg = _aggregate(matrix.indptr, matrix.indices, n)
    order = np.argsort(agg, kind="stable")
    start = np.searchsorted(agg[order], np.arange(n_agg + 1))
    q, kept, coarse = _fit_candidates(order, start, np.ascontiguousarray(candidates))
    k = candidates.shape[1]
    offset = np.concatenate([[0], np.cumsum(kept)])
    width = kept[agg]  # basis vectors of each node's aggregate
    rows = np.repeat(np.arange(n), width)
    within = np.arange(rows.size) - np.repeat(np.cumsum(width) - width, width)
    cols = offset[agg][rows] + within
    vals = q[rows, within]
    t = sp.csr_matrix((vals, (rows, cols)), shape=(n, int(offset[-1])))
    keep_rows = (np.arange(k)[None, :] < kept[:, None]).ravel()
    return t, coarse[keep_rows]


def _spectral_radius(matrix: sp.csr_matrix, inv_diag: np.ndarray) -> float:
    """Largest eigenvalue of D^-1 A (symmetric form D^-1/2 A D^-1/2)."""
    s = np.sqrt(inv_diag)
    scaled = sp.diags(s) @ matrix @ sp.diags(s)
    if scaled.shape[0] <= 50:
        return float(np.max(np.abs(np.linalg.eigvalsh(scaled.toarray()))))
    rng = np.random.default_rng(0)
    try:
        value = spla.eigsh(
            scaled,
            k=1,
            which="LA",
            tol=1e-3,
            maxiter=200,
            v0=rng.random(scaled.shape[0]),
            return_eigenvectors=False,
        )
        return float(value[0])
    except spla.ArpackNoConvergence as exc:  # pragma: no cover - rare
        if len(exc.eigenvalues):
            return float(np.max(exc.eigenvalues))
        return float(abs(scaled).sum(axis=1).max())  # Gershgorin bound


class SmoothedAggregation:
    """Smoothed-aggregation hierarchy of a symmetric positive (semi-)definite matrix.

    ``candidates`` (n x k) are the near-null modes (default: constants). Calling the
    object applies one V-cycle from a zero initial guess. ``dofs``: the degree of
    freedom ``3 node + component`` of each unknown, for unknowns that are components of
    the nodes of a mesh (the sweeps of the first level then work by nodes, see
    :class:`Smoother`); ``blocks``: the blocks of the matrix, where they have been made
    already (:class:`NodeBlocks`; :meth:`refresh` then gives the sweeps of the first
    level the values that they have taken since).
    """

    def __init__(self, matrix, candidates=None, *, max_coarse: int = 500, max_levels: int = 20,
                 dofs=None, blocks=None):  # fmt: skip
        a = sp.csr_matrix(matrix, dtype=np.float64)
        a.sum_duplicates()
        a.sort_indices()
        n = a.shape[0]
        b = np.ones((n, 1)) if candidates is None else np.asarray(candidates, dtype=float)
        if b.ndim == 1:
            b = b[:, None]
        b = b[:, np.linalg.norm(b, axis=0) > 0] if b.shape[1] else np.ones((n, 1))
        self.levels: list[dict] = []
        while a.shape[0] > max_coarse and len(self.levels) < max_levels - 1:
            diag = a.diagonal()
            inv_diag = np.where(diag > 0, 1.0 / np.where(diag > 0, diag, 1.0), 0.0)
            t, coarse_b = _tentative(a, b)
            if t.shape[1] == 0 or t.shape[1] >= a.shape[0]:
                break
            rho = _spectral_radius(a, inv_diag)
            omega = 4.0 / 3.0 / rho if rho > 0 else 0.0
            p = (t - omega * (sp.diags(inv_diag) @ (a @ t))).tocsr()
            r = p.T.tocsr()
            first = not self.levels
            smoother = Smoother(a, dofs if first else None, blocks if first else None)
            self.levels.append({"A": a, "P": p, "R": r, "smoother": smoother})
            a = (r @ a @ p).tocsr()
            a.sum_duplicates()
            a.sort_indices()
            b = coarse_b
        self.levels.append({"A": a})
        self._coarse = self._pseudo_inverse(a)

    def refresh(self, blocks: NodeBlocks) -> bool:
        """The sweeps of the first level for the values that the blocks of the matrix
        have now (see :meth:`Smoother.refresh`); the interpolations and the coarser
        levels are kept. The cycle stays a symmetric positive definite preconditioner:
        its coarse correction adds a positive term to the symmetric sweeps of the
        matrix as it is."""
        return len(self.levels) > 1 and self.levels[0]["smoother"].refresh(blocks)

    @staticmethod
    def _pseudo_inverse(a: sp.csr_matrix):
        if a.shape[0] > 3000:  # coarsening stalled: sparse factorization
            lu = spla.splu(a.tocsc())
            return spla.LinearOperator(a.shape, matvec=lu.solve)
        dense = a.toarray()
        dense = 0.5 * (dense + dense.T)
        w, v = sla.eigh(dense)
        cut = 1e-12 * max(float(np.abs(w).max()) if w.size else 0.0, 1e-300)
        inv = np.where(np.abs(w) > cut, 1.0 / np.where(np.abs(w) > cut, w, 1.0), 0.0)
        return (v * inv) @ v.T

    def operator_complexity(self) -> float:
        nnz = [lvl["A"].nnz for lvl in self.levels]
        return float(sum(nnz) / nnz[0]) if nnz[0] else 1.0

    def _cycle(self, level: int, b: np.ndarray) -> np.ndarray:
        lvl = self.levels[level]
        if level == len(self.levels) - 1:
            if isinstance(self._coarse, np.ndarray):
                return dense_product(self._coarse, b)
            return self._coarse @ b
        a, smoother = lvl["A"], lvl["smoother"]
        if smoother.coloured:  # (the cycle of this level in the order of the colours)
            bs = smoother.sorted(b)
            xs = smoother.start(bs)
            residual = smoother.unsorted(smoother.residual(xs))
            coarse = product(lvl["P"], self._cycle(level + 1, product(lvl["R"], residual)))
            xs += smoother.sorted(coarse)
            smoother.relax(xs, bs, False)
            return smoother.unsorted(xs)
        x = np.zeros_like(b)
        smoother.sweep(x, b, True)
        residual = b - product(a, x)
        x += product(lvl["P"], self._cycle(level + 1, product(lvl["R"], residual)))
        smoother.sweep(x, b, False)
        return x

    def __call__(self, b: np.ndarray) -> np.ndarray:
        return self._cycle(0, np.ascontiguousarray(b, dtype=np.float64))
