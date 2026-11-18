"""The kernels of the iterative solvers of the finite element problems: the product of a
sparse matrix on several threads, the Gauss-Seidel sweeps in the order of the colours of
the unknowns or of the nodes (3 x 3 blocks), the two-level preconditioner that they
make, the reduction of a tangent by the places of its stored values, the products
that leave the threads of BLAS asleep between the compiled kernels, the matrix by blocks
of nodes (its product, its values taken from another matrix), the sweeps that follow
them, and the preconditioner kept from a step of a loading path to the next."""

import numba
import numpy as np
import pytest
import scipy.sparse as sp

from helpers import box_mesh
from morphogen import settings
from morphogen.homogenization import Material, Plasticity, amg, solvers
from morphogen.homogenization.fe import Reduction
from morphogen.homogenization.nonlinear import NonlinearFEProblem


def spd(n=3000, seed=0):
    """A sparse symmetric positive definite matrix above the size of the coloured sweeps
    (some 85 stored values per row, as the stiffness of 10-node tetrahedra)."""
    g = sp.random(n, n, density=3.0 / n, random_state=seed, format="csr")
    g = g + sp.diags(np.ones(n))
    a = (g @ g.T @ g @ g.T + 0.5 * sp.identity(n)).tocsr()
    a.sum_duplicates()
    a.sort_indices()
    assert a.nnz >= amg.COLOURED
    return a


def displacements(n_nodes=1100, seed=0):
    """A sparse symmetric positive definite matrix with the pattern of the displacements
    of a mesh: the three components of the nodes, 3 x 3 blocks between coupled nodes."""
    rng = np.random.default_rng(seed)
    g = sp.random(n_nodes, n_nodes, density=4.0 / n_nodes, random_state=seed, format="csr")
    g = sp.kron(g + sp.identity(n_nodes), np.ones((3, 3)), format="csr")
    g.data = rng.normal(size=g.nnz)
    a = (g @ g.T + 0.5 * sp.identity(3 * n_nodes)).tocsr()
    a.sum_duplicates()
    a.sort_indices()
    return a


#: the kinds of unknowns of the sweeps: any; the three components of every node; the
#: components of the nodes without those that a condition holds
KINDS = ["unknowns", "nodes", "held"]


def system(kind):
    """A matrix above the size of the coloured sweeps and the degrees of freedom of its
    unknowns (none: unknowns of any kind)."""
    if kind == "unknowns":
        return spd(), None
    a = displacements()
    dofs = np.arange(a.shape[0])
    if kind == "held":  # a component, two components and a whole node are not unknowns
        held = [3 * 5, 3 * 17 + 1, 3 * 40, 3 * 40 + 2, 3 * 77, 3 * 77 + 1, 3 * 77 + 2, 3 * 500 + 2]
        dofs = np.setdiff1d(dofs, held)
        a = a[dofs][:, dofs].tocsr()
    assert a.nnz >= amg.COLOURED
    return a, dofs


def in_colour_order(smoother, a):
    """The matrix of a smoother with its unknowns at their places in the order of the
    colours (empty rows and columns at the places of held components)."""
    n = a.shape[0]
    assert np.unique(smoother.place).size == n and smoother.place.max() < smoother.size
    e = sp.csr_matrix((np.ones(n), (smoother.place, np.arange(n))), shape=(smoother.size, n))
    ap = (e @ a @ e.T).tocsr()
    ap.sort_indices()
    return e, ap


def test_product_on_several_threads():
    """The product of a large CSR matrix and a vector on several threads is that of SciPy
    to the bit (each row summed in the order of its stored values); small matrices, and
    matrices or vectors that are not in double precision, are left to SciPy."""
    rng = np.random.default_rng(0)
    a = spd()
    x = rng.normal(size=a.shape[1])
    assert a.nnz >= amg.THREADED and np.array_equal(amg.product(a, x), a @ x)
    wide = sp.random(700, 5000, density=0.02, random_state=1, format="csr")
    y = rng.normal(size=5000)
    assert wide.nnz >= amg.THREADED and np.array_equal(amg.product(wide, y), wide @ y)
    small = sp.random(50, 50, density=0.2, random_state=2, format="csr")
    assert np.array_equal(amg.product(small, y[:50]), small @ y[:50])
    single = a.astype(np.float32)
    assert np.array_equal(amg.product(single, x), single @ x)


def colours(order, starts):
    """The colour of each unknown, from the unknowns sorted by colour."""
    colour = np.empty(order.size, dtype=int)
    colour[order] = np.repeat(np.arange(starts.size - 1), np.diff(starts))
    return colour


def test_colours_of_the_unknowns():
    """The greedy colouring of the graph of a matrix: the unknowns sorted by colour are a
    permutation, and no stored value couples two unknowns of a colour — on either side
    of the diagonal: a value stored on one side only (a product of sparse matrices drops
    the exact zeros) separates its two unknowns as well."""
    a = spd()
    n = a.shape[0]
    order, starts = amg._colours(a.indptr, a.indices, n)
    assert np.array_equal(np.sort(order), np.arange(n))
    assert starts[0] == 0 and starts[-1] == n and np.all(np.diff(starts) > 0)
    colour = colours(order, starts)
    rows = np.repeat(np.arange(n), np.diff(a.indptr))
    coupled = rows != a.indices
    assert np.all(colour[rows[coupled]] != colour[a.indices[coupled]])
    assert starts.size - 1 <= np.diff(a.indptr).max() + 1
    for c in range(starts.size - 1):  # within a colour, the unknowns in their order
        assert np.all(np.diff(order[starts[c] : starts[c + 1]]) > 0)
    # two unknowns of a colour, coupled by a value stored in the row of the first only
    assert amg._one_sided(a.indptr, a.indices) == 0
    assert amg._graph(a)[1] is a.indices
    i, j = order[starts[0]], order[starts[0] + 1]
    assert i < j and a[i, j] == 0.0 and a[j, i] == 0.0
    b = (a + sp.csr_matrix(([1e-3], ([i], [j])), shape=a.shape)).tocsr()
    b.sort_indices()
    assert amg._one_sided(b.indptr, b.indices) == 1
    plain = colours(*amg._colours(b.indptr, b.indices, n))
    assert plain[i] == plain[j]  # (the colouring of the rows alone does not see it)
    indptr, indices = amg._graph(b)
    assert amg._one_sided(indptr, indices) == 0 and indices.size == b.nnz + 1
    smoother = amg.Smoother(b)
    separated = colours(np.argsort(smoother.place), smoother.starts)
    assert separated[i] != separated[j]
    rows = np.repeat(np.arange(n), np.diff(b.indptr))
    coupled = rows != b.indices
    assert np.all(separated[rows[coupled]] != separated[b.indices[coupled]])


def test_colours_of_the_nodes():
    """With the degrees of freedom of its unknowns, a matrix is coloured by nodes: no
    stored value couples two nodes of a colour, the three components of a node are at
    three places that follow each other, and a component that a condition holds leaves
    its place empty."""
    for kind in ("nodes", "held"):
        a, dofs = system(kind)
        smoother = amg.Smoother(a, dofs)
        assert smoother.coloured and smoother.block == 3
        place, starts = smoother.place, smoother.starts
        n_nodes = np.unique(dofs // 3).size
        assert smoother.size == 3 * n_nodes and starts[-1] == n_nodes
        assert (smoother.size == dofs.size) == (kind == "nodes")
        assert np.array_equal(place % 3, dofs % 3)  # the components in their order
        node = place // 3
        assert np.array_equal(np.unique(node), np.arange(n_nodes))
        assert np.all((np.diff(dofs // 3) == 0) == (np.diff(node) == 0))  # a node, one place
        colour = np.searchsorted(starts, node, side="right") - 1
        rows = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
        apart = node[rows] != node[a.indices]
        assert np.all(colour[rows[apart]] != colour[a.indices[apart]])
        assert starts.size - 1 < amg.Smoother(a).starts.size - 1  # fewer colours than unknowns'
    # unknowns that are not in the order of their degrees of freedom: coloured one by one
    a, dofs = system("nodes")
    assert amg.Smoother(a, dofs[::-1].copy()).block == 1


@pytest.mark.parametrize("kind", KINDS)
def test_sweeps_in_the_order_of_the_colours(kind):
    """The sweeps of a large matrix are Gauss-Seidel sweeps in the order of the colours
    (of the unknowns, or of the nodes with their three components one after the other):
    those of the matrix sorted by colour, taken in the order of its unknowns on one
    thread (the copy that the sweeps read is in single precision: to 1e-5). The forward
    sweep from zero and the residual that follows it read half of the matrix each; a
    small matrix is relaxed in the order of its unknowns."""
    rng = np.random.default_rng(1)
    a, dofs = system(kind)
    smoother = amg.Smoother(a, dofs)
    assert smoother.coloured and smoother.block == (1 if dofs is None else 3)
    e, ap = in_colour_order(smoother, a)
    n = smoother.size
    b = rng.normal(size=a.shape[0])
    bs = smoother.sorted(b)
    assert np.array_equal(bs, e @ b) and np.array_equal(smoother.unsorted(bs), b)
    # the forward sweep from zero
    ref = np.zeros(n)
    amg.gauss_seidel(ap.indptr, ap.indices, ap.data, ref, bs, True)
    x = smoother.start(bs)
    scale = np.abs(ref).max()
    assert np.allclose(x, ref, rtol=0, atol=1e-5 * scale)
    # its residual: the part of the matrix above the diagonal
    assert np.allclose(smoother.residual(x), bs - ap @ x, rtol=0, atol=1e-5 * np.abs(bs).max())
    # a backward sweep, then a forward one, from this state
    for forward in (False, True):
        amg.gauss_seidel(ap.indptr, ap.indices, ap.data, ref, bs, forward)
        smoother.relax(x, bs, forward)
        assert np.allclose(x, ref, rtol=0, atol=1e-5 * scale)
    # the same sweeps on vectors in the order of the unknowns
    y = smoother.unsorted(x)
    smoother.sweep(y, b, False)
    smoother.relax(x, bs, False)
    assert np.array_equal(y, smoother.unsorted(x))
    # a small matrix: the sweep of its unknowns in their order
    small = (a[:60][:, :60] + sp.identity(60)).tocsr()
    plain = amg.Smoother(small, None if dofs is None else np.arange(60))
    assert not plain.coloured
    u, v = np.zeros(60), np.zeros(60)
    plain.sweep(u, b[:60], True)
    amg.gauss_seidel(small.indptr, small.indices, small.data, v, b[:60], True)
    assert np.array_equal(u, v)


def aggregation(n, size=8):
    """A piecewise constant interpolation: an unknown of the coarse space for ``size``
    consecutive unknowns."""
    rows = np.arange(n)
    return sp.csr_matrix((np.ones(n), (rows, rows // size)), shape=(n, -(-n // size)))


@pytest.mark.parametrize("kind", KINDS)
def test_two_level_preconditioner(kind, monkeypatch):
    """The two-level preconditioner with coloured sweeps (by unknowns, by nodes) is
    symmetric positive definite (a forward sweep, a coarse correction, the backward
    sweep: to the single precision of the copy of its sweeps), and conjugate gradients
    converge with it as with the sweeps in the order of the unknowns."""
    rng = np.random.default_rng(2)
    a, dofs = system(kind)
    n = a.shape[0]
    p = aggregation(n)
    pre = solvers.TwoLevelPreconditioner(a, p, None, dofs)
    assert pre.smoother.coloured and pre.smoother.block == (1 if dofs is None else 3)
    assert pre.p.shape == (pre.smoother.size, p.shape[1])
    u, v = rng.normal(size=(2, n))
    pu, pv = pre(u), pre(v)
    assert abs(u @ pv - v @ pu) < 1e-5 * np.sqrt((u @ pu) * (v @ pv))
    assert u @ pu > 0.0 and v @ pv > 0.0
    b = rng.normal(size=n)
    x, its, res = solvers.conjugate_gradient(lambda w: a @ w, pre, b, 1e-10, 300)
    assert res <= 1e-10 and np.linalg.norm(b - a @ x) <= 2e-10 * np.linalg.norm(b)
    monkeypatch.setattr(amg, "COLOURED", 10**12)  # the sweeps in the order of the unknowns
    plain = solvers.TwoLevelPreconditioner(a, p, None, dofs)
    assert not plain.smoother.coloured
    y, its_plain, _ = solvers.conjugate_gradient(lambda w: a @ w, plain, b, 1e-10, 300)
    assert abs(its - its_plain) <= max(2, 0.25 * its_plain)
    assert np.allclose(x, y, rtol=0, atol=1e-8 * np.abs(y).max())


@pytest.mark.parametrize("kind", KINDS)
def test_sweeps_do_not_depend_on_the_threads(kind):
    """The unknowns (the nodes) of a colour are relaxed independently: the sweeps, the
    products and the preconditioner give the same result to the bit on one thread and on
    several."""
    rng = np.random.default_rng(3)
    a, dofs = system(kind)
    n = a.shape[0]
    b = rng.normal(size=n)
    pre = solvers.TwoLevelPreconditioner(a, aggregation(n), None, dofs)
    several = numba.get_num_threads()
    try:
        results = []
        for threads in (1, several):
            numba.set_num_threads(threads)
            bs = pre.smoother.sorted(b)
            x = pre.smoother.start(bs)
            pre.smoother.relax(x, bs, False)
            results.append((x, pre(b), amg.product(a, b)))
    finally:
        numba.set_num_threads(several)
    for one, more in zip(*results, strict=True):
        assert np.array_equal(one, more)


@pytest.mark.parametrize("weighted", [False, True])
def test_reduction_of_a_tangent(weighted):
    """The reduced matrix ``T^T K T`` by the places of the stored values of ``K``
    (:class:`~morphogen.homogenization.fe.Reduction`) is that of the products of sparse
    matrices, to round-off, for a reduction with one stored value per row at most (ones:
    matching periodic faces and kinematic conditions; or weights) and rows without any
    (fixed unknowns); none is built for a reduction that interpolates."""
    rng = np.random.default_rng(4)
    n, m = 500, 300
    k = sp.random(n, n, density=0.04, random_state=5, format="csr")
    k = (k + k.T).tocsr()
    k.sort_indices()
    column = rng.integers(-1, m, size=n)  # the reduced unknown of each row (-1: none)
    rows = np.flatnonzero(column >= 0)
    values = rng.normal(size=rows.size) if weighted else np.ones(rows.size)
    t = sp.csr_matrix((values, (rows, column[rows])), shape=(n, m))
    plan = Reduction.of(t, k.indptr, k.indices)
    reduced = plan(k.data)
    reference = (t.T @ k @ t).tocsr()
    assert reduced.shape == (m, m) and reduced.has_sorted_indices
    assert abs(reduced - reference).max() <= 1e-14 * abs(reference).max()
    # another matrix of the same pattern, with the same plan
    other = sp.csr_matrix((rng.normal(size=k.nnz), k.indices, k.indptr), shape=k.shape)
    assert abs(plan(other.data) - (t.T @ other @ t)).max() <= 1e-13 * abs(other).max()
    interpolating = sp.csr_matrix(sp.random(n, m, density=0.02, random_state=6))
    assert Reduction.of(interpolating, k.indptr, k.indices) is None


def test_products_without_the_threads_of_blas():
    """The products of long arrays between two compiled kernels (dot products, norms,
    products of a matrix and a vector or a matrix, the dense matrix of the coarsest level
    of a multigrid cycle) are those of NumPy to round-off, without its BLAS."""
    rng = np.random.default_rng(5)
    n = 4000
    a, b = rng.normal(size=(2, n))
    tall, wide = rng.normal(size=(n, 6)), rng.normal(size=(6, n))
    assert abs(solvers.dot(a, b) - a @ b) <= 1e-12 * np.linalg.norm(a) * np.linalg.norm(b)
    assert abs(solvers.norm(a) - np.linalg.norm(a)) <= 1e-13 * np.linalg.norm(a)
    for x, y in ((a, b), (tall, a[:6]), (a, tall), (wide, tall), (tall.T, a), (tall, wide[:, :9])):
        found, reference = solvers.times(x, y), x @ y
        assert np.shape(found) == np.shape(reference)
        assert np.abs(found - reference).max() <= 1e-12 * np.abs(reference).max()
    dense = rng.normal(size=(300, 300))
    x = rng.normal(size=300)
    found = amg.dense_product(dense, x)
    assert np.abs(found - dense @ x).max() <= 1e-13 * np.abs(dense @ x).max()
    several = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        assert np.array_equal(amg.dense_product(dense, x), found)
    finally:
        numba.set_num_threads(several)
    assert np.array_equal(amg.dense_product(np.asfortranarray(dense), x), found)


def test_blas_on_one_thread():
    """Within a loading path or a Krylov solve the BLAS of NumPy and SciPy runs on one
    thread (its idle threads would take the cores of the compiled kernels), and has its
    threads back afterwards, also after an exception."""
    libraries = settings._openblas()
    if not libraries:
        pytest.skip("no OpenBLAS found in the process")

    def threads():
        return [get() for get, _ in libraries]

    before = threads()
    with settings.blas_on_one_thread():
        assert threads() == [1] * len(libraries)
        with settings.blas_on_one_thread():
            assert threads() == [1] * len(libraries)
        assert threads() == [1] * len(libraries)
    assert threads() == before
    with pytest.raises(ZeroDivisionError), settings.blas_on_one_thread():
        1 / 0
    assert threads() == before

    seen = []

    def operator(v):
        seen.append(threads())
        return 2.0 * v

    b = np.arange(1.0, 6.0)
    x, _, _ = solvers.conjugate_gradient(operator, lambda r: r, b, 1e-12, 10)
    assert np.allclose(x, b / 2.0) and seen and all(t == [1] * len(libraries) for t in seen)
    assert threads() == before
    assert solvers.conjugate_gradient.__name__ == "conjugate_gradient"
    assert solvers.gmres.__doc__ and solvers.minres.__doc__


def another(a, seed=7):
    """Another symmetric positive definite matrix with the stored values of ``a`` at the
    same places (the arrays of its pattern shared): ``D a D`` with a positive diagonal."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.5, 2.0, size=a.shape[0])
    rows = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
    return sp.csr_matrix((d[rows] * a.data * d[a.indices], a.indices, a.indptr), shape=a.shape)


@pytest.mark.parametrize("kind", ["nodes", "held"])
def test_blocks_of_the_nodes(kind):
    """The matrix by 3 x 3 blocks of nodes: its product is that of its stored values to
    the bit, on one thread and on several; it takes the values of another matrix of the
    same pattern at their places, found once for the matrices that share the arrays of
    their pattern, and those of a matrix with fewer stored values; a matrix with a value
    outside its blocks leaves it as it is."""
    rng = np.random.default_rng(6)
    a, dofs = system(kind)
    n = a.shape[0]
    blocks = amg.NodeBlocks.of(a, dofs)
    assert blocks.n == n and blocks.size == 3 * np.unique(dofs // 3).size
    assert (blocks.size == n) == (kind == "nodes")
    assert np.all(np.diff(blocks.slot) > 0) and blocks.slot[-1] < blocks.size
    x = rng.normal(size=n)
    x[::7] = 0.0
    assert np.array_equal(blocks.product(x), a @ x)
    several = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        assert np.array_equal(blocks.product(x), a @ x)
    finally:
        numba.set_num_threads(several)
    # the values of another matrix, at the places of its stored values
    b = another(a)
    assert blocks.refresh(b) and np.array_equal(blocks.product(x), b @ x)
    places = blocks._place
    c = another(a, seed=8)
    assert blocks.refresh(c) and blocks._place is places  # (the same pattern: found once)
    assert np.array_equal(blocks.product(x), c @ x)
    # fewer stored values: the others are zeros of the blocks
    fewer = c.copy()
    fewer.data[rng.random(fewer.nnz) < 0.2] = 0.0
    fewer.eliminate_zeros()
    assert fewer.nnz < c.nnz and blocks.refresh(fewer) and blocks._place is not places
    assert np.array_equal(blocks.product(x), fewer @ x)
    # a value outside the blocks: the blocks stay those of the last matrix
    coupled = set(zip(*blocks_pattern(blocks), strict=True))
    i, j = next((i, j) for i in range(n) for j in range(n - 1, i, -1)
                if (blocks.slot[i] // 3, blocks.slot[j] // 3) not in coupled)  # fmt: skip
    outside = (c + sp.csr_matrix(([1.0], ([i], [j])), shape=c.shape)).tocsr()
    assert not blocks.refresh(outside) and np.array_equal(blocks.product(x), fewer @ x)
    assert not blocks.refresh(c.astype(np.float32)) and not blocks.refresh(c[:-3][:, :-3])
    # where the blocks would not serve
    assert amg.NodeBlocks.of(a, None) is None
    assert amg.NodeBlocks.of(a, dofs[::-1].copy()) is None
    small = a[:30][:, :30].tocsr()
    assert amg.NodeBlocks.of(small, dofs[:30]) is None
    with pytest.raises(ValueError, match="order"):
        amg.NodeBlocks(a, dofs[::-1].copy())


def blocks_pattern(blocks):
    """The rows and the columns of the stored blocks."""
    rows = np.repeat(np.arange(blocks.indptr.size - 1), np.diff(blocks.indptr))
    return rows.tolist(), blocks.indices.tolist()


@pytest.mark.parametrize("kind", ["nodes", "held"])
def test_sweeps_follow_the_blocks(kind):
    """The sweeps by nodes take the values that the blocks of their matrix have taken
    since (the tangent of another Newton iteration): they are then the sweeps of that
    matrix to the bit, with the same colours; sweeps do not follow other blocks than
    their own, nor those of the unknowns coloured one by one."""
    rng = np.random.default_rng(9)
    a, dofs = system(kind)
    blocks = amg.NodeBlocks.of(a, dofs)
    smoother = amg.Smoother(a, dofs, blocks)
    assert smoother.block == 3
    b = another(a)
    assert blocks.refresh(b) and smoother.refresh(blocks)
    fresh = amg.Smoother(b, dofs)
    assert np.array_equal(smoother.place, fresh.place)
    assert np.array_equal(smoother.starts, fresh.starts)
    for mine, theirs in zip((*smoother.lower, *smoother.upper, smoother.diag),
                            (*fresh.lower, *fresh.upper, fresh.diag), strict=True):  # fmt: skip
        assert np.array_equal(mine, theirs)
    rhs = smoother.sorted(rng.normal(size=a.shape[0]))
    x, y = smoother.start(rhs), fresh.start(rhs)
    assert np.array_equal(x, y) and np.array_equal(smoother.residual(x), fresh.residual(y))
    smoother.relax(x, rhs, False)
    fresh.relax(y, rhs, False)
    assert np.array_equal(x, y)
    assert not smoother.refresh(amg.NodeBlocks.of(b, dofs))  # (not the blocks of its copy)
    assert not amg.Smoother(a).refresh(blocks)
    plain, _ = system("unknowns")
    assert not amg.Smoother(plain).refresh(blocks)


@pytest.mark.parametrize("kind", ["nodes", "held"])
def test_solver_of_another_matrix(kind):
    """A solver updated with another matrix of the same pattern (the tangent of the next
    Newton iteration) keeps its coarse levels and gives its blocks, and with them its
    sweeps, the values of the new matrix: its products are those of the new matrix to the
    bit, its sweeps those of a solver built for it, and it solves the new system; with a
    matrix that is not symmetric the sweeps stay those of the last symmetric one, and
    GMRES solves it."""
    rng = np.random.default_rng(10)
    a, dofs = system(kind)
    n = a.shape[0]
    p = aggregation(n)
    solver = solvers.LinearSolver(a, method="amg", tol=1e-10, coarse=(p, None), dofs=dofs)
    assert solver.blocks is not None and solver._precondition.smoother.block == 3
    rhs = rng.normal(size=n)
    x, info = solver.solve(rhs)
    assert info.iterations > 0 and np.linalg.norm(a @ x - rhs) <= 2e-10 * np.linalg.norm(rhs)
    assert np.array_equal(solver._product(rhs), a @ rhs)
    b = another(a)
    new = solver.updated(b)
    assert new.blocks is solver.blocks and new._precondition is solver._precondition
    assert np.array_equal(new._product(rhs), b @ rhs)
    fresh = solvers.LinearSolver(b, method="amg", tol=1e-10, coarse=(p, None), dofs=dofs)
    assert np.array_equal(new._precondition.smoother.diag, fresh._precondition.smoother.diag)
    y, info_new = new.solve(rhs)
    assert np.linalg.norm(b @ y - rhs) <= 2e-10 * np.linalg.norm(rhs)
    z, info_fresh = fresh.solve(rhs)
    assert info_new.iterations <= 2 * info_fresh.iterations + 2  # (the coarse levels of a)
    assert np.allclose(y, z, rtol=0, atol=1e-7 * np.abs(z).max())
    # a matrix that is not symmetric: the blocks follow, the sweeps stay
    rows = np.repeat(np.arange(n), np.diff(b.indptr))
    skew = sp.csr_matrix((b.data * np.where(rows < b.indices, 1.02, 1.0), b.indices, b.indptr),
                         shape=b.shape)  # fmt: skip
    diag = new._precondition.smoother.diag.copy()
    last = new.updated(skew, symmetric=False)
    assert not last.symmetric and np.array_equal(last._product(rhs), skew @ rhs)
    assert np.array_equal(last._precondition.smoother.diag, diag)
    w, info_last = last.solve(rhs)
    assert info_last.iterations > 0
    assert np.linalg.norm(skew @ w - rhs) <= 2e-10 * np.linalg.norm(rhs)
    # another pattern: other blocks, for the products
    other = (b + sp.identity(n)).tocsr()
    wider = new.updated(other)
    assert np.array_equal(wider._product(rhs), other @ rhs)


def test_wear_of_a_preconditioner():
    """A multigrid preconditioner is kept from a step of a loading path to the next until
    the solves of a step take :data:`~morphogen.homogenization.solvers.WORN` times the
    iterations per decade of their residual of the step for which it was built; a
    factorization and a diagonal are built with their solver."""
    rng = np.random.default_rng(11)
    a, dofs = system("nodes")
    n = a.shape[0]
    rhs = rng.normal(size=n)
    solver = solvers.LinearSolver(a, method="amg", tol=1e-8, coarse=(aggregation(n), None),
                                  dofs=dofs)  # fmt: skip
    assert not solver.worn()  # (no solve yet: nothing known)
    x, info = solver.solve(rhs)
    rates = solver._wear["rates"]
    assert len(rates) == 1 and rates[0] == pytest.approx(info.iterations / -np.log10(info.residual))
    solver.solve(rhs, x0=x)  # (a solve from a guess says nothing of the preconditioner)
    assert len(solver._wear["rates"]) == 1
    assert not solver.worn() and solver._wear["reference"] == pytest.approx(rates[0])
    new = solver.updated(another(a))
    assert new._wear is solver._wear and not new._wear["rates"]
    reference = new._wear["reference"]
    new._wear["rates"].extend([1.2 * reference, 1.3 * reference])  # (a mean below WORN)
    assert not new.worn() and not new._wear["rates"]
    new._wear["rates"].extend([1.3 * reference, 1.4 * reference])
    assert new.worn()
    assert solvers.WORN == pytest.approx(1.3)
    small = a[:300][:, :300].tocsc().tocsr()
    assert solvers.LinearSolver(small, method="direct").worn()
    assert solvers.LinearSolver(small, method="cg").worn()


def test_preconditioner_kept_along_a_path(monkeypatch):
    """Along a loading path solved by multigrid, the preconditioner of a step is kept for
    the next ones while it serves — its sweeps take the values of each tangent —, and
    built again once it is worn: fewer preconditioners than steps, the curve of the
    direct solver within the tolerance of Newton's method, with or without the rule."""
    mesh = box_mesh((5, 5, 5), upper=(1.0, 1.0, 1.0), order=2, perturb=0.15,
                    phase=lambda c: (c[:, 0] > 0.6).astype(int))  # fmt: skip
    soft = Material(young=70e3, poisson=0.3,
                    plasticity=Plasticity(yield_stress=100.0, hardening=500.0))  # fmt: skip
    mats = {0: soft, 1: Material(young=400e3, poisson=0.2)}
    path = {"times": [0, 10, 20], "increments": [5, 3], "strain": {"11": [0, 0.01, 0.006]},
            "stress": {"22": 0, "33": 0}}  # fmt: skip
    count = {"built": 0, "refreshed": 0}
    build, refresh = solvers.TwoLevelPreconditioner.__init__, amg.Smoother.refresh

    def built(self, *args, **kwargs):
        count["built"] += 1
        build(self, *args, **kwargs)

    def refreshed(self, blocks):
        done = refresh(self, blocks)
        count["refreshed"] += done
        return done

    monkeypatch.setattr(solvers.TwoLevelPreconditioner, "__init__", built)
    monkeypatch.setattr(amg.Smoother, "refresh", refreshed)
    direct = NonlinearFEProblem(mesh, mats, solver="direct").run(path)
    kept = NonlinearFEProblem(mesh, mats, solver="amg").run(path)
    steps, corrections = kept.iterations.size, int(kept.iterations.sum())
    assert steps == 8 and kept.info["linear_iterations"] > 0
    assert 1 <= count["built"] < steps
    assert count["refreshed"] == corrections - count["built"]
    scale = np.abs(direct.stress).max()
    assert np.abs(kept.stress - direct.stress).max() < 1e-6 * scale
    # a preconditioner worn at once: built again every other step (the step after it
    # was built gives the iterations that the next ones are compared with)
    first = dict(count)
    monkeypatch.setattr(solvers, "WORN", 0.0)
    again = NonlinearFEProblem(mesh, mats, solver="amg").run(path)
    assert count["built"] - first["built"] >= steps // 2 > first["built"]
    assert np.abs(again.stress - direct.stress).max() < 1e-6 * scale
