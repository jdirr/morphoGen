"""Linear solvers for the symmetric positive definite systems of the FE problems.

* ``direct``: sparse LU factorization (SuperLU, from SciPy), reused for every right-hand
  side. Exact and fast for small systems; its memory grows quickly in 3D.
* ``amg``: conjugate gradient preconditioned by smoothed-aggregation algebraic multigrid
  (:mod:`morphogen.homogenization.amg`), with the rigid-body modes (or constants) as
  near-null space. For 10-node tetrahedra, the preconditioner has two levels: a
  Gauss-Seidel sweep on the quadratic system, and an AMG cycle on the linear system of
  the corner nodes (Galerkin projection). The hierarchy is reused for every right-hand
  side.
* ``cg``: conjugate gradient with a Jacobi preconditioner (slow on large or
  high-contrast problems; for checks).
* ``auto``: ``direct`` up to :data:`DIRECT_LIMIT` unknowns, ``amg`` above.

The tangents of the nonlinear problems whose behaviours have nonsymmetric consistent
tangents (``symmetric=False``: crystal and porous plasticity, the Armstrong-Frederick
hardening, learned laws that are not potentials, MFront behaviours without a symmetric
tangent) are solved by sparse LU (``direct``) or by GMRES with deflated restarts
(:func:`gmres`) with the preconditioner of their symmetric part (``amg``, ``cg``); along
the loading paths of the voxels GMRES recycles its space from a solve to the next once
one restarts (GCRO-DR, :class:`Recycling`) and stops where it stagnates
(:data:`GMRES_STAGNATION`).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit, prange

from morphogen.homogenization.amg import NodeBlocks, SmoothedAggregation, Smoother, product
from morphogen.settings import without_blas_threads
from morphogen.stopping import checkpoint

log = logging.getLogger("morphogen")

DIRECT_LIMIT = 6_000
#: A multigrid preconditioner kept from a step of a loading path to the next is built again
#: when the iterations that a solve takes for a decade of its residual have grown by this
#: factor since it was built (:meth:`LinearSolver.worn`): building it costs as much as
#: the iterations that a third more would add to a step
WORN = 1.3
#: The dimension of the Krylov spaces of :func:`gmres` before it restarts (the vectors of
#: its basis kept in memory)
GMRES_RESTART = 50
#: The approximate eigenvectors that :func:`gmres` keeps at its restarts: the harmonic Ritz
#: vectors of the harmonic Ritz values of least modulus (deflated restarting; 0: the plain
#: restarts)
GMRES_DEFLATE = 20
#: The cycles over which the residual of a solve of :func:`gmres` along a loading path must
#: at least halve (``stagnation``): 240 iterations of restarted cycles. Measured on the
#: softening composites of 16^3 voxels (Rousselier and GTN at high triaxialities), it stops
#: all the solves that would end at their ``maxiter`` above their tolerance (35 of them,
#: after about 300 iterations instead of 1,320) and none of the 95 that converge after
#: restarts (up to 528 iterations, after slow cycles); six cycles would stop three of
#: these, three cycles eight
GMRES_STAGNATION = 8
METHODS = ("auto", "direct", "amg", "cg")
#: Solvers of finite element problems: morphoGen's linear solvers, or FEniCSx
#: (:mod:`morphogen.homogenization.fenicsx`).
SOLVERS = (*METHODS, "fenicsx")
#: Right-hand sides smaller than this fraction of their scale (the norm they would have
#: without cancellation between elements) are round-off: the affine field solves the
#: problem (homogeneous medium, laminate loaded along its layers...).
NEGLIGIBLE = 1e-12


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two vectors in a single thread. BLAS (``a @ b``) splits long
    vectors over its own threads, which compete with those of the compiled kernels and
    slow the solvers down several times — during the product and for a tenth of a second
    after it, while its idle threads wait for the next one (OpenBLAS); this sum is also
    the same for any number of threads."""
    return float(np.einsum("i,i->", a, b))


def norm(a: np.ndarray) -> float:
    """Euclidean norm of a vector (see :func:`dot`)."""
    return math.sqrt(dot(a, a))


#: The sums of the products of two arrays of one or two dimensions
_TIMES = {(1, 1): "i,i->", (2, 1): "ij,j->i", (1, 2): "i,ij->j", (2, 2): "ij,jk->ik"}


def times(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b`` for arrays of one or two dimensions, in a single thread: the products
    of long arrays between two compiled kernels (see :func:`dot`)."""
    return np.einsum(_TIMES[a.ndim, b.ndim], a, b)


def choose_method(method: str, n: int) -> str:
    """Concrete solver for ``n`` unknowns (resolves ``auto``)."""
    if method not in METHODS:
        raise ValueError(f"solver: one of {METHODS}, got {method!r}")
    if method != "auto":
        return method
    return "direct" if n <= DIRECT_LIMIT else "amg"


@dataclass
class SolveInfo:
    method: str
    n_dofs: int
    iterations: int
    residual: float
    setup_s: float
    solve_s: float

    def to_dict(self) -> dict:
        return {
            "solver": self.method,
            "n_dofs": self.n_dofs,
            "iterations": self.iterations,
            "residual": self.residual,
            "setup_s": round(self.setup_s, 4),
            "solve_s": round(self.solve_s, 4),
        }


@without_blas_threads
def conjugate_gradient(apply, precondition, rhs, tol, maxiter):
    """Preconditioned conjugate gradient from ``x = 0``.

    ``apply(v)`` and ``precondition(r)`` return new arrays. Stops when the residual is
    below ``tol`` times the norm of ``rhs``; returns (x, iterations, relative residual).
    A direction of negative curvature (an indefinite operator: the tangent of softening
    points) hands the system to :func:`minres`, from the current iterate or from zero,
    whichever leaves the smaller residual.
    Each iteration is a checkpoint of :mod:`morphogen.stopping`.
    """
    x = np.zeros_like(rhs)
    norm_b = norm(rhs)
    if norm_b == 0.0:
        return x, 0, 0.0
    r = rhs.copy()
    z = precondition(r)
    p = z.copy()
    rz = dot(r, z)
    res = 1.0
    it = 0
    for it in range(1, maxiter + 1):
        checkpoint()
        ap = apply(p)
        pap = dot(p, ap)
        if pap <= 0.0:
            norm_r = norm(r)
            if pap < 0.0 and norm_r > 0.0:
                if norm_r > norm_b:  # an iterate worse than zero
                    x[:] = 0.0
                    r = rhs.copy()
                    norm_r = norm_b
                dx, more, rel = minres(apply, precondition, r, tol * norm_b / norm_r,
                                       maxiter - it)  # fmt: skip
                return x + dx, it + more, rel * norm_r / norm_b
            break
        alpha = rz / pap
        x += alpha * p
        r -= alpha * ap
        res = norm(r) / norm_b
        if res <= tol:
            break
        z = precondition(r)
        rz_new = dot(r, z)
        if rz_new <= 0.0:  # the residual is round-off the preconditioner cancels
            break
        p *= rz_new / rz
        p += z
        rz = rz_new
    return x, it, res


@without_blas_threads
def minres(apply, precondition, rhs, tol, maxiter):
    """Preconditioned MINRES (Paige & Saunders 1975) from ``x = 0``, for a symmetric
    operator that may be indefinite (the tangent of softening points) and a symmetric
    positive definite preconditioner.

    Each cycle stops when the residual in the norm of the preconditioner falls below the
    target; the residual is then computed in the Euclidean norm, and another cycle starts
    from it while it is above ``tol`` times the norm of ``rhs``. Returns (x, iterations,
    relative residual), as :func:`conjugate_gradient`.
    """
    x = np.zeros_like(rhs)
    norm_b = norm(rhs)
    if norm_b == 0.0:
        return x, 0, 0.0
    r = rhs.copy()
    res = 1.0
    total = 0
    while total < maxiter:
        dx, its = _minres_cycle(apply, precondition, r, tol / res, maxiter - total)
        total += its
        x += dx
        r = rhs - apply(x)
        res = norm(r) / norm_b
        if res <= tol or its == 0:
            break
    return x, total, res


def _minres_cycle(apply, precondition, rhs, tol, maxiter):
    """One cycle of :func:`minres` from ``x = 0``: (x, iterations)."""
    x = np.zeros_like(rhs)
    r1 = rhs.copy()
    y = precondition(r1)
    beta1 = dot(r1, y)
    if not beta1 > 0.0:
        return x, 0
    beta1 = math.sqrt(beta1)
    oldb, beta, dbar, epsln, phibar, cs, sn = 0.0, beta1, 0.0, 0.0, beta1, -1.0, 0.0
    w = np.zeros_like(rhs)
    w2 = np.zeros_like(rhs)
    r2 = r1
    it = 0
    for it in range(1, maxiter + 1):
        checkpoint()
        v = y / beta
        y = apply(v)
        if it >= 2:
            y -= (beta / oldb) * r1
        alfa = dot(v, y)
        y -= (alfa / beta) * r2
        r1 = r2
        r2 = y
        y = precondition(r2)
        oldb = beta
        beta = dot(r2, y)
        if beta < 0.0:  # a preconditioner that is not positive definite
            break
        beta = math.sqrt(beta)
        # the previous rotation, then the next one
        oldeps = epsln
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        epsln = sn * beta
        dbar = -cs * beta
        gamma = max(math.hypot(gbar, beta), 1e-300)
        cs = gbar / gamma
        sn = beta / gamma
        phi = cs * phibar
        phibar = sn * phibar
        w1 = w2
        w2 = w
        w = (v - oldeps * w1 - delta * w2) / gamma
        x += phi * w
        if abs(phibar) <= tol * beta1 or beta == 0.0:
            break
    return x, it


#: The sums of the Gram-Schmidt passes of :func:`gmres` run by blocks of at least _BLOCK
#: values, at most _MAX_BLOCKS of them: their number is set by the length of the vectors,
#: so that the sums are the same for any number of threads, and whether one thread or
#: several sweep them. Vectors of a single block (shorter than _PARALLEL) are swept by one
#: thread (waking the threads would take longer; from two blocks on, the threads take a
#: GMRES iteration on 24^3 voxels from 15.7 to 9.3 ms, 6.5 ms for the conjugate gradients)
_BLOCK = 4096
_MAX_BLOCKS = 256
_PARALLEL = 2 * _BLOCK
#: A Gram-Schmidt pass of :func:`gmres` that leaves less than this fraction of the norm of
#: the vector is done again (Daniel, Gragg, Kaufman & Stewart 1976: twice is enough)
_REORTHOGONALIZE = 1.0 / math.sqrt(2.0)


@njit(cache=True)
def _block_dots(basis, k, w, lo, hi, out):
    """``out[:k] = basis[:k, lo:hi] w[lo:hi]``, by four interleaved partial sums."""
    for i in range(k):
        a0 = a1 = a2 = a3 = 0.0
        t = lo
        while t + 4 <= hi:
            a0 += basis[i, t] * w[t]
            a1 += basis[i, t + 1] * w[t + 1]
            a2 += basis[i, t + 2] * w[t + 2]
            a3 += basis[i, t + 3] * w[t + 3]
            t += 4
        while t < hi:
            a0 += basis[i, t] * w[t]
            t += 1
        out[i] = (a0 + a1) + (a2 + a3)


@njit(parallel=True, cache=True)
def _parallel_dots(basis, k, w, partial):
    n, n_blocks = w.size, partial.shape[0]
    for b in prange(n_blocks):
        _block_dots(basis, k, w, b * n // n_blocks, (b + 1) * n // n_blocks, partial[b])


@njit(cache=True)
def _serial_dots(basis, k, w, partial):
    n, n_blocks = w.size, partial.shape[0]
    for b in range(n_blocks):
        _block_dots(basis, k, w, b * n // n_blocks, (b + 1) * n // n_blocks, partial[b])


def _dots(basis, k: int, w, out) -> None:
    """``out[:k] = basis[:k] w``: the dot products of ``w`` with the first ``k`` vectors of
    ``basis`` (its rows), in one sweep of them."""
    n = w.size
    partial = np.empty((min(max(n // _BLOCK, 1), _MAX_BLOCKS), k))
    (_parallel_dots if n >= _PARALLEL else _serial_dots)(basis, k, w, partial)
    out[:k] = partial.sum(axis=0)


@njit(cache=True)
def _block_subtract(basis, k, coef, w, lo, hi):
    for i in range(k):
        c = coef[i]
        for t in range(lo, hi):
            w[t] -= c * basis[i, t]


@njit(parallel=True, cache=True)
def _parallel_subtract(basis, k, coef, w, n_blocks):
    n = w.size
    for b in prange(n_blocks):
        _block_subtract(basis, k, coef, w, b * n // n_blocks, (b + 1) * n // n_blocks)


def _subtract(basis, k: int, coef, w) -> None:
    """``w -= basis[:k]^T coef[:k]``, in place (the vectors in the order of the basis)."""
    n = w.size
    if n >= _PARALLEL:
        _parallel_subtract(basis, k, coef, w, min(max(n // _BLOCK, 1), _MAX_BLOCKS))
    else:
        _block_subtract(basis, k, coef, w, 0, n)


#: The columns of the basis that :func:`_combine` turns at once
_CHUNK = 256


@njit(cache=True)
def _block_combine(basis, rows, p, lo, hi):
    """``basis[i, lo:hi] = sum_j p[j, i] basis[j, lo:hi]`` for the columns ``i`` of ``p`` and
    ``j < rows``, in place, by chunks of the columns of the basis."""
    cols = p.shape[1]
    acc = np.empty((cols, _CHUNK))
    for start in range(lo, hi, _CHUNK):
        width = min(_CHUNK, hi - start)
        acc[:, :width] = 0.0
        for j in range(rows):
            for i in range(cols):
                c = p[j, i]
                for t in range(width):
                    acc[i, t] += c * basis[j, start + t]
        for i in range(cols):
            for t in range(width):
                basis[i, start + t] = acc[i, t]


@njit(parallel=True, cache=True)
def _parallel_combine(basis, rows, p, n_blocks):
    n = basis.shape[1]
    for b in prange(n_blocks):
        _block_combine(basis, rows, p, b * n // n_blocks, (b + 1) * n // n_blocks)


def _combine(basis, rows: int, p) -> None:
    """``basis[:c] = p^T basis[:rows]`` in place (``p``: rows x c): the vectors that a
    deflated restart of :func:`gmres` keeps, each sum in the order of the basis whatever
    the threads."""
    n = basis.shape[1]
    p = np.ascontiguousarray(p, dtype=np.float64)
    if n >= _PARALLEL:
        _parallel_combine(basis, rows, p, min(max(n // _BLOCK, 1), _MAX_BLOCKS))
    else:
        _block_combine(basis, rows, p, 0, n)


def _deflation(h, c, y, m: int, k: int):
    """The start of the cycle that follows a full cycle of :func:`gmres` (``m`` iterations)
    in deflated restarting (Morgan 2002): ``h`` is the Arnoldi matrix of the cycle ((m + 1)
    x m, in its basis), ``c`` the coordinates of its first residual, ``y`` the solution of
    its least squares problem. The harmonic Ritz vectors of the ``k`` harmonic Ritz values
    of least modulus (the eigenvectors of ``H + h_{m+1,m}^2 H^-T e_m e_m^T``, ``H`` the square
    part of ``h``; a complex pair kept whole by the real and imaginary parts of its vectors)
    and the residual of the least squares problem, ``z = c - h y``, span the next start:
    their harmonic residuals are all parallel to ``z``, so that the Arnoldi relation holds
    on them. Returns their orthonormal basis ``P`` ((m + 1) x (k' + 1): the harmonic Ritz
    vectors, then ``z``), the Arnoldi matrix on it, ``P^T h P[:m, :k']``, a full block, and
    the coordinates ``P^T z`` of the residual; None when ``H`` is singular or the vectors
    are not finite or not independent of ``z`` (the cycle then restarts from the residual
    alone)."""
    z = c - h @ y[:m]
    hm = h[:m, :m]
    em = np.zeros(m)
    em[-1] = 1.0
    try:
        a = hm.copy()
        a[:, -1] += h[m, m - 1] ** 2 * np.linalg.solve(hm.T, em)
        theta, g = np.linalg.eig(a)
    except np.linalg.LinAlgError:
        return None
    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(g))):
        return None
    vectors = []
    for i in np.argsort(np.abs(theta), kind="stable"):
        if len(vectors) >= k:
            break
        if theta[i].imag > 0.0:  # a pair: its conjugate gives the same two real vectors
            vectors += [g[:, i].real, g[:, i].imag]
        elif theta[i].imag == 0.0:
            vectors.append(g[:, i].real)
    q, tri = np.linalg.qr(np.column_stack(vectors))
    diag = np.abs(np.diag(tri))
    q = q[:, diag > 1e-12 * diag.max()]
    kept = q.shape[1]
    p = np.zeros((m + 1, kept + 1))
    p[:m, :kept] = q
    perp = z.copy()
    for _ in range(2):
        perp -= p[:, :kept] @ (p[:, :kept].T @ perp)
    size = float(np.linalg.norm(perp))
    if not size > 1e-12 * float(np.linalg.norm(z)):
        return None
    p[:, kept] = perp / size
    return p, p.T @ h @ p[:m, :kept], p.T @ z


@without_blas_threads
def gmres(apply, precondition, rhs, tol, maxiter, restart: int = GMRES_RESTART,
          deflate: int = GMRES_DEFLATE, recycle: Recycling | None = None,
          stagnation: int = 0):  # fmt: skip
    """GMRES (Saad & Schultz 1986) from ``x = 0``, preconditioned on the right, with
    deflated restarts (Morgan 2002): the solver of the tangent operators that are not
    symmetric (the full consistent tangents of the points), with the preconditioner of
    :func:`conjugate_gradient` (``apply(v)`` and ``precondition(r)`` return new arrays).

    Each cycle builds an orthonormal basis of the Krylov space of ``apply(precondition(.))``
    by classical Gram-Schmidt (each pass sweeps the basis once for all its dot products,
    summed by blocks that do not depend on the threads), passed again where it cancels
    most of the vector (twice is enough: orthogonal to round-off), and minimizes the
    residual over it by the Givens rotations of its Hessenberg matrix. Preconditioned on
    the right, the residual minimized is that of the system, ``rhs - apply(x)``: the cycle
    stops when it falls below ``tol`` times the norm of ``rhs``, the test of the conjugate
    gradients. After ``restart`` iterations (the vectors kept) ``x`` is updated, and the
    next cycle starts from the harmonic Ritz vectors of the ``deflate`` harmonic Ritz values
    of least modulus and from the residual (:func:`_deflation`; their combinations replace
    the first vectors of the basis): the approximate eigenvectors of the eigenvalues
    nearest zero, which a plain restart loses and the next cycle must find again (restarted
    GMRES stalls on them: the indefinite tangents of softening points), stay in the space,
    and the cycle minimizes the residual over them and the Krylov vectors of the residual.
    The Hessenberg matrix of such a cycle has a full first block, factorized by Householder
    reflections, its next columns by rotations. A cycle that does not decrease the residual
    ends the solve; after deflated restarts the residual that reaches the tolerance is
    computed again (a product) and, if it is above it, the next cycle starts from it alone,
    as with ``deflate=0`` (the plain restarted GMRES: each cycle from the residual computed
    again) or a singular Hessenberg matrix. A solve that does not restart is that of the
    plain GMRES. Returns (x, iterations, relative residual); each iteration takes a product
    by the operator and one by the preconditioner, each cycle one more by the
    preconditioner, each residual computed again one more by the operator. Each iteration
    is a checkpoint of :mod:`morphogen.stopping`.

    ``recycle`` (a :class:`Recycling` shared by a sequence of solves): a solve that
    restarts leaves it the vectors of its last deflated restart, and the next solves of the
    sequence start from them by GCRO-DR (:func:`_gcro_dr`); the solves before the first
    restart are those without it, to the bit. ``stagnation``: a solve whose residual has
    not been halved over its last ``stagnation`` cycles stops there (0: never), an inexact
    correction of Newton's method that more cycles would improve only slowly.
    """
    x = np.zeros_like(rhs)
    norm_b = norm(rhs)
    if norm_b == 0.0:
        return x, 0, 0.0
    m = max(1, min(int(restart), int(maxiter)))
    k_max = min(int(deflate), m - 2)
    if recycle is not None and recycle.u is not None and k_max > 0:
        out = _gcro_dr(_HOST, apply, precondition, rhs, tol, maxiter, m, k_max, recycle,
                       stagnation)  # fmt: skip
        if out is not None:
            return out
    ends = [1.0]  # the relative residuals at the ends of the cycles
    basis = np.empty((m + 1, rhs.size))
    h = np.zeros((m + 1, m))  # the Arnoldi matrix of the cycle
    tri = np.zeros((m + 1, m))  # its triangular factor
    cs, sn, g, coef = np.zeros(m), np.zeros(m), np.zeros(m + 1), np.zeros(m + 1)
    c = np.zeros(m + 1)  # the coordinates of the residual at the start of the cycle
    q0 = np.ones((1, 1))  # the orthogonal factor of the block of the kept vectors
    beta, total, res = norm_b, 0, 1.0
    s, deflated = 0, False  # the kept vectors (and the residual after them)
    np.divide(rhs, beta, out=basis[0])
    c[0] = beta
    while total < maxiter:
        g[:] = 0.0
        g[: s + 1] = q0.T @ c[: s + 1]
        k = s
        for j in range(s, min(m, s + maxiter - total)):
            checkpoint()
            w = apply(precondition(basis[j]))
            h[:, j] = 0.0
            before = norm(w)
            for _ in range(2):
                _dots(basis, j + 1, w, coef)
                _subtract(basis, j + 1, coef, w)
                h[: j + 1, j] += coef[: j + 1]
                after = norm(w)
                if after > _REORTHOGONALIZE * before:  # orthogonal enough
                    break
                before = after
            h[j + 1, j] = after
            col = tri[:, j]
            col[:] = h[:, j]
            if s:
                col[: s + 1] = q0.T @ col[: s + 1]
            for i in range(s, j):  # the rotations of the previous columns
                a, b = col[i], col[i + 1]
                col[i] = cs[i] * a + sn[i] * b
                col[i + 1] = cs[i] * b - sn[i] * a
            gamma = math.hypot(col[j], after)
            if not gamma > 0.0:  # the operator vanishes on the space (or is not finite)
                break
            cs[j], sn[j] = col[j] / gamma, after / gamma
            col[j], col[j + 1] = gamma, 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] *= cs[j]
            k = j + 1
            total += 1
            res = abs(g[k]) / norm_b
            if res <= tol or after == 0.0:
                break
            np.divide(w, after, out=basis[j + 1])
        if k == 0:
            break
        y = np.zeros(m + 1)
        for i in range(k - 1, -1, -1):  # the triangular system of the rotated Hessenberg
            y[i] = (g[i] - tri[i, i + 1 : k] @ y[i + 1 : k]) / tri[i, i]
        combined = np.zeros_like(rhs)
        _subtract(basis, k, -y, combined)
        x += precondition(combined)
        if total >= maxiter or (res <= tol and not deflated):
            break
        previous = beta
        full = res > tol and k == m and h[m, m - 1] > 0.0
        if full and stagnation:
            ends.append(res)
            if _stagnates(ends, stagnation):
                break
        start = _deflation(h, c, y, m, k_max) if full and k_max > 0 else None
        if start is not None:
            p, hk, ck = start
            beta = res * norm_b
            if not beta < previous:
                break
            s = p.shape[1] - 1
            _combine(basis, m + 1, p)
            h[:] = 0.0
            h[: s + 1, :s] = hk
            c[:] = 0.0
            c[: s + 1] = ck
            q0, r0 = np.linalg.qr(hk, mode="complete")
            tri[:] = 0.0
            tri[: s + 1, :s] = r0
            deflated = True
            continue
        r = rhs - apply(x)
        beta = norm(r)
        res = beta / norm_b
        if res <= tol or not beta < previous:
            break
        if deflated and recycle is not None:  # before the plain cycle overwrites them
            _keep(_HOST, recycle, basis, h[: s + 1, :s])
        np.divide(r, beta, out=basis[0])
        c[:] = 0.0
        c[0] = beta
        s, q0, deflated = 0, np.ones((1, 1)), False
    if deflated and recycle is not None:
        _keep(_HOST, recycle, basis, h[: s + 1, :s])
    return x, total, res


def _stagnates(ends, cycles: int) -> bool:
    """Whether the relative residuals ``ends`` at the ends of the cycles of a solve (1 at
    its start) have not been halved over its last ``cycles`` cycles."""
    return len(ends) > cycles and ends[-1] > 0.5 * ends[-1 - cycles]


class Recycling:
    """The space that :func:`gmres` recycles from a solve to the next of a sequence of
    systems whose operators change little, those of the Newton iterations of a loading
    path: GCRO-DR (Parks, de Sturler, Mackey, Johnson & Maiti 2006), the recycled vectors
    approximate eigenvectors of the eigenvalues nearest zero of the preconditioned
    operator, which restarted GMRES must otherwise find again in each solve. Empty until a
    solve of the sequence restarts, so that the solves before are those of GMRES with
    deflated restarts to the bit and cost nothing more: that solve leaves it the harmonic
    Ritz vectors of its last deflated restart, and each later solve minimizes its residual
    over them and the Krylov vectors of the operator projected out of their images, and
    keeps the harmonic Ritz vectors of that space at each of its own restarts. The solves
    of the same operator reuse the images of the recycled vectors; a new operator
    (:meth:`renew`: the tangent of the next Newton iteration) has them computed again by its
    first solve, ``k`` products (:attr:`products`), if the last solve filled a cycle
    (:attr:`needed`), and drops the space otherwise. The vectors live where those of the
    solves do (on the GPU for :func:`morphogen.homogenization.gpu.gmres`)."""

    def __init__(self):
        #: The recycled vectors (rows), in the variable of the preconditioned operator (the
        #: corrections are their images by the preconditioner), or None
        self.u = None
        #: Their images by the preconditioned operator, orthonormal
        self.c = None
        #: Whether :attr:`c` are the images by the operator of the next solve
        self.current = False
        #: Whether the last solve filled a cycle (it restarted, or would have without the
        #: recycled vectors): otherwise a new operator drops them (:meth:`renew`)
        self.needed = False
        #: The products by the operators spent on the images of the recycled vectors, and
        #: the solves started from them
        self.products = 0
        self.solves = 0

    def renew(self) -> None:
        """The next solves are those of another operator: the recycled vectors are kept if
        the last solve filled a cycle (:attr:`needed`), and their images will be computed
        again."""
        if not self.needed:
            self.clear()
        self.current = False

    def clear(self) -> None:
        """Drop the recycled space: the next solves are those of :func:`gmres` without it."""
        self.u = self.c = None
        self.current = False


class _Host:
    """The operations on the vectors of :func:`_gcro_dr` and :func:`_keep` on the host: the
    kernels of :func:`gmres`, whose sums do not depend on the threads."""

    norm = staticmethod(norm)
    zeros_like = staticmethod(np.zeros_like)

    @staticmethod
    def empty(rows: int, n: int) -> np.ndarray:
        return np.empty((rows, n))

    @staticmethod
    def dots(basis, k: int, w) -> np.ndarray:
        out = np.empty(k)
        _dots(basis, k, w, out)
        return out

    @staticmethod
    def subtract(basis, k: int, coef, w) -> None:
        _subtract(basis, k, np.ascontiguousarray(coef, dtype=np.float64), w)

    @staticmethod
    def combine(basis, rows: int, p) -> None:
        _combine(basis, rows, p)


_HOST = _Host()


def _keep(ops, recycle: Recycling, basis, hk) -> None:
    """The space that a solve of :func:`gmres` which restarted leaves to ``recycle``: the
    ``s`` vectors ``W`` of its last deflated start (``basis[:s]``), whose images by the
    preconditioned operator are ``basis[:s + 1] hk`` (the Arnoldi relation of the deflated
    restart): with ``hk = Q R``, the vectors ``W R^-1`` and their images ``basis[:s + 1]
    Q``, orthonormal."""
    s = hk.shape[1]
    if s == 0:
        return
    q, r = np.linalg.qr(hk)
    diag = np.abs(np.diag(r))
    if not diag.min() > 1e-12 * diag.max():
        return
    p = np.zeros((s + 1, 2 * s))
    p[:s, :s] = sla.solve_triangular(r, np.eye(s))
    p[:, s:] = q
    work = ops.empty(2 * s, basis.shape[1])
    work[: s + 1] = basis[: s + 1]
    ops.combine(work, s + 1, p)
    recycle.u, recycle.c, recycle.current, recycle.needed = work[:s], work[s:], True, True


def _orthonormalize(ops, rows, k: int):
    """Gram-Schmidt of the ``k`` vectors ``rows`` in place, each pass done again where it
    cancels most of the vector: their triangular factor (``rows = Q R``), or None when they
    are not independent."""
    r = np.zeros((k, k))
    largest = 0.0
    for i in range(k):
        w = rows[i]
        before = ops.norm(w)
        largest = max(largest, before)
        after = before
        for _ in range(2 if i else 0):
            coef = ops.dots(rows, i, w)
            ops.subtract(rows, i, coef, w)
            r[:i, i] += coef
            after = ops.norm(w)
            if after > _REORTHOGONALIZE * before:
                break
            before = after
        if not after > 1e-10 * largest:
            return None
        r[i, i] = after
        w /= after
    return r


def _recycled_space(h, e, m: int, k: int, keep: int):
    """The space that a full cycle of :func:`_gcro_dr` recycles: the harmonic Ritz vectors
    of its ``keep`` harmonic Ritz values of least modulus with respect to its images (a
    complex pair kept whole), the solutions ``z`` of ``G^T G z = theta G^T W^T V z``
    (Parks et al. 2006), ``G = h`` the matrix of the cycle from its solution space ``V``
    (the ``k`` recycled vectors, scaled to unit norms, and the ``m - k`` Krylov vectors) to
    its image space ``W`` (the images of the recycled vectors and the ``m - k + 1`` Krylov
    vectors, orthonormal), ``W^T V = [[e[:k], 0], [e[k:], I]]`` (``e = W^T U``). Returns
    ``(Z, Q)``: the new vectors ``V Z`` and their images ``W Q``, orthonormal (``G Z = Q``);
    None when the problem is singular or its vectors are not independent."""
    t = np.zeros((m + 1, m))
    t[:, :k] = e
    rows = np.arange(k, m)
    t[rows, rows] = 1.0
    try:
        theta, z = sla.eig(h.T @ h, h.T @ t)
    except (np.linalg.LinAlgError, ValueError):
        return None
    vectors = []
    for i in np.argsort(np.abs(theta), kind="stable"):
        if len(vectors) >= keep:
            break
        if not (np.isfinite(theta[i]) and np.all(np.isfinite(z[:, i]))):
            continue
        if theta[i].imag > 0.0:  # a pair: its conjugate gives the same two real vectors
            vectors += [z[:, i].real, z[:, i].imag]
        elif theta[i].imag == 0.0:
            vectors.append(z[:, i].real)
    if not vectors:
        return None
    p, tri = np.linalg.qr(np.column_stack(vectors))
    diag = np.abs(np.diag(tri))
    p = p[:, diag > 1e-12 * diag.max()]
    q, tri = np.linalg.qr(h @ p)
    diag = np.abs(np.diag(tri))
    if not diag.min() > 1e-12 * diag.max():
        return None
    return p @ sla.solve_triangular(tri, np.eye(tri.shape[0])), q


def _gcro_dr(ops, apply, precondition, rhs, tol, maxiter, m: int, k_max: int,
             recycle: Recycling, stagnation: int):  # fmt: skip
    """GCRO-DR (Parks, de Sturler, Mackey, Johnson & Maiti 2006): :func:`gmres` from the
    space of ``recycle`` (``ops``: the operations on the vectors, :class:`_Host` or those of
    the GPU). With ``U`` its ``k`` vectors and ``C`` their orthonormal images by the
    preconditioned operator ``A M^-1`` (computed again for a new operator by ``k`` products
    and Gram-Schmidt, ``U`` turned with them), each cycle projects its residual out of
    ``C``, builds from it the Krylov vectors ``V`` of ``(I - C C^T) A M^-1`` (``m - k`` of
    them, each orthogonalized against ``C`` and the previous ones by the passes of
    :func:`gmres`, which give the coefficients ``B = C^T A M^-1 V``) and minimizes the
    residual over ``[U, V]``, whose images are ``[C, V] G`` with ``G = [[D, B], [0, H]]``
    (``D`` the inverse norms of the ``U``, ``H`` the Hessenberg matrix): the least squares
    problem of :func:`gmres` with a diagonal first block and the same rotations. A full
    cycle keeps the harmonic Ritz vectors of ``[U, V]`` (:func:`_recycled_space`) as the
    next ``U``, combined in place with their images and the residual that starts the next
    cycle (``m + k + 2`` vectors at most). After more than one cycle the residual that reaches the
    tolerance is computed again (a product), and the cycles go on from it while it is
    above. Returns (x, iterations, relative residual), the products of the images counted
    as iterations; None when the images are not independent (the space is dropped)."""
    norm_b = ops.norm(rhs)
    n = rhs.size
    k = recycle.u.shape[0]
    rows = m + 1 + max(k, k_max + 1)  # U, C and the Krylov vectors (a pair kept whole)
    work = ops.empty(rows, n)
    work[:k] = recycle.u
    current = recycle.current
    if current:
        work[k : 2 * k] = recycle.c
    recycle.u = recycle.c = None  # the space lives in work while the solve runs
    total = 0
    if not current:  # the images by the new operator
        for i in range(k):
            checkpoint()
            work[k + i] = apply(precondition(work[i]))
        total = k
        recycle.products += k
        r = _orthonormalize(ops, work[k : 2 * k], k)
        if r is None:
            recycle.clear()
            return None
        ops.combine(work, k, sla.solve_triangular(r, np.eye(k)))
    recycle.solves += 1
    x = ops.zeros_like(rhs)
    coords = np.zeros(m + 1)  # the residual at the start of the cycle, on [C, V]
    start, ends, res, cycles = rhs, [1.0], 1.0, 0
    while True:
        basis = work[k:]  # C, then the Krylov vectors
        if start is not None:  # the residual projected out of C
            basis[k] = start
            coords[:] = 0.0
            before = ops.norm(basis[k])
            for _ in range(2):
                coef = ops.dots(basis, k, basis[k])
                ops.subtract(basis, k, coef, basis[k])
                coords[:k] += coef
                after = ops.norm(basis[k])
                if after > _REORTHOGONALIZE * before:
                    break
                before = after
            coords[k] = after
            if after > 0.0:
                basis[k] /= after
            start = None
        d = np.array([ops.norm(work[i]) for i in range(k)])
        h = np.zeros((m + 1, m))
        h[np.arange(k), np.arange(k)] = 1.0 / d
        tri = h.copy()
        g = coords.copy()
        cs, sn = np.zeros(m), np.zeros(m)
        kend = k
        res = abs(g[k]) / norm_b
        for j in range(k, min(m, k + maxiter - total) if res > tol else k):
            checkpoint()
            w = apply(precondition(basis[j]))
            before = ops.norm(w)
            for _ in range(2):
                coef = ops.dots(basis, j + 1, w)
                ops.subtract(basis, j + 1, coef, w)
                h[: j + 1, j] += coef
                after = ops.norm(w)
                if after > _REORTHOGONALIZE * before:
                    break
                before = after
            h[j + 1, j] = after
            col = tri[:, j]
            col[:] = h[:, j]
            for i in range(k, j):
                a, b = col[i], col[i + 1]
                col[i] = cs[i] * a + sn[i] * b
                col[i + 1] = cs[i] * b - sn[i] * a
            gamma = math.hypot(col[j], after)
            if not gamma > 0.0:
                break
            cs[j], sn[j] = col[j] / gamma, after / gamma
            col[j], col[j + 1] = gamma, 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] *= cs[j]
            kend = j + 1
            total += 1
            res = abs(g[kend]) / norm_b
            if res <= tol or after == 0.0:
                break
            basis[j + 1] = w / after
        cycles += 1
        y = np.zeros(m)
        for i in range(kend - 1, -1, -1):
            y[i] = (g[i] - tri[i, i + 1 : kend] @ y[i + 1 : kend]) / tri[i, i]
        combined = ops.zeros_like(rhs)
        ops.subtract(work, k, -y[:k] / d, combined)
        if kend > k:
            ops.subtract(basis[k:], kend - k, -y[k:kend], combined)
        x += precondition(combined)
        if res <= tol or total >= maxiter or kend < m:
            if res <= tol and cycles > 1:
                r = rhs - apply(x)
                true = ops.norm(r) / norm_b
                if true > tol and true < ends[-1] and total < maxiter:
                    ends.append(true)
                    start = r
                    continue
                res = true
            break
        ends.append(res)
        if not ends[-1] < ends[-2] or (stagnation and _stagnates(ends, stagnation)):
            break
        e = np.empty((m + 1, k))  # W^T U, scaled as the first columns of h
        for i in range(k):
            e[:, i] = ops.dots(basis, m + 1, work[i]) / d[i]
        z = coords - h @ y  # the residual of the cycle on [C, V]
        new = _recycled_space(h, e, m, k, k_max)
        kk = k if new is None else new[1].shape[1]
        p = np.zeros((k + m + 1, 2 * kk + 1))
        if new is None:  # U and C kept
            p[np.arange(k), np.arange(k)] = 1.0
            q = np.eye(m + 1, k)
        else:
            zs, q = new
            p[:k, :kk] = zs[:k] / d[:, None]
            p[2 * k : k + m, :kk] = zs[k:]
        cq = q.T @ z
        perp = z - q @ cq
        perp -= q @ (q.T @ perp)
        size = float(np.linalg.norm(perp))
        p[k:, kk : 2 * kk] = q
        if size > 0.0:
            p[k:, 2 * kk] = perp / size
        ops.combine(work, k + m + 1, p)
        k = kk
        coords[:] = 0.0
        coords[:k] = cq
        coords[k] = size
    recycle.u, recycle.c, recycle.current = work[:k].copy(), work[k : 2 * k].copy(), True
    recycle.needed = cycles > 1 or kend == m
    return x, total, res


class TwoLevelPreconditioner:
    """Symmetric two-level preconditioner of ``A`` with a coarse space ``P``.

    A forward Gauss-Seidel sweep on ``A``, a correction by one AMG V-cycle on the
    Galerkin matrix ``P^T A P``, then a backward sweep (so the preconditioner is
    symmetric). With ``P`` the interpolation of the linear (4-node) elements into the
    quadratic (10-node) ones, the coarse matrix is the stiffness of the linear
    elements, on which smoothed aggregation is efficient. ``dofs``, ``coarse_dofs``:
    the degree of freedom ``3 node + component`` of each unknown of the matrix and of
    the coarse space, where they are components of the nodes of a mesh (the sweeps then
    work by nodes, see :class:`~morphogen.homogenization.amg.Smoother`); ``blocks``:
    the blocks of the matrix, where they have been made already
    (:class:`~morphogen.homogenization.amg.NodeBlocks`).
    """

    def __init__(self, matrix, prolongation, near_null=None, dofs=None, coarse_dofs=None,
                 blocks=None):  # fmt: skip
        self.matrix = sp.csr_matrix(matrix)
        self.matrix.sort_indices()
        self.p = sp.csr_matrix(prolongation)
        self.pt = self.p.T.tocsr()
        coarse = (self.pt @ self.matrix @ self.p).tocsr()
        self.amg = SmoothedAggregation(coarse, _candidates(near_null), dofs=coarse_dofs)
        self.smoother = Smoother(self.matrix, dofs, blocks)
        if self.smoother.coloured:
            # the cycle in the order of the colours of the smoother: the interpolation
            # with its rows at their places in that order
            rows = self.p.tocoo()
            shape = (self.smoother.size, self.p.shape[1])
            places = (self.smoother.place[rows.row], rows.col)
            self.p = sp.csr_matrix((rows.data, places), shape=shape)
            self.pt = self.p.T.tocsr()

    def refresh(self, blocks: NodeBlocks) -> bool:
        """The sweeps for the values that the blocks of the matrix have now (see
        :meth:`~morphogen.homogenization.amg.Smoother.refresh`); the coarse space and its
        multigrid cycle are kept. The preconditioner stays symmetric positive definite:
        the coarse correction adds a positive term to the symmetric sweeps of the matrix
        as it is."""
        return self.smoother.refresh(blocks)

    def __call__(self, r: np.ndarray) -> np.ndarray:
        smoother = self.smoother
        if smoother.coloured:
            b = smoother.sorted(np.ascontiguousarray(r, dtype=float))
            x = smoother.start(b)
            residual = smoother.residual(x)
            x += product(self.p, self.amg(product(self.pt, residual)))
            smoother.relax(x, b, False)
            return smoother.unsorted(x)
        x = np.zeros_like(r)
        smoother.sweep(x, r, True)
        residual = r - product(self.matrix, x)
        x += product(self.p, self.amg(product(self.pt, residual)))
        smoother.sweep(x, r, False)
        return x


def _candidates(near_null):
    if near_null is None:
        return None
    b = np.asarray(near_null, dtype=float)
    return b[:, np.linalg.norm(b, axis=0) > 0]


class LinearSolver:
    """Solver of A x = b for a fixed matrix A and several right-hand sides.

    ``near_null``: low-energy modes (columns) of A, used by algebraic multigrid.
    ``coarse``: ``(P, near_null_coarse)``, a coarse space for the two-level ``amg``
    preconditioner (see :class:`TwoLevelPreconditioner`). ``symmetric``: false for a
    matrix that is not symmetric (the tangent of nonsymmetric consistent tangents), which
    the iterative methods solve by :func:`gmres` with the preconditioner of its symmetric
    part (LU factorizes it as it is). ``dofs``: the degree of freedom ``3 node +
    component`` of each unknown, where the unknowns are components of the displacements
    of the nodes of a mesh (the sweeps of the multigrid preconditioners then work by
    nodes, see :class:`~morphogen.homogenization.amg.Smoother`; ``coarse`` may carry
    those of its coarse space as a third item). The products of the iterative methods
    then read the matrix by the 3 x 3 blocks of its nodes (:attr:`blocks`:
    :class:`~morphogen.homogenization.amg.NodeBlocks`, the same products to the bit).
    """

    def __init__(
        self,
        matrix,
        *,
        method="auto",
        tol=1e-8,
        maxiter=None,
        near_null=None,
        coarse=None,
        symmetric: bool = True,
        dofs=None,
    ):
        self.matrix = sp.csr_matrix(matrix)
        n = self.matrix.shape[0]
        method = choose_method(method, n)
        self.method = method
        self.tol = float(tol)
        self.maxiter = maxiter
        self.symmetric = bool(symmetric)
        t0 = time.perf_counter()
        self._precondition = None
        self._dofs = dofs
        #: The matrix by blocks of nodes, for the products of the iterative methods
        self.blocks = None
        #: The iterations per decade of the residual of the solves since the last step
        #: of a path, and those of the step for which the preconditioner was built
        self._wear = {"rates": [], "reference": None}
        if n and method != "direct":
            # (rows in the order of their columns: the order in which the blocks sum them)
            self.matrix.sort_indices()
            self.blocks = NodeBlocks.of(self.matrix, dofs)
        # the preconditioners are those of the symmetric part (which has its own blocks)
        part = self.matrix if self.symmetric else (0.5 * (self.matrix + self.matrix.T)).tocsr()
        shared = self.blocks if self.symmetric else None
        if n == 0:
            pass
        elif method == "direct":
            self._lu = spla.splu(
                self.matrix.tocsc(), permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True}
            )
        elif method == "amg":
            if coarse is not None:
                p, coarse_null = coarse[:2]
                coarse_dofs = coarse[2] if len(coarse) > 2 else None
                self._precondition = TwoLevelPreconditioner(part, p, coarse_null, dofs,
                                                            coarse_dofs, shared)  # fmt: skip
                self.method = "amg-2level"
            else:
                self._precondition = SmoothedAggregation(part, _candidates(near_null), dofs=dofs,
                                                         blocks=shared)  # fmt: skip
        else:
            d = self.matrix.diagonal()
            inv = 1.0 / np.where(d > 0, d, 1.0)
            self._precondition = lambda r: inv * r
        self.setup_s = time.perf_counter() - t0

    def updated(self, matrix, symmetric: bool | None = None) -> LinearSolver:
        """Solver of another matrix of the same size (a new tangent operator): direct
        methods factorize it, iterative ones keep this preconditioner (``symmetric``: whether
        the matrix is, by default as this one). This solver is not to be used afterwards:
        the blocks of its matrix take the values of the new one, and with them the sweeps
        of a multigrid preconditioner that work by nodes (for a symmetric matrix of the
        same pattern) — the interpolations and the coarse levels are kept, and the
        preconditioner stays symmetric positive definite."""
        if self.method == "direct" or self._precondition is None:
            return LinearSolver(matrix, method="direct", tol=self.tol, maxiter=self.maxiter)
        out = object.__new__(LinearSolver)
        out.__dict__.update(self.__dict__)
        out.matrix = sp.csr_matrix(matrix)
        out.setup_s = 0.0
        if symmetric is not None:
            out.symmetric = bool(symmetric)
        if self.blocks is not None:
            if not self.blocks.refresh(out.matrix):  # (other stored values: other blocks)
                out.blocks = NodeBlocks.of(out.matrix, self._dofs)
            elif out.symmetric and hasattr(self._precondition, "refresh"):
                self._precondition.refresh(self.blocks)
        return out

    def worn(self) -> bool:
        """Whether the preconditioner is to be built again, asked at the start of a step
        of a loading path, before the solver is :meth:`updated`: true for a factorization
        or a diagonal (built with the solver); for a multigrid preconditioner, when the
        solves of the last step took :data:`WORN` times the iterations per decade of
        their residual that those of the step for which it was built took (the tangent
        has moved away from the one whose coarse levels it holds)."""
        if not isinstance(self._precondition, TwoLevelPreconditioner | SmoothedAggregation):
            return True
        wear = self._wear
        rates, wear["rates"] = wear["rates"], []
        if not rates:
            return False
        last = sum(rates) / len(rates)
        if wear["reference"] is None:
            wear["reference"] = last
            return False
        return last > WORN * wear["reference"]

    def _product(self, x: np.ndarray) -> np.ndarray:
        """``A x``: by the blocks of the nodes where the matrix has them."""
        if self.blocks is not None and x.ndim == 1:
            return self.blocks.product(x)
        return product(self.matrix, x)

    def solve(
        self, rhs: np.ndarray, x0: np.ndarray | None = None, tol: float | None = None
    ) -> tuple[np.ndarray, SolveInfo]:
        """Solution of ``A x = rhs``; iterative methods start from ``x0`` if given and
        stop at the relative residual ``tol`` (default: that of the solver)."""
        n = self.matrix.shape[0]
        b = np.asarray(rhs, dtype=float)
        t0 = time.perf_counter()
        norm_b = norm(b.ravel())
        if n == 0 or norm_b == 0.0:
            return np.zeros(n), SolveInfo(self.method, n, 0, 0.0, self.setup_s, 0.0)
        tol = self.tol if tol is None else float(tol)
        iterations = 0
        if self.method == "direct":
            x = self._lu.solve(b)
        else:
            default = 50_000 if self.method == "cg" else 2_000
            start = None if x0 is None or not np.any(x0) else np.asarray(x0, dtype=float)
            r = b if start is None else b - self._product(start)
            norm_r = norm(r)
            if norm_r <= tol * norm_b:
                x = np.zeros(n) if start is None else start.copy()
            else:
                krylov = conjugate_gradient if self.symmetric else gmres
                x, iterations, _ = krylov(
                    self._product,
                    self._precondition,
                    r,
                    tol * norm_b / norm_r,
                    self.maxiter or default,
                )
                if start is not None:
                    x += start
        residual = norm((b - self._product(x)).ravel()) / norm_b
        if iterations and start is None:  # (the wear of the preconditioner: see worn)
            decades = max(-math.log10(max(residual, 1e-300)), 0.5)
            self._wear["rates"].append(iterations / decades)
        if self.method != "direct" and residual > 10 * tol:
            log.warning(
                "%s solver stopped at relative residual %.2e (tolerance %.1e)",
                self.method,
                residual,
                tol,
            )
        info = SolveInfo(
            self.method, n, iterations, residual, self.setup_s, time.perf_counter() - t0
        )
        return x, info
