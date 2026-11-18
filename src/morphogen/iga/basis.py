"""B-spline basis functions: values and derivatives, knot vectors and the operators of
their refinement.

A knot vector ``U = (u_0 <= ... <= u_m)`` and a degree ``p`` define ``n = m - p``
functions ``N_(i,p)`` by the recurrence of Cox and de Boor::

    N_(i,0)(u) = 1 on [u_i, u_(i+1)), 0 elsewhere,
    N_(i,p)(u) = (u - u_i) / (u_(i+p) - u_i) N_(i,p-1)(u)
               + (u_(i+p+1) - u) / (u_(i+p+1) - u_(i+1)) N_(i+1,p-1)(u),

piecewise polynomials of degree ``p``, non-negative, summing to one, each supported by
``p + 1`` spans, and ``p - k`` times differentiable at a knot repeated ``k`` times (Piegl &
Tiller 1997, ch. 2). The knot vectors here are *open*: their first and last knots are
repeated ``p + 1`` times, so that the functions interpolate at the two ends. In a span
``[u_s, u_(s+1))`` only the functions ``s - p ... s`` are non-zero: they are computed
there together (:func:`basis`), with their derivatives by::

    N'_(i,p) = p (N_(i,p-1) / (u_(i+p) - u_i) - N_(i+1,p-1) / (u_(i+p+1) - u_(i+1))).

Refinement. A space of splines contains that of a coarser knot vector (knots inserted:
*h*-refinement) or of a lower degree with the same continuity (the multiplicity of every
knot raised with the degree: *p*-refinement; both: *k*-refinement). A function keeps its
shape and gets new coefficients, linear in the old ones: :func:`embedding` gives the
matrix ``T`` of that map (``c_new = T c_old``), found by matching the two expansions at the
Gauss points of the new elements — exact since the spaces are nested, whatever the
continuity. Rational curves and surfaces (NURBS) refine through their homogeneous control
points.
"""

from __future__ import annotations

import numpy as np

#: Entries of a refinement operator below this are round-off (its rows sum to one)
_ROUNDOFF = 1e-13


def open_knots(n: int, degree: int, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    """The open uniform knot vector of ``n`` functions of degree ``degree`` on ``[lower,
    upper]``: ``n - degree`` elements of equal length."""
    n, degree = int(n), int(degree)
    if degree < 1 or n < degree + 1:
        raise ValueError("open_knots: a degree of 1 or more and at least degree + 1 functions")
    inner = np.linspace(lower, upper, n - degree + 1)[1:-1]
    return np.concatenate([np.full(degree + 1, lower), inner, np.full(degree + 1, upper)])


def check_knots(knots, degree: int) -> np.ndarray:
    """The knot vector as an array, checked: non-decreasing, open, no knot repeated more
    than ``degree`` times inside (continuous functions)."""
    knots = np.asarray(knots, dtype=float)
    degree = int(degree)
    if degree < 1:
        raise ValueError("degree: 1 or more")
    if knots.ndim != 1 or len(knots) < 2 * (degree + 1):
        raise ValueError(f"knots: at least {2 * (degree + 1)} knots for degree {degree}")
    if np.any(np.diff(knots) < 0):
        raise ValueError("knots: a non-decreasing sequence")
    if knots[degree] != knots[0] or knots[-degree - 1] != knots[-1] or knots[0] == knots[-1]:
        raise ValueError(f"knots: open (the end knots repeated {degree + 1} times)")
    inner, counts = np.unique(knots[degree + 1 : -degree - 1], return_counts=True)
    if inner.size and (counts.max() > degree or inner[0] == knots[0] or inner[-1] == knots[-1]):
        raise ValueError(f"knots: inner knots repeated at most {degree} times")
    return knots


def n_functions(knots, degree: int) -> int:
    """The number of functions of a knot vector."""
    return len(knots) - int(degree) - 1


def breakpoints(knots) -> np.ndarray:
    """The distinct knots: the ends of the elements."""
    return np.unique(np.asarray(knots, dtype=float))


def greville(knots, degree: int) -> np.ndarray:
    """The Greville abscissae ``(u_(i+1) + ... + u_(i+p)) / p`` of the functions: where
    each one is largest, and the parameters of the control points of the identity."""
    knots = np.asarray(knots, dtype=float)
    p = int(degree)
    n = n_functions(knots, p)
    return np.array([knots[i + 1 : i + p + 1].mean() for i in range(n)])


def spans(knots, degree: int, u) -> np.ndarray:
    """The span ``s`` of each parameter (``u_s <= u < u_(s+1)``; the last span for the
    last knot). Parameters outside the knots are an error."""
    knots = np.asarray(knots, dtype=float)
    u = np.atleast_1d(np.asarray(u, dtype=float))
    tolerance = 1e-12 * (knots[-1] - knots[0])
    if u.size and (u.min() < knots[0] - tolerance or u.max() > knots[-1] + tolerance):
        raise ValueError(f"parameters outside the knots [{knots[0]:g}, {knots[-1]:g}]")
    last = n_functions(knots, degree) - 1
    return np.clip(np.searchsorted(knots, u, side="right") - 1, int(degree), last)


def _values(knots: np.ndarray, degree: int, u: np.ndarray, span: np.ndarray) -> np.ndarray:
    """The ``degree + 1`` functions of that degree that are non-zero in the span of each
    parameter (points x degree + 1): the recurrence of Cox and de Boor, by columns."""
    m = len(u)
    values = np.zeros((m, degree + 1))
    values[:, 0] = 1.0
    left = np.empty((m, degree + 1))
    right = np.empty((m, degree + 1))
    for j in range(1, degree + 1):
        left[:, j] = u - knots[span + 1 - j]
        right[:, j] = knots[span + j] - u
        saved = np.zeros(m)
        for r in range(j):
            ratio = values[:, r] / (right[:, r + 1] + left[:, j - r])
            values[:, r] = saved + right[:, r + 1] * ratio
            saved = left[:, j - r] * ratio
        values[:, j] = saved
    return values


def _derived(knots, degree: int, u, span, order: int) -> np.ndarray:
    """The derivatives of order ``order`` of the non-zero functions of ``degree`` (points
    x degree + 1), from those of order ``order - 1`` of the degree below."""
    if order == 0:
        return _values(knots, degree, u, span)
    out = np.zeros((len(u), degree + 1))
    if order > degree:
        return out
    lower = _derived(knots, degree - 1, u, span, order - 1)  # functions span - degree + 1 ...
    for a in range(degree + 1):
        i = span - degree + a
        if a >= 1:
            out[:, a] += degree * lower[:, a - 1] / (knots[i + degree] - knots[i])
        if a <= degree - 1:
            out[:, a] -= degree * lower[:, a] / (knots[i + degree + 1] - knots[i + 1])
    return out


def basis(knots, degree: int, u, derivatives: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The spans of the parameters ``u`` and the functions that are non-zero there with
    their derivatives: ``values[k, j, a]`` is the derivative of order ``k`` of the function
    ``span[j] - degree + a`` at ``u[j]`` (``k = 0 ... derivatives``, ``a = 0 ... degree``)."""
    knots = np.asarray(knots, dtype=float)
    u = np.atleast_1d(np.asarray(u, dtype=float))
    degree = int(degree)
    span = spans(knots, degree, u)
    u = np.clip(u, knots[0], knots[-1])
    values = np.stack([_derived(knots, degree, u, span, k) for k in range(int(derivatives) + 1)])
    return span, values


def collocation(knots, degree: int, u, derivative: int = 0) -> np.ndarray:
    """The matrix of the functions (or of their derivatives of an order) at parameters:
    points x functions."""
    span, values = basis(knots, degree, u, derivative)
    out = np.zeros((len(span), n_functions(knots, degree)))
    rows = np.arange(len(span))
    for a in range(int(degree) + 1):
        out[rows, span - int(degree) + a] = values[derivative, :, a]
    return out


def gauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The Gauss-Legendre rule of ``n`` points on ``[0, 1]``: points and weights."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * (x + 1.0), 0.5 * w


def refined_knots(knots, degree: int, subdivisions: int = 1, elevate: int = 0) -> np.ndarray:
    """The knot vector of a refinement: every element cut into ``subdivisions`` equal
    parts (new knots, once each), and the degree raised by ``elevate`` with the
    continuity of the knots kept (their multiplicities raised as much)."""
    knots = check_knots(knots, degree)
    subdivisions, elevate = int(subdivisions), int(elevate)
    if subdivisions < 1 or elevate < 0:
        raise ValueError("refinement: subdivisions >= 1 and elevate >= 0")
    points, counts = np.unique(knots, return_counts=True)
    out = []
    for k, (point, count) in enumerate(zip(points, counts, strict=True)):
        out.extend([point] * (count + elevate))
        if k + 1 < len(points) and subdivisions > 1:
            out.extend(np.linspace(point, points[k + 1], subdivisions + 1)[1:-1])
    return np.array(out)


def embedding(knots, degree: int, new_knots, new_degree: int) -> np.ndarray:
    """The matrix ``T`` (new functions x old functions) of the coefficients of the old
    functions in a refined space that contains them: ``sum_i c_i N_i = sum_j (T c)_j
    N'_j``. Found by least squares at the Gauss points of the new elements, where the two
    expansions agree exactly; an error if the new space does not contain the old one."""
    knots = check_knots(knots, degree)
    new_knots = check_knots(new_knots, new_degree)
    ends = breakpoints(new_knots)
    points, _ = gauss(int(new_degree) + 1)
    u = (ends[:-1, None] + np.diff(ends)[:, None] * points[None, :]).ravel()
    new = collocation(new_knots, new_degree, u)
    old = collocation(knots, degree, u)
    matrix, *_ = np.linalg.lstsq(new, old, rcond=None)
    if np.abs(new @ matrix - old).max() > 1e-9:
        raise ValueError("embedding: the new space does not contain the old one")
    matrix[np.abs(matrix) < _ROUNDOFF] = 0.0
    return matrix


__all__ = [
    "basis",
    "breakpoints",
    "check_knots",
    "collocation",
    "embedding",
    "gauss",
    "greville",
    "n_functions",
    "open_knots",
    "refined_knots",
    "spans",
]
