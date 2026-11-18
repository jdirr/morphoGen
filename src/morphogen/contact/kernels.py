"""Kernels of the contact of surfaces (Numba): the functions of the faces, the point of a
face nearest a point, the forces of a slave point on a master face or on a rigid
obstacle, with Coulomb's friction, and their tangents.

A *slave point* is an integration point of a face of the slave surface, of weight ``w``
(its share of the reference area of the face). Its state (:data:`STATE` values) is kept
from the end of a step to the next: the multiplier of the augmented Lagrangian, the point
of the master surface that faced it (a face and its coordinates; for an obstacle, its
position relative to the obstacle), the tangential traction, the energy that friction
dissipated per unit area, its gap and its pressure.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

#: The parameters of a contact: the penalty of the normal gap, the stiffness of the
#: stick, the coefficient of friction, the deepest penetration looked for, the gap below
#: which two surfaces touch, the size of the faces, and the distance within which a face
#: of a surface that lies behind a point of it, at rest, is across a wall from the point;
#: how far beyond the rim of the master a point still faces it (:data:`RIM`), and whether
#: the weight of a point is the share of its face that the other surface sees
#: (:func:`_share`: the points at the nodes)
P_STIFF, P_STICK, P_MU, P_DEPTH, P_TOUCH, P_SIZE, P_BEHIND, P_RIM, P_SHARE = range(9)
#: How far beyond the rim of the master surface (in the coordinates of its faces) a slave
#: point still faces it, on the face continued: nothing for the points inside the faces
#: of the slave; a twentieth of a face for those at its nodes, which may lie on the rim
#: of the master (two bodies of the same section) and would otherwise face it or not
#: with the rounding of each iteration
RIM, RIM_NODES = 1e-12, 0.05
#: How far outside a master face (in its coordinates) a slave point stays on it
SLACK = 0.05
#: Two faces face each other when the cosine of the angle of their outward normals is
#: below this (more than 104 degrees apart): faces at right angles along an edge of a body,
#: a little distorted, do not
FACING = -0.25
#: The same for a surface against itself: the walls of a notch that closes touch at its
#: bottom while their normals are still far from opposite (the two walls of the corner of
#: a hexagonal cell: 60 degrees apart at rest), and what tells them from the two sides of
#: an edge of the body is where they lay at rest (:func:`_across`)
FACING_OWN = 0.6
#: The state of a slave point (the last: the share of its weight that presses)
S_LAMBDA, S_FACE, S_XI, S_ETA, S_TX, S_TY, S_TZ, S_WORK, S_GAP, S_PRESSURE, S_SHARE = range(11)
STATE = 11
#: The kinds of rigid obstacles
PLANE, SPHERE, CYLINDER, CONE = 0, 1, 2, 3

#: The points of the faces of three nodes (exact to degree 2) and of six (to degree 4):
#: ξ, η and weights that add up to 1
POINTS3 = np.array([[1 / 6, 1 / 6, 1 / 3], [2 / 3, 1 / 6, 1 / 3], [1 / 6, 2 / 3, 1 / 3]])
#: The points at the nodes of the faces, for surfaces whose corners and edges touch
#: first: the vertices of a face of three nodes (exact to degree 1); the vertices, the
#: middles of the edges and the centroid of a face of six (exact to degree 3)
NODES3 = np.array([[0.0, 0.0, 1 / 3], [1.0, 0.0, 1 / 3], [0.0, 1.0, 1 / 3]])
NODES6 = np.array(
    [
        [0.0, 0.0, 1 / 20],
        [1.0, 0.0, 1 / 20],
        [0.0, 1.0, 1 / 20],
        [0.5, 0.0, 2 / 15],
        [0.5, 0.5, 2 / 15],
        [0.0, 0.5, 2 / 15],
        [1 / 3, 1 / 3, 9 / 20],
    ]
)
#: The rules of the contact points of a surface
RULES = ("gauss", "nodes")
_GA, _GB = 0.445948490915965, 0.091576213509771
POINTS6 = np.array(
    [
        [_GA, _GA, 0.223381589678011],
        [1.0 - 2.0 * _GA, _GA, 0.223381589678011],
        [_GA, 1.0 - 2.0 * _GA, 0.223381589678011],
        [_GB, _GB, 0.109951743655322],
        [1.0 - 2.0 * _GB, _GB, 0.109951743655322],
        [_GB, 1.0 - 2.0 * _GB, 0.109951743655322],
    ]
)


@njit(cache=True)
def shapes(n, xi, eta, shape, dxi, deta):
    """The functions of the nodes of a face of ``n`` nodes (3, or 6: the corners, then the
    middles of the edges 0-1, 1-2 and 2-0) at a point, and their derivatives."""
    zeta = 1.0 - xi - eta
    if n == 3:
        shape[0], shape[1], shape[2] = zeta, xi, eta
        dxi[0], dxi[1], dxi[2] = -1.0, 1.0, 0.0
        deta[0], deta[1], deta[2] = -1.0, 0.0, 1.0
        return
    shape[0] = zeta * (2.0 * zeta - 1.0)
    shape[1] = xi * (2.0 * xi - 1.0)
    shape[2] = eta * (2.0 * eta - 1.0)
    shape[3] = 4.0 * zeta * xi
    shape[4] = 4.0 * xi * eta
    shape[5] = 4.0 * eta * zeta
    dxi[0] = 1.0 - 4.0 * zeta
    dxi[1] = 4.0 * xi - 1.0
    dxi[2] = 0.0
    dxi[3] = 4.0 * (zeta - xi)
    dxi[4] = 4.0 * eta
    dxi[5] = -4.0 * eta
    deta[0] = 1.0 - 4.0 * zeta
    deta[1] = 0.0
    deta[2] = 4.0 * eta - 1.0
    deta[3] = -4.0 * xi
    deta[4] = 4.0 * xi
    deta[5] = 4.0 * (zeta - eta)


@njit(cache=True)
def _closest_flat(a, b, c, p):
    """The coordinates (ξ, η) of the point of the triangle ``a b c`` nearest ``p``
    (Ericson 2005, §5.1.5: the regions of its corners, of its edges and of its face)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ab[0] * ap[0] + ab[1] * ap[1] + ab[2] * ap[2]
    d2 = ac[0] * ap[0] + ac[1] * ap[1] + ac[2] * ap[2]
    if d1 <= 0.0 and d2 <= 0.0:
        return 0.0, 0.0
    bp = p - b
    d3 = ab[0] * bp[0] + ab[1] * bp[1] + ab[2] * bp[2]
    d4 = ac[0] * bp[0] + ac[1] * bp[1] + ac[2] * bp[2]
    if d3 >= 0.0 and d4 <= d3:
        return 1.0, 0.0
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return d1 / (d1 - d3), 0.0
    cp = p - c
    d5 = ab[0] * cp[0] + ab[1] * cp[1] + ab[2] * cp[2]
    d6 = ac[0] * cp[0] + ac[1] * cp[1] + ac[2] * cp[2]
    if d6 >= 0.0 and d5 <= d6:
        return 0.0, 1.0
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return 0.0, d2 / (d2 - d6)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return 1.0 - w, w
    denom = 1.0 / (va + vb + vc)
    return vb * denom, vc * denom


@njit(cache=True)
def _at(n, x, xi, eta, point, a, b):
    """The point of a face at (ξ, η) and its two tangents."""
    shape = np.empty(n)
    dxi = np.empty(n)
    deta = np.empty(n)
    shapes(n, xi, eta, shape, dxi, deta)
    for c in range(3):
        point[c] = 0.0
        a[c] = 0.0
        b[c] = 0.0
        for i in range(n):
            point[c] += shape[i] * x[i, c]
            a[c] += dxi[i] * x[i, c]
            b[c] += deta[i] * x[i, c]


@njit(cache=True)
def _edge_point(x, edge, t, point, tangent, curvature):
    """The point of an edge of a face of six nodes at ``t`` (edge 0: from the corner 0 to
    1, 1: from 1 to 2, 2: from 2 to 0), its derivative and its second derivative."""
    i, j, k = edge, (edge + 1) % 3, 3 + edge  # its ends and its middle
    n0 = (1.0 - t) * (1.0 - 2.0 * t)
    n1 = t * (2.0 * t - 1.0)
    n2 = 4.0 * t * (1.0 - t)
    for c in range(3):
        point[c] = n0 * x[i, c] + n1 * x[j, c] + n2 * x[k, c]
        tangent[c] = (
            (4.0 * t - 3.0) * x[i, c] + (4.0 * t - 1.0) * x[j, c] + (4.0 - 8.0 * t) * x[k, c]
        )
        curvature[c] = 4.0 * x[i, c] + 4.0 * x[j, c] - 8.0 * x[k, c]


@njit(cache=True)
def closest(n, x, p, slack, rim, over=RIM):
    """The coordinates (ξ, η) of the point of a face of ``n`` nodes (``x``: their
    positions) nearest ``p``, the square of their distance, and how it lies: 1, the foot
    of the perpendicular from ``p`` (the gap is then along the normal of the face); 0, a
    point of an edge that the face shares with another of its surface (a point over a
    ridge); -1, a point of the rim of the surface (``rim``: which of the edges 0-1, 1-2
    and 2-0 of the face no other face shares): ``p`` is beyond the surface. A face of six
    nodes may be curved: Newton's method on its surface from the nearest point of the
    triangle of its corners, then, where it leaves the face, on its three edges.
    ``slack``: how far outside the face (in its coordinates), across the edges that it
    shares, the foot of the perpendicular is still taken, on the face continued: a point
    that hovers over an edge then stays on one face; ``over``: likewise across the edges
    of the rim (:data:`RIM`)."""
    # how far out each edge lets the foot of the perpendicular be
    out0 = over if rim[0] else slack  # the edge 0-1: η = 0
    out1 = over if rim[1] else slack  # the edge 1-2: ξ + η = 1
    out2 = over if rim[2] else slack  # the edge 2-0: ξ = 0
    point = np.empty(3)
    a = np.empty(3)
    b = np.empty(3)
    if n == 3:
        # the foot of the perpendicular on the plane of the triangle
        for c in range(3):
            a[c] = x[1, c] - x[0, c]
            b[c] = x[2, c] - x[0, c]
            point[c] = p[c] - x[0, c]
        d00 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
        d01 = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        d11 = b[0] * b[0] + b[1] * b[1] + b[2] * b[2]
        d20 = point[0] * a[0] + point[1] * a[1] + point[2] * a[2]
        d21 = point[0] * b[0] + point[1] * b[1] + point[2] * b[2]
        det = d00 * d11 - d01 * d01
        xi = (d11 * d20 - d01 * d21) / det
        eta = (d00 * d21 - d01 * d20) / det
        free = xi >= -out2 and eta >= -out0 and xi + eta <= 1.0 + out1
        if not free:
            xi, eta = _closest_flat(x[0], x[1], x[2], p)
        _at(3, x, xi, eta, point, a, b)
        d2 = (point[0] - p[0]) ** 2 + (point[1] - p[1]) ** 2 + (point[2] - p[2]) ** 2
        return xi, eta, d2, 1 if free else _on_rim(xi, eta, rim)
    xi, eta = _closest_flat(x[0], x[1], x[2], p)
    # the second derivatives of the position: constant on a face of six nodes
    xx = np.empty(3)
    yy = np.empty(3)
    xy = np.empty(3)
    for c in range(3):
        xx[c] = 4.0 * (x[0, c] + x[1, c]) - 8.0 * x[3, c]
        yy[c] = 4.0 * (x[0, c] + x[2, c]) - 8.0 * x[5, c]
        xy[c] = 4.0 * (x[0, c] - x[3, c] + x[4, c] - x[5, c])
    u, v = xi, eta
    free = False
    for _ in range(20):
        _at(6, x, u, v, point, a, b)
        r0, r1, r2 = point[0] - p[0], point[1] - p[1], point[2] - p[2]
        g1 = r0 * a[0] + r1 * a[1] + r2 * a[2]
        g2 = r0 * b[0] + r1 * b[1] + r2 * b[2]
        aa = a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
        bb = b[0] * b[0] + b[1] * b[1] + b[2] * b[2]
        ab = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        h11 = aa + r0 * xx[0] + r1 * xx[1] + r2 * xx[2]
        h22 = bb + r0 * yy[0] + r1 * yy[1] + r2 * yy[2]
        h12 = ab + r0 * xy[0] + r1 * xy[1] + r2 * xy[2]
        det = h11 * h22 - h12 * h12
        if det <= 1e-12 * aa * bb or h11 <= 0.0:  # not convex there: Gauss-Newton
            h11, h22, h12 = aa, bb, ab
            det = h11 * h22 - h12 * h12
        du = -(h22 * g1 - h12 * g2) / det
        dv = -(h11 * g2 - h12 * g1) / det
        u += du
        v += dv
        if u < -0.5 or v < -0.5 or u + v > 1.5:
            break  # far outside the face: its edges
        if abs(du) + abs(dv) < 1e-13:
            free = u >= -out2 - 1e-12 and v >= -out0 - 1e-12 and u + v <= 1.0 + out1 + 1e-12
            break
    if free:
        _at(6, x, u, v, point, a, b)
        d2 = (point[0] - p[0]) ** 2 + (point[1] - p[1]) ** 2 + (point[2] - p[2]) ** 2
        return u, v, d2, 1
    best = math.inf
    for edge in range(3):
        if edge == 0:
            t = xi
        elif edge == 1:
            t = eta
        else:
            t = 1.0 - eta
        t = min(max(t, 0.0), 1.0)
        for _ in range(20):
            _edge_point(x, edge, t, point, a, b)
            r0, r1, r2 = point[0] - p[0], point[1] - p[1], point[2] - p[2]
            g = r0 * a[0] + r1 * a[1] + r2 * a[2]
            h = a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + r0 * b[0] + r1 * b[1] + r2 * b[2]
            if h <= 0.0:
                h = a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
            new = min(max(t - g / h, 0.0), 1.0)
            if abs(new - t) < 1e-13:
                t = new
                break
            t = new
        _edge_point(x, edge, t, point, a, b)
        d2 = (point[0] - p[0]) ** 2 + (point[1] - p[1]) ** 2 + (point[2] - p[2]) ** 2
        if d2 < best:
            best = d2
            if edge == 0:
                xi, eta = t, 0.0
            elif edge == 1:
                xi, eta = 1.0 - t, t
            else:
                xi, eta = 0.0, 1.0 - t
    return xi, eta, best, _on_rim(xi, eta, rim)


@njit(cache=True)
def _on_rim(xi, eta, rim):
    """-1 if a point of the boundary of a face lies on an edge of the rim of its surface,
    0 otherwise."""
    if rim[0] and eta <= 1e-12:
        return -1
    if rim[1] and xi + eta >= 1.0 - 1e-12:
        return -1
    if rim[2] and xi <= 1e-12:
        return -1
    return 0


@njit(cache=True)
def _normal(a, b, n):
    """The unit normal ``a × b`` of two tangents; returns the norm of the product."""
    n[0] = a[1] * b[2] - a[2] * b[1]
    n[1] = a[2] * b[0] - a[0] * b[2]
    n[2] = a[0] * b[1] - a[1] * b[0]
    norm = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
    n[0] /= norm
    n[1] /= norm
    n[2] /= norm
    return norm


@njit(cache=True)
def _share(ns, xs, xi, eta, normal) -> float:
    """The share of the face of a slave point that the other surface sees: the cosine of
    the angle of the inward normal of the face, at the point, and the ``normal`` of the
    other surface there (1 for two surfaces that lie on each other, 0 for a face seen
    edge-on or from behind). The points at the nodes of the faces carry it: a node of an
    edge of a body belongs to the faces of both sides of the edge, and only what each
    shows to the other surface presses on it (the side of a block that stands on a plane
    carries nothing)."""
    p = np.empty(3)
    a = np.empty(3)
    b = np.empty(3)
    own = np.empty(3)
    _at(ns, xs, xi, eta, p, a, b)
    _normal(a, b, own)
    return max(0.0, -(own[0] * normal[0] + own[1] * normal[1] + own[2] * normal[2]))


@njit(cache=True)
def _friction(par, pressure, slip, normal, old, predict, traction):
    """The tangential traction of a slave point that moved by ``slip`` along the master
    since the end of the last step (a radial return on the traction of then, turned into
    the tangent plane of now); returns the energy dissipated per unit area."""
    mu, stick = par[P_MU], par[P_STICK]
    traction[0] = traction[1] = traction[2] = 0.0
    if mu <= 0.0:
        return 0.0
    # the traction of the last step in the tangent plane, its norm kept
    t0, t1, t2 = old[S_TX], old[S_TY], old[S_TZ]
    before = math.sqrt(t0 * t0 + t1 * t1 + t2 * t2)
    along = t0 * normal[0] + t1 * normal[1] + t2 * normal[2]
    t0 -= along * normal[0]
    t1 -= along * normal[1]
    t2 -= along * normal[2]
    now = math.sqrt(t0 * t0 + t1 * t1 + t2 * t2)
    if now > 0.0:
        t0 *= before / now
        t1 *= before / now
        t2 *= before / now
    along = slip[0] * normal[0] + slip[1] * normal[1] + slip[2] * normal[2]
    t0 += stick * (slip[0] - along * normal[0])
    t1 += stick * (slip[1] - along * normal[1])
    t2 += stick * (slip[2] - along * normal[2])
    norm = math.sqrt(t0 * t0 + t1 * t1 + t2 * t2)
    limit = mu * max(pressure, 0.0)
    if norm <= limit or predict:
        traction[0], traction[1], traction[2] = t0, t1, t2
        return 0.0
    scale = limit / norm
    traction[0], traction[1], traction[2] = scale * t0, scale * t1, scale * t2
    # the slip of the return: (trial - traction) / stick, along the traction
    return limit * (norm - limit) / stick


@njit(cache=True)
def _touched(par, old) -> bool:
    """Whether a point pressed or touched the master at the end of the last step."""
    return old[S_PRESSURE] > 0.0 or old[S_GAP] <= par[P_TOUCH]


@njit(cache=True)
def pair_point(ns, nm, xs, xm, xs_old, xm_old, rim, xi, eta, weight, par, old, anchored, anchor,
               predict, f, new):  # fmt: skip
    """The forces (``f``: the nodes of the slave face, then those of the master face) of a
    slave point at (``xi``, ``eta``) of its face against a master face, and its state
    ``new``. ``anchored``: the point of the master that faced it at the end of the last
    step lies on this face (at the coordinates of ``old``); otherwise ``anchor`` is its
    position now; a point that no point of the master faced then rubs by the move of
    the two points that touch now since then (``xs_old``, ``xm_old``: the nodes of the two
    faces at the end of the last step). ``rim``: the edges of the master face on the rim
    of its surface.
    ``predict``: a point that touched stays closed and sticks. Returns whether the point
    presses."""
    shape = np.empty(ns)
    dxi = np.empty(ns)
    deta = np.empty(ns)
    shapes(ns, xi, eta, shape, dxi, deta)
    p = np.zeros(3)
    for i in range(ns):
        for c in range(3):
            p[c] += shape[i] * xs[i, c]
    u, v, _, _ = closest(nm, xm, p, SLACK, rim, par[P_RIM])
    there = np.empty(3)
    a = np.empty(3)
    b = np.empty(3)
    _at(nm, xm, u, v, there, a, b)
    normal = np.empty(3)
    _normal(a, b, normal)
    gap = (
        (p[0] - there[0]) * normal[0]
        + (p[1] - there[1]) * normal[1]
        + (p[2] - there[2]) * normal[2]
    )
    touched = _touched(par, old)
    closed = predict and touched
    pressure = old[S_LAMBDA] - par[P_STIFF] * gap
    for k in range(STATE):
        new[k] = old[k]
    new[S_XI], new[S_ETA], new[S_GAP] = u, v, gap
    new[S_TX] = new[S_TY] = new[S_TZ] = 0.0
    new[S_PRESSURE] = 0.0
    new[S_SHARE] = 1.0
    for k in range(3 * (ns + nm)):
        f[k] = 0.0
    if pressure <= 0.0 and not closed:
        return False
    if par[P_SHARE] > 0.0:  # a point at a node: what its face shows to the master
        new[S_SHARE] = _share(ns, xs, xi, eta, normal)
    weight = weight * new[S_SHARE]
    new[S_PRESSURE] = max(pressure, 0.0)
    traction = np.zeros(3)
    shape_m = np.empty(nm)
    shapes(nm, u, v, shape_m, np.empty(nm), np.empty(nm))
    if par[P_MU] > 0.0:
        slip = np.empty(3)
        if not touched:  # it comes into contact: the move of the two points over the step
            for c in range(3):
                slip[c] = p[c] - there[c]
                for i in range(ns):
                    slip[c] -= shape[i] * xs_old[i, c]
                for i in range(nm):
                    slip[c] += shape_m[i] * xm_old[i, c]
        elif anchored:
            _at(nm, xm, old[S_XI], old[S_ETA], slip, a, b)
            for c in range(3):
                slip[c] = there[c] - slip[c]
        else:
            for c in range(3):
                slip[c] = there[c] - anchor[c]
        rubbed = _friction(par, pressure, slip, normal, old, closed, traction)
        new[S_WORK] = old[S_WORK] + new[S_SHARE] * rubbed
        new[S_TX], new[S_TY], new[S_TZ] = traction[0], traction[1], traction[2]
    for c in range(3):
        force = weight * (pressure * normal[c] - traction[c])  # on the slave point
        for i in range(ns):
            f[3 * i + c] = -shape[i] * force
        for i in range(nm):
            f[3 * (ns + i) + c] = shape_m[i] * force
    return True


@njit(cache=True)
def _slave_normal(ns, xs, xi, eta, p, normal):
    """A slave point and the outward normal of its face there."""
    a = np.empty(3)
    b = np.empty(3)
    _at(ns, xs, xi, eta, p, a, b)
    _normal(a, b, normal)


@njit(cache=True)
def _across(nm, xm, p, rim, par) -> bool:
    """Whether a face of a surface is across a wall of solid from a point of the same
    surface: at rest (``xm``: the nodes of the face, ``p``: the point, both in the
    reference positions), the point lies behind the face, within the distance
    :data:`P_BEHIND` of it. The two sides of a thin ligament, back to back, look like two
    surfaces that went through each other by its thickness; surfaces that do come to
    touch were in front of each other at rest."""
    u, v, d2, _ = closest(nm, xm, p, 0.0, rim)
    if d2 > par[P_BEHIND] * par[P_BEHIND]:
        return False
    there = np.empty(3)
    a = np.empty(3)
    b = np.empty(3)
    normal = np.empty(3)
    _at(nm, xm, u, v, there, a, b)
    _normal(a, b, normal)
    gap = ((p[0] - there[0]) * normal[0] + (p[1] - there[1]) * normal[1]
           + (p[2] - there[2]) * normal[2])  # fmt: skip
    return gap < -par[P_TOUCH]


@njit(parallel=True, cache=True)
def pair_forces(x, x_old, x_ref, ident, slave, master, rims, face, xi, eta, weight, candidates,
                last, par, old, new, own, tangent, predict, fe, ke, nodes, active):  # fmt: skip
    """The forces and tangents of the slave points of a pair of surfaces in the positions
    ``x`` of the nodes (``x_old``: at the end of the last step; ``x_ref``: at rest;
    ``ident``: the node that each is, itself or, for the image of a node of a periodic
    cell, the one that it is the same as across a face of the cell). For
    each point, the master face among its ``candidates`` whose nearest point is the
    closest, among those that face it (opposite normals) within the deepest penetration
    looked for, and not beyond the rim of the master surface
    (``rims``: the edges of each master face that no other shares); ``own``: the two
    surfaces are one (the faces that share a node with that of the point are left out,
    and those that it has gone through but that lay behind it at rest, across a wall of
    solid: :func:`_across`).
    ``last``: the face of each point at the
    last evaluation (-1: none), which it keeps while the foot of its perpendicular stays on
    it or within :data:`SLACK` of it: a point that hovers over an edge between two faces,
    where the tangent of the forces breaks, would otherwise go from one to the other at
    each iteration. ``fe`` (points x dofs), ``ke`` (points x
    dofs x dofs, by central differences of the forces), ``nodes`` (the nodes of the dofs)
    and ``active`` (whether the point presses)."""
    ns, nm = slave.shape[1], master.shape[1]
    nd = 3 * (ns + nm)
    for q in prange(face.shape[0]):
        xs = np.empty((ns, 3))
        for i in range(ns):
            xs[i] = x[slave[face[q], i]]
        p = np.empty(3)
        normal_s = np.empty(3)
        _slave_normal(ns, xs, xi[q], eta[q], p, normal_s)
        xm = np.empty((nm, 3))
        there = np.empty(3)
        a = np.empty(3)
        b = np.empty(3)
        normal = np.empty(3)
        p_ref = np.empty(3)  # the point at rest, for the faces that it has gone through
        if own:
            for i in range(ns):
                xs[i] = x_ref[slave[face[q], i]]
            _slave_normal(ns, xs, xi[q], eta[q], p_ref, normal)
            for i in range(ns):
                xs[i] = x[slave[face[q], i]]
        best, best_d2, best_gap = -1, math.inf, math.inf
        before = int(old[q, S_FACE])
        apart = FACING_OWN if own else FACING
        if last[q] >= 0:  # the face of the last evaluation, while the point is over it
            m = last[q]
            for i in range(nm):
                xm[i] = x[master[m, i]]
            u, v, d2, how = closest(nm, xm, p, SLACK, rims[m], par[P_RIM])
            if how == 1:
                _at(nm, xm, u, v, there, a, b)
                _normal(a, b, normal)
                gap = ((p[0] - there[0]) * normal[0] + (p[1] - there[1]) * normal[1]
                       + (p[2] - there[2]) * normal[2])  # fmt: skip
                facing = (normal[0] * normal_s[0] + normal[1] * normal_s[1]
                          + normal[2] * normal_s[2]) < apart  # fmt: skip
                if facing and gap >= -par[P_DEPTH]:
                    through = False
                    if own and gap < 0.0:
                        for i in range(nm):
                            xm[i] = x_ref[master[m, i]]
                        through = _across(nm, xm, p_ref, rims[m], par)
                    if not through:
                        best, best_d2, best_gap = m, d2, gap
        kept = best >= 0
        for k in range(candidates.shape[1]):
            m = candidates[q, k]
            if m < 0 or kept:
                continue
            if own:
                shared = False
                for i in range(3):
                    for j in range(3):
                        if ident[slave[face[q], i]] == ident[master[m, j]]:
                            shared = True
                if shared:
                    continue
            for i in range(nm):
                xm[i] = x[master[m, i]]
            u, v, d2, how = closest(nm, xm, p, 0.0, rims[m], par[P_RIM])
            if how < 0:
                continue  # beyond the rim of the surface
            if how == 0:
                # on an edge that two faces share: over a ridge of the surface (the point
                # is then within 60 degrees of the normal of the face), or beside the
                # face, a little further than the corner where the rim meets that edge
                _at(nm, xm, u, v, there, a, b)
                _normal(a, b, normal)
                along = ((p[0] - there[0]) * normal[0] + (p[1] - there[1]) * normal[1]
                         + (p[2] - there[2]) * normal[2])  # fmt: skip
                beside = d2 - along * along
                if beside > (0.02 * par[P_SIZE]) ** 2 and along * along < 0.25 * d2:
                    continue
            # a point on an edge is as near two faces: it stays on that of the last step,
            # or goes to the first of them, whatever the rounding and the order of the search
            margin = 1e-9 * best_d2 + 1e-300
            if d2 > best_d2 + margin:
                continue
            if d2 >= best_d2 - margin and (best == before or (m != before and m > best)):
                continue
            _at(nm, xm, u, v, there, a, b)
            _normal(a, b, normal)
            if (normal[0] * normal_s[0] + normal[1] * normal_s[1]
                    + normal[2] * normal_s[2]) >= apart:  # fmt: skip
                continue  # a face that does not face the point: the other side of a wall
            gap = ((p[0] - there[0]) * normal[0] + (p[1] - there[1]) * normal[1]
                   + (p[2] - there[2]) * normal[2])  # fmt: skip
            if gap < -par[P_DEPTH]:
                continue
            if own and gap < 0.0:
                for i in range(nm):
                    xm[i] = x_ref[master[m, i]]
                if _across(nm, xm, p_ref, rims[m], par):
                    continue  # the other side of a ligament, not a face gone through
            best, best_d2, best_gap = m, d2, gap
        active[q] = False
        for k in range(STATE):
            new[q, k] = old[q, k]
        new[q, S_FACE] = best
        new[q, S_GAP] = best_gap
        new[q, S_PRESSURE] = 0.0
        new[q, S_TX] = new[q, S_TY] = new[q, S_TZ] = 0.0
        if best < 0:
            continue
        xs_old = np.empty((ns, 3))
        xm_old = np.empty((nm, 3))
        for i in range(ns):
            xs_old[i] = x_old[slave[face[q], i]]
        for i in range(nm):
            xm[i] = x[master[best, i]]
            xm_old[i] = x_old[master[best, i]]
        anchored = before == best
        anchor = np.zeros(3)
        if not anchored and before >= 0:  # the point that faced it, on another face
            xo = np.empty((nm, 3))
            for i in range(nm):
                xo[i] = x[master[before, i]]
            _at(nm, xo, old[q, S_XI], old[q, S_ETA], anchor, a, b)
        state = old[q].copy()
        if before < 0:  # no point faced it: nothing to rub against yet
            state[S_GAP] = math.inf
            state[S_PRESSURE] = 0.0
        f = np.empty(nd)
        fresh = np.empty(STATE)
        pressed = pair_point(ns, nm, xs, xm, xs_old, xm_old, rims[best], xi[q], eta[q], weight[q],
                             par, state, anchored, anchor, False, f, fresh)  # fmt: skip
        fresh[S_FACE] = best
        new[q] = fresh
        closed = predict and _touched(par, state)
        if not (pressed or closed):
            continue
        active[q] = True
        fe[q] = f
        for i in range(ns):
            nodes[q, i] = slave[face[q], i]
        for i in range(nm):
            nodes[q, ns + i] = master[best, i]
        if not tangent:
            continue
        size = math.sqrt(best_d2) + math.sqrt(
            (xm[1, 0] - xm[0, 0]) ** 2 + (xm[1, 1] - xm[0, 1]) ** 2 + (xm[1, 2] - xm[0, 2]) ** 2
        )
        step = 1e-7 * size
        fp = np.empty(nd)
        fm = np.empty(nd)
        dummy = np.empty(STATE)
        for j in range(nd):
            node, c = j // 3, j % 3
            target = xs if node < ns else xm
            row = node if node < ns else node - ns
            keep = target[row, c]
            target[row, c] = keep + step
            pair_point(ns, nm, xs, xm, xs_old, xm_old, rims[best], xi[q], eta[q], weight[q],
                       par, state, anchored, anchor, closed, fp, dummy)  # fmt: skip
            target[row, c] = keep - step
            pair_point(ns, nm, xs, xm, xs_old, xm_old, rims[best], xi[q], eta[q], weight[q],
                       par, state, anchored, anchor, closed, fm, dummy)  # fmt: skip
            target[row, c] = keep
            for i in range(nd):
                ke[q, i, j] = (fp[i] - fm[i]) / (2.0 * step)


@njit(cache=True)
def obstacle_distance(kind, geometry, p, normal):
    """The signed distance of the point ``p`` to a rigid obstacle (positive outside it)
    and its outward normal there. ``geometry``: a point (0-2), a unit axis (3-5), a size
    (6: the radius of a sphere or of a cylinder, the half-angle of a cone) and a sign (7:
    -1 for the inside of a sphere or of a cylinder that holds the body). A plane: its
    point and its outward normal; a sphere: its centre; a cylinder: a point of its axis;
    a cone: its apex and its axis, from the apex into the cone."""
    d0, d1, d2 = p[0] - geometry[0], p[1] - geometry[1], p[2] - geometry[2]
    a0, a1, a2 = geometry[3], geometry[4], geometry[5]
    if kind == PLANE:
        normal[0], normal[1], normal[2] = a0, a1, a2
        return d0 * a0 + d1 * a1 + d2 * a2
    sign = geometry[7]
    if kind == SPHERE:
        r = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        if r > 0.0:
            normal[0], normal[1], normal[2] = sign * d0 / r, sign * d1 / r, sign * d2 / r
        else:
            normal[0], normal[1], normal[2] = sign * a0, sign * a1, sign * a2
        return sign * (r - geometry[6])
    along = d0 * a0 + d1 * a1 + d2 * a2
    r0, r1, r2 = d0 - along * a0, d1 - along * a1, d2 - along * a2
    r = math.sqrt(r0 * r0 + r1 * r1 + r2 * r2)
    if kind == CYLINDER:
        if r > 0.0:
            normal[0], normal[1], normal[2] = sign * r0 / r, sign * r1 / r, sign * r2 / r
        else:
            normal[0], normal[1], normal[2] = 0.0, 0.0, 0.0
        return sign * (r - geometry[6])
    # a cone: its side where the nearest point is on it, its apex otherwise
    cos, sin = math.cos(geometry[6]), math.sin(geometry[6])
    if along * cos + r * sin <= 0.0:  # behind the apex
        full = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        if full > 0.0:
            normal[0], normal[1], normal[2] = d0 / full, d1 / full, d2 / full
        else:
            normal[0], normal[1], normal[2] = -a0, -a1, -a2
        return full
    if r > 0.0:
        e0, e1, e2 = r0 / r, r1 / r, r2 / r
    else:  # on the axis, inside: any radial direction
        e0, e1, e2 = _any_normal(a0, a1, a2)
    normal[0] = cos * e0 - sin * a0
    normal[1] = cos * e1 - sin * a1
    normal[2] = cos * e2 - sin * a2
    return r * cos - along * sin


@njit(cache=True)
def _any_normal(a0, a1, a2):
    """A unit vector normal to the unit vector ``a``."""
    if abs(a0) < 0.9:
        b0, b1, b2 = 1.0, 0.0, 0.0
    else:
        b0, b1, b2 = 0.0, 1.0, 0.0
    along = b0 * a0 + b1 * a1 + b2 * a2
    b0, b1, b2 = b0 - along * a0, b1 - along * a1, b2 - along * a2
    norm = math.sqrt(b0 * b0 + b1 * b1 + b2 * b2)
    return b0 / norm, b1 / norm, b2 / norm


@njit(cache=True)
def obstacle_point(ns, xs, xi, eta, weight, kind, geometry, par, old, predict, f, new):
    """The forces (the nodes of its face) of a slave point against a rigid obstacle and
    its state ``new``; its state keeps, in place of a point of a master face, its position
    relative to the obstacle at the end of the last step. Returns whether it presses."""
    shape = np.empty(ns)
    shapes(ns, xi, eta, shape, np.empty(ns), np.empty(ns))
    p = np.zeros(3)
    for i in range(ns):
        for c in range(3):
            p[c] += shape[i] * xs[i, c]
    normal = np.empty(3)
    gap = obstacle_distance(kind, geometry, p, normal)
    touched = _touched(par, old)
    closed = predict and touched
    pressure = old[S_LAMBDA] - par[P_STIFF] * gap
    for k in range(STATE):
        new[k] = old[k]
    new[S_GAP] = gap
    new[S_TX] = new[S_TY] = new[S_TZ] = 0.0
    new[S_PRESSURE] = 0.0
    # where it is, relative to the obstacle
    new[S_FACE], new[S_XI], new[S_ETA] = p[0] - geometry[0], p[1] - geometry[1], p[2] - geometry[2]
    new[S_SHARE] = 1.0
    for k in range(3 * ns):
        f[k] = 0.0
    if pressure <= 0.0 and not closed:
        return False
    if par[P_SHARE] > 0.0:  # a point at a node: what its face shows to the obstacle
        new[S_SHARE] = _share(ns, xs, xi, eta, normal)
    weight = weight * new[S_SHARE]
    new[S_PRESSURE] = max(pressure, 0.0)
    traction = np.zeros(3)
    if par[P_MU] > 0.0:
        slip = np.empty(3)
        slip[0] = new[S_FACE] - old[S_FACE]
        slip[1] = new[S_XI] - old[S_XI]
        slip[2] = new[S_ETA] - old[S_ETA]
        rubbed = _friction(par, pressure, slip, normal, old, closed, traction)
        new[S_WORK] = old[S_WORK] + new[S_SHARE] * rubbed
        new[S_TX], new[S_TY], new[S_TZ] = traction[0], traction[1], traction[2]
    for c in range(3):
        force = weight * (pressure * normal[c] - traction[c])  # on the slave point
        for i in range(ns):
            f[3 * i + c] = -shape[i] * force
    return True


@njit(parallel=True, cache=True)
def obstacle_forces(x, slave, face, xi, eta, weight, kind, geometry, par, old, new, tangent,
                    predict, fe, ke, active):  # fmt: skip
    """The forces and tangents (central differences) of the slave points of a surface
    against a rigid obstacle in the positions ``x`` of the nodes."""
    ns = slave.shape[1]
    nd = 3 * ns
    for q in prange(face.shape[0]):
        xs = np.empty((ns, 3))
        for i in range(ns):
            xs[i] = x[slave[face[q], i]]
        f = np.empty(nd)
        fresh = np.empty(STATE)
        pressed = obstacle_point(ns, xs, xi[q], eta[q], weight[q], kind, geometry, par, old[q],
                                 False, f, fresh)  # fmt: skip
        new[q] = fresh
        closed = predict and _touched(par, old[q])
        active[q] = pressed or closed
        if not active[q]:
            continue
        fe[q] = f
        if not tangent:
            continue
        step = 1e-7 * math.sqrt(
            (xs[1, 0] - xs[0, 0]) ** 2 + (xs[1, 1] - xs[0, 1]) ** 2 + (xs[1, 2] - xs[0, 2]) ** 2
        )
        fp = np.empty(nd)
        fm = np.empty(nd)
        dummy = np.empty(STATE)
        for j in range(nd):
            node, c = j // 3, j % 3
            keep = xs[node, c]
            xs[node, c] = keep + step
            obstacle_point(ns, xs, xi[q], eta[q], weight[q], kind, geometry, par, old[q],
                           closed, fp, dummy)  # fmt: skip
            xs[node, c] = keep - step
            obstacle_point(ns, xs, xi[q], eta[q], weight[q], kind, geometry, par, old[q],
                           closed, fm, dummy)  # fmt: skip
            xs[node, c] = keep
            for i in range(nd):
                ke[q, i, j] = (fp[i] - fm[i]) / (2.0 * step)
