"""Finite elements on conforming tetrahedral meshes (4-node and 10-node tetrahedra).

The unknown is the fluctuation ``w`` of ``u = E x + w`` (``T = G . x + w`` in
conduction), constrained by the boundary conditions of :mod:`morphogen.homogenization.core`:

``kubc``
    ``w = 0`` at the nodes of the faces of the box;
``mixed``
    the normal (or the prescribed shear) components of ``w`` vanish on the faces
    concerned (Dirrenberger et al. 2014);
``subc``
    tractions ``Sigma n`` (heat flux ``Q . n``) on the faces of the box, rigid motions
    removed by 3-2-1 supports;
``periodic``
    ``w`` takes the same value at a node of an upper face and at its image on the
    lower face. :func:`morphogen.mesh.conforming.conforming_mesh` gives opposite faces
    matching nodes for periodic microstructures, and the conditions are then exact. On
    meshes with non-matching faces, the image value is interpolated with the shape
    functions of the face element that contains it: an approximation (the patch test is
    not passed exactly), reported in the log. On porous meshes whose opposite faces
    differ (voxel meshes of a pore phase), a node whose image falls in a pore of the
    opposite face lies on a free surface, and stays free.

Meshes of whole periodic cells (polycrystals, ``mesh.whole``) are bounded by pairs of
translated facets instead of the faces of the box: ``w`` takes the same value at the
translates of a node (``mesh.periodic_links()``), and the kinematic and static
conditions apply on that boundary (``mesh.outer_faces()``, with their own normals);
mixed conditions, defined on the faces of the box, need a mesh of the box.

Porous media are meshed without their pores (or their pore elements are discarded when
the pore phase is a void): averages are taken over the box with zero stress in the
pores, as for the FFT solver. Elements are integrated exactly: one point for 4-node
elements, four points for 10-node elements with straight edges.

The 10-node elements are isoparametric: a mid-edge node may leave the middle of its
edge (:func:`morphogen.mesh.curved.curved_mesh` puts those of the interfaces on the
implicit surfaces of the phases). Such an element is its straight tetrahedron moved by
``sum_k N_k d_k``, with ``N_k`` the function of a mid-edge node and ``d_k`` its
departure from the middle of its edge: at an integration point, the gradients of the
shape functions are those of the straight element times the inverse of ``F = I + sum_k
d_k (x) grad N_k``, and the volume element is multiplied by ``det F``. Linear fields are
in the space of the elements whatever the departures (the elements pass the patch test
up to the four-point rule, exact for straight elements only).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.sparse as sp
from numba import njit, prange
from scipy.spatial import cKDTree

from morphogen.homogenization.core import (
    Loading,
    MacroscopicLoadings,
    PhaseStatistics,
    Solution,
    boundary_condition,
)
from morphogen.homogenization.elements import VOIGT, voigt_to_tensor
from morphogen.homogenization.materials import Material, material_table, transport_physics
from morphogen.homogenization.solvers import (
    NEGLIGIBLE,
    LinearSolver,
    SolveInfo,
    choose_method,
    norm,
    times,
)
from morphogen.mesh.tetra import TET_EDGES, TetMesh

log = logging.getLogger("morphogen")

_A, _B = 0.5854101966249685, 0.1381966011250105
GAUSS = {
    1: (np.array([[0.25, 0.25, 0.25, 0.25]]), np.array([1.0])),
    2: (
        np.array([[_A, _B, _B, _B], [_B, _A, _B, _B], [_B, _B, _A, _B], [_B, _B, _B, _A]]),
        np.full(4, 0.25),
    ),
}
EDGES = np.ascontiguousarray(TET_EDGES, dtype=np.int64)
#: The integrations of the elastic problems: every point, or the volume change of each
#: element as its mean (10-node tetrahedra; :mod:`morphogen.homogenization.dilatation`)
INTEGRATIONS = ("full", "bbar")
#: Poisson's ratio from which a phase is nearly incompressible (a warning with ``full``)
NEARLY_INCOMPRESSIBLE = 0.49


def integration_of(value) -> str:
    """The name of an integration: ``"full"``, or ``"bbar"`` (also written ``"reduced"``,
    ``"selective"``, ``"b-bar"``, ``"f-bar"``)."""
    name = "full" if value is None else str(value).lower().replace("-", "").replace("_", "")
    name = {"reduced": "bbar", "selective": "bbar", "fbar": "bbar"}.get(name, name)
    if name not in INTEGRATIONS:
        raise ValueError(
            "integration: 'full' or 'bbar' (the volume change of each element as its mean: "
            f"selective reduced integration), got {value!r}"
        )
    return name


def nearly_incompressible(materials, phases, limit: float = NEARLY_INCOMPRESSIBLE) -> list[int]:
    """The phases whose Poisson's ratio is at least ``limit`` (isotropic elastic
    materials): those that lock the elements integrated at every point."""
    out = []
    for p in phases:
        material = materials.get(int(p))
        poisson = getattr(material, "poisson", None)
        if material is not None and not material.void and poisson is not None:
            if float(poisson) >= limit:
                out.append(int(p))
    return out


@njit(cache=True, inline="always")
def _corner_gradients(x, g):
    """Gradients of the barycentric coordinates (4 x 3) and the volume of a tetrahedron."""
    a00 = x[1, 0] - x[0, 0]
    a01 = x[2, 0] - x[0, 0]
    a02 = x[3, 0] - x[0, 0]
    a10 = x[1, 1] - x[0, 1]
    a11 = x[2, 1] - x[0, 1]
    a12 = x[3, 1] - x[0, 1]
    a20 = x[1, 2] - x[0, 2]
    a21 = x[2, 2] - x[0, 2]
    a22 = x[3, 2] - x[0, 2]
    det = (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )
    inv = 1.0 / det
    # rows of A^-1 are the gradients of L1, L2, L3
    g[1, 0] = (a11 * a22 - a12 * a21) * inv
    g[1, 1] = (a02 * a21 - a01 * a22) * inv
    g[1, 2] = (a01 * a12 - a02 * a11) * inv
    g[2, 0] = (a12 * a20 - a10 * a22) * inv
    g[2, 1] = (a00 * a22 - a02 * a20) * inv
    g[2, 2] = (a02 * a10 - a00 * a12) * inv
    g[3, 0] = (a10 * a21 - a11 * a20) * inv
    g[3, 1] = (a01 * a20 - a00 * a21) * inv
    g[3, 2] = (a00 * a11 - a01 * a10) * inv
    for d in range(3):
        g[0, d] = -(g[1, d] + g[2, d] + g[3, d])
    return det / 6.0


@njit(cache=True, inline="always")
def _shape_gradients(order, gl, bary, out):
    """Gradients (n x 3) of the shape functions at a point of barycentric coordinates."""
    if order == 1:
        for a in range(4):
            for d in range(3):
                out[a, d] = gl[a, d]
        return
    for a in range(4):
        f = 4.0 * bary[a] - 1.0
        for d in range(3):
            out[a, d] = f * gl[a, d]
    for k in range(6):
        i = EDGES[k, 0]
        j = EDGES[k, 1]
        for d in range(3):
            out[4 + k, d] = 4.0 * (bary[i] * gl[j, d] + bary[j] * gl[i, d])


@njit(cache=True, inline="always")
def _geometry(nodes, elements, e, x, gl, dev):
    """The corners ``x`` of element ``e``, the gradients ``gl`` of their barycentric
    coordinates and, for a 10-node element, the departures ``dev`` (6 x 3) of its mid-edge
    nodes from the middles of its straight edges. Returns the volume of the straight
    tetrahedron and whether the element is curved (a departure beyond the round-off of
    coordinates read from a file)."""
    for a in range(4):
        for d in range(3):
            x[a, d] = nodes[elements[e, a], d]
    vol = _corner_gradients(x, gl)
    curved = False
    if elements.shape[1] == 10:
        for k in range(6):
            i = EDGES[k, 0]
            j = EDGES[k, 1]
            m = elements[e, 4 + k]
            length2 = 0.0
            dev2 = 0.0
            for d in range(3):
                dv = nodes[m, d] - 0.5 * (x[i, d] + x[j, d])
                dev[k, d] = dv
                dev2 += dv * dv
                length2 += (x[j, d] - x[i, d]) ** 2
            if dev2 > 1e-20 * length2:
                curved = True
    return vol, curved


@njit(cache=True, inline="always")
def _point_gradients(order, curved, gl, dev, bary, out):
    """Gradients (n x 3) of the shape functions of an element at a point of barycentric
    coordinates, and the ratio of its volume element to that of its straight tetrahedron
    (1 for a straight element). A curved 10-node element is the straight one moved by
    ``sum_k N_k d_k`` (``N_k`` the function of a mid-edge node, ``d_k`` its departure):
    with ``F = I + sum_k d_k (x) grad N_k``, the gradients are those of the straight
    element times ``F^-1`` and the ratio is ``det F``."""
    _shape_gradients(order, gl, bary, out)
    if not curved:
        return 1.0
    f00 = 1.0
    f01 = 0.0
    f02 = 0.0
    f10 = 0.0
    f11 = 1.0
    f12 = 0.0
    f20 = 0.0
    f21 = 0.0
    f22 = 1.0
    for k in range(6):
        g0 = out[4 + k, 0]
        g1 = out[4 + k, 1]
        g2 = out[4 + k, 2]
        f00 += dev[k, 0] * g0
        f01 += dev[k, 0] * g1
        f02 += dev[k, 0] * g2
        f10 += dev[k, 1] * g0
        f11 += dev[k, 1] * g1
        f12 += dev[k, 1] * g2
        f20 += dev[k, 2] * g0
        f21 += dev[k, 2] * g1
        f22 += dev[k, 2] * g2
    c00 = f11 * f22 - f12 * f21
    c01 = f02 * f21 - f01 * f22
    c02 = f01 * f12 - f02 * f11
    det = f00 * c00 + f10 * c01 + f20 * c02
    inv = 1.0 / det
    i00 = c00 * inv
    i01 = c01 * inv
    i02 = c02 * inv
    i10 = (f12 * f20 - f10 * f22) * inv
    i11 = (f00 * f22 - f02 * f20) * inv
    i12 = (f02 * f10 - f00 * f12) * inv
    i20 = (f10 * f21 - f11 * f20) * inv
    i21 = (f01 * f20 - f00 * f21) * inv
    i22 = (f00 * f11 - f01 * f10) * inv
    for a in range(10):
        g0 = out[a, 0]
        g1 = out[a, 1]
        g2 = out[a, 2]
        out[a, 0] = g0 * i00 + g1 * i10 + g2 * i20
        out[a, 1] = g0 * i01 + g1 * i11 + g2 * i21
        out[a, 2] = g0 * i02 + g1 * i12 + g2 * i22
    return det


@njit(cache=True, inline="always")
def _element_matrix(order, curved, gl, dev, vol, nd, d, gp, gw, grad, db, ke):
    """Element matrix of a tetrahedron: conductivity tensor or elasticity matrix ``d``."""
    nn = grad.shape[0]
    ke[:, :] = 0.0
    for g in range(gp.shape[0]):
        w = gw[g] * vol * _point_gradients(order, curved, gl, dev, gp[g], grad)
        if nd == 1:
            for b in range(nn):
                for i in range(3):
                    db[b, i, 0] = d[i, 0] * grad[b, 0] + d[i, 1] * grad[b, 1] + d[i, 2] * grad[b, 2]
            for a in range(nn):
                for b in range(nn):
                    ke[a, b] += w * (
                        grad[a, 0] * db[b, 0, 0]
                        + grad[a, 1] * db[b, 1, 0]
                        + grad[a, 2] * db[b, 2, 0]
                    )
            continue
        # D B_b (6 x 3) for each node, B_a^T (D B_b) for each pair
        for b in range(nn):
            gx, gy, gz = grad[b, 0], grad[b, 1], grad[b, 2]
            for r in range(6):
                db[b, r, 0] = d[r, 0] * gx + d[r, 4] * gz + d[r, 5] * gy
                db[b, r, 1] = d[r, 1] * gy + d[r, 3] * gz + d[r, 5] * gx
                db[b, r, 2] = d[r, 2] * gz + d[r, 3] * gy + d[r, 4] * gx
        for a in range(nn):
            gx, gy, gz = grad[a, 0], grad[a, 1], grad[a, 2]
            for b in range(nn):
                for c2 in range(3):
                    ke[3 * a, 3 * b + c2] += w * (
                        gx * db[b, 0, c2] + gz * db[b, 4, c2] + gy * db[b, 5, c2]
                    )
                    ke[3 * a + 1, 3 * b + c2] += w * (
                        gy * db[b, 1, c2] + gz * db[b, 3, c2] + gx * db[b, 5, c2]
                    )
                    ke[3 * a + 2, 3 * b + c2] += w * (
                        gz * db[b, 2, c2] + gy * db[b, 3, c2] + gx * db[b, 4, c2]
                    )


@njit(cache=True)
def _mean_dilatation_matrix(order, curved, gl, dev, vol, d, gp, gw, grad, db, ke, dp, c, q):
    """Elasticity matrix of a tetrahedron whose volume change is its mean (``B``-bar: see
    :mod:`morphogen.homogenization.dilatation`): with ``P = I - 1 (x) 1 / 3`` and the mean
    ``c`` of the divergence operator over the element, ``sum_g w_g B_g^T (P D P) B_g + q_L
    c^T + c q_R^T + k V c c^T / 9``, ``q = V B(c)^T (D 1 - k 1 / 3) / 3``, ``k = 1 . D
    1``."""
    nn = grad.shape[0]
    cm = np.empty(6)
    mc = np.empty(6)
    for r in range(6):
        cm[r] = d[r, 0] + d[r, 1] + d[r, 2]
        mc[r] = d[0, r] + d[1, r] + d[2, r]
    k = cm[0] + cm[1] + cm[2]
    for r in range(6):
        mr = 1.0 if r < 3 else 0.0
        for s in range(6):
            ms = 1.0 if s < 3 else 0.0
            dp[r, s] = d[r, s] - (cm[r] * ms + mr * mc[s]) / 3.0 + k * mr * ms / 9.0
    _element_matrix(order, curved, gl, dev, vol, 3, dp, gp, gw, grad, db, ke)
    total = 0.0
    for b in range(nn):
        for i in range(3):
            c[3 * b + i] = 0.0
    for g in range(gp.shape[0]):
        w = gw[g] * vol * _point_gradients(order, curved, gl, dev, gp[g], grad)
        total += w
        for b in range(nn):
            for i in range(3):
                c[3 * b + i] += w * grad[b, i]
    for r in range(6):
        mr = 1.0 if r < 3 else 0.0
        cm[r] = (cm[r] - k * mr / 3.0) / 3.0
        mc[r] = (mc[r] - k * mr / 3.0) / 3.0
    for b in range(nn):
        gx, gy, gz = c[3 * b] / total, c[3 * b + 1] / total, c[3 * b + 2] / total
        c[3 * b], c[3 * b + 1], c[3 * b + 2] = gx, gy, gz
        # q_L = V B(c)^T v_L (the first three), q_R = V B(c)^T v_R (the last three)
        q[3 * b] = total * (gx * cm[0] + gz * cm[4] + gy * cm[5])
        q[3 * b + 1] = total * (gy * cm[1] + gz * cm[3] + gx * cm[5])
        q[3 * b + 2] = total * (gz * cm[2] + gy * cm[3] + gx * cm[4])
        q[3 * (nn + b)] = total * (gx * mc[0] + gz * mc[4] + gy * mc[5])
        q[3 * (nn + b) + 1] = total * (gy * mc[1] + gz * mc[3] + gx * mc[5])
        q[3 * (nn + b) + 2] = total * (gz * mc[2] + gy * mc[3] + gx * mc[4])
    for a in range(3 * nn):
        for b in range(3 * nn):
            ke[a, b] += q[a] * c[b] + c[a] * q[3 * nn + b] + k * total * c[a] * c[b] / 9.0


@njit(cache=True)
def _fill(nodes, elements, order, nd, mat, matrices, gp, gw, indptr_n, indices_n, data, bbar):
    """Add the element matrices into the CSR values (serial: elements share rows);
    ``bbar``: those of the mean dilatation (elasticity)."""
    n_el, nn = elements.shape
    x = np.empty((4, 3))
    gl = np.empty((4, 3))
    dev = np.empty((6, 3))
    grad = np.empty((nn, 3))
    db = np.empty((nn, 6, 3))
    ke = np.empty((nn * nd, nn * nd))
    pos = np.empty((nn, nn), dtype=np.int64)
    dp = np.empty((6, 6))
    mean = np.empty(3 * nn)  # of the divergence operator
    rank = np.empty(6 * nn)  # the vectors of the term of rank two
    for e in range(n_el):
        vol, curved = _geometry(nodes, elements, e, x, gl, dev)
        if bbar:
            _mean_dilatation_matrix(
                order, curved, gl, dev, vol, matrices[mat[e]], gp, gw, grad, db, ke, dp, mean, rank
            )
        else:
            _element_matrix(order, curved, gl, dev, vol, nd, matrices[mat[e]], gp, gw, grad, db, ke)
        for a in range(nn):
            na = elements[e, a]
            lo = indptr_n[na]
            hi = indptr_n[na + 1]
            for b in range(nn):
                nb = elements[e, b]
                # binary search of nb in the sorted neighbours of na
                left, right = lo, hi
                while left < right:
                    mid = (left + right) // 2
                    if indices_n[mid] < nb:
                        left = mid + 1
                    else:
                        right = mid
                pos[a, b] = left - lo
        for a in range(nn):
            na = elements[e, a]
            length = indptr_n[na + 1] - indptr_n[na]
            for c in range(nd):
                start = nd * nd * indptr_n[na] + c * nd * length
                for b in range(nn):
                    p = pos[a, b]
                    for c2 in range(nd):
                        data[start + p * nd + c2] += ke[nd * a + c, nd * b + c2]


@njit(parallel=True, cache=True)
def _expand_columns(indptr_n, indices_n, nd):
    """Dof-level column indices of the node-level pattern."""
    n = indptr_n.size - 1
    out = np.empty(nd * nd * indptr_n[-1], dtype=np.int64)
    for na in prange(n):
        length = indptr_n[na + 1] - indptr_n[na]
        for c in range(nd):
            start = nd * nd * indptr_n[na] + c * nd * length
            for j in range(length):
                nb = indices_n[indptr_n[na] + j]
                for c2 in range(nd):
                    out[start + j * nd + c2] = nb * nd + c2
    return out


N_ELASTIC = 27
N_THERMAL = 15


@njit(cache=True, inline="always")
def _strain(order, gl, bary, elements, e, u, grad, eps):
    """Strain (engineering shears) at a point of the straight element ``e``."""
    _shape_gradients(order, gl, bary, grad)
    _strain_of(grad, elements, e, u, eps)


@njit(cache=True, inline="always")
def _strain_of(grad, elements, e, u, eps):
    """Strain (engineering shears) of element ``e`` from the gradients of its shape
    functions at a point."""
    nn = elements.shape[1]
    eps[:] = 0.0
    for a in range(nn):
        k = elements[e, a]
        ux, uy, uz = u[3 * k], u[3 * k + 1], u[3 * k + 2]
        gx, gy, gz = grad[a, 0], grad[a, 1], grad[a, 2]
        eps[0] += gx * ux
        eps[1] += gy * uy
        eps[2] += gz * uz
        eps[3] += gz * uy + gy * uz
        eps[4] += gz * ux + gx * uz
        eps[5] += gy * ux + gx * uy


@njit(cache=True, inline="always")
def _gradient_of(grad, elements, e, t, gt):
    """Gradient of the potential in element ``e`` from the gradients of its shape
    functions at a point."""
    nn = elements.shape[1]
    gt[:] = 0.0
    for a in range(nn):
        k = elements[e, a]
        for d in range(3):
            gt[d] += grad[a, d] * t[k]


@njit(cache=True, inline="always")
def _product(m, v, out):
    n = v.shape[0]
    for r in range(n):
        s = 0.0
        for c in range(n):
            s += m[r, c] * v[c]
        out[r] = s


N_ELASTIC = 27
N_THERMAL = 15


@njit(parallel=True, cache=True)
def _elastic_sums(nodes, elements, order, group, n_groups, mat, matrices, eigen, dt, u, gp, gw):
    """Integrals over each group of the strain (tensor components), the stress
    ``D (eps - dt alpha)``, the energy density ``sigma : (eps - dt alpha)`` and their
    squares (``eigen[mat_e]``: the thermal strain of a unit change of temperature)."""
    n_el, nn = elements.shape
    n_chunks = min(max(n_el, 1), 256)
    partial = np.zeros((n_chunks, n_groups, N_ELASTIC))
    for ch in prange(n_chunks):
        x = np.empty((4, 3))
        gl = np.empty((4, 3))
        dev = np.empty((6, 3))
        grad = np.empty((nn, 3))
        eps = np.empty(6)
        mech = np.empty(6)
        sig = np.empty(6)
        for e in range(ch * n_el // n_chunks, (ch + 1) * n_el // n_chunks):
            vol, curved = _geometry(nodes, elements, e, x, gl, dev)
            acc = partial[ch, group[e]]
            dm = matrices[mat[e]]
            for g in range(gp.shape[0]):
                w = gw[g] * vol * _point_gradients(order, curved, gl, dev, gp[g], grad)
                _strain_of(grad, elements, e, u, eps)
                for r in range(6):
                    mech[r] = eps[r] - dt * eigen[mat[e], r]
                _product(dm, mech, sig)
                dens = 0.0
                for r in range(6):
                    dens += sig[r] * mech[r]
                for r in range(6):
                    et = eps[r] if r < 3 else 0.5 * eps[r]
                    acc[r] += w * et
                    acc[6 + r] += w * sig[r]
                    acc[13 + r] += w * et * et
                    acc[19 + r] += w * sig[r] * sig[r]
                acc[12] += w * dens
                acc[25] += w * dens * dens
                acc[26] += w
    return partial.sum(axis=0)


@njit(parallel=True, cache=True)
def _thermal_sums(nodes, elements, order, group, n_groups, mat, matrices, t, gp, gw):
    n_el, nn = elements.shape
    n_chunks = min(max(n_el, 1), 256)
    partial = np.zeros((n_chunks, n_groups, N_THERMAL))
    for ch in prange(n_chunks):
        x = np.empty((4, 3))
        gl = np.empty((4, 3))
        dev = np.empty((6, 3))
        grad = np.empty((nn, 3))
        gt = np.empty(3)
        q = np.empty(3)
        for e in range(ch * n_el // n_chunks, (ch + 1) * n_el // n_chunks):
            vol, curved = _geometry(nodes, elements, e, x, gl, dev)
            acc = partial[ch, group[e]]
            km = matrices[mat[e]]
            for g in range(gp.shape[0]):
                w = gw[g] * vol * _point_gradients(order, curved, gl, dev, gp[g], grad)
                _gradient_of(grad, elements, e, t, gt)
                _product(km, gt, q)
                dens = q[0] * gt[0] + q[1] * gt[1] + q[2] * gt[2]
                for d in range(3):
                    flux = -q[d]
                    acc[d] += w * gt[d]
                    acc[3 + d] += w * flux
                    acc[7 + d] += w * gt[d] * gt[d]
                    acc[10 + d] += w * flux * flux
                acc[6] += w * dens
                acc[13] += w * dens * dens
                acc[14] += w
    return partial.sum(axis=0)


@njit(parallel=True, cache=True)
def _elastic_fields(nodes, elements, order, mat, matrices, eigen, dt, u, gp, gw):
    """Element means of strain (tensor components), stress and energy density."""
    n_el, nn = elements.shape
    strain = np.zeros((n_el, 6))
    stress = np.zeros((n_el, 6))
    energy = np.zeros(n_el)
    for e in prange(n_el):
        x = np.empty((4, 3))
        gl = np.empty((4, 3))
        dev = np.empty((6, 3))
        grad = np.empty((nn, 3))
        eps = np.empty(6)
        mech = np.empty(6)
        sig = np.empty(6)
        _, curved = _geometry(nodes, elements, e, x, gl, dev)
        dm = matrices[mat[e]]
        total = 0.0
        for g in range(gp.shape[0]):
            w = gw[g] * _point_gradients(order, curved, gl, dev, gp[g], grad)
            _strain_of(grad, elements, e, u, eps)
            for r in range(6):
                mech[r] = eps[r] - dt * eigen[mat[e], r]
            _product(dm, mech, sig)
            total += w
            for r in range(6):
                strain[e, r] += w * (eps[r] if r < 3 else 0.5 * eps[r])
                stress[e, r] += w * sig[r]
                energy[e] += w * sig[r] * mech[r]
        for r in range(6):
            strain[e, r] /= total
            stress[e, r] /= total
        energy[e] /= total
    return strain, stress, energy


@njit(cache=True)
def _thermal_forces(nodes, elements, order, mat, matrices, eigen, gp, gw):
    """Nodal forces of a unit change of temperature, ``sum_e int B^T D alpha dV``, and
    the norm of the element contributions (before they cancel between elements)."""
    n_el, nn = elements.shape
    out = np.zeros(3 * nodes.shape[0])
    norm2 = 0.0
    x = np.empty((4, 3))
    gl = np.empty((4, 3))
    dev = np.empty((6, 3))
    grad = np.empty((nn, 3))
    sig = np.empty(6)
    for e in range(n_el):
        a = eigen[mat[e]]
        if a[0] == 0.0 and a[1] == 0.0 and a[2] == 0.0 and a[3] == 0.0 and a[4] == 0.0:
            if a[5] == 0.0:
                continue
        vol, curved = _geometry(nodes, elements, e, x, gl, dev)
        _product(matrices[mat[e]], a, sig)
        for g in range(gp.shape[0]):
            w = gw[g] * vol * _point_gradients(order, curved, gl, dev, gp[g], grad)
            for b in range(nn):
                k = elements[e, b]
                gx, gy, gz = grad[b, 0], grad[b, 1], grad[b, 2]
                fx = w * (gx * sig[0] + gz * sig[4] + gy * sig[5])
                fy = w * (gy * sig[1] + gz * sig[3] + gx * sig[5])
                fz = w * (gz * sig[2] + gy * sig[3] + gx * sig[4])
                out[3 * k] += fx
                out[3 * k + 1] += fy
                out[3 * k + 2] += fz
                norm2 += fx * fx + fy * fy + fz * fz
    return out, np.sqrt(norm2)


@njit(parallel=True, cache=True)
def _thermal_fields(nodes, elements, order, mat, matrices, t, gp, gw):
    """Element means of the temperature gradient, heat flux and dissipation density."""
    n_el, nn = elements.shape
    grad = np.zeros((n_el, 3))
    flux = np.zeros((n_el, 3))
    dissipation = np.zeros(n_el)
    for e in prange(n_el):
        x = np.empty((4, 3))
        gl = np.empty((4, 3))
        dev = np.empty((6, 3))
        shape_grad = np.empty((nn, 3))
        gt = np.empty(3)
        q = np.empty(3)
        _, curved = _geometry(nodes, elements, e, x, gl, dev)
        km = matrices[mat[e]]
        total = 0.0
        for g in range(gp.shape[0]):
            w = gw[g] * _point_gradients(order, curved, gl, dev, gp[g], shape_grad)
            _gradient_of(shape_grad, elements, e, t, gt)
            _product(km, gt, q)
            total += w
            for d in range(3):
                grad[e, d] += w * gt[d]
                flux[e, d] -= w * q[d]
                dissipation[e] += w * q[d] * gt[d]
        for d in range(3):
            grad[e, d] /= total
            flux[e, d] /= total
        dissipation[e] /= total
    return grad, flux, dissipation


@njit(cache=True)
def _node_elements(elements, n_nodes):
    """Elements around each node (CSR: pointers, element indices)."""
    n_el, nn = elements.shape
    ptr = np.zeros(n_nodes + 1, dtype=np.int64)
    for e in range(n_el):
        for a in range(nn):
            ptr[elements[e, a] + 1] += 1
    for i in range(n_nodes):
        ptr[i + 1] += ptr[i]
    fill = ptr[:-1].copy()
    idx = np.empty(ptr[-1], dtype=np.int64)
    for e in range(n_el):
        for a in range(nn):
            node = elements[e, a]
            idx[fill[node]] = e
            fill[node] += 1
    return ptr, idx


@njit(cache=True, inline="always")
def _neighbours(elements, ptr, idx, i, buf):
    """Sorted distinct nodes of the elements around node ``i`` (in ``buf``); count."""
    nn = elements.shape[1]
    c = 0
    for q in range(ptr[i], ptr[i + 1]):
        e = idx[q]
        for a in range(nn):
            buf[c] = elements[e, a]
            c += 1
    view = buf[:c]
    view.sort()
    u = 0
    for j in range(c):
        if j == 0 or view[j] != view[j - 1]:
            buf[u] = view[j]
            u += 1
    return u


@njit(parallel=True, cache=True)
def _pattern(elements, ptr, idx, n_nodes):
    nn = elements.shape[1]
    lengths = np.zeros(n_nodes, dtype=np.int64)
    for i in prange(n_nodes):
        buf = np.empty((ptr[i + 1] - ptr[i]) * nn, dtype=np.int64)
        lengths[i] = _neighbours(elements, ptr, idx, i, buf)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    for i in range(n_nodes):
        indptr[i + 1] = indptr[i] + lengths[i]
    indices = np.empty(indptr[-1], dtype=np.int64)
    for i in prange(n_nodes):
        buf = np.empty((ptr[i + 1] - ptr[i]) * nn, dtype=np.int64)
        u = _neighbours(elements, ptr, idx, i, buf)
        indices[indptr[i] : indptr[i] + u] = buf[:u]
    return indptr, indices


def node_pattern(elements: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Node-level CSR pattern (row pointers, sorted column indices) of a mesh."""
    elements = np.ascontiguousarray(elements, dtype=np.int64)
    ptr, idx = _node_elements(elements, n_nodes)
    return _pattern(elements, ptr, idx, n_nodes)


#: The elements of a colour are shared between the threads by chunks: this many at most
#: (their work arrays are allocated once for each, before the loops)
CHUNKS = 64


@njit(cache=True)
def _element_colours(elements, ptr, idx):
    """Greedy colouring of the elements (``ptr``, ``idx``: those around each node): each
    takes the smallest colour that the elements before it with a node of its own do not
    have. Returns the elements sorted by colour (in their order within a colour) and
    the offsets of the colours in that list."""
    n_el, nn = elements.shape
    colour = np.full(n_el, -1, dtype=np.int64)
    most = 0
    for i in range(ptr.size - 1):
        most = max(most, ptr[i + 1] - ptr[i])
    taken = np.full(nn * most + 2, -1, dtype=np.int64)  # the element that last met a colour
    count = 0
    for e in range(n_el):
        for a in range(nn):
            node = elements[e, a]
            for q in range(ptr[node], ptr[node + 1]):
                c = colour[idx[q]]
                if c >= 0:
                    taken[c] = e
        c = 0
        while taken[c] == e:
            c += 1
        colour[e] = c
        count = max(count, c + 1)
    starts = np.zeros(count + 1, dtype=np.int64)
    for e in range(n_el):
        starts[colour[e] + 1] += 1
    for c in range(count):
        starts[c + 1] += starts[c]
    order = np.empty(n_el, dtype=np.int64)
    place = starts[:count].copy()
    for e in range(n_el):
        order[place[colour[e]]] = e
        place[colour[e]] += 1
    return order, starts


@njit(cache=True)
def chunk_slots(colours):
    """The offsets of the chunks of each colour (:data:`CHUNKS` at most, one element at
    least) in the list of all the chunks: the place of the sums of each chunk, added in
    this order whatever the number of threads."""
    slots = np.zeros(colours.size, dtype=np.int64)
    for c in range(colours.size - 1):
        slots[c + 1] = slots[c] + min(colours[c + 1] - colours[c], CHUNKS)
    return slots


def element_colours(elements: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """The elements of a mesh sorted by colour, and the offsets of the colours in that
    list: two elements of a colour have no node in common. The loops that add the
    contributions of the elements at their nodes (a stiffness, nodal forces) take the
    elements of a colour in parallel, without two threads writing at one place, and
    each place receives the contributions of its elements colour by colour: the same
    sums for any number of threads."""
    elements = np.ascontiguousarray(elements, dtype=np.int64)
    ptr, idx = _node_elements(elements, n_nodes)
    return _element_colours(elements, ptr, idx)


def assemble(mesh: TetMesh, nd: int, mat, matrices, integration: str = "full") -> sp.csr_matrix:
    """Global matrix (all nodes and components) of a tetrahedral mesh: element ``e``
    has the elasticity matrix (``nd = 3``) or conductivity tensor (``nd = 1``)
    ``matrices[mat[e]]``. ``integration``: ``"bbar"`` takes the volume change of each
    10-node tetrahedron as its mean (elasticity; :data:`INTEGRATIONS`)."""
    bbar = integration_of(integration) == "bbar" and nd == 3 and mesh.order == 2
    indptr_n, indices_n = node_pattern(mesh.elements, mesh.n_nodes)
    gp, gw = GAUSS[mesh.order]
    data = np.zeros(nd * nd * indptr_n[-1])
    _fill(
        mesh.nodes,
        mesh.elements,
        mesh.order,
        nd,
        np.ascontiguousarray(mat, dtype=np.int64),
        np.ascontiguousarray(matrices, dtype=float),
        gp,
        gw,
        indptr_n,
        indices_n,
        data,
        bbar,
    )
    indices = _expand_columns(indptr_n, indices_n, nd)
    lengths = np.repeat(np.diff(indptr_n) * nd, nd)
    indptr = np.zeros(nd * mesh.n_nodes + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    if indptr[-1] < 2**31 - 1:
        indices, indptr = indices.astype(np.int32), indptr.astype(np.int32)
    n = nd * mesh.n_nodes
    return sp.csr_matrix((data, indices, indptr), shape=(n, n))


def _face_weights(order: int, bary: np.ndarray) -> np.ndarray:
    """Shape functions of a 3- or 6-node triangle at barycentric coordinates (k x 3)."""
    if order == 1:
        return bary
    l0, l1, l2 = bary[:, 0], bary[:, 1], bary[:, 2]
    return np.stack(
        [
            l0 * (2 * l0 - 1),
            l1 * (2 * l1 - 1),
            l2 * (2 * l2 - 1),
            4 * l0 * l1,
            4 * l1 * l2,
            4 * l2 * l0,
        ],
        axis=1,
    )


@njit(parallel=True, cache=True)
def _places(indptr, indices, column, r_indptr, r_indices, place):
    """The place in a reduced matrix (CSR pattern ``r_indptr``, ``r_indices``, its rows
    sorted) of each stored value of a matrix (``indptr``, ``indices``): row and column
    are those of the reduced unknowns ``column`` of its own (-1: none, no place)."""
    for i in prange(indptr.size - 1):
        a = column[i]
        for k in range(indptr[i], indptr[i + 1]):
            b = column[indices[k]]
            found = -1
            if a >= 0 and b >= 0:
                lo, hi = r_indptr[a], r_indptr[a + 1]
                while lo < hi:
                    mid = (lo + hi) // 2
                    if r_indices[mid] < b:
                        lo = mid + 1
                    else:
                        hi = mid
                found = lo
            place[k] = found


@njit(cache=True)
def _sources(indptr, indices, place, weight, weighted, n_out):
    """The stored values of ``K`` that go to each place of the reduced matrix (their
    indices, in their order, and the offsets of the places in that list), with the
    product of the values of ``T`` of their row and of their column when these are not
    all ones (``weighted``)."""
    offsets = np.zeros(n_out + 1, dtype=np.int64)
    for k in range(place.size):
        if place[k] >= 0:
            offsets[place[k] + 1] += 1
    for p in range(n_out):
        offsets[p + 1] += offsets[p]
    sources = np.empty(offsets[n_out], dtype=place.dtype)
    factors = np.empty(offsets[n_out] if weighted else 1)
    fill = offsets[:n_out].copy()
    for i in range(indptr.size - 1):
        for k in range(indptr[i], indptr[i + 1]):
            p = place[k]
            if p >= 0:
                sources[fill[p]] = k
                if weighted:
                    factors[fill[p]] = weight[i] * weight[indices[k]]
                fill[p] += 1
    return offsets, sources, factors


@njit(parallel=True, cache=True)
def _gather(offsets, sources, factors, weighted, data, out):
    """``out = T^T K T``: at each place of the reduced matrix, the sum of the stored
    values of ``K`` (``data``) that go there, in their order (:func:`_sources`); the
    places are shared between the threads."""
    for p in prange(out.size):
        s = 0.0
        for q in range(offsets[p], offsets[p + 1]):
            if weighted:
                s += factors[q] * data[sources[q]]
            else:
                s += data[sources[q]]
        out[p] = s


class Reduction:
    """The reduced matrices ``T^T K T`` of the matrices ``K`` of one pattern, for a
    reduction ``T`` that has one stored value per row at most (periodic faces that match,
    kinematic conditions): each stored value of ``K`` goes to one place of the reduced
    matrix, found once (:func:`_places`), and each place sums the values that go there
    (:func:`_gather`, on several threads). The tangent of a loading path is reduced at
    each linearization; the two products of sparse matrices and their conversions take
    much longer. The reduced matrices share the arrays of their pattern. :meth:`of`
    returns none for a ``T`` whose rows interpolate several unknowns (periodic faces
    that do not match)."""

    def __init__(self, t: sp.csr_matrix, indptr: np.ndarray, indices: np.ndarray):
        n_rows, n_red = t.shape
        has = np.diff(t.indptr) == 1
        column = np.full(n_rows, -1, dtype=np.int64)
        column[has] = t.indices
        self.weight = np.zeros(n_rows)
        self.weight[has] = t.data
        self.weighted = bool(np.any(t.data != 1.0))
        # the pattern of the reduced matrix: that of the product for a K of ones
        ones = sp.csr_matrix((np.ones(indices.size), indices, indptr), shape=(n_rows, n_rows))
        pattern = (t.T @ ones @ t).tocsr()
        pattern.sum_duplicates()
        pattern.sort_indices()
        self.shape = (n_red, n_red)
        self.indptr, self.indices = pattern.indptr, pattern.indices
        small = max(pattern.nnz, indices.size) < 2**31 - 1
        place = np.empty(indices.size, dtype=np.int32 if small else np.int64)
        _places(indptr, indices, column, self.indptr, self.indices, place)
        self._sources = _sources(indptr, indices, place, self.weight, self.weighted, pattern.nnz)

    @classmethod
    def of(cls, t, indptr, indices) -> Reduction | None:
        t = sp.csr_matrix(t)
        t.sum_duplicates()
        if t.nnz and int(np.diff(t.indptr).max()) > 1:
            return None
        return cls(t, indptr, indices)

    def __call__(self, data: np.ndarray) -> sp.csr_matrix:
        """``T^T K T`` for the stored values ``data`` of ``K``."""
        out = np.empty(self.indices.size)
        _gather(*self._sources, self.weighted, np.ascontiguousarray(data, dtype=float), out)
        reduced = sp.csr_matrix((out, self.indices, self.indptr), shape=self.shape)
        reduced.has_sorted_indices = True
        return reduced


def periodic_map(mesh: TetMesh) -> tuple[sp.csr_matrix, np.ndarray, dict]:
    """Matrix T (nodes x masters) with ``w = T w_masters`` for periodic fluctuations.

    Nodes of the upper faces are slaves. A slave with an image node on the lower face
    takes its value; the others are interpolated in the lower face element containing
    their image. A node whose image lies off the elements of the lower face, farther than
    a quarter of the longest edge of the nearest of them, is on a free surface: the
    material of the cell meets a pore of the next one there (porous voxel meshes whose
    opposite faces differ), and it stays free, unless another node lies at its place
    modulo the periods (an image through an edge or a corner of the box), whose class it
    joins. Returns T, the master nodes and the counts ``matched`` and ``interpolated`` of
    slave nodes, ``outside``: interpolated nodes found just outside the lower face (the
    nearest element is used), and ``free``: the nodes of the upper faces left free. Whole
    periodic cells: the translates of a node take its value.
    """
    n = mesh.n_nodes
    if mesh.whole:  # translates of the nodes of the boundary: their first node
        slave, master, _ = mesh.periodic_links()
        target = np.arange(n)
        target[slave] = master
        masters = np.flatnonzero(target == np.arange(n))
        position = np.full(n, -1)
        position[masters] = np.arange(masters.size)
        t = sp.csr_matrix((np.ones(n), (np.arange(n), position[target])), shape=(n, masters.size))
        counts = {"matched": int(slave.size), "interpolated": 0, "outside": 0, "free": 0}
        return t, masters, counts
    rows, cols, vals = [], [], []
    slave_of = np.full(n, -1)
    freed = np.zeros(n, dtype=bool)
    outside = 0
    n_matched = 0
    n_interpolated = 0
    for axis in range(3):
        upper = mesh.face_nodes(axis, 1)
        faces = mesh.boundary_faces(axis, 0)
        if upper.size == 0:
            continue
        if faces.size == 0:
            raise ValueError(
                f"periodic conditions: the lower face normal to axis {axis} carries no "
                "element while the upper one does"
            )
        keep = slave_of[upper] < 0  # each node is the slave of its first upper face
        upper = upper[keep]
        slave_of[upper] = axis
        freed[upper] = False
        others = [d for d in range(3) if d != axis]
        # nodes with an image node on the lower face (matching meshes): weight 1
        lower = np.unique(faces)
        dist, near = cKDTree(mesh.nodes[lower][:, others]).query(mesh.nodes[upper][:, others])
        matched = dist <= 1e-7 * float(np.max(mesh.size))
        rows.append(upper[matched])
        cols.append(lower[near[matched]])
        vals.append(np.ones(int(matched.sum())))
        n_matched += int(matched.sum())
        upper = upper[~matched]
        if upper.size == 0:
            continue
        corners = mesh.nodes[faces[:, :3]][:, :, others]  # (k, 3, 2)
        tree = cKDTree(corners.mean(axis=1))
        pts = mesh.nodes[upper][:, others]
        k = min(12, len(faces))
        _, cand = tree.query(pts, k=k)
        cand = cand.reshape(len(pts), k)
        # off the lower face (a pore of the next cell there): free nodes of a free surface
        gap, edge = _distance_to_triangles(pts, corners[cand])
        loose = gap > 0.25 * edge
        slave_of[upper[loose]] = -1
        freed[upper[loose]] = True
        upper, pts, cand = upper[~loose], pts[~loose], cand[~loose]
        n_interpolated += upper.size
        if upper.size == 0:
            continue
        a, b, c = corners[cand, 0], corners[cand, 1], corners[cand, 2]
        v0, v1, v2 = b - a, c - a, pts[:, None, :] - a
        d00 = (v0 * v0).sum(-1)
        d01 = (v0 * v1).sum(-1)
        d11 = (v1 * v1).sum(-1)
        d20 = (v2 * v0).sum(-1)
        d21 = (v2 * v1).sum(-1)
        den = d00 * d11 - d01 * d01
        l1 = (d11 * d20 - d01 * d21) / den
        l2 = (d00 * d21 - d01 * d20) / den
        bary = np.stack([1 - l1 - l2, l1, l2], axis=-1)
        score = bary.min(axis=-1)
        best = np.argmax(score, axis=1)
        idx = np.arange(len(pts))
        chosen = cand[idx, best]
        lam = bary[idx, best]
        bad = score[idx, best] < -1e-6
        outside += int(bad.sum())
        lam = np.clip(lam, 0.0, None)
        lam /= lam.sum(axis=1, keepdims=True)
        w = _face_weights(mesh.order, lam)
        masters = faces[chosen]
        rows.append(np.repeat(upper, w.shape[1]))
        cols.append(masters.ravel())
        vals.append(w.ravel())
    if freed.any():  # the free nodes at the place of others modulo the periods (diagonally)
        joined = _free_classes(mesh, np.flatnonzero(freed), slave_of)
        for node, master in joined:
            rows.append(np.array([node]))
            cols.append(np.array([master]))
            vals.append(np.ones(1))
            slave_of[node], freed[node] = 0, False
    n_free = int(freed.sum())
    free = np.flatnonzero(slave_of < 0)
    rows.append(free)
    cols.append(free)
    vals.append(np.ones(free.size))
    p = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )
    p.eliminate_zeros()
    t = p.copy()
    for _ in range(4):  # resolve the slaves whose masters are slaves (edges, corners)
        support = np.unique(t.indices)
        if not np.any(slave_of[support] >= 0):
            break
        t = t @ p
        t.data[np.abs(t.data) < 1e-14] = 0.0
        t.eliminate_zeros()
    masters = np.flatnonzero(slave_of < 0)
    used = np.unique(t.indices)
    if np.any(slave_of[used] >= 0):
        raise RuntimeError("periodic conditions: could not resolve the slave nodes")
    position = np.full(n, -1)
    position[masters] = np.arange(masters.size)
    t = sp.csr_matrix((t.data, position[t.indices], t.indptr), shape=(n, masters.size))
    counts = {"matched": n_matched, "interpolated": n_interpolated, "outside": outside}
    return t, masters, {**counts, "free": n_free}


def _free_classes(mesh: TetMesh, loose: np.ndarray, slave_of: np.ndarray) -> list:
    """The free nodes of the upper faces (``loose``) at the place of other nodes modulo the
    periods of the box (their images through an edge or a corner, along a diagonal): one
    master for each class of them, the node of the class that is a master already, else its
    first free node; returns the pairs (free node, its master) of the others."""
    size = np.asarray(mesh.size, dtype=float)
    tol = 1e-7 * float(size.max())
    on = mesh.boundary_nodes()
    wrapped = np.mod(mesh.nodes[on] - mesh.lower, size)
    wrapped[np.abs(wrapped - size) <= tol] = 0.0
    tree = cKDTree(wrapped, boxsize=size)
    index = np.full(mesh.n_nodes, -1)
    index[on] = np.arange(on.size)
    free = set(int(p) for p in loose)
    out, done = [], set()
    for p, near in zip(loose, tree.query_ball_point(wrapped[index[loose]], tol), strict=True):
        if int(p) in done:
            continue
        members = on[np.asarray(near, dtype=np.int64)]
        masters = [int(m) for m in members if slave_of[m] < 0 and int(m) not in free]
        own = sorted(int(m) for m in members if int(m) in free)
        master = masters[0] if masters else own[0]
        out += [(m, master) for m in own if m != master]
        done.update(own)
    return out


def _distance_to_triangles(points: np.ndarray, triangles: np.ndarray):
    """The distance from each point (m x 2) to the nearest of its candidate triangles
    (m x k x 3 x 2: zero inside one) and the longest edge of that triangle."""
    p = points[:, None, :]
    gaps, sides = [], []
    for i in range(3):  # the distance to each edge, and the side of the point
        a, d = triangles[:, :, i], triangles[:, :, (i + 1) % 3] - triangles[:, :, i]
        s = np.clip(((p - a) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-300), 0.0, 1.0)
        gaps.append(np.linalg.norm(p - a - s[..., None] * d, axis=-1))
        sides.append(d[..., 0] * (p - a)[..., 1] - d[..., 1] * (p - a)[..., 0])
    sides = np.array(sides)
    inside = np.all(sides >= 0, axis=0) | np.all(sides <= 0, axis=0)
    gap = np.where(inside, 0.0, np.min(gaps, axis=0))
    nearest = np.argmin(gap, axis=1)
    idx = np.arange(len(points))
    tri = triangles[idx, nearest]
    edges = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=-1).max(axis=1)
    return gap[idx, nearest], edges


def _outer_normals(mesh: TetMesh) -> tuple[np.ndarray, np.ndarray]:
    """Faces bounding the volume element and their outward normals times their areas."""
    faces = mesh.outer_faces()
    p = mesh.nodes[faces[:, :3]]
    return faces, 0.5 * np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]).reshape(-1, 3)


#: A rule of the triangle exact to degree 4 (Dunavant): area coordinates and weights
_T1, _T2 = (0.445948490915965, 0.108103018168070), (0.091576213509771, 0.816847572980459)
_TRIANGLE = (
    np.array(
        [
            [_T1[0], _T1[0], _T1[1]],
            [_T1[0], _T1[1], _T1[0]],
            [_T1[1], _T1[0], _T1[0]],
            [_T2[0], _T2[0], _T2[1]],
            [_T2[0], _T2[1], _T2[0]],
            [_T2[1], _T2[0], _T2[0]],
        ]
    ),
    np.array([0.223381589678011] * 3 + [0.109951743655322] * 3),
)


def _face_loads(mesh: TetMesh) -> tuple[np.ndarray, np.ndarray]:
    """The faces bounding the volume element and, for each of their nodes, the integral
    over the face of its shape function times the outward normal (faces x nodes x 3):
    the consistent nodal loads of a uniform traction (flux) are their products by the
    stress (flux). Straight faces: a third of the area vector at the corners (3 nodes) or
    at the mid-edge nodes (6 nodes); faces with curved edges (the nodes of an interface
    moved within a face of the box): the six nodes, integrated on the curved triangle."""
    faces, normal = _outer_normals(mesh)
    loads = np.zeros((len(faces), faces.shape[1], 3))
    if mesh.order == 1:
        loads[:] = normal[:, None, :] / 3.0
        return faces, loads
    loads[:, 3:] = normal[:, None, :] / 3.0
    if not mesh.curved:
        return faces, loads
    x = mesh.nodes[faces]  # (faces, 6, 3)
    ends = x[:, [0, 1, 2]], x[:, [1, 2, 0]]
    bent = np.flatnonzero(np.abs(x[:, 3:] - 0.5 * (ends[0] + ends[1])).max(axis=(1, 2)) > 0.0)
    if not bent.size:
        return faces, loads
    points, weights = _TRIANGLE
    shape = _face_weights(2, points)  # (points, 6)
    l0, l1, l2 = points[:, 0], points[:, 1], points[:, 2]
    zero = np.zeros_like(l0)
    # derivatives of the six functions along the two edges from the first corner
    d1 = np.stack([1 - 4 * l0, 4 * l1 - 1, zero, 4 * (l0 - l1), 4 * l2, -4 * l2], axis=1)
    d2 = np.stack([1 - 4 * l0, zero, 4 * l2 - 1, -4 * l1, 4 * l1, 4 * (l0 - l2)], axis=1)
    t1 = np.einsum("qa,fad->fqd", d1, x[bent])
    t2 = np.einsum("qa,fad->fqd", d2, x[bent])
    area = np.cross(t1, t2)  # (faces, points, 3): the area vector per unit reference area
    loads[bent] = 0.5 * np.einsum("q,qa,fqd->fad", weights, shape, area)
    return faces, loads


def linear_interpolation(mesh: TetMesh) -> tuple[sp.csr_matrix, np.ndarray]:
    """Interpolation (nodes x corner nodes) of the 4-node element functions of a
    10-node mesh into its 10-node functions, and the corner nodes."""
    corners = np.unique(mesh.elements[:, :4])
    index = np.full(mesh.n_nodes, -1)
    index[corners] = np.arange(corners.size)
    if mesh.order == 1:
        return sp.identity(mesh.n_nodes, format="csr"), corners
    mids, first = np.unique(mesh.elements[:, 4:].ravel(), return_index=True)
    ends = mesh.elements[:, TET_EDGES].reshape(-1, 2)[first]
    rows = np.concatenate([corners, mids, mids])
    cols = np.concatenate([np.arange(corners.size), index[ends[:, 0]], index[ends[:, 1]]])
    vals = np.concatenate([np.ones(corners.size), np.full(2 * mids.size, 0.5)])
    p = sp.csr_matrix((vals, (rows, cols)), shape=(mesh.n_nodes, corners.size))
    return p, corners


class _FEProblem(MacroscopicLoadings):
    physics = ""
    nd = 0

    def __init__(
        self,
        mesh: TetMesh,
        materials: dict[int, Material],
        bc: str = "kubc",
        *,
        solver: str = "auto",
        tol: float = 1e-8,
        maxiter: int | None = None,
        regularization: float = 1e-12,
        physics: str | None = None,
        integration: str = "full",
    ):
        if not isinstance(mesh, TetMesh):
            raise TypeError("the FE solver needs a TetMesh (see morphogen.mesh.conforming)")
        #: ``"full"``, or ``"bbar"``: the volume change of each 10-node tetrahedron as its
        #: mean (elastic problems; :mod:`morphogen.homogenization.dilatation`)
        self.integration = integration_of(integration)
        if physics is not None:
            if transport_physics(physics) != (self.nd == 1):
                kind = "a transport phenomenon" if self.nd == 1 else "'elastic'"
                raise ValueError(f"{type(self).__name__}: physics {kind}, got {physics!r}")
            self.physics = physics
        self.bc = boundary_condition(bc)
        if self.bc == "mixed" and mesh.whole:
            raise ValueError(
                "mixed boundary conditions are defined on the faces of the box: mesh the "
                "polycrystal with cells='box'"
            )
        self.box_volume = mesh.box_volume
        phases = [int(p) for p in np.unique(mesh.phase)]
        missing = [p for p in phases if p not in materials]
        if missing:
            raise ValueError(f"no material for phase(s) {missing}; phases present: {phases}")
        # the phases inert for the physics: voids, and the insulators of a transport
        voids = [p for p in phases if materials[p].inert(self.physics)]
        solid_phases = [p for p in phases if p not in voids]
        # pores: void phases, or parts of the box left out of the mesh
        covered = float(np.abs(mesh.volumes()).sum()) / mesh.box_volume
        self.has_voids = bool(voids) or covered < 1.0 - 1e-9
        if self.bc == "subc" and self.has_voids:
            raise ValueError(
                "static uniform (Neumann) boundary conditions cannot be applied to porous "
                "media (Dirrenberger et al. 2014, sec. 3.4.2); use 'mixed' instead"
            )
        if voids:
            mesh = mesh.subset(solid_phases) if solid_phases else None
        self.floating_fraction = 0.0
        if mesh is not None and self.has_voids and mesh.n_elements:
            mesh = self._drop_floating(mesh)
        self.empty = mesh is None or mesh.n_elements == 0
        if self.empty:
            log.warning("no load-carrying material in the sample: its apparent properties are zero")
        self.mesh = mesh
        self.solver_method = solver
        self.tol = tol
        self.maxiter = maxiter
        self.regularization = float(regularization) if self.has_voids else 0.0
        self.groups = tuple(solid_phases)
        if not self.empty:
            # material matrix of every element (one per grain of an oriented crystal)
            self.table = material_table(
                mesh.phase,
                materials,
                self.physics,
                grains=mesh.cell,
                orientations=mesh.orientations,
            )
            self.groups = self.table.phases
            self.group = np.ascontiguousarray(self.table.stat)
            self.mat = np.ascontiguousarray(self.table.mat)
            locking = nearly_incompressible(materials, solid_phases) if self.nd == 3 else []
            if locking and (self.integration == "full" or mesh.order == 1):
                log.warning(
                    "phase(s) %s nearly incompressible (Poisson's ratio of %.2f or more): "
                    "elements integrated at every point lock; take 10-node elements with "
                    "integration='bbar' (the volume change of each element as its mean)",
                    locking,
                    NEARLY_INCOMPRESSIBLE,
                )
        self._systems: dict = {}
        self._matrix = None
        self._units: dict = {}
        self._forces = None

    def _drop_floating(self, mesh: TetMesh) -> TetMesh:
        """Remove the clusters of elements that touch no face of the box (they carry no
        load); ``floating_fraction`` is their share of the meshed volume."""
        labels, n_comp = mesh.components()
        if n_comp == 1 and mesh.boundary_nodes().size:
            return mesh
        on_boundary = np.zeros(mesh.n_nodes, dtype=bool)
        on_boundary[mesh.boundary_nodes()] = True
        touching = np.zeros(n_comp, dtype=bool)
        hit = on_boundary[mesh.elements].any(axis=1)
        touching[np.unique(labels[hit])] = True
        keep = touching[labels]
        if keep.all():
            return mesh
        vol = np.abs(mesh.volumes())
        self.floating_fraction = float(vol[~keep].sum() / vol.sum())
        sub = mesh._derived(mesh.nodes, mesh.elements[keep], keep)
        return sub.subset(np.unique(sub.phase)) if sub.n_elements else sub

    # -- constraints ------------------------------------------------------------------
    def _face_mask(self, axis: int) -> np.ndarray:
        on = np.zeros(self.mesh.n_nodes, dtype=bool)
        on[self.mesh.face_nodes(axis, 0)] = True
        on[self.mesh.face_nodes(axis, 1)] = True
        return on

    def _support_nodes(self) -> list[int]:
        """3-2-1 supports: nodes nearest to three corners of the box."""
        m = self.mesh
        lo, hi = m.lower, m.upper
        targets = [lo, np.array([hi[0], lo[1], lo[2]]), np.array([lo[0], hi[1], lo[2]])]
        tree = cKDTree(m.nodes)
        return [int(tree.query(t)[1]) for t in targets]

    def _stiffness(self) -> sp.csr_matrix:
        if self._matrix is None:
            self._matrix = assemble(
                self.mesh, self.nd, self.mat, self.table.matrices, self.integration
            )
        return self._matrix

    def _reduction(self, key):
        """Reduced unknowns of a constraint key: the matrix ``T`` of ``w = T x`` and the
        dof that each reduced unknown represents."""
        nd = self.nd
        n = self.mesh.n_nodes
        if key[0] == "periodic":
            t_nodes, masters, counts = periodic_map(self.mesh)
            if counts["interpolated"]:
                log.warning(
                    "periodic conditions on non-matching faces: %d of %d nodes interpolated "
                    "(%d outside the opposite face); the results are approximate. Mesh "
                    "periodic microstructures with conforming_mesh(..., periodic=True).",
                    counts["interpolated"],
                    counts["interpolated"] + counts["matched"],
                    counts["outside"],
                )
            t = sp.kron(t_nodes, sp.identity(nd), format="csr") if nd > 1 else t_nodes
            # each reduced unknown is the value of a master dof
            rep = (masters[:, None] * nd + np.arange(nd)).ravel()
            fixed = np.zeros(t.shape[1], dtype=bool)
            if not self.has_voids:
                fixed[:nd] = True  # translations: fluctuation zero at one master node
            keep = np.flatnonzero(~fixed)
            return t[:, keep].tocsr(), rep[keep]
        rep = np.flatnonzero(~self._constraints(key))
        t = sp.csr_matrix((np.ones(rep.size), (rep, np.arange(rep.size))), shape=(n * nd, rep.size))
        return t, rep

    def _reduce(self, key, k, t, rep) -> sp.csr_matrix:
        """Reduced matrix ``T^T K T`` (regularized for porous media)."""
        reduced = (t.T @ k @ t).tocsr() if key[0] == "periodic" else k[rep][:, rep].tocsr()
        return self._regularized(reduced)

    def _regularized(self, reduced: sp.csr_matrix) -> sp.csr_matrix:
        """A reduced matrix with the shift of its diagonal that regularizes porous media."""
        if self.regularization > 0 and reduced.shape[0]:
            diag = reduced.diagonal()
            shift = self.regularization * float(np.mean(diag))
            reduced = (reduced + sp.diags(np.full(diag.size, shift))).tocsr()
        return reduced

    def _linear_solver(self, reduced, t, rep, symmetric: bool = True) -> LinearSolver:
        """The solver of a reduced matrix (``symmetric``: false for the tangent of
        nonsymmetric consistent tangents, see :class:`LinearSolver`)."""
        method = choose_method(self.solver_method, reduced.shape[0])
        near_null = self._near_null(t)
        coarse = None
        if method == "amg" and self.mesh.order == 2:
            coarse = self._coarse_space(t, rep, near_null)
        return LinearSolver(
            reduced,
            method=method,
            tol=self.tol,
            maxiter=self.maxiter,
            near_null=near_null,
            coarse=coarse,
            symmetric=symmetric,
            dofs=rep if self.nd == 3 else None,
        )

    def _system(self, key):
        """Reduced system of a constraint key: ``w = T x`` and the solver of
        ``T^T K T x = T^T f``."""
        if key in self._systems:
            return self._systems[key]
        t, rep = self._reduction(key)
        reduced = self._reduce(key, self._stiffness(), t, rep)
        self._systems[key] = (t, self._linear_solver(reduced, t, rep))
        return self._systems[key]

    def _coarse_space(self, t, rep, near_null):
        """Interpolation of the reduced unknowns of the linear elements (corner nodes)
        into those of the quadratic elements, the near-null modes of the former and the
        degrees of freedom that they stand for (displacements; none otherwise).

        ``rep`` is the dof represented by each reduced unknown: the coarse unknowns are
        those carried by corner nodes; the values of the other corner dofs follow from
        the constraints (``T``), and the mid-edge values from linear interpolation.
        """
        nd = self.nd
        p_nodes, corners = linear_interpolation(self.mesh)
        p_full = sp.kron(p_nodes, sp.identity(nd), format="csr") if nd > 1 else p_nodes
        corner_dofs = (corners[:, None] * nd + np.arange(nd)).ravel()
        on_corner = np.zeros(t.shape[0], dtype=bool)
        on_corner[corner_dofs] = True
        cols = np.flatnonzero(on_corner[rep])
        p = (p_full[rep] @ t[corner_dofs][:, cols]).tocsr()
        p.eliminate_zeros()
        return p, near_null[cols], rep[cols] if nd == 3 else None

    def _solve(self, key, rhs_full, scale=None):
        """Fluctuation solving the reduced system; a load below ``NEGLIGIBLE`` times
        ``scale`` (the size of its parts before they cancel) is only round-off: the
        fluctuation is zero (the affine field is the solution)."""
        t, solver = self._system(key)
        b = t.T @ rhs_full
        if scale is not None and norm(b) <= NEGLIGIBLE * scale:
            info = SolveInfo(solver.method, b.size, 0, 0.0, solver.setup_s, 0.0)
            return np.zeros(t.shape[0]), info
        x, info = solver.solve(b)
        return t @ x, info

    def _phase_statistics(self, sums, n_mean, names, sizes) -> list[PhaseStatistics]:
        out = []
        vol_index = sums.shape[1] - 1
        for g, phase in enumerate(self.groups):
            vol = sums[g, vol_index]
            if vol <= 0:
                continue
            mean, std, pos = {}, {}, 0
            for name, size in zip(names, sizes, strict=True):
                m = sums[g, pos : pos + size] / vol
                sq = sums[g, n_mean + pos : n_mean + pos + size] / vol
                mean[name] = m
                std[name] = np.sqrt(np.maximum(sq - m * m, 0.0))
                pos += size
            out.append(PhaseStatistics(phase, float(vol / self.box_volume), mean, std))
        return out

    def _empty_solution(self, loading: Loading, t0: float) -> Solution:
        elastic = self.physics == "elastic"
        values = np.where(loading.strain_controlled, loading.array, 0.0)
        strain = voigt_to_tensor(values, strain=True) if elastic else values
        return Solution(
            physics=self.physics,
            bc=self.bc,
            loading=loading,
            strain=strain,
            stress=np.zeros((3, 3)) if elastic else np.zeros(3),
            energy=0.0,
            material_fraction=0.0,
            phases=[],
            info=SolveInfo("none", 0, 0, 0.0, 0.0, time.perf_counter() - t0),
        )


class FEElasticProblem(_FEProblem):
    """Linear (thermo-)elasticity on a tetrahedral mesh (small strains); crystals are
    turned by the orientations of the grains of polycrystal meshes."""

    physics = "elastic"
    nd = 3

    def _key(self, x: np.ndarray):
        """Constraint key of a native loading (strain components, or stresses)."""
        if self.bc != "mixed":
            return (self.bc,)
        shear = np.flatnonzero(x[3:] != 0)
        if shear.size == 0:
            return ("mixed", "normal")
        if shear.size == 1 and np.all(x[:3] == 0):
            a, b = VOIGT[3 + shear[0]]
            return ("mixed", "shear", min(a, b), max(a, b))
        raise ValueError(
            "mixed boundary conditions are defined for diagonal (triaxial) strains and for "
            "a single shear component (Dirrenberger et al. 2014, sec. 3.4.3)"
        )

    def _constraints(self, key) -> np.ndarray:
        n = self.mesh.n_nodes
        con = np.zeros((n, 3), dtype=bool)
        if key[0] == "kubc":
            con[self.mesh.boundary_nodes(), :] = True
        elif key[0] == "subc":
            a, b, c = self._support_nodes()
            con[a, :] = True
            con[b, 1:] = True
            con[c, 2] = True
        elif key[1] == "normal":
            for d in range(3):
                con[self._face_mask(d), d] = True
        else:
            a, b = key[2], key[3]
            con[self._face_mask(a), b] = True
            con[self._face_mask(b), a] = True
            if not self.has_voids:
                third = 3 - a - b
                con[self._support_nodes()[0], third] = True
        return con.ravel()

    def _near_null(self, t):
        x = self.mesh.nodes - self.mesh.nodes.mean(axis=0)
        n = self.mesh.n_nodes
        modes = []
        for c in range(3):
            m = np.zeros((n, 3))
            m[:, c] = 1.0
            modes.append(m)
        if self.bc != "periodic":
            for a, b in ((0, 1), (1, 2), (2, 0)):
                r = np.zeros((n, 3))
                r[:, a] = -x[:, b]
                r[:, b] = x[:, a]
                modes.append(r)
        full = np.stack([m.ravel() for m in modes], axis=1)
        # least-squares representation of the modes in the reduced unknowns
        tt = (t.T @ t).diagonal()
        return (t.T @ full) / np.where(tt > 0, tt, 1.0)[:, None]

    def _thermal_forces(self) -> tuple[np.ndarray, float]:
        """Nodal forces of a unit change of temperature and their scale."""
        if self._forces is None:
            gp, gw = GAUSS[self.mesh.order]
            self._forces = _thermal_forces(
                self.mesh.nodes,
                self.mesh.elements,
                self.mesh.order,
                self.mat,
                self.table.matrices,
                self.table.eigen,
                gp,
                gw,
            )
        return self._forces

    def _field(self, x: np.ndarray, dt: float):
        """Displacement for the native loading ``x`` (strain, or stress for ``subc``,
        Voigt order) and the change of temperature ``dt``."""
        key = self._key(x)
        m = self.mesh
        if self.native == "strain":
            e = voigt_to_tensor(x, strain=True)
            u_aff = times(m.nodes, e.T).ravel()
            rhs = -(self._stiffness() @ u_aff)
            scale = norm(self._stiffness().diagonal() * u_aff)
        else:
            u_aff = np.zeros(3 * m.n_nodes)
            rhs = self._traction_load(voigt_to_tensor(x, strain=False))
            scale = None
        if dt:
            forces, size = self._thermal_forces()
            rhs = rhs + dt * forces
            scale = None if scale is None else float(np.hypot(scale, abs(dt) * size))
        w, info = self._solve(key, rhs, scale)
        return u_aff + w, info

    def _sums(self, u: np.ndarray, dt: float) -> np.ndarray:
        m = self.mesh
        gp, gw = GAUSS[m.order]
        return _elastic_sums(
            m.nodes,
            m.elements,
            m.order,
            self.group,
            len(self.groups),
            self.mat,
            self.table.matrices,
            self.table.eigen,
            float(dt),
            u,
            gp,
            gw,
        )

    def _dual(self, u: np.ndarray, x: np.ndarray, dt: float) -> np.ndarray:
        """Average stress (strain loadings) or strain (stress loadings, engineering
        shears) of a displacement, in the Voigt order."""
        total = self._sums(u, dt).sum(axis=0) / self.box_volume
        if self.native == "strain":
            return total[6:12].copy()
        return total[0:6] * np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    def _evaluate(self, loading, u, x, dt, info, fields, t0) -> Solution:
        m = self.mesh
        sums = self._sums(u, dt)
        total = sums.sum(axis=0)
        stress = voigt_to_tensor(total[6:12] / self.box_volume, strain=False)
        if self.native == "strain":
            strain = voigt_to_tensor(x, strain=True)
        else:
            strain = voigt_to_tensor(total[0:6] / self.box_volume, strain=False)
        fld = None
        if fields:
            gp, gw = GAUSS[m.order]
            es, ss, en = _elastic_fields(
                m.nodes,
                m.elements,
                m.order,
                self.mat,
                self.table.matrices,
                self.table.eigen,
                float(dt),
                u,
                gp,
                gw,
            )
            fld = {"strain": es, "stress": ss, "energy": en, "displacement": u.reshape(-1, 3)}
        info.solve_s = time.perf_counter() - t0
        return Solution(
            physics="elastic",
            bc=self.bc,
            loading=loading,
            strain=strain,
            stress=stress,
            energy=float(total[12] / self.box_volume),
            material_fraction=float(total[26] / self.box_volume),
            phases=self._phase_statistics(sums, 13, ("strain", "stress", "energy"), (6, 6, 1)),
            info=info,
            fields=fld,
            support=m if fields else None,
        )

    def _traction_load(self, sigma: np.ndarray) -> np.ndarray:
        m = self.mesh
        load = np.zeros((m.n_nodes, 3))
        # consistent loads: corners A/3 (3-node), mid-edge nodes A/3 (6-node, straight)
        faces, loads = _face_loads(m)
        np.add.at(load, faces, loads @ sigma.T)
        return load.ravel()


class FEThermalProblem(_FEProblem):
    """Steady linear heat conduction on a tetrahedral mesh, or with ``physics`` any other
    transport phenomenon of a scalar potential (:data:`~morphogen.homogenization.
    materials.TRANSPORT`: ``"electrical"``, ``"dielectric"``, ``"magnetic"``,
    ``"diffusion"``), with the coefficient of the materials for it; the phases whose
    coefficient is zero (insulators) are left out as pores."""

    physics = "thermal"
    nd = 1

    def _key(self, x: np.ndarray):
        if self.bc != "mixed":
            return (self.bc,)
        nonzero = np.flatnonzero(x != 0)
        if nonzero.size > 1:
            raise ValueError(
                "mixed thermal boundary conditions (MTBC) need a gradient along one axis"
            )
        return ("mixed", int(nonzero[0]) if nonzero.size else 0)

    def _constraints(self, key) -> np.ndarray:
        con = np.zeros(self.mesh.n_nodes, dtype=bool)
        if key[0] == "kubc":
            con[self.mesh.boundary_nodes()] = True
        elif key[0] == "subc":
            con[self._support_nodes()[0]] = True
        else:
            con |= self._face_mask(key[1])
        return con

    def _near_null(self, t):
        ones = np.ones((self.mesh.n_nodes, 1))
        tt = (t.T @ t).diagonal()
        return (t.T @ ones) / np.where(tt > 0, tt, 1.0)[:, None]

    def _field(self, x: np.ndarray, dt: float = 0.0):
        """Temperature for the native loading ``x`` (gradient, or flux for ``subc``)."""
        key = self._key(x)
        m = self.mesh
        if self.native == "gradient":
            t_aff = times(m.nodes, x)
            rhs = -(self._stiffness() @ t_aff)
            scale = norm(self._stiffness().diagonal() * t_aff)
        else:
            t_aff = np.zeros(m.n_nodes)
            rhs = self._flux_load(x)
            scale = None
        w, info = self._solve(key, rhs, scale)
        return t_aff + w, info

    def _sums(self, temp: np.ndarray) -> np.ndarray:
        m = self.mesh
        gp, gw = GAUSS[m.order]
        return _thermal_sums(
            m.nodes,
            m.elements,
            m.order,
            self.group,
            len(self.groups),
            self.mat,
            self.table.matrices,
            temp,
            gp,
            gw,
        )

    def _dual(self, temp: np.ndarray, x: np.ndarray, dt: float = 0.0) -> np.ndarray:
        """Average flux (gradient loadings) or gradient (flux loadings)."""
        total = self._sums(temp).sum(axis=0) / self.box_volume
        return (total[3:6] if self.native == "gradient" else total[0:3]).copy()

    def _evaluate(self, loading, temp, x, dt, info, fields, t0) -> Solution:
        m = self.mesh
        sums = self._sums(temp)
        total = sums.sum(axis=0)
        if self.native == "gradient":
            grad = np.array(x, dtype=float)
        else:
            grad = total[0:3] / self.box_volume
        fld = None
        if fields:
            gp, gw = GAUSS[m.order]
            eg, ef, ed = _thermal_fields(
                m.nodes, m.elements, m.order, self.mat, self.table.matrices, temp, gp, gw
            )
            fld = {"gradient": eg, "flux": ef, "dissipation": ed, "temperature": temp}
        info.solve_s = time.perf_counter() - t0
        return Solution(
            physics=self.physics,
            bc=self.bc,
            loading=loading,
            strain=grad,
            stress=total[3:6] / self.box_volume,
            energy=float(total[6] / self.box_volume),
            material_fraction=float(total[14] / self.box_volume),
            phases=self._phase_statistics(sums, 7, ("gradient", "flux", "dissipation"), (3, 3, 1)),
            info=info,
            fields=fld,
            support=m if fields else None,
        )

    def _flux_load(self, flux: np.ndarray) -> np.ndarray:
        m = self.mesh
        load = np.zeros(m.n_nodes)
        faces, loads = _face_loads(m)
        np.add.at(load, faces, -(loads @ flux))
        return load
