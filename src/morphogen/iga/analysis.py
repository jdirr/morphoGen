"""Isogeometric analysis of linear elasticity and conduction on NURBS assemblies.

The isoparametric concept of Hughes, Cottrell & Bazilevs (2005): the rational functions
``R_a`` that describe the geometry (:mod:`morphogen.iga.geometry`) interpolate the
unknowns too, ``u = sum_a R_a u_a``, the unknowns attached to the control points. The
geometry of the analysis is that of the design — a circle stays a circle at any
refinement —, and the functions are smooth across the elements of a patch (``C^(p-1)``
after a *k*-refinement), which gives more accuracy per unknown than finite elements of
the same degree.

Discretization. The elements are the knot spans of each patch; the integrals are those
of Gauss-Legendre with ``p + 1`` points per direction. At a point, the derivatives of the
functions with respect to the parameters give the Jacobian ``J = sum_a P_a (x) dR_a/dxi``
and the gradients ``dR_a/dx = dR_a/dxi J^-1``; the stiffness is ``K = sum int B^T D B det
J dxi``, as for finite elements, with ``B`` the strain operator of these gradients.
Strains and stresses are written in Voigt's notation with engineering shears: ``(11, 22,
12)`` in the plane, ``(11, 22, 33, 23, 31, 12)`` in space (the order of the rest of
morphoGen). In the plane, ``plane="strain"`` takes the stiffness of the plane components
of the three-dimensional law, ``plane="stress"`` its condensation at zero out-of-plane
stresses.

Conditions. The functions of an open knot vector interpolate at the ends of a patch:
the control points of a side carry the values of the unknown along it, so that a
uniform value (a clamped or symmetric side) is prescribed on them exactly; tractions
are integrated on the sides.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from morphogen.iga import basis as bs
from morphogen.iga.geometry import Multipatch, Patch

PLANES = ("strain", "stress")
PHYSICS = ("elastic", "conduction")
#: Elements of a patch integrated at once (memory of the arrays of their points)
_CHUNK = 4096


def elastic_matrix(material, ndim: int, plane: str = "strain") -> np.ndarray:
    """The stiffness of a material in Voigt's notation for an analysis of ``ndim``
    dimensions: the 6 x 6 matrix of a solid; in the plane, its plane components (plane
    strain) or its condensation at zero out-of-plane stresses (plane stress). ``material``:
    a :class:`~morphogen.homogenization.materials.Material`, or a 6 x 6 (or, in the plane,
    3 x 3) array."""
    if hasattr(material, "matrix"):
        full = np.asarray(material.matrix("elastic"), dtype=float)
    else:
        full = np.asarray(material, dtype=float)
    if ndim == 3:
        if full.shape != (6, 6):
            raise ValueError("elastic_matrix: a 6 x 6 stiffness for a solid")
        return full
    if full.shape == (3, 3):
        return full
    if full.shape != (6, 6):
        raise ValueError("elastic_matrix: a 6 x 6 stiffness, or 3 x 3 in the plane")
    if plane not in PLANES:
        raise ValueError(f"plane: one of {PLANES}")
    kept, out = [0, 1, 5], [2, 3, 4]
    if plane == "strain":
        return full[np.ix_(kept, kept)]
    coupling = full[np.ix_(kept, out)]
    return full[np.ix_(kept, kept)] - coupling @ np.linalg.solve(
        full[np.ix_(out, out)], full[np.ix_(out, kept)]
    )


def conduction_matrix(material, ndim: int) -> np.ndarray:
    """The conductivity tensor of a material for an analysis of ``ndim`` dimensions: a
    :class:`~morphogen.homogenization.materials.Material`, a number or a tensor."""
    if hasattr(material, "matrix"):
        full = np.asarray(material.matrix("thermal"), dtype=float)
    else:
        full = np.asarray(material, dtype=float)
    if full.ndim == 0:
        return float(full) * np.eye(ndim)
    return full[:ndim, :ndim]


@dataclass
class Points:
    """The integration points of a group of elements of a patch: the control points of
    each element (``nodes``: elements x functions), the weights of the points times the
    Jacobian (``weights``: elements x points), their positions, the functions there and
    their gradients (elements x points x functions (x dim)), and the Jacobian
    determinants."""

    patch: int
    nodes: np.ndarray
    weights: np.ndarray
    x: np.ndarray
    values: np.ndarray
    gradients: np.ndarray
    jacobian: np.ndarray


def _directions(patch: Patch, extra: int):
    """For each direction of a patch: the functions and their derivatives at the Gauss
    points of its elements (elements x points x degree + 1), the weights of the points
    and the indices of these functions (elements x degree + 1)."""
    out = []
    for knots, degree in zip(patch.knots, patch.degrees, strict=True):
        ends = bs.breakpoints(knots)
        points, weights = bs.gauss(degree + 1 + extra)
        u = ends[:-1, None] + np.diff(ends)[:, None] * points[None, :]
        span, values = bs.basis(knots, degree, u.ravel(), 1)
        shape = (len(ends) - 1, len(points), degree + 1)
        index = span.reshape(u.shape)[:, :1] - degree + np.arange(degree + 1)[None, :]
        out.append((values[0].reshape(shape), values[1].reshape(shape),
                    np.diff(ends)[:, None] * weights[None, :], index))  # fmt: skip
    return out


def _tensor(directions, which: int | None, rows: np.ndarray) -> np.ndarray:
    """The products of the functions of the directions for the elements ``rows`` of the
    first one (all those of the others), the direction ``which`` differentiated:
    elements x points x functions."""
    factors = [d[1] if k == which else d[0] for k, d in enumerate(directions)]
    factors[0] = factors[0][rows]
    if len(factors) == 2:
        out = np.einsum("iga,jhb->ijghab", *factors)
    else:
        out = np.einsum("iga,jhb,kfc->ijkghfabc", *factors)
    n = len(factors)
    elements = int(np.prod(out.shape[:n]))
    points = int(np.prod(out.shape[n : 2 * n]))
    return out.reshape(elements, points, -1)


def patch_points(model: Multipatch, k: int, extra: int = 0) -> Iterator[Points]:
    """The integration points of patch ``k`` of an assembly, by groups of elements
    (``extra`` Gauss points per direction beyond ``degree + 1``)."""
    patch, ids = model.patches[k], model.ids[k]
    ndim = patch.ndim
    if patch.dim != ndim or ndim not in (2, 3):
        raise ValueError("analysis: surfaces of the plane or solids")
    directions = _directions(patch, extra)
    others = int(np.prod([len(d[2]) for d in directions[1:]]))
    step = max(1, _CHUNK // others)
    for start in range(0, len(directions[0][2]), step):
        rows = np.arange(start, min(start + step, len(directions[0][2])))
        index = [d[3] for d in directions]
        index[0] = index[0][rows]
        if ndim == 2:
            nodes = ids[index[0][:, None, :, None], index[1][None, :, None, :]]
            w = np.einsum("ig,jh->ijgh", directions[0][2][rows], directions[1][2])
        else:
            nodes = ids[index[0][:, None, None, :, None, None],
                        index[1][None, :, None, None, :, None],
                        index[2][None, None, :, None, None, :]]  # fmt: skip
            w = np.einsum("ig,jh,kf->ijkghf", directions[0][2][rows], directions[1][2],
                          directions[2][2])  # fmt: skip
        n_el = int(np.prod(nodes.shape[:ndim]))
        nodes = nodes.reshape(n_el, -1)
        w = w.reshape(n_el, -1)
        weight = model.weights[nodes]  # elements x functions
        values = _tensor(directions, None, rows) * weight[:, None, :]
        total = values.sum(axis=2)
        values /= total[:, :, None]
        derived = np.empty((*values.shape, ndim))
        for d in range(ndim):
            dn = _tensor(directions, d, rows) * weight[:, None, :]
            derived[..., d] = (dn - values * dn.sum(axis=2)[:, :, None]) / total[:, :, None]
        control = model.nodes[nodes]  # elements x functions x dim
        x = np.einsum("eqa,eak->eqk", values, control)
        jac = np.einsum("eqad,eak->eqkd", derived, control)
        det = np.linalg.det(jac)
        gradients = np.einsum("eqad,eqdk->eqak", derived, np.linalg.inv(jac))
        yield Points(k, nodes, w * det, x, values, gradients, det)


def strain_operator(gradients: np.ndarray) -> np.ndarray:
    """The strain operator ``B`` of the gradients of the functions (... x functions x
    dim): ... x strains x (functions dim), in Voigt's notation with engineering shears."""
    *lead, n, dim = gradients.shape
    rows = 3 if dim == 2 else 6
    b = np.zeros((*lead, rows, n, dim))
    if dim == 2:
        b[..., 0, :, 0] = gradients[..., 0]
        b[..., 1, :, 1] = gradients[..., 1]
        b[..., 2, :, 0] = gradients[..., 1]
        b[..., 2, :, 1] = gradients[..., 0]
    else:
        gx, gy, gz = gradients[..., 0], gradients[..., 1], gradients[..., 2]
        b[..., 0, :, 0] = gx
        b[..., 1, :, 1] = gy
        b[..., 2, :, 2] = gz
        b[..., 3, :, 1] = gz
        b[..., 3, :, 2] = gy
        b[..., 4, :, 0] = gz
        b[..., 4, :, 2] = gx
        b[..., 5, :, 0] = gy
        b[..., 5, :, 1] = gx
    return b.reshape(*lead, rows, n * dim)


class Analysis:
    """The discretization of an assembly for a physics (see the module): ``"elastic"``
    (``plane``: ``"strain"`` or ``"stress"`` for surfaces) or ``"conduction"``, with the
    ``materials`` of the phases of its patches (:class:`~morphogen.homogenization.
    materials.Material`, or their matrices); ``extra`` Gauss points per direction beyond
    ``degree + 1``."""

    def __init__(self, model: Multipatch, materials, *, physics: str = "elastic",
                 plane: str = "strain", extra: int = 0):  # fmt: skip
        if physics not in PHYSICS:
            raise ValueError(f"physics: one of {PHYSICS}")
        if isinstance(model, Patch):
            model = Multipatch(model)
        self.model = model
        self.physics = physics
        self.plane = plane
        self.extra = int(extra)
        self.dim = model.dim
        #: Unknowns per control point, and components of the gradient-like measure
        self.nd = self.dim if physics == "elastic" else 1
        missing = [p for p in model.phases if p not in materials]
        if missing:
            raise ValueError(f"no material for phase(s) {missing}")
        if physics == "elastic":
            self.matrices = {p: elastic_matrix(materials[p], self.dim, plane) for p in model.phases}
        else:
            self.matrices = {p: conduction_matrix(materials[p], self.dim) for p in model.phases}
        self.n_dofs = self.nd * model.n_nodes

    # -- the points and their operators ---------------------------------------------------

    def points(self) -> Iterator[tuple[Points, np.ndarray, np.ndarray, np.ndarray]]:
        """The groups of integration points with their operator ``B`` (strains, or
        gradients: elements x points x components x unknowns of the element), the
        matrix of their material and the unknowns of their elements."""
        for k, patch in enumerate(self.model.patches):
            for pts in patch_points(self.model, k, self.extra):
                if np.any(pts.jacobian <= 0.0):
                    raise ValueError(
                        f"patch {k}: the Jacobian of its geometry is not positive everywhere "
                        "(a patch turned over, or folded)"
                    )
                if self.physics == "elastic":
                    b = strain_operator(pts.gradients)
                else:
                    b = np.swapaxes(pts.gradients, 2, 3)
                dofs = (self.nd * pts.nodes[:, :, None] + np.arange(self.nd)).reshape(
                    len(pts.nodes), -1
                )
                yield pts, b, self.matrices[patch.phase], dofs

    def matrix(self) -> sp.csr_matrix:
        """The stiffness (conductivity) matrix of all the unknowns."""
        rows, cols, values = [], [], []
        for pts, b, d, dofs in self.points():
            ke = np.einsum("eq,eqsi,st,eqtj->eij", pts.weights, b, d, b, optimize=True)
            n = dofs.shape[1]
            rows.append(np.repeat(dofs, n, axis=1).ravel())
            cols.append(np.tile(dofs, (1, n)).ravel())
            values.append(ke.ravel())
        matrix = sp.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_dofs, self.n_dofs),
        )
        return matrix.tocsr()

    def measure(self) -> float:
        """The area (volume) of the assembly."""
        return float(sum(pts.weights.sum() for pts, _, _, _ in self.points()))

    def uniform_loads(self) -> tuple[np.ndarray, np.ndarray]:
        """The nodal forces ``int B^T D E`` of the unit uniform strains (gradients) ``E``
        (unknowns x components), and ``int D`` (components x components)."""
        n = next(iter(self.matrices.values())).shape[0]
        loads = np.zeros((self.n_dofs, n))
        total = np.zeros((n, n))
        for pts, b, d, dofs in self.points():
            fe = np.einsum("eq,eqsi,st->eit", pts.weights, b, d)
            for j in range(n):
                np.add.at(loads[:, j], dofs, fe[:, :, j])
            total += pts.weights.sum() * d
        return loads, total

    # -- loads ------------------------------------------------------------------------------

    def side_load(self, patch: int, axis: int, end: int, traction) -> np.ndarray:
        """The nodal forces of a traction (a flux, for conduction) on a side of a
        patch: a vector, or a function ``traction(x, n)`` of the points (points x dim)
        and of the outward unit normals there."""
        p = self.model.patches[patch]
        local = self.model.side_nodes(patch, axis, end)
        f = np.zeros((self.model.n_nodes, self.nd))
        values, x, tangents, w, nodes, params = _side_points(p.side(axis, end), self.extra)
        if self.dim == 2:
            normal = np.stack([tangents[..., 1, 0], -tangents[..., 0, 0]], axis=-1)
        else:
            normal = np.cross(tangents[..., 0], tangents[..., 1])
        size = np.linalg.norm(normal, axis=-1)
        normal = normal / size[..., None]
        # outwards: along the direction of the patch that leaves it through this side
        full = np.insert(params.reshape(-1, p.ndim - 1), axis, p.knots[axis][-end], axis=1)
        leaving = p.jacobian(full)[1][:, :, axis].reshape(x.shape) * (1.0 if end else -1.0)
        normal = normal * np.sign(np.einsum("eqk,eqk->eq", normal, leaving))[..., None]
        if callable(traction):
            t = np.asarray(traction(x.reshape(-1, self.dim), normal.reshape(-1, self.dim)))
            t = t.reshape(*x.shape[:2], self.nd)
        else:
            t = np.broadcast_to(np.asarray(traction, dtype=float), (*x.shape[:2], self.nd))
        fe = np.einsum("eq,eqa,eqc->eac", w * size, values, t)
        np.add.at(f, local[nodes], fe)
        return f.ravel()

    # -- solution ---------------------------------------------------------------------------

    def dofs(self, nodes, components=None) -> np.ndarray:
        """The unknowns of control points: all their components, or those given."""
        nodes = np.atleast_1d(np.asarray(nodes, dtype=np.int64))
        components = range(self.nd) if components is None else np.atleast_1d(components)
        return np.concatenate([self.nd * nodes + int(c) for c in components])

    def solve(self, forces, fixed, values=0.0, matrix=None) -> np.ndarray:
        """The unknowns under nodal ``forces`` with the unknowns ``fixed`` prescribed to
        ``values`` (0 by default)."""
        k = self.matrix() if matrix is None else matrix
        fixed = np.unique(np.asarray(fixed, dtype=np.int64))
        u = np.zeros(self.n_dofs)
        u[fixed] = values
        free = np.setdiff1d(np.arange(self.n_dofs), fixed)
        rhs = np.asarray(forces, dtype=float)[free] - (k[free][:, fixed] @ u[fixed])
        u[free] = spla.spsolve(k[free][:, free].tocsc(), rhs)
        return u

    # -- fields -----------------------------------------------------------------------------

    def fields(self, u, counts: int = 4) -> list[dict]:
        """For each patch, on a grid of parameters (``counts`` per element and direction):
        the positions, the unknown and, for the elastic problems, the strains and the
        stresses there (Voigt's notation); for conduction, the gradient and the flux."""
        u = np.asarray(u, dtype=float).reshape(-1, self.nd)
        out = []
        for k, patch in enumerate(self.model.patches):
            params = _grid(patch, counts)
            values, gradients, x = _functions_at(self.model, k, params)
            nodes = self.model.ids[k].ravel()
            d = self.matrices[patch.phase]
            if self.physics == "elastic":
                b = strain_operator(gradients)
                measure = b @ u[nodes].ravel()
                flux = measure @ d.T
                names = ("strain", "stress")
            else:
                measure = np.einsum("mak,a->mk", gradients, u[nodes, 0])
                flux = -measure @ d.T
                names = ("gradient", "flux")
            out.append(
                {
                    "x": x,
                    "u": values @ u[nodes],
                    names[0]: measure,
                    names[1]: flux,
                    "shape": params.shape[:-1],
                    "parameters": params.reshape(-1, patch.ndim),
                }
            )
        return out

    def at(self, u, patch: int, params) -> dict:
        """The position, the unknown and its measures at parameters of a patch (points x
        ndim), as :meth:`fields`."""
        params = np.atleast_2d(np.asarray(params, dtype=float))
        u = np.asarray(u, dtype=float).reshape(-1, self.nd)
        values, gradients, x = _functions_at(self.model, patch, params)
        nodes = self.model.ids[patch].ravel()
        d = self.matrices[self.model.patches[patch].phase]
        if self.physics == "elastic":
            measure = strain_operator(gradients) @ u[nodes].ravel()
            return {"x": x, "u": values @ u[nodes], "strain": measure, "stress": measure @ d.T}
        measure = np.einsum("mak,a->mk", gradients, u[nodes, 0])
        return {"x": x, "u": values @ u[nodes], "gradient": measure, "flux": -measure @ d.T}

    def error(self, u, exact: Callable | None = None, exact_stress: Callable | None = None):
        """The errors of a solution against an exact one: ``l2``, the relative error of
        the unknown in the mean square (``exact(x)``: points x unknowns), and ``energy``,
        that of the stresses in the energy norm (``exact_stress(x)``: points x
        components, Voigt's notation); each for the function given."""
        u = np.asarray(u, dtype=float)
        num = den = num_e = den_e = 0.0
        for pts, b, d, dofs in self.points():
            flat = pts.x.reshape(-1, self.dim)
            if exact is not None:
                ref = np.asarray(exact(flat)).reshape(*pts.x.shape[:2], self.nd)
                ue = u[dofs].reshape(len(dofs), -1, self.nd)
                uh = np.einsum("eqa,eac->eqc", pts.values, ue)
                num += float(np.einsum("eq,eqc,eqc->", pts.weights, uh - ref, uh - ref))
                den += float(np.einsum("eq,eqc,eqc->", pts.weights, ref, ref))
            if exact_stress is not None:
                sigma = np.einsum("st,eqti,ei->eqs", d, b, u[dofs])
                ref_s = np.asarray(exact_stress(flat)).reshape(sigma.shape)
                compliance = np.linalg.inv(d)
                diff = sigma - ref_s
                num_e += float(np.einsum("eq,eqs,st,eqt->", pts.weights, diff, compliance, diff))
                den_e += float(np.einsum("eq,eqs,st,eqt->", pts.weights, ref_s, compliance, ref_s))
        out = {}
        if exact is not None:
            out["l2"] = float(np.sqrt(num / den))
        if exact_stress is not None:
            out["energy"] = float(np.sqrt(num_e / den_e))
        return out


def _grid(patch: Patch, counts) -> np.ndarray:
    """A grid of parameters of a patch: ``counts`` per element and direction, the ends
    of the elements included (... x ndim)."""
    axes = []
    for knots in patch.knots:
        ends = bs.breakpoints(knots)
        inside = ends[:-1, None] + np.diff(ends)[:, None] * np.arange(counts)[None, :] / counts
        axes.append(np.append(inside.ravel(), ends[-1]))
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def _functions_at(model: Multipatch, k: int, params):
    """The rational functions of all the control points of patch ``k`` and their
    gradients at parameters (points x functions (x dim)), and the positions."""
    patch = model.patches[k]
    params = np.asarray(params, dtype=float).reshape(-1, patch.ndim)
    letters = "abc"[: patch.ndim]
    spec = ",".join(f"m{c}" for c in letters) + f"->m{letters}"
    weight = patch.weights.ravel()

    def product(which):
        matrices = [
            bs.collocation(patch.knots[d], patch.degrees[d], params[:, d], int(d == which))
            for d in range(patch.ndim)
        ]
        return np.einsum(spec, *matrices).reshape(len(params), -1) * weight[None, :]

    values = product(None)
    total = values.sum(axis=1)
    values /= total[:, None]
    derived = np.empty((*values.shape, patch.ndim))
    for d in range(patch.ndim):
        dn = product(d)
        derived[..., d] = (dn - values * dn.sum(axis=1)[:, None]) / total[:, None]
    control = model.nodes[model.ids[k].ravel()]
    x = values @ control
    jac = np.einsum("mad,ak->mkd", derived, control)
    gradients = np.einsum("mad,mdk->mak", derived, np.linalg.inv(jac))
    return values, gradients, x


def _side_points(side: Patch, extra: int):
    """The integration points of a side (a curve or a surface in a space of one more
    dimension): the functions (elements x points x functions), the positions, the
    tangents (elements x points x dim x ndim), the weights of the points, the control
    points of the elements (indices in the raveled side) and the parameters of the
    points (elements x points x ndim)."""
    directions = _directions(side, extra)
    ids = np.arange(side.weights.size).reshape(side.shape)
    parameters = []
    for knots, degree in zip(side.knots, side.degrees, strict=True):
        ends = bs.breakpoints(knots)
        points, _ = bs.gauss(degree + 1 + extra)
        parameters.append(ends[:-1, None] + np.diff(ends)[:, None] * points[None, :])
    if side.ndim == 1:
        n, dn, w, index = directions[0]
        nodes = ids[index]
        values, derived = n.copy(), dn[..., None].copy()
        params = parameters[0][..., None]
    else:
        rows = np.arange(len(directions[0][2]))
        index = [d[3] for d in directions]
        nodes = ids[index[0][:, None, :, None], index[1][None, :, None, :]]
        nodes = nodes.reshape(-1, nodes.shape[2] * nodes.shape[3])
        w = np.einsum("ig,jh->ijgh", directions[0][2], directions[1][2]).reshape(len(nodes), -1)
        values = _tensor(directions, None, rows)
        derived = np.stack([_tensor(directions, d, rows) for d in range(2)], axis=-1)
        first = np.broadcast_to(parameters[0][:, None, :, None],
                                (len(parameters[0]), len(parameters[1]),
                                 parameters[0].shape[1], parameters[1].shape[1]))  # fmt: skip
        second = np.broadcast_to(parameters[1][None, :, None, :], first.shape)
        params = np.stack([first, second], axis=-1).reshape(len(nodes), -1, 2)
    weight = side.weights.ravel()[nodes]
    values = values * weight[:, None, :]
    total = values.sum(axis=2)
    values /= total[:, :, None]
    for d in range(side.ndim):
        dn = derived[..., d] * weight[:, None, :]
        derived[..., d] = (dn - values * dn.sum(axis=2)[:, :, None]) / total[:, :, None]
    control = side.points.reshape(-1, side.dim)[nodes]
    x = np.einsum("eqa,eak->eqk", values, control)
    tangents = np.einsum("eqad,eak->eqkd", derived, control)
    return values, x, tangents, w, nodes, params


__all__ = [
    "PHYSICS",
    "PLANES",
    "Analysis",
    "Points",
    "conduction_matrix",
    "elastic_matrix",
    "patch_points",
    "strain_operator",
]
