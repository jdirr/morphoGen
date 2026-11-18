"""Mixed-mode cracks in 3D, in small and finite strains: the energy release rate along the
front by the G-θ method, its partition into the modes I, II and III by the interaction
integrals, the direction of kinking, and the propagation of kinked cracks on meshes
conforming to them, rebuilt by MMG at each step (as Z-cracks).

*Problem.* The crack (a planar :class:`~morphogen.fracture.lefm.Crack` or a
:class:`KinkedCrack`) is opened on a mesh conforming to it (its lips duplicated,
:func:`~morphogen.fracture.lefm.crack_faces`), the displacements prescribed on node sets;
the bulk is linear elastic in small strains or a compressible neo-Hookean solid in finite
strains (total Lagrangian), the lips free or in frictionless contact (a penalty against
their interpenetration, under shear and closure): :func:`solve_crack`.

*Energy release rate* (the G-θ method of Destuynder & Djaoua 1981, as in Code_Aster and
Z-set): for a virtual advance ``θ = q(r) w(s) d`` of the front (``d`` the direction of
propagation, ``q`` from 1 within ``r_in`` of the front to 0 beyond ``r_out``, ``w`` a weight
along the front), ``∫ G w ds = ∫ (Hᵀ P - W I) : ∇θ dV`` with the displacement gradient
``H``, the first Piola-Kirchhoff stress ``P`` and the energy density ``W`` of the reference
configuration (Eshelby's energy-momentum tensor): exact for hyperelastic solids in finite
strains, the J-integral in small strains. The weights are the piecewise linear functions
of stations along the front, giving ``G(s)`` (a small system with the mass matrix of the
front), or 1, giving its mean. ``θ`` stays tangent to the lips: its domain does not reach
the mouth of the crack, nor the last kink of a kinked one. The check: the energy of the
body at fixed prescribed displacements, ``G = -dU/dA`` (:func:`energy_release_rate`).

*Modes* (the interaction integral of Yau, Wang & Corten 1980, in 3D as Moran & Shih 1987):
with the same ``θ``, the cross term of the energy-momentum tensors of the body and of the
asymptotic field of a unit factor of each mode (plane strain for I and II, antiplane for
III, in the axes ``(d, n, t = d × n)`` of the front), ``M = ∫ (σ_ij u'_i,k + σ'_ij u_i,k -
σ' : ε δ_kj) θ_k,j dV``, gives ``K_I = E' M_I / 2``, ``K_II = E' M_II / 2`` and ``K_III =
μ M_III`` (``E' = E / (1 - ν²)``, the isotropic elasticity of the material at the front,
the Voigt-Reuss-Hill estimate of an anisotropic one such as a crystal):
more accurate than the jumps of the lips, whose fit by ``sqrt(r)`` on linear elements
biases the mixity. Along a curved front the auxiliary fields, those of a straight front in
the turning axes of the front, are neither compatible nor balanced: the divergence of the
cross tensor, ``∫ (σ_ij (u'_i,kj - ε'_ij,k) + σ'_ij,j u_i,k) θ_k dV``, is added (Gosz &
Moran 2002), which makes the factors independent of the domain (zero along a straight
front, where it is skipped). Behind the front the auxiliary fields of an element are those
of its side of the crack (the angle continued across the plane of the crack): an element
flat along the lips, whose centroid lies within a step of differences of the plane (or
across it, where the mesh moved the lips), would otherwise difference the two sides of the
auxiliary displacement. The energy release rate of the G-θ method is shared in the
proportions ``K_I² / E'``, ``K_II² / E'``, ``K_III² / 2μ`` (in small strains, it converges
with their sum). The mean rotation of the domain, which adds nothing to the integrals, is
removed from the displacement gradient (the auxiliary stress is not in discrete
equilibrium: without it, a rigid rotation would bias the factors). In finite strains,
where the modes do not separate, the integral takes the stress and the displacement
gradient turned back by the rotation of the polar decomposition of the mean deformation
gradient around the front: an estimate of the mixity of the remote field. Lips in contact
carry tractions, whose work in the auxiliary fields, ``∫ t · (∇u' θ) dS`` over the lips,
is subtracted (their work in the actual field, that of the penalty, is neglected in
``G``); frictionless, it only concerns the mode I, which lips closed at the front do not
have: ``K_I ≥ 0`` (the closure of the singular integral near the front being the least
accurate part).

*Kinking* (Erdogan & Sih 1963): the direction of the largest hoop stress, ``θ = 2 atan((K_I
- sqrt(K_I² + 8 K_II²)) / (4 K_II))`` from ``d`` towards ``n`` (-70.5° in pure mode II); the
tearing mode III, which twists the front, counts in ``G`` but does not turn the path here.

*Morphologies* (crack meshes whose phases are those of the materials and whose sides of the
crack are apart, :func:`~morphogen.fracture.lefm.crack_mesh` with ``morphology``): the
material at the front is that of the elements in which the points of the front lie; the
interaction integrals, their ``G`` and the driving force at the tip (``energy="tip"``) take
``θ`` zero beyond ``r_in`` at the nodes outside that material (its interfaces with the
others, the other materials, the walls of the pores), so that they are those of the
material at the front, the configurational forces of the inclusions and of the pores in the
domain left out (the inhomogeneity term of Simha, Fischer, Kolednik & Chen 2003); the far J
(``"work"``, ``"free"``) keeps them. A front grown into a pore goes through it
(:func:`through_voids`).

*Inelastic bulks* (``bulk="points"``, or automatically when a phase is inelastic: the
behaviours of morphoGen's points, :mod:`morphogen.fracture.bulk`: von Mises
elasto-viscoplasticity, crystal plasticity, MFront laws, in small and finite strains).
The energy-momentum tensor then takes either the work density ``w = ∫ P : dF`` (the J of
the deformation theory of plasticity, independent of the domain under monotonic
proportional loading, and the release of the work of the loads at fixed displacements,
``-dW/dA``, the J of Begley & Landes), or the free energy ``ψ``: its divergence is then
``-∂ψ/∂α · ∇α`` in the zone where the internal variables ``α`` vary, so that ``∫ (Hᵀ P - ψ
I) : ∇θ dV`` grows with the domain, and adding ``-∫ ∂ψ/∂α · ∇α · θ dV`` gives the driving force
at the tip (the configurational force of Simha, Fischer, Shan, Chen & Kolednik 2008),
smaller than the far J by the shielding of the plastic zone: ``energy="work"``, ``"free"``
or ``"tip"`` of :meth:`CrackSolution.release_rate` (the same for elastic bodies).

*Propagation* (:func:`propagate_kinked`): at each step the body is meshed again conforming
to the crack (MMG through the signed distance to its path), solved, its ``G`` and modes
computed, and its front advanced by an increment in the direction of kinking (by Paris'
law ``da/dN = C ΔK^m``, ``ΔK = sqrt(E' G)``, the cycles of the increment). With a toughness
of the phases and of their interfaces (:mod:`morphogen.fracture.toughness`), each advance is
the candidate of the largest ``G(θ) / Γ(θ)``, the kink of the bulk or a deflection along an
interface ahead (the first-order kinked factors of Cotterell & Rice), and each crack has the
critical load factor ``sqrt(Γ / G)`` of its advance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from morphogen.fracture.cohesive import (
    CohesiveProblem,
    CohesiveResult,
    Contact,
    Interfaces,
    _tet_gradients,
    lip_tractions,
)
from morphogen.fracture.lefm import crack_faces, line_in_box
from morphogen.mesh.tetra import TetMesh

_COUNT = 400  # points of the front
#: The three points of a face of the lips (those of its interface element)
_LIP_POINTS = ((1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3))
#: The points and weights of Gauss on [0, 1] of the graded rule of the lips
_LIP_GAUSS = (
    0.5 * (np.polynomial.legendre.leggauss(5)[0] + 1.0),
    0.5 * np.polynomial.legendre.leggauss(5)[1],
)


@dataclass(frozen=True)
class KinkedCrack:
    """A crack with a straight front along ``axis``, whose path in the planes normal to the
    front is the polyline of ``points`` (from its mouth, on the boundary of the body, to its
    front): an edge or through crack that kinks as it grows (:meth:`grown`); the normal of
    each segment ``e`` is ``axis × e`` (towards the side ``+``)."""

    points: tuple
    axis: tuple = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        t = np.asarray(self.axis, dtype=float)
        t = t / np.linalg.norm(t)
        p = np.atleast_2d(np.asarray(self.points, dtype=float))
        p = p - np.outer((p - p[0]) @ t, t)  # in the plane of the first point
        if len(p) < 2 or np.any(np.linalg.norm(np.diff(p, axis=0), axis=1) <= 0):
            raise ValueError("KinkedCrack: at least two distinct points")
        object.__setattr__(self, "axis", tuple(t))
        object.__setattr__(self, "points", tuple(map(tuple, p)))

    @classmethod
    def of(cls, crack) -> KinkedCrack:
        """The kinked crack of a straight planar :class:`~morphogen.fracture.lefm.Crack`."""
        p, d, n = (np.asarray(v, dtype=float) for v in (crack.point, crack.direction, crack.normal))
        return cls((tuple(p), tuple(p + crack.size * d)), tuple(np.cross(d, n)))

    def _segments(self):
        p = np.asarray(self.points)
        e = np.diff(p, axis=0)
        lengths = np.linalg.norm(e, axis=1)
        e = e / lengths[:, None]
        return p, e, lengths, np.cross(np.asarray(self.axis), e)

    def _plane(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        t = np.asarray(self.axis)
        return x - np.outer((x - self.points[0]) @ t, t)

    @property
    def size(self) -> float:
        """The length of the path."""
        return float(self._segments()[2].sum())

    @property
    def last(self) -> float:
        """The length of the last segment."""
        return float(self._segments()[2][-1])

    @property
    def tip(self) -> np.ndarray:
        return np.asarray(self.points[-1])

    def phi(self, x) -> np.ndarray:
        """Signed distance to the path, its first and last segments extended. The points
        nearest a kink take the side of the sum of the normals of its two segments (its
        pseudo-normal; that of one segment turns the side of the points ahead of it beyond
        a kink of 90° or more, such as a deflection along an interface)."""
        y = self._plane(x)
        p, e, lengths, normals = self._segments()
        best = np.full(len(y), np.inf)
        value = np.zeros(len(y))
        last = len(e) - 1
        for i in range(len(e)):
            tau = (y - p[i]) @ e[i]
            low = -np.inf if i == 0 else 0.0
            high = np.inf if i == last else lengths[i]
            v = y - p[i] - np.outer(np.clip(tau, low, high), e[i])
            dist = np.linalg.norm(v, axis=1)
            side = v @ normals[i]
            if i > 0:  # nearest the kink at its start
                side = np.where(tau < low, v @ (normals[i - 1] + normals[i]), side)
            if i < last:  # nearest the kink at its end
                side = np.where(tau > high, v @ (normals[i] + normals[i + 1]), side)
            closer = dist < best
            best[closer] = dist[closer]
            value[closer] = np.where(side >= 0.0, dist, -dist)[closer]
        return value

    def psi(self, x) -> np.ndarray:
        """Distance beyond the front along the last segment (negative behind it)."""
        _, e, _, _ = self._segments()
        return (self._plane(x) - self.tip) @ e[-1]

    def local(self, x):
        """The coordinates of points along the axes ``d`` and ``n`` of the front."""
        _, e, _, normals = self._segments()
        y = self._plane(x) - self.tip
        return y @ e[-1], y @ normals[-1]

    def frame(self, x):
        """As :meth:`~morphogen.fracture.lefm.Crack.frame`: the distance ``r`` to the front,
        the axes ``n``, ``d``, ``t`` of the last segment and the coordinate ``s`` along the
        front."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        _, e, _, normals = self._segments()
        r = np.hypot(*self.local(x))
        axes = (normals[-1], e[-1], np.asarray(self.axis))
        n, d, t = (np.broadcast_to(a, x.shape).copy() for a in axes)
        return r, n, d, t, (x - self.points[0]) @ np.asarray(self.axis)

    def front(self, lower, upper, count: int = _COUNT):
        """Points of the front inside the box ``[lower, upper]`` and their lengths."""
        return line_in_box(self.tip, np.asarray(self.axis), lower, upper, count)

    def grown(self, da: float, angle: float = 0.0) -> KinkedCrack:
        """The crack advanced by ``da`` at ``angle`` (radians) from its last segment towards
        its normal (a new segment, or the last one longer if ``angle`` is 0), or its last
        segment shortened by ``-da`` (``da < 0``). The new front is rounded to 10⁻¹⁰: the
        meshes of MMG change with the last digits of the crack, so that rounding makes a
        growth reproducible whatever the load at which its angles were computed."""
        _, e, lengths, normals = self._segments()
        da = float(da)
        if da < 0 and (angle or -da >= (1 - 1e-9) * lengths[-1]):
            raise ValueError("KinkedCrack.grown: shortened within its last segment, straight")
        if da <= 0 or not angle:
            tip = np.round(self.tip + da * e[-1], 10)
            return KinkedCrack((*self.points[:-1], tuple(tip)), self.axis)
        step = math.cos(angle) * e[-1] + math.sin(angle) * normals[-1]
        return KinkedCrack((*self.points, tuple(np.round(self.tip + da * step, 10))), self.axis)


def _reach(crack) -> float:
    """How far from the front the domain of ``θ`` may go, tangent to the lips."""
    if hasattr(crack, "reach"):
        return float(crack.reach)
    return crack.last if isinstance(crack, KinkedCrack) else float(crack.size)


def _closed(crack) -> bool:
    """Whether the front of a crack is a closed curve (a penny, a surface crack inside
    the body)."""
    return bool(getattr(crack, "closed", getattr(crack, "kind", None) == "penny"))


def _curved(crack) -> bool:
    """Whether the front of a crack turns (the auxiliary fields then turn with it)."""
    return bool(getattr(crack, "curved", getattr(crack, "kind", None) == "penny"))


def _hats(s, at, closed: bool) -> np.ndarray:
    """The piecewise linear functions of the stations ``at`` at the coordinates ``s``
    (points x stations; angles of a closed front); ``at`` None: one function, 1."""
    s = np.asarray(s, dtype=float)
    if at is None:
        return np.ones((len(s), 1))
    if closed:
        h = 2 * np.pi / len(at)
        dist = np.abs((s[:, None] - at[None, :] + np.pi) % (2 * np.pi) - np.pi)
    else:
        h = at[1] - at[0]
        dist = np.abs(s[:, None] - at[None, :])
    return np.clip(1.0 - dist / h, 0.0, None)


def _local(crack, x):
    """The coordinates of points along the axes ``d`` and ``n`` of the front."""
    if hasattr(crack, "local"):
        return crack.local(x)
    return crack.psi(x), crack.phi(x)


def _auxiliary(x1, x2, mu: float, nu: float, reference=None) -> np.ndarray:
    """The displacements (points x modes x 3) of the asymptotic fields of unit factors
    ``K_I``, ``K_II``, ``K_III`` at the coordinates ``(x1, x2)`` along ``(d, n)`` (plane
    strain; Williams); ``reference``: angles from which the angles of the points are
    continued (across the plane of the crack behind the front: the side of each point)."""
    r = np.hypot(x1, x2)
    angle = np.arctan2(x2, x1)
    if reference is not None:
        angle = angle + 2 * np.pi * np.round((reference - angle) / (2 * np.pi))
    half = 0.5 * angle
    c = np.sqrt(r / (2 * np.pi)) / (2 * mu)
    sin, cos = np.sin(half), np.cos(half)
    kappa = 3 - 4 * nu
    u = np.zeros((len(r), 3, 3))
    u[:, 0, 0] = c * cos * (kappa - 1 + 2 * sin**2)
    u[:, 0, 1] = c * sin * (kappa + 1 - 2 * cos**2)
    u[:, 1, 0] = c * sin * (kappa + 1 + 2 * cos**2)
    u[:, 1, 1] = -c * cos * (kappa - 1 - 2 * sin**2)
    u[:, 2, 2] = 4 * c * sin
    return u


def _auxiliary_gradients(x1, x2, mu: float, nu: float, reference=None) -> np.ndarray:
    """The gradients (points x modes x 3 x 3, in the axes of the front) of the auxiliary
    displacements (central differences, on the branch of ``reference``, by default that of
    the points)."""
    step = 1e-5 * np.maximum(np.hypot(x1, x2), 1e-300)
    reference = np.arctan2(x2, x1) if reference is None else reference
    g = np.zeros((len(step), 3, 3, 3))
    for j, (a, b) in enumerate(((1.0, 0.0), (0.0, 1.0))):
        plus = _auxiliary(x1 + a * step, x2 + b * step, mu, nu, reference)
        minus = _auxiliary(x1 - a * step, x2 - b * step, mu, nu, reference)
        g[:, :, :, j] = (plus - minus) / (2 * step)[:, None, None]
    return g


def _auxiliary_fields(crack, x, mu: float, nu: float, reference=None):
    """The gradients of the auxiliary displacements and their stresses (points x modes x 3
    x 3, global axes) at the points ``x``: the fields of a straight front in the axes of
    the nearest point of the front (on the branches of ``reference``)."""
    local = _auxiliary_gradients(*_local(crack, x), mu, nu, reference)
    strain = 0.5 * (local + np.swapaxes(local, 2, 3))
    lam = 2 * mu * nu / (1 - 2 * nu)
    stress = 2 * mu * strain + lam * np.einsum("emii->em", strain)[:, :, None, None] * np.eye(3)
    _, n, d, t, _ = crack.frame(x)
    axes = np.stack([d, n, t], axis=2)  # columns: the axes of the front
    turn = "eai,emij,ebj->emab"
    return np.einsum(turn, axes, local, axes), np.einsum(turn, axes, stress, axes)


def _auxiliary_derivatives(crack, x, mu: float, nu: float, reference):
    """The derivatives along the global axes (last index) of the auxiliary gradients and
    stresses (central differences, on the branches of ``reference``: a step across the
    plane of the crack behind the front, from an element flat along the lips, would
    difference the two sides): terms of the curvature of a curved front."""
    step = 1e-3 * np.maximum(crack.frame(x)[0], 1e-300)
    dg = np.empty((len(x), 3, 3, 3, 3))
    ds = np.empty_like(dg)
    for j in range(3):
        shift = np.zeros(3)
        shift[j] = 1.0
        g_plus, s_plus = _auxiliary_fields(crack, x + step[:, None] * shift, mu, nu, reference)
        g_minus, s_minus = _auxiliary_fields(crack, x - step[:, None] * shift, mu, nu, reference)
        dg[..., j] = (g_plus - g_minus) / (2 * step)[:, None, None, None]
        ds[..., j] = (s_plus - s_minus) / (2 * step)[:, None, None, None]
    return dg, ds


def _auxiliary_along_lips(x1, side: float, mu: float, nu: float) -> np.ndarray:
    """The derivatives along ``d`` (points x modes x 3, in the axes of the front) of the
    auxiliary displacements on the side ``+`` (``side`` 1) or ``-`` (-1) of the lips."""
    step = 1e-5 * np.maximum(np.abs(x1), 1e-300)
    on = np.full(len(step), 0.0 if side > 0 else -0.0)  # the angle ±π behind the front
    plus = _auxiliary(x1 + step, on, mu, nu)
    minus = _auxiliary(x1 - step, on, mu, nu)
    return (plus - minus) / (2 * step)[:, None, None]


def element_sides(mesh: TetMesh, crack) -> np.ndarray:
    """The side of a crack of each element of a mesh (1 or -1): that of its phase on a mesh
    of :func:`~morphogen.fracture.lefm.crack_mesh` (the median sign of ``φ`` over it), or of
    its label ``mesh.side`` on a crack mesh of a morphology, otherwise the sign of ``φ`` at
    its centroid."""
    phi = crack.phi(mesh.nodes[mesh.elements[:, :4]].mean(axis=1))
    labels = None
    if mesh.side is not None:
        labels = mesh.side
    elif mesh.metadata.get("crack_sides") == "phase":
        labels = mesh.phase
    if labels is not None:
        side = np.ones(mesh.n_elements)
        for p in np.unique(labels):
            sel = labels == p
            side[sel] = 1.0 if np.median(phi[sel]) >= 0 else -1.0
        return side
    return np.where(phi >= 0, 1.0, -1.0)


def _same_material(a, b) -> bool:
    if a is b:
        return True
    try:
        return bool(a == b)
    except (TypeError, ValueError):  # materials that do not compare
        return False


def material_groups(phase, materials: dict) -> np.ndarray:
    """The material of each element (of its ``phase``) as a label: the first phase of an
    equal material (phases of the same material make one group)."""
    phase = np.asarray(phase, dtype=np.int64)
    out = phase.copy()
    first: list[int] = []
    for p in np.unique(phase).tolist():
        rep = next((q for q in first if _same_material(materials.get(q), materials.get(p))), p)
        if rep == p:
            first.append(p)
        out[phase == p] = rep
    return out


def material_nodes(mesh: TetMesh, materials: dict, phase: int, lips=None) -> np.ndarray:
    """The nodes (a mask) inside the material of the phase ``phase`` of a crack mesh of a
    morphology: those whose elements all have it (the phases of the same material alike),
    off the walls of the pores (the free faces off the faces of the box and off the
    ``lips``, an :class:`~morphogen.fracture.cohesive.Interfaces` of the opened crack)."""
    from morphogen.fracture.lefm import boundary_faces

    tets = mesh.elements[:, :4]
    material = materials[int(phase)]
    same = [int(p) for p in np.unique(mesh.phase)
            if int(p) in materials and _same_material(materials[int(p)], material)]  # fmt: skip
    other = ~np.isin(mesh.phase, same)
    inside = np.ones(mesh.n_nodes, dtype=bool)
    inside[tets[other].ravel()] = False
    free = boundary_faces(mesh)
    x, tol = mesh.nodes[free], mesh.tolerance()
    wall = np.ones(len(free), dtype=bool)
    for axis in range(3):
        for value in (mesh.lower[axis], mesh.upper[axis]):
            wall &= ~np.all(np.abs(x[:, :, axis] - value) <= tol, axis=1)
    if lips is not None and lips.n_elements:
        keys = {tuple(k) for k in np.sort(np.vstack([lips.minus, lips.plus]), axis=1).tolist()}
        wall &= ~np.array([tuple(k) in keys for k in np.sort(free, axis=1).tolist()], dtype=bool)
    inside[free[wall].ravel()] = False
    return inside


def _isotropic_moduli(material) -> tuple[float, float]:
    """Young's modulus and Poisson's ratio of an isotropic material, or the isotropic
    estimate of an anisotropic one (a crystal): the means of the Voigt and Reuss averages
    of its bulk and shear moduli (Hill)."""
    if material.young is not None and material.poisson is not None:
        return float(material.young), float(material.poisson)
    from morphogen.homogenization.materials import reuss_average, voigt_average

    d = material.matrix("elastic")
    (kv, gv), (kr, gr) = voigt_average(d), reuss_average(d)
    k, g = 0.5 * (kv + kr), 0.5 * (gv + gr)
    return 9.0 * k * g / (3.0 * k + g), (3.0 * k - 2.0 * g) / (2.0 * (3.0 * k + g))


def _as_materials(materials, mesh: TetMesh) -> dict:
    """A dict of the materials of the phases (one material: all the phases)."""
    if hasattr(materials, "young"):
        return {int(p): materials for p in np.unique(mesh.phase)}
    return dict(materials)


def _moduli(materials, phase):
    """Lamé constants of the elements (isotropic materials)."""
    mu = np.zeros(len(phase))
    lam = np.zeros(len(phase))
    for p, m in materials.items():
        if m.void or m.young is None:
            continue
        sel = phase == p
        mu[sel] = m.young / (2 * (1 + m.poisson))
        lam[sel] = m.young * m.poisson / ((1 + m.poisson) * (1 - 2 * m.poisson))
    return mu, lam


# -- solution --------------------------------------------------------------------------------


@dataclass
class CrackSolution:
    """A solved cracked body: its opened ``mesh``, the lips ``faces`` (pairs of faces), the
    ``crack``, the ``materials`` of the phases, the kinematics, the final ``displacement``
    (nodes x 3), the ``steps`` of the solver, the final nodal ``forces``, the stiffness of
    the ``contact`` of the lips and the ``bulk`` of points with its final state (inelastic
    bodies; :class:`~morphogen.fracture.bulk.PointBulk`)."""

    mesh: TetMesh
    faces: Interfaces
    crack: object
    materials: dict
    finite: bool
    displacement: np.ndarray
    steps: CohesiveResult = field(repr=False)
    forces: np.ndarray | None = field(default=None, repr=False)
    contact: float | None = None
    bulk: object | None = field(default=None, repr=False)
    friction: float = 0.0
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def stresses(self, energy: str = "work"):
        """The displacement gradient ``H``, the first Piola-Kirchhoff stress ``P`` (Cauchy
        in small strains) and the energy density ``W`` of each element, its volume and the
        gradients of its shape functions: ``W`` the elastic energy of an elastic body; for
        a bulk of points, their work density (``energy="work"``) or their free energy
        (``"free"`` or ``"tip"``)."""
        if energy not in ("work", "free", "tip"):
            raise ValueError("energy: 'work', 'free' or 'tip'")
        key = "free" if self.bulk is not None and energy != "work" else "work"
        if key in self._cache:
            return self._cache[key]
        mesh = self.mesh
        grads, volumes = _tet_gradients(mesh.nodes, mesh.elements)
        h = np.einsum("eai,eaj->eij", self.displacement[mesh.elements], grads)
        if self.bulk is not None:
            p = self.bulk.stress_tensors()
            w = self.bulk.work.copy() if key == "work" else self.bulk.free_energy()
        elif self.finite:
            mu, lam = _moduli(self.materials, mesh.phase)
            f = np.eye(3) + h
            finv_t = np.transpose(np.linalg.inv(f), (0, 2, 1))
            lnj = np.log(np.maximum(np.linalg.det(f), 1e-300))
            p = mu[:, None, None] * (f - finv_t) + (lam * lnj)[:, None, None] * finv_t
            w = 0.5 * mu * (np.einsum("eij,eij->e", f, f) - 3.0) - mu * lnj + 0.5 * lam * lnj**2
        else:
            from morphogen.homogenization.materials import material_table

            table = material_table(
                mesh.phase,
                self.materials,
                "elastic",
                grains=mesh.cell,
                orientations=mesh.orientations,
            )  # crystals: their grains
            d = table.matrices[np.maximum(table.mat, 0)] * (table.mat >= 0)[:, None, None]
            eps = np.stack([h[:, 0, 0], h[:, 1, 1], h[:, 2, 2], h[:, 1, 2] + h[:, 2, 1],
                            h[:, 2, 0] + h[:, 0, 2], h[:, 0, 1] + h[:, 1, 0]], axis=1)  # fmt: skip
            sig = np.einsum("eij,ej->ei", d, eps)  # Voigt (11, 22, 33, 23, 31, 12)
            w = 0.5 * np.einsum("ei,ei->e", sig, eps)
            p = np.empty_like(h)
            for k, (a, b) in enumerate(((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))):
                p[:, a, b] = p[:, b, a] = sig[:, k]
        self._cache[key] = (h, p, w, volumes, grads)
        return self._cache[key]

    def energy(self, kind: str = "work") -> float:
        """The elastic energy of an elastic body; for a bulk of points, the work of its
        stresses (``kind="work"``: the stored and dissipated energies, the work of the
        loads at fixed displacements) or its free energy (``"free"``)."""
        _, _, w, volumes, _ = self.stresses(kind)
        return float(volumes @ w)

    def potential(self) -> float:
        """The potential energy: the elastic energy (the work of the bulk) less the work of
        the forces (dead loads)."""
        if self.forces is None:
            return self.energy()
        return self.energy() - float(self.forces @ self.displacement.ravel())

    def area(self) -> float:
        """The area of the crack (its opened faces, in the reference configuration)."""
        return float(self.faces.areas(self.mesh).sum())

    def element_size(self) -> float:
        """The mean length of the edges of the elements; on a mesh refined around the
        front (:func:`~morphogen.fracture.lefm.refine_front`), of the elements that reach
        within half their size of it: the size that the domains of the integrals and the
        stations along the front are counted in."""
        if "element_size" not in self._cache:
            x = self.mesh.nodes[self.mesh.elements[:, :4]]
            edges = x[:, [1, 2, 3, 2, 3, 3]] - x[:, [0, 0, 0, 1, 1, 2]]
            lengths = np.linalg.norm(edges, axis=2).mean(axis=1)
            if self.mesh.metadata.get("front_size"):
                r = self.crack.frame(self.mesh.nodes)[0][self.mesh.elements[:, :4]]
                near = r.min(axis=1) < 0.5 * lengths
                if np.any(near):
                    lengths = lengths[near]
            self._cache["element_size"] = float(np.mean(lengths))
        return self._cache["element_size"]

    def _stations(self, stations):
        """The stations along the front (None: one, the whole front), whether it is closed,
        and the coordinates and lengths of its points."""
        lower, upper = self.mesh.nodes.min(axis=0), self.mesh.nodes.max(axis=0)
        points, lengths = self.crack.front(lower, upper, _COUNT)
        if not len(points):
            raise ValueError("no front inside the body")
        s = self.crack.frame(points)[4]
        closed = _closed(self.crack)
        if stations is None:
            return None, closed, s, lengths
        count = int(stations)
        if count < 2:
            raise ValueError("stations: at least two")
        if closed:
            if len(points) < _COUNT:
                raise ValueError("stations: a closed front inside the body")
            at = np.linspace(-np.pi, np.pi, count, endpoint=False)
            spacing = float(lengths.sum()) / count
        else:
            half = 0.5 * float(lengths.max())
            at = np.linspace(s.min() - half, s.max() + half, count)
            spacing = at[1] - at[0]
        if spacing < 2.0 * self.element_size():
            raise ValueError("stations: at least two elements apart along the front")
        return at, closed, s, lengths

    def _radius(self, value, default: float) -> float:
        return default * self.element_size() if value is None else float(value)

    def front_phase(self) -> int:
        """The phase at the front (its material: that of the front): of the elements in which
        most of the points of the front lie (or of the element nearest the front, if none of
        them lies in the body)."""
        if "front_phase" not in self._cache:
            from morphogen.fracture.ductile import locate

            mesh = self.mesh
            points, _ = self.crack.front(mesh.nodes.min(axis=0), mesh.nodes.max(axis=0), 64)
            if len(points):
                element, _ = locate(points, mesh)
                phases, counts = np.unique(mesh.phase[element], return_counts=True)
                phase = int(phases[np.argmax(counts)])
            else:
                centroids = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
                phase = int(mesh.phase[int(np.argmin(self.crack.frame(centroids)[0]))])
            self._cache["front_phase"] = phase
        return self._cache["front_phase"]

    def _theta(self, r_in, r_out, stations, restrict: bool = False):
        """The virtual advances ``θ`` of the stations at the nodes (stations x nodes x 3),
        the stations, whether the front is closed, the mass matrix of the front and the
        elements where ``θ`` is not zero; with ``restrict``, on a crack mesh of a morphology
        (its sides apart from its phases), zero beyond ``r_in`` at the nodes outside the
        material at the front (:func:`material_nodes`; see :meth:`release_rate`)."""
        r_out = min(self._radius(r_out, 4.0), _reach(self.crack))
        r_in = min(self._radius(r_in, 1.0), 0.5 * r_out)
        r, _, d, _, s_nodes = self.crack.frame(self.mesh.nodes)
        q = np.clip((r_out - r) / (r_out - r_in), 0.0, 1.0)
        if restrict and self.mesh.side is not None:
            if "material_nodes" not in self._cache:
                self._cache["material_nodes"] = material_nodes(
                    self.mesh, self.materials, self.front_phase(), self.faces
                )
            q = np.where(self._cache["material_nodes"] | (r <= r_in), q, 0.0)
        at, closed, s_front, lengths = self._stations(stations)
        theta = np.einsum("nk,ni->kni", _hats(s_nodes, at, closed) * q[:, None], d)
        front = _hats(s_front, at, closed)
        mass = np.einsum("pk,pl,p->kl", front, front, lengths)
        active = np.flatnonzero(np.any(q[self.mesh.elements] > 0, axis=1))
        return theta, at, closed, mass, active

    def release_rate(
        self, r_in=None, r_out=None, stations: int | None = None, energy: str = "work"
    ):
        """The energy release rate by the G-θ method (see the module): its mean along the
        front, or ``(s, G)`` at ``stations`` points along it (the coordinate ``s``: a length,
        an angle on a penny; at least two elements apart); ``q`` from 1 within ``r_in`` (by
        default one element) to 0 beyond ``r_out`` (four), within the reach of the lips.
        ``energy`` of an inelastic bulk (see the module): ``"work"``, the J of the
        deformation theory; ``"free"``, the configurational force on the domain; ``"tip"``,
        the driving force at the tip.

        On a crack mesh of a morphology, the domain of ``"work"`` and ``"free"`` takes in
        the other materials and the walls of the pores within it, their configurational
        forces with it (the far J); that of ``"tip"`` is the material at the front: ``θ``
        vanishes beyond ``r_in`` at the nodes of its interfaces with the other materials,
        in them and on the walls of the pores, which leaves their configurational forces
        out (the inhomogeneity term of Simha, Fischer, Kolednik & Chen 2003)."""
        return self._release_rate(r_in, r_out, stations, energy, energy == "tip")

    def _release_rate(self, r_in, r_out, stations, energy: str, restrict: bool):
        theta, at, _, mass, active = self._theta(r_in, r_out, stations, restrict)
        h, p, w, volumes, grads = (a[active] for a in self.stresses(energy))
        elements = self.mesh.elements[active]
        body = None
        if energy == "tip" and self.bulk is not None:
            body = self.bulk.configurational()[active]  # Div (ψ I - Hᵀ P)
        work = np.empty(len(theta))
        for k, field_k in enumerate(theta):
            g_theta = np.einsum("eai,eaj->eij", field_k[elements], grads)
            eshelby = np.einsum("eik,eij,ekj->e", h, p, g_theta) - w * np.einsum("eii->e", g_theta)
            work[k] = volumes @ eshelby
            if body is not None:
                work[k] -= volumes @ np.einsum("ek,ek->e", body, field_k[elements].mean(axis=1))
        if self.contact and self.friction:  # the work of the tractions of the lips that rub
            work -= self._lips_work(theta)
        g = np.linalg.solve(mass, work)
        return float(g[0]) if at is None else (at, g)

    def _sides(self) -> np.ndarray:
        """The side of the crack of each element (see :func:`element_sides`)."""
        return element_sides(self.mesh, self.crack)

    def _mean_gradients(self, active, at, closed: bool) -> np.ndarray:
        """The mean displacement gradients (stations x 3 x 3) of the elements ``active``
        around the front."""
        h, _, _, volumes, _ = self.stresses()
        centroids = self.mesh.nodes[self.mesh.elements[active]].mean(axis=1)
        weights = _hats(self.crack.frame(centroids)[4], at, closed) * volumes[active, None]
        return np.einsum("ek,eij->kij", weights, h[active]) / weights.sum(axis=0)[:, None, None]

    def _lips(self, theta, mu: float, nu: float) -> np.ndarray:
        """The work (stations x modes) of the tractions of the lips in contact on the
        bodies in the virtual advances ``θ`` of the auxiliary displacements, ``∫ t · (∇u'
        θ) dS`` over the two lips (see the module). The auxiliary gradient is singular at
        the front, as ``r^(-1/2)``: each face is integrated at points graded towards its
        corner nearest the front, then towards the next (``s = v²`` and ``t = w²`` in the
        triangle collapsed on that corner, which make the integrand smooth whether the face
        touches the front by a corner or by an edge), with the tractions of the linear
        field through those of its three points."""
        minus, plus = self.faces.minus[:, :3], self.faces.plus[:, :3]
        u = self.displacement
        xr = self.mesh.nodes[minus]
        reference = np.cross(xr[:, 1] - xr[:, 0], xr[:, 2] - xr[:, 0])
        area = 0.5 * np.linalg.norm(reference, axis=1)
        if self.friction:  # with the shear of the lips that stick or slide
            tractions = self._lip_tractions()[0]
        else:
            x = xr + 0.5 * (u[minus] + u[plus]) if self.finite else xr  # the frame of the law
            normal = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)
            shapes = np.array([[1.0 - xi - eta, xi, eta] for xi, eta in _LIP_POINTS])
            jumps = np.einsum("gi,eij->egj", shapes, u[plus] - u[minus])
            opening = np.einsum("egj,ej->eg", jumps, normal)
            tractions = (self.contact * np.minimum(opening, 0.0))[:, :, None] * normal[:, None]
        # the corners of each face from the nearest to the front
        behind = -_local(self.crack, xr.reshape(-1, 3))[0].reshape(-1, 3)
        order = np.argsort(behind, axis=1)
        rows = np.arange(len(minus))
        work = np.zeros((len(theta), 3))
        for v, wv in zip(_LIP_GAUSS[0], _LIP_GAUSS[1], strict=True):
            for w, ww in zip(_LIP_GAUSS[0], _LIP_GAUSS[1], strict=True):
                s, t = v * v, w * w
                shape = np.empty((len(minus), 3))  # of the corners of each face
                shape[rows, order[:, 0]] = 1.0 - s
                shape[rows, order[:, 1]] = s * (1.0 - t)
                shape[rows, order[:, 2]] = s * t
                weight = 8.0 * v**3 * w * wv * ww * area
                # the tractions: linear through those of the points (1/6, 1/6), (2/3, 1/6)
                # and (1/6, 2/3)
                traction = (
                    tractions[:, 0]
                    + 2.0 * (shape[:, 1] - 1.0 / 6.0)[:, None] * (tractions[:, 1] - tractions[:, 0])
                    + 2.0 * (shape[:, 2] - 1.0 / 6.0)[:, None] * (tractions[:, 2] - tractions[:, 0])
                )
                point = np.einsum("ei,eij->ej", shape, xr)
                x1, _ = _local(self.crack, point)
                _, n, d, tangent, _ = self.crack.frame(point)
                axes = np.stack([d, n, tangent], axis=2)
                # the faces "plus" of the interface elements lie on either side of the crack
                orient = np.where(np.einsum("ej,ej->e", reference, n) >= 0, 1.0, -1.0)
                jump_aux = _auxiliary_along_lips(x1, 1.0, mu, nu) - _auxiliary_along_lips(
                    x1, -1.0, mu, nu
                )
                jump_aux = orient[:, None, None] * jump_aux  # plus less minus
                along = np.einsum("eai,emi->ema", axes, jump_aux)  # the jump of ∂u'/∂d, global
                # on the face plus the traction -t, on the face minus the traction t
                per_face = -weight[:, None] * np.einsum("ea,ema->em", traction, along)
                for k, field_k in enumerate(theta):
                    advance = np.einsum("ei,eij->ej", shape, field_k[minus])
                    work[k] += np.einsum("e,em->m", np.einsum("ej,ej->e", advance, d), per_face)
        return work

    def _lip_tractions(self):
        """The tractions on the side ``-`` of the lips that rub (and the jumps of the
        displacement) at the three points of each face: those of the law at the end of
        the solution."""
        if "lips" not in self._cache:
            if self.faces.order != 1:
                raise ValueError("the driving forces of lips that rub: a mesh of 4-node tetrahedra")
            law = Contact(float(self.contact), friction=float(self.friction))
            self._cache["lips"] = lip_tractions(
                self.mesh.nodes, self.faces, self.displacement, law, self.steps, self.finite
            )
        return self._cache["lips"]

    def _closed_at_front(self, r_in=None) -> bool:
        """Whether the lips that rub press on each other at all their points within
        ``r_in`` of the front (by default one element): lips closed there open in no mode
        I, whatever the errors of quadrature of its interaction integral."""
        tractions, _ = self._lip_tractions()
        minus = self.faces.minus[:, :3]
        x = self.mesh.nodes[minus]
        shapes = np.array([[1.0 - xi - eta, xi, eta] for xi, eta in _LIP_POINTS])
        points = np.einsum("gi,eic->egc", shapes, x).reshape(-1, 3)
        near = self.crack.frame(points)[0] <= self._radius(r_in, 1.0)
        if not near.any():
            return False
        normal = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        pressure = -np.einsum("egc,ec->eg", tractions, normal).ravel()
        return bool(np.all(pressure[near] > 0.0))

    def _lips_work(self, theta) -> np.ndarray:
        """The work (stations) of the tractions of the lips that rub on the two bodies in
        the virtual advances ``θ`` of the actual displacement, ``∫ t · (∇u θ) dS`` over the
        two lips: ``-∫ t · (∇[u] θ) dS`` with the jump ``[u]`` of the displacement, whose
        gradient along the lips is that of the linear functions of each face."""
        minus, plus = self.faces.minus, self.faces.plus
        tractions, _ = self._lip_tractions()
        xr = self.mesh.nodes[minus]
        e1, e2 = xr[:, 1] - xr[:, 0], xr[:, 2] - xr[:, 0]
        area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        # the gradients of the functions of the nodes in the plane of each face
        metric = np.stack(
            [
                np.stack([np.einsum("ei,ei->e", e1, e1), np.einsum("ei,ei->e", e1, e2)], axis=1),
                np.stack([np.einsum("ei,ei->e", e1, e2), np.einsum("ei,ei->e", e2, e2)], axis=1),
            ],
            axis=1,
        )
        dual = np.einsum("eab,ebi->eai", np.linalg.inv(metric), np.stack([e1, e2], axis=1))
        grads = np.concatenate([-dual.sum(axis=1, keepdims=True), dual], axis=1)  # e x 3 x 3
        jump = self.displacement[plus] - self.displacement[minus]  # at the nodes
        gradient = np.einsum("eac,eai->eci", jump, grads)  # d[u]_c / dx_i
        work = np.zeros(len(theta))
        for g, (xi, eta) in enumerate(((1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3))):
            shape = np.array([1.0 - xi - eta, xi, eta])
            for k, field_k in enumerate(theta):
                advance = np.einsum("i,eij->ej", shape, field_k[minus])
                along = np.einsum("eci,ei->ec", gradient, advance)
                work[k] -= (area / 3) @ np.einsum("ec,ec->e", tractions[:, g], along)
        return work

    def modes(self, r_in=None, r_out=None, stations: int | None = None, release_rate=None) -> dict:
        """The stress intensity factors of the three modes by the interaction integrals,
        the energy release rate ``G`` by the G-θ method (or the value given, the same at all
        the stations) shared between them (the same domain), and the angle of kinking (see
        the module): floats, or arrays at the ``stations`` along the front (``"s"``). On a
        crack mesh of a morphology, the domain is the material at the front (that of
        :meth:`release_rate` with ``energy="tip"``), whose moduli the auxiliary fields
        take."""
        theta, at, closed, mass, active = self._theta(r_in, r_out, stations, True)
        mesh = self.mesh
        h, p, _, volumes, grads = (a[active] for a in self.stresses())
        elements = mesh.elements[active]
        centroids = mesh.nodes[elements].mean(axis=1)
        material = self.materials[self.front_phase()]
        young, nu = _isotropic_moduli(material)
        mu = young / (2 * (1 + nu))
        x1, x2 = _local(self.crack, centroids)
        side = self._sides()[active]  # behind the front, the branch of the side
        reference = np.where(x1 < 0, np.arctan2(side * np.abs(x2), x1), np.arctan2(x2, x1))
        g_aux, s_aux = _auxiliary_fields(self.crack, centroids, mu, nu, reference)
        curved = _curved(self.crack)  # straight fronts: zero
        if curved:
            dg, ds = _auxiliary_derivatives(self.crack, centroids, mu, nu, reference)
            de = 0.5 * (dg + np.swapaxes(dg, 2, 3))  # of the auxiliary strain
        mean = self._mean_gradients(active, at, closed)
        m = np.empty((len(theta), 3))
        for k, field_k in enumerate(theta):
            g_theta = np.einsum("eai,eaj->eij", field_k[elements], grads)
            if self.finite:  # turned back by the rotation of the mean deformation gradient
                u, _, vt = np.linalg.svd(np.eye(3) + mean[k])
                rot = u @ vt
                pt = np.einsum("ji,ejk->eik", rot, p)
                ht = np.einsum("ji,ejk->eik", rot, np.eye(3) + h) - np.eye(3)
            else:  # less the mean rotation, which adds nothing but errors of quadrature
                pt, ht = p, h - 0.5 * (mean[k] - mean[k].T)
            eps = 0.5 * (ht + np.swapaxes(ht, 1, 2))
            cross = (
                np.einsum("eij,emik,ekj->em", pt, g_aux, g_theta)
                + np.einsum("emij,eik,ekj->em", s_aux, ht, g_theta)
                - np.einsum("emij,eij->em", s_aux, eps) * np.einsum("eii->e", g_theta)[:, None]
            )
            if curved:  # the divergence of the cross tensor: incompatibility, unbalance
                advance = field_k[elements].mean(axis=1)  # θ at the centroids
                cross = cross + (
                    np.einsum("eij,emikj,ek->em", pt, dg, advance)
                    - np.einsum("eij,emijk,ek->em", pt, de, advance)
                    + np.einsum("emijj,eik,ek->em", ds, ht, advance)
                )
            m[k] = volumes @ cross
        if self.contact:
            m -= self._lips(theta, mu, nu)
        e_prime = young / (1 - nu**2)
        k_int = np.linalg.solve(mass, m) * np.array([e_prime / 2, e_prime / 2, mu])
        if self.contact:  # lips closed at the front: no opening
            k_int[:, 0] = np.maximum(k_int[:, 0], 0.0)
            if self.friction and self._closed_at_front(r_in):
                k_int[:, 0] = 0.0
        moduli = np.array([e_prime, e_prime, 2 * mu])
        shares = k_int**2 / moduli
        if release_rate is None:
            g = self._release_rate(r_in, r_out, stations, "work", True)
            g = np.atleast_1d(g if at is None else g[1])
        else:
            g = np.full(len(theta), float(release_rate))
        total = shares.sum(axis=1, keepdims=True)
        split = g[:, None] * np.divide(shares, total, out=np.zeros_like(shares), where=total > 0)
        out = {
            "K_I": k_int[:, 0],
            "K_II": k_int[:, 1],
            "K_III": k_int[:, 2],
            "G": g,
            "G_I": split[:, 0],
            "G_II": split[:, 1],
            "G_III": split[:, 2],
            "angle": np.array([kink_angle(a, b) for a, b in k_int[:, :2]]),
        }
        if at is None:
            return {key: float(v[0]) for key, v in out.items()}
        out["s"] = at
        return out


def solve_crack(
    mesh: TetMesh,
    crack,
    materials,
    prescribed,
    load: float = 1.0,
    *,
    kinematics: str = "small",
    steps: int | None = None,
    contact: float | bool | None = None,
    friction: float = 0.0,
    forces=None,
    sides=None,
    bulk: str = "auto",
    initial: CrackSolution | None = None,
    start: float | None = None,
    transfer: str = "projection",
    equilibrate: bool = True,
    **options,
) -> CrackSolution:
    """Open the crack on its conforming mesh and solve the body (see the module):
    ``materials`` a material (all the phases) or a dict of the phases; ``prescribed`` the
    entries ``(nodes, components, value)`` of
    :meth:`~morphogen.fracture.cohesive.CohesiveProblem.run`, or a function of the opened
    mesh returning them, and ``forces`` nodal forces (3 x nodes, or a function of the opened
    mesh), both grown to ``load`` times their values in ``steps`` (by default 1, or 10 in
    finite strains, with contact or an inelastic bulk); ``contact``: the penalty stiffness of
    the contact of the lips (``True``: ``50 E / h``; none by default) and ``friction`` their
    coefficient of friction (Coulomb: lips in contact stick or slide; see the module);
    ``bulk``: that of
    :class:`~morphogen.fracture.cohesive.CohesiveProblem` (inelastic phases: their points);
    ``options``: those of the solver (``tol``, ``max_newton``, ``duration``...).

    ``initial``: the solution of another crack (another mesh) whose state goes on: its
    displacement, and the internal variables of its points, are transferred to this mesh by
    ``transfer`` (:func:`~morphogen.fracture.ductile.transfer_state`), brought to
    equilibrium (``equilibrate``) at the load ``start`` (by default the last of
    ``initial``), and the loading resumes from there up to ``load``."""
    if friction and not contact:
        raise ValueError("solve_crack: friction is that of lips in contact (contact=True)")
    bonded = None
    if friction:  # with the ligament ahead of the front, which keeps the lips closed there
        opened, faces, bonded = _faces_and_ligament(mesh, crack, sides)
    else:
        opened, faces = crack_faces(mesh, crack, sides)
    materials = _as_materials(materials, opened)
    if callable(prescribed):
        prescribed = prescribed(opened)
    if callable(forces):
        forces = forces(opened)
    if contact:
        if contact is True:
            x = opened.nodes[opened.elements]
            size = float(np.mean(np.linalg.norm(x[:, 1] - x[:, 0], axis=1)))
            young = max(_stiffest(m) for m in materials.values() if not m.void)
            contact = 50.0 * young / size
        interfaces, law = faces, Contact(float(contact), friction=float(friction))
    else:
        empty = np.zeros((0, 3), dtype=np.int64)
        interfaces, law = Interfaces(empty, empty, np.zeros((0, 2), dtype=np.int64)), Contact(1.0)
    problem = CohesiveProblem(opened, interfaces, materials, law, kinematics=kinematics, bulk=bulk)
    if steps is None:
        steps = 10 if kinematics == "finite" or contact or problem.bulk is not None else 1
    if initial is not None:
        from morphogen.fracture.ductile import transfer_displacement, transfer_state

        start = float(initial.steps.loads[-1]) if start is None else float(start)
        if problem.bulk is not None and initial.bulk is not None:
            u0 = transfer_state(initial, opened, problem.bulk, crack, transfer)
        else:
            u0 = transfer_displacement(initial, opened, crack)
        options = {**options, "start": start, "initial": u0, "equilibrate": equilibrate}
    if bonded is not None:
        options = {**options, "history": np.where(bonded, -1.0, 0.0)}
    result = problem.run(prescribed, load, steps=steps, forces=forces, **options)
    if bonded is not None:  # the lips alone: their faces, the state of their points
        faces = faces.subset(~bonded)
        for name in ("history", "damage", "slip", "friction"):
            setattr(result, name, getattr(result, name)[~bonded])
    final = None if forces is None else load * np.asarray(forces, dtype=float).ravel()
    return CrackSolution(opened, faces, crack, materials, kinematics == "finite",
                         result.displacement, result, final,
                         float(contact) if contact else None, bulk=problem.bulk,
                         friction=float(friction) if contact else 0.0)  # fmt: skip


def _faces_and_ligament(mesh: TetMesh, crack, sides=None):
    """The mesh opened along the lips of a crack (:func:`~morphogen.fracture.lefm.
    crack_faces`) and along the faces of its surface ahead of its front that touch them
    (the ligament: one row of faces), the interface elements of both and the mask of
    those of the ligament. Lips whose nodes of the front are shared cannot press on each
    other there (the separation of a shared node is zero, and with it the pressure of the
    penalty and Coulomb's limit): with the ligament opened and bonded by the same
    penalty, the nodes of the front are split and the lips rub up to it."""
    from morphogen.fracture.cohesive import insert_interfaces, interior_faces

    if sides is None:
        if mesh.side is not None:
            sides = mesh.side
        elif mesh.metadata.get("crack_sides") == "phase":
            sides = mesh.phase
        else:
            sides = crack.phi(mesh.nodes[mesh.elements[:, :4]].mean(axis=1)) > 0
    sides = np.asarray(sides)
    corners, pairs = interior_faces(mesh)
    between = sides[pairs[:, 0]] != sides[pairs[:, 1]]
    lips = between & (crack.psi(mesh.nodes[corners].mean(axis=1)) < 0.0)
    touched = np.zeros(mesh.n_nodes, dtype=bool)
    touched[corners[lips]] = True
    ligament = between & ~lips & touched[corners].any(axis=1)
    opened, faces = insert_interfaces(mesh, lips | ligament)
    return opened, faces, ligament[lips | ligament]


def _stiffest(material) -> float:
    """Young's modulus of a material, or the largest normal stiffness of its matrix."""
    if material.young:
        return float(material.young)
    return float(np.max(np.diag(np.asarray(material.stiffness, dtype=float))[:3]))


def kink_angle(k1: float, k2: float) -> float:
    """The direction of the largest hoop stress (Erdogan & Sih), in radians from the
    direction of propagation towards the normal of the crack."""
    if abs(k2) <= 1e-12 * max(abs(k1), 1e-300):
        return 0.0
    return 2.0 * math.atan((k1 - math.sqrt(k1 * k1 + 8.0 * k2 * k2)) / (4.0 * k2))


def energy_release_rate(solve, crack, delta: float) -> float:
    """``G = -(Π(a + δ) - Π(a - δ)) / (A(a + δ) - A(a - δ))``, the potential energies at fixed
    prescribed displacements and forces, ``solve(crack)`` returning the
    :class:`CrackSolution` of a crack: the check of the G-θ method (the mean along the
    front)."""
    plus, minus = solve(crack.grown(delta)), solve(crack.grown(-delta))
    return -(plus.potential() - minus.potential()) / (plus.area() - minus.area())


@dataclass
class KinkedGrowth:
    """The steps of a propagation: the cracks, their energy release rates, stress
    intensity factors (steps x 3), angles of their advances and cycles (Paris), with
    ``keep`` the solutions; with a toughness, the critical load factor of each crack
    (``loads``: ``sqrt(Γ / G)`` of its advance at the load of the growth) and its advance
    (``advances``: the :class:`~morphogen.fracture.toughness.Advance` chosen; that of the last
    crack is not taken)."""

    cracks: list
    rates: np.ndarray
    factors: np.ndarray
    angles: np.ndarray
    cycles: np.ndarray
    solutions: list = field(default_factory=list, repr=False)
    loads: np.ndarray | None = None
    advances: list = field(default_factory=list, repr=False)

    def rows(self) -> list[dict]:
        rows = [
            {"step": k, "length": c.size, "x_tip": float(c.tip[0]), "y_tip": float(c.tip[1]),
             "z_tip": float(c.tip[2]), "G": float(g), "K_I": float(f[0]), "K_II": float(f[1]),
             "K_III": float(f[2]), "angle_deg": math.degrees(a), "N": float(n)}
            for k, (c, g, f, a, n) in enumerate(zip(self.cracks, self.rates, self.factors,
                                                    self.angles, self.cycles, strict=True))
        ]  # fmt: skip
        if self.advances:
            for row, advance, load in zip(rows, self.advances, self.loads, strict=True):
                row.update(_advance_columns(advance))
                row["load_critical"] = float(load)
        return rows


def _advance_columns(advance) -> dict:
    """The columns of an advance chosen by a toughness: into the bulk or along an
    interface, what it meets, its toughness and its ratio ``G(θ) / Γ(θ)``."""
    return {"path": advance.kind, "met": advance.label, "toughness": advance.toughness,
            "ratio": advance.ratio}  # fmt: skip


def propagate_kinked(
    mesh_of,
    crack: KinkedCrack,
    materials,
    prescribed,
    load: float = 1.0,
    *,
    increment: float,
    steps: int,
    kinematics: str = "small",
    paris: tuple[float, float] | None = None,
    contact: float | bool | None = None,
    toughness=None,
    keep: bool = False,
    **options,
) -> KinkedGrowth:
    """Grow a kinked crack (see the module): ``mesh_of(crack)`` meshes the body conforming
    to it (:func:`~morphogen.fracture.lefm.crack_mesh`), ``prescribed`` (entries, or a
    function of the opened mesh) and ``load`` load it; each of the ``steps`` advances the
    front by ``increment`` in the direction of kinking; ``paris = (C, m)`` counts the
    cycles; ``options`` those of :func:`solve_crack`.

    ``toughness``: a :class:`~morphogen.fracture.toughness.Toughness` (or one value, or a
    dict of the phases): each advance is then the candidate of the largest ``G(θ) / Γ(θ)``,
    the kink of the bulk or along an interface within an increment ahead
    (:func:`~morphogen.fracture.toughness.advance_candidates`), and the critical load factor
    of each crack is ``load sqrt(Γ / G)`` of its advance (``G`` growing as the square of the
    load: small strains); one value for all the phases without interfaces gives the advances
    in the direction of kinking."""
    from morphogen.fracture.toughness import Toughness, advance_candidates, choose_advance

    if not increment > 0:
        raise ValueError("propagate_kinked: a positive increment")
    rule = Toughness.of(toughness)
    cracks, rates, factors, angles, cycles, solutions = [], [], [], [], [], []
    loads, advances = [], []
    n = 0.0
    for step in range(int(steps) + 1):
        solution = solve_crack(mesh_of(crack), crack, materials, prescribed, load,
                               kinematics=kinematics, contact=contact, **options)  # fmt: skip
        modes = solution.modes()
        choice = None
        if rule is not None:
            choice = choose_advance(advance_candidates(solution, rule, increment, modes=modes))
            ratio = choice.ratio
            loads.append(float(load) / math.sqrt(ratio) if ratio > 0 else math.inf)
            advances.append(choice)
        cracks.append(crack)
        rates.append(modes["G"])
        factors.append((modes["K_I"], modes["K_II"], modes["K_III"]))
        angles.append(modes["angle"] if choice is None else choice.angle)
        cycles.append(n)
        if keep:
            solutions.append(solution)
        if step == steps:
            break
        grown = crack.grown(increment, modes["angle"]) if choice is None else choice.crack
        if paris is not None:  # the moduli of the material at the front
            young, nu = _isotropic_moduli(solution.materials[solution.front_phase()])
            e_prime = young / (1 - nu**2)
            c, exponent = paris
            bulk = choice is None or choice.kind == "bulk"
            length = increment if bulk else grown.size - crack.size  # along an interface
            n += length / (c * math.sqrt(e_prime * max(modes["G"], 1e-300)) ** exponent)
        crack = through_voids(grown, solution.mesh, increment)
    return KinkedGrowth(cracks, np.array(rates), np.array(factors), np.array(angles),
                        np.array(cycles), solutions,
                        None if rule is None else np.array(loads), advances)  # fmt: skip


def inside_elements(mesh: TetMesh, points, k: int = 16, tol: float = 1e-9) -> np.ndarray:
    """Whether points (rows) lie in an element of a mesh of linear tetrahedra (among the
    ``k`` whose centroids are nearest; ``tol``: of their barycentric coordinates)."""
    from scipy.spatial import cKDTree

    points = np.atleast_2d(np.asarray(points, dtype=float))
    x = mesh.nodes[mesh.elements[:, :4]]
    origin = x[:, 0]
    count = min(k, mesh.n_elements)
    _, near = cKDTree(x.mean(axis=1)).query(points, k=count)
    near = near.reshape(len(points), count)
    edges = np.stack([x[:, 1] - origin, x[:, 2] - origin, x[:, 3] - origin], axis=2)[near]
    local = np.linalg.solve(edges, (points[:, None, :] - origin[near])[..., None])[..., 0]
    smallest = np.minimum(local.min(axis=2), 1.0 - local.sum(axis=2))
    return np.any(smallest >= -tol, axis=1)


def through_voids(crack, mesh: TetMesh, increment: float, count: int = 64):
    """The crack grown on through the voids of a mesh (its pores) in which its front would
    end: its last segment lengthened by halves of ``increment`` until most of the ``count``
    points of its front, and most of the points a quarter of ``increment`` behind them (on
    its lips), lie in elements again (:func:`inside_elements`), or its front leaves the box.
    A front that ran into a pore so goes through it, the crack then emanating from its far
    wall, its front a quarter to three quarters of ``increment`` beyond it."""
    lower, upper = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)
    span = float(np.max(upper - lower))
    for _ in range(int(4 * span / max(float(increment), 1e-300)) + 1):
        points, _ = crack.front(lower, upper, count)
        if not len(points):
            return crack
        behind = points - 0.25 * float(increment) * crack.frame(points)[2]
        inside = inside_elements(mesh, np.vstack([points, behind])).reshape(2, -1)
        if np.all(2 * np.count_nonzero(inside, axis=1) > len(points)):
            return crack
        crack = crack.grown(0.5 * float(increment))
    return crack
