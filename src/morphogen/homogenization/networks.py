"""The effective stiffness of random periodic frameworks (fibre networks, open foams,
lattices with defects) as periodic frames of Euler–Bernoulli beams (or of Timoshenko beams,
the option of :mod:`morphogen.homogenization.frames`), and the softening of networks of
non-identical beams (Ban, Barocas, Shephard & Picu 2016).

A framework (:class:`morphogen.design.frames.Framework`: nodes in a periodic box, bars from
a node to an image of another; ``delaunay_network``, ``open_foam`` and ``random_lattice``
keep theirs in their metadata, of which :func:`network_frame` makes the frame of a
realization) becomes a frame of beams of circular sections,
:meth:`SpaceFrame.from_framework <morphogen.homogenization.frames.SpaceFrame.from_framework>`
(:meth:`Frame.from_framework <morphogen.homogenization.frames.Frame.from_framework>` in the
plane). The displacement of a node is ``E X + w`` and its rotation ``θ``, ``w`` and ``θ``
periodic, so that a beam from node ``i`` to the image of node ``j`` (vector ``d``) moves
its second end by ``E d + w_j - w_i`` and ``θ_j - θ_i`` relative to its first. The periodic
problem of each macroscopic strain is solved on the sparse stiffness of the fluctuations
(SuperLU, the translations held at the first node; without bending and torsion the
rotations are left out: a pin-jointed truss), for the strains of morphoGen's Voigt order
``11, 22, 33, 23, 31, 12`` with engineering shears (``11, 22, 12`` in the plane), that of
:func:`morphogen.homogenization.postprocess.engineering_constants` (the Mandel form is ``W
C W``, ``W = diag(1, 1, 1, √2, √2, √2)``). The effective stiffness is the energy of the
solutions ``u_I`` per unit volume (per unit area in the plane),

    C_IJ = (1 / V) Σ_beams u_I · K_e u_J,

the sum of the parts stored in the stretching, bending and torsion of the beams, each
positive semi-definite: the energy split of any macroscopic strain. The bending of a
Timoshenko beam is a uniform curvature (``EI (θ1 - θ2)² / l``, no shear force) and the
rotation of its chord ``a = (θ1 + θ2) / 2 - (v2 - v1) / l``, of stiffness ``k_a = 12 EI / l
(1 + Φ)``, the bending and shear compliances in series, whose energy is split between
bending and transverse shear as ``1 : Φ`` (the part ``shear``). The affine estimate is
the stretching of affinely deformed fibres (Cox 1952), ``C_aff = (1 / V) Σ E A L N Nᵀ``,
``N = (n1², n2², n3², n2 n3, n3 n1, n1 n2)`` for a fibre along ``n``: for isotropic
orientations ``E = φ E_s / 6`` and ``ν = 1/4`` in space, ``φ E_s / 3`` and ``1/3`` in the
plane, ``φ = Σ A L / V`` the fibre volume fraction (the fibre volume per unit area in the
plane). The non-affinity of a macroscopic strain ``E`` is the mean square non-affine
motion of the ends of the beams over their mean square length, ``Γ = Σ |w_j - w_i|² / (|E|²
Σ l²)``: the measure of Head, Levine & MacKintosh (2003) and Hatami-Marbini & Picu (2008)
at the scale of the segments, zero for an affine deformation (that of the nodes
themselves, ``⟨|w - w̄|²⟩``, grows with the size of a plane cell, as the fluctuations of a
random medium in two dimensions, and the motions relative to the length of each beam are
dominated by the shortest); and the ratio of the Young's modulus to that of the affine
estimate. Beam by beam, the axial force and the energy of each part under a uniaxial
stress (:func:`beam_forces`) show the force chains and the beams that bend.

Networks of non-identical beams (Ban et al. 2016). The stiffness ``k_i`` of a mode of the
beams (``axial``, ``bending``, or ``modulus``: all of them, the Young's modulus of each
fibre) is drawn with mean ``k_0`` and coefficient of variation ``CV``. The Young's modulus
of the network ``E(k)`` is the minimum, over the fluctuations and the free macroscopic
strains (the lateral contraction of a uniaxial stress), of an energy linear in each
``k_i``: it is concave in each, so that the network softens on average, and to second
order in ``CV``

    (E_0 - ⟨E⟩) / E_0 = s CV²,   s = -(k_0² / 2 E_0) Σ_i ∂²E/∂k_i²,
    σ_E² / E_0² = v CV²,          v = (k_0 / E_0)² Σ_i (∂E/∂k_i)²,

with ``∂E/∂k_i = U · K_i' U / V`` (``U`` the solution of the uniaxial stress, ``K_i'``
the stiffness of the mode of beam ``i`` per unit ``k``) and ``∂²E/∂k_i² = -(2 / V k_0²)
(f_i · K⁻¹ f_i + g_i · (V C')⁻¹ g_i)``, ``f_i = k_0 K_i' U`` the force of the beam on the
fluctuations, ``g_iJ = u_J · f_i`` on the free strains and ``C'`` their stiffness, plus
``U · K_i'' U / V`` where the stiffness of the beam is not linear in its factor (the ``EI``
and ``κGA`` of Timoshenko beams, concave: ``k_a'' = -2 k_a Φ / (1 + Φ)²``): computed
exactly for a frame (:func:`softening_coefficients`) and against replicas of the stiffness
(:func:`softening`). The mode ``area`` scales ``EA`` and ``κGA`` together, the area of the
general sections of Ban et al.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

from morphogen.homogenization.elements import VOIGT
from morphogen.homogenization.frames import Frame, SpaceFrame
from morphogen.seeding import SeedLike, make_rng

#: The modes of the beams whose stiffness varies among them: ``axial`` (EA), ``bending``
#: (EI), ``modulus`` (the Young's modulus: EA, EI, GJ and κGA alike) or ``area`` (the
#: area of a general section: EA and the shear stiffness κGA of Timoshenko beams)
MODES = ("axial", "bending", "modulus", "area")
#: The laws of the stiffness of the beams, of mean 1 and coefficient of variation CV:
#: ``normal`` (truncated at 1e-3), ``lognormal`` and ``bimodal`` (1 ± CV, equally likely)
LAWS = ("normal", "lognormal", "bimodal")
#: The Voigt order of the strains in the plane (engineering shear)
PLANE_VOIGT = ((0, 0), (1, 1), (0, 1))
#: Beams whose fluctuation forces are solved together (the second derivatives)
BATCH = 256
#: The smallest pivot of the factored stiffness of a truss, relative to the largest,
#: without mechanisms (the zero-energy modes of a Voronoi truss: 5e-19). Beams with rigid
#: joints have none when connected; the short edges of Poisson-Voronoi foams stiffen
#: some dofs by 1e16 times.
MECHANISM = 1e-12


def _hermite(lengths: np.ndarray) -> np.ndarray:
    """The bending stiffness per unit ``EI`` of Euler–Bernoulli beams of ``lengths`` on
    ``(v1, θ1, v2, θ2)`` (m x 4 x 4)."""
    base = np.array([[12, 6, -12, 6], [6, 4, -6, 2], [-12, -6, 12, -6], [6, 2, -6, 4]], float)
    powers = np.array([[0, 1, 0, 1], [1, 2, 1, 2], [0, 1, 0, 1], [1, 2, 1, 2]])
    length = lengths[:, None, None]
    return base * length**powers / length**3


def _timoshenko(lengths: np.ndarray) -> dict[str, np.ndarray]:
    """The two parts of the bending of beams of ``lengths`` on ``(v1, θ1, v2, θ2)`` (m x 4
    x 4): the ``curvature`` ``(θ1 - θ2)² / L`` per unit ``EI`` (uniform moment, no shear
    force) and the ``chord`` rotation ``a = (θ1 + θ2) / 2 - (v2 - v1) / L``, ``a²``, per
    unit of its stiffness ``k_a = 12 EI / L(1 + Φ)``, the bending ``l / 12 EI`` and the
    shear ``1 / κGA l`` compliances in series: ``EI curvature + (12 EI / L) chord`` is the
    Hermite stiffness of :func:`_hermite`."""
    m = len(lengths)
    c = np.array([0.0, 1.0, 0.0, -1.0])
    a = np.empty((m, 4))
    a[:, 0], a[:, 1], a[:, 2], a[:, 3] = 1.0 / lengths, 0.5, -1.0 / lengths, 0.5
    return {
        "curvature": np.outer(c, c)[None] / lengths[:, None, None],
        "chord": np.einsum("mi,mj->mij", a, a),
    }


def _plane_units(d: np.ndarray, timoshenko: bool = False) -> dict[str, np.ndarray]:
    """The stiffness of plane beams of vectors ``d`` per unit ``EA`` and ``EI`` in global
    axes (m x 6 x 6, dofs ``u1 v1 θ1 u2 v2 θ2``), as :func:`frames.beam_matrices`, and for
    Timoshenko beams the ``curvature`` and ``chord`` parts of the bending
    (:func:`_timoshenko`)."""
    length = np.hypot(d[:, 0], d[:, 1])
    c, s = d[:, 0] / length, d[:, 1] / length
    m = len(d)
    b = np.zeros((m, 6))
    b[:, 0], b[:, 1], b[:, 3], b[:, 4] = -c, -s, c, s
    p = np.zeros((m, 4, 6))  # (v1, θ1, v2, θ2) of the local axes
    p[:, 0, 0], p[:, 0, 1], p[:, 1, 2] = -s, c, 1.0
    p[:, 2, 3], p[:, 2, 4], p[:, 3, 5] = -s, c, 1.0
    out = {
        "axial": np.einsum("ma,mb->mab", b, b) / length[:, None, None],
        "bending": np.einsum("mia,mij,mjb->mab", p, _hermite(length), p),
    }
    if timoshenko:
        for key, local in _timoshenko(length).items():
            out[key] = np.einsum("mia,mij,mjb->mab", p, local, p)
    return out


def _space_units(d: np.ndarray, timoshenko: bool = False) -> dict[str, np.ndarray]:
    """The stiffness of beams in space of vectors ``d`` per unit ``EA``, ``EI`` and ``GJ``
    in global axes (m x 12 x 12, dofs ``u v w θx θy θz`` of each end), as
    :func:`frames.space_beam_matrices` (the bending, the same about both axes of the
    section, does not depend on their choice), and for Timoshenko beams the ``curvature``
    and ``chord`` parts of the bending in both planes (:func:`_timoshenko`)."""
    length = np.linalg.norm(d, axis=1)
    ex = d / length[:, None]
    helper = np.where((np.abs(ex[:, 2]) < 0.9)[:, None], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    ey = np.cross(helper, ex)
    ey /= np.linalg.norm(ey, axis=1)[:, None]
    ez = np.cross(ex, ey)
    m = len(d)
    stretch, twist = np.zeros((m, 12)), np.zeros((m, 12))
    stretch[:, 0:3], stretch[:, 6:9] = -ex, ex
    twist[:, 3:6], twist[:, 9:12] = -ex, ex
    plane_y, plane_z = np.zeros((m, 4, 12)), np.zeros((m, 4, 12))
    for row, start in enumerate((0, 3, 6, 9)):  # (v, θz) and (w, -θy) of both ends
        rotation = start in (3, 9)
        plane_y[:, row, start : start + 3] = ez if rotation else ey
        plane_z[:, row, start : start + 3] = -ey if rotation else ez
    hermite = _hermite(length)
    inv = 1.0 / length[:, None, None]
    out = {
        "axial": np.einsum("ma,mb->mab", stretch, stretch) * inv,
        "bending": np.einsum("mia,mij,mjb->mab", plane_y, hermite, plane_y)
        + np.einsum("mia,mij,mjb->mab", plane_z, hermite, plane_z),
        "torsion": np.einsum("ma,mb->mab", twist, twist) * inv,
    }
    if timoshenko:
        for key, local in _timoshenko(length).items():
            out[key] = np.einsum("mia,mij,mjb->mab", plane_y, local, plane_y) + np.einsum(
                "mia,mij,mjb->mab", plane_z, local, plane_z
            )
    return out


def _rigid_arms(axes: np.ndarray, rigid: np.ndarray) -> np.ndarray:
    """The motion of the ends of the flexible parts of beams of unit ``axes`` with rigid
    ends of lengths ``rigid`` (m x 2) from the dofs of their nodes (m x n x n): ``u + θ ×
    a₁ e`` at the first, ``u - θ × a₂ e`` at the second, the rotations unchanged."""
    m, dim = axes.shape
    per_node = 6 if dim == 3 else 3
    out = np.broadcast_to(np.eye(2 * per_node), (m, 2 * per_node, 2 * per_node)).copy()
    for end, sign in ((0, -1.0), (1, 1.0)):
        arm = sign * rigid[:, end]
        u, t = end * per_node, end * per_node + dim
        if dim == 3:  # θ × (-arm e) = arm [e]x θ
            ex, ey, ez = axes.T
            skew = np.zeros((m, 3, 3))
            skew[:, 0, 1], skew[:, 0, 2], skew[:, 1, 2] = -ez, ey, -ex
            skew[:, 1, 0], skew[:, 2, 0], skew[:, 2, 1] = ez, -ey, ex
            out[:, u : u + 3, t : t + 3] = arm[:, None, None] * skew
        else:  # θ z × (-arm e) = -arm θ (-e2, e1)
            out[:, u, t] = arm * axes[:, 1]
            out[:, u + 1, t] = -arm * axes[:, 0]
    return out


class _Periodic:
    """The periodic problem of a frame: its beams (vectors, lengths, dofs), their stiffness
    per unit of each mode, the affine motion of the unit strains and the sparse assembly.
    Beams with rigid ends take the stiffness of their flexible part (``flexible`` long),
    moved by the rigid arms of their nodes."""

    def __init__(self, frame) -> None:
        if not isinstance(frame, (Frame, SpaceFrame)):
            raise TypeError("frame_stiffness: a Frame or a SpaceFrame")
        self.frame = frame
        self.space = isinstance(frame, SpaceFrame)
        self.dim = 3 if self.space else 2
        self.vectors = np.asarray(frame.vectors(), dtype=float)
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        if len(self.lengths) == 0 or np.any(self.lengths <= 0):
            raise ValueError("frame_stiffness: beams of positive length")
        self.dofs = frame._dofs()
        self.n_dofs = int(frame.n_dofs)
        self.volume = float(abs(np.linalg.det(frame.lattice)))
        self.timoshenko = getattr(frame, "shear", None) is not None
        units = _space_units if self.space else _plane_units
        rigid = getattr(frame, "rigid", None)
        if rigid is None:
            self.flexible = self.lengths
            self.units = units(self.vectors, self.timoshenko)
        else:
            rigid = np.asarray(rigid, dtype=float).reshape(len(self.lengths), 2)
            self.flexible = self.lengths - rigid.sum(axis=1)
            if np.any(rigid < 0) or np.any(self.flexible <= 0):
                raise ValueError("frame_stiffness: rigid ends non-negative, shorter than the beams")
            shortened = self.vectors * (self.flexible / self.lengths)[:, None]
            flexible = units(shortened, self.timoshenko)
            arms = _rigid_arms(self.vectors / self.lengths[:, None], rigid)
            self.units = {
                k: np.einsum("mia,mij,mjb->mab", arms, v, arms) for k, v in flexible.items()
            }
        self.strains = VOIGT if self.space else PLANE_VOIGT
        per_node = 6 if self.space else 3
        m, n = self.dofs.shape
        self.affine = np.zeros((m, n, len(self.strains)))
        for j, (a, b) in enumerate(self.strains):
            e = np.zeros((self.dim, self.dim))
            e[a, b] = e[b, a] = 1.0 if a == b else 0.5
            self.affine[:, per_node : per_node + self.dim, j] = self.vectors @ e.T
        self.rows = np.repeat(self.dofs, n, axis=1).ravel()
        self.cols = np.tile(self.dofs, (1, n)).ravel()
        graph = sp.coo_matrix(
            (np.ones(len(self.dofs)), (frame.beams[:, 0], frame.beams[:, 1])),
            shape=(len(frame.nodes),) * 2,
        )
        if connected_components(graph, directed=False)[0] > 1:
            raise ValueError("frame_stiffness: the frame has disconnected parts")
        free = np.ones(self.n_dofs, dtype=bool)
        free[: self.dim] = False  # the translations of the first node
        torsion = frame.torsion if self.space else np.zeros(1)
        self.truss = not np.any(frame.bending) and not np.any(torsion)
        if self.truss:  # pin joints: the rotations are left out
            free[np.arange(self.n_dofs) % per_node >= self.dim] = False
        self.free = free

    def chord(self, bending, shear) -> tuple[np.ndarray, np.ndarray]:
        """The shear parameter ``Φ = 12 EI / (κGA l²)`` and the stiffness ``k_a = 12 EI /
        l(1 + Φ)`` of the chord rotation of Timoshenko beams (m each), ``l`` their flexible
        length."""
        bending = np.asarray(bending, dtype=float)
        phi = 12.0 * bending / (np.asarray(shear, dtype=float) * self.flexible**2)
        return phi, 12.0 * bending / (self.flexible * (1.0 + phi))

    def stiffness(self, axial, bending, torsion=None, shear=None) -> dict[str, np.ndarray]:
        """The element stiffness of each mode (m x n x n) for these beam stiffnesses: for
        Timoshenko beams (``shear`` the ``κGA`` of each), the energy of the chord rotation
        ``k_a a² / 2`` split between ``bending`` and ``shear`` as their compliances in
        series, ``1 : Φ``."""
        out = {"axial": np.asarray(axial, dtype=float)[:, None, None] * self.units["axial"]}
        if shear is None:
            out["bending"] = np.asarray(bending, dtype=float)[:, None, None] * self.units["bending"]
        else:
            phi, k = self.chord(bending, shear)
            chord = self.units["chord"]
            out["bending"] = (
                np.asarray(bending, dtype=float)[:, None, None] * self.units["curvature"]
                + (k / (1.0 + phi))[:, None, None] * chord
            )
            out["shear"] = (k * phi / (1.0 + phi))[:, None, None] * chord
        if self.space:
            out["torsion"] = np.asarray(torsion, dtype=float)[:, None, None] * self.units["torsion"]
        return out

    def values(self) -> dict:
        """The stiffness of the beams of the frame (the arguments of :meth:`stiffness`)."""
        f = self.frame
        return {
            "axial": f.axial,
            "bending": f.bending,
            "torsion": f.torsion if self.space else None,
            "shear": f.shear if self.timoshenko else None,
        }

    def of_frame(self) -> dict[str, np.ndarray]:
        return self.stiffness(**self.values())

    def derivatives(self, modes: dict, mode: str):
        """The first and second derivatives of the element stiffness in the factor of a
        mode of each beam (m x n x n; the second None where it is linear in the factor):
        the first is the stiffness of the parts of the energy that the mode carries; the
        chord stiffness ``k_a`` of a Timoshenko beam, the bending and shear compliances in
        series, is concave in the factor of either, ``k_a'' = -2 k_a Φ / (1 + Φ)²``."""
        if mode == "axial":
            return modes["axial"], None
        if mode == "modulus":
            return sum(modes.values()), None
        first = modes["bending"] if mode == "bending" else modes["axial"]
        if not self.timoshenko:
            return first, None
        if mode == "area":
            first = first + modes["shear"]
        phi, k = self.chord(self.frame.bending, self.frame.shear)
        return first, (-2.0 * k * phi / (1.0 + phi) ** 2)[:, None, None] * self.units["chord"]

    def factor(self, ke: np.ndarray):
        """The LU factors of the stiffness of the free fluctuations; for a truss, a pivot at
        round-off relative to the largest (a zero-energy mode) refused."""
        k = sp.coo_matrix((ke.ravel(), (self.rows, self.cols)), shape=(self.n_dofs,) * 2)
        k = k.tocsr()[self.free][:, self.free].tocsc()
        message = (
            "frame_stiffness: the frame has mechanisms (a truss below rigidity, "
            "disconnected parts or beams without stiffness)"
        )
        try:
            lu = spla.splu(k)
        except RuntimeError as exc:  # exactly singular
            raise ValueError(message) from exc
        pivots = np.abs(lu.U.diagonal())
        if self.truss and pivots.min() <= MECHANISM * pivots.max():
            raise ValueError(message)
        return lu

    def scatter(self, forces: np.ndarray) -> np.ndarray:
        """Element forces (m x n x k) assembled on the free dofs (free x k)."""
        out = np.zeros((self.n_dofs, forces.shape[2]))
        np.add.at(out, self.dofs, forces)
        return out[self.free]

    def solve(self, ke: np.ndarray):
        """The element displacements of the unit strains (m x n x strains), the periodic
        fluctuations (dofs x strains) and the LU factors."""
        lu = self.factor(ke)
        rhs = self.scatter(np.einsum("mab,mbj->maj", ke, self.affine))
        w = np.zeros((self.n_dofs, len(self.strains)))
        w[self.free] = -lu.solve(rhs)
        return w[self.dofs] + self.affine, w, lu

    def energy(self, u: np.ndarray, ke: np.ndarray) -> np.ndarray:
        """``(1 / V) Σ u_I · K_e u_J`` (strains x strains, symmetrized)."""
        out = np.einsum("mai,mab,mbj->ij", u, ke, u) / self.volume
        return 0.5 * (out + out.T)


def _strain_norm(strain: np.ndarray, dim: int) -> float:
    """The norm of a strain tensor given by its Voigt components (engineering shears)."""
    normal = strain[:dim]
    shear = strain[dim:]
    return float(np.sqrt(normal @ normal + 0.5 * shear @ shear))


@dataclass
class FrameStiffness:
    """The effective stiffness of a periodic frame (:func:`frame_stiffness`): the
    ``stiffness`` (6 x 6, Voigt order 11, 22, 33, 23, 31, 12 with engineering shears; 3 x 3,
    11, 22, 12, per unit area in the plane) and its parts stored in the ``stretching``,
    ``bending`` and ``torsion`` of the beams (their sum), the ``affine`` estimate (the
    stretching of affinely deformed beams), the sums over the beams of the products of
    the non-affine motions of their ends under the unit strains over that of the squares
    of their lengths (``gram``), the periodic ``fluctuations`` themselves (dofs x
    strains), the mean length of the beams, the fibre volume fraction (fibre volume per
    unit area in the plane; None when the sections are unknown), the dimension and, for
    Timoshenko beams, the part stored in their transverse ``shear`` (None for
    Euler–Bernoulli beams)."""

    stiffness: np.ndarray
    stretching: np.ndarray
    bending: np.ndarray
    torsion: np.ndarray
    affine: np.ndarray
    gram: np.ndarray
    fluctuations: np.ndarray
    mean_length: float
    density: float | None
    dimension: int
    shear: np.ndarray | None = None

    def compliance(self) -> np.ndarray:
        return np.linalg.inv(self.stiffness)

    def uniaxial(self, axis: int = 0) -> np.ndarray:
        """The macroscopic strain (Voigt) of a uniaxial stress along ``axis``, its normal
        component along the axis equal to 1."""
        s = self.compliance()
        return s[:, axis] / s[axis, axis]

    def young(self, axis: int = 0) -> float:
        """The Young's modulus along ``axis``."""
        return float(1.0 / self.compliance()[axis, axis])

    def poisson(self, axis: int = 0, lateral: int = 1) -> float:
        """The Poisson's ratio: the contraction along ``lateral`` under a stress along
        ``axis``."""
        s = self.compliance()
        return float(-s[axis, lateral] / s[axis, axis])

    def engineering(self) -> dict[str, float]:
        """Young's moduli, Poisson's ratios and shear moduli (as
        :func:`morphogen.homogenization.postprocess.engineering_constants`)."""
        if self.dimension == 3:
            from morphogen.homogenization.postprocess import engineering_constants

            return engineering_constants(self.stiffness)
        s = self.compliance()
        return {
            "E1": float(1.0 / s[0, 0]),
            "E2": float(1.0 / s[1, 1]),
            "nu12": float(-s[0, 1] / s[0, 0]),
            "nu21": float(-s[0, 1] / s[1, 1]),
            "G12": float(1.0 / s[2, 2]),
        }

    def _strain(self, strain) -> np.ndarray:
        if strain is None:
            return self.uniaxial(0)
        e = np.asarray(strain, dtype=float)
        if e.shape == (self.dimension, self.dimension):
            order = VOIGT if self.dimension == 3 else PLANE_VOIGT
            e = np.array([e[a, b] * (1.0 if a == b else 2.0) for a, b in order])
        if e.shape != (len(self.stiffness),):
            raise ValueError("strain: Voigt components (engineering shears) or a tensor")
        return e

    def split(self, strain=None) -> dict[str, float]:
        """The fractions of the energy of a macroscopic strain (Voigt components with
        engineering shears, or a tensor; by default that of a uniaxial stress along x)
        stored in the stretching, bending and torsion of the beams (and their transverse
        shear for Timoshenko beams)."""
        e = self._strain(strain)
        total = float(e @ self.stiffness @ e)
        parts = {"stretching": self.stretching, "bending": self.bending}
        if self.dimension == 3:
            parts["torsion"] = self.torsion
        if self.shear is not None:
            parts["shear"] = self.shear
        return {k: float(e @ v @ e) / total for k, v in parts.items()}

    def nonaffinity(self, strain=None) -> float:
        """The mean square non-affine motion of the ends of the beams over their mean
        square length under a macroscopic strain ``E`` (by default that of a uniaxial stress
        along x), ``Σ |w_j - w_i|² / (|E|² Σ l²)``: zero for an affine deformation."""
        e = self._strain(strain)
        return float(e @ self.gram @ e) / _strain_norm(e, self.dimension) ** 2

    def affinity(self, axis: int = 0) -> float:
        """The Young's modulus along ``axis`` over that of the affine estimate (1 for an
        affine deformation of slender beams)."""
        try:
            affine = 1.0 / np.linalg.inv(self.affine)[axis, axis]
        except np.linalg.LinAlgError:  # beams along too few directions
            return float("nan")
        return self.young(axis) / float(affine)

    def by_axis(self) -> list[dict]:
        """The values of a uniaxial stress along each axis: the ``axis`` (x, y, z), the
        Young's modulus, the Poisson's ratio (the contraction along the next axis), the
        ratio to the affine estimate (``affinity``), the fractions of the energy of
        :meth:`split` and the ``nonaffinity``."""
        dim = self.dimension
        out = []
        for a in range(dim):
            strain = self.uniaxial(a)
            out.append(
                {
                    "axis": "xyz"[a],
                    "young": self.young(a),
                    "poisson": self.poisson(a, (a + 1) % dim),
                    "affinity": self.affinity(a),
                    **self.split(strain),
                    "nonaffinity": self.nonaffinity(strain),
                }
            )
        return out

    def summary(self, young: float = 1.0) -> dict[str, float]:
        """The main values, the moduli relative to the Young's modulus ``young`` of the
        beams (the mean over the axes of :meth:`by_axis` for the moduli, Poisson's ratios,
        stretching fractions and non-affinity)."""
        rows = self.by_axis()

        def mean(key: str) -> float:
            return float(np.mean([row[key] for row in rows]))

        out = {
            "frame_young": mean("young") / young,
            "frame_poisson": mean("poisson"),
            "frame_affinity": mean("affinity"),
            "frame_stretching": mean("stretching"),
            "frame_nonaffinity": mean("nonaffinity"),
        }
        if self.density:
            out["frame_young_per_density"] = out["frame_young"] / self.density
        return out


def frame_stiffness(frame) -> FrameStiffness:
    """The effective stiffness of a periodic frame (:class:`Frame` or :class:`SpaceFrame`)
    by its sparse periodic problem, its split into stretching, bending and torsion and the
    fluctuations of the nodes (the module docstring gives the conventions)."""
    problem = _Periodic(frame)
    modes = problem.of_frame()
    ke = sum(modes.values())
    u, w, _ = problem.solve(ke)
    parts = {k: problem.energy(u, v) for k, v in modes.items()}
    dim = problem.dim
    per_node = 6 if problem.space else 3
    ends = w[problem.dofs]  # beams x dofs x strains
    jump = ends[:, per_node : per_node + dim] - ends[:, :dim]
    gram = np.einsum("mai,maj->ij", jump, jump) / np.sum(problem.lengths**2)
    n = problem.vectors / problem.lengths[:, None]
    voigt = np.column_stack([n[:, a] * n[:, b] for a, b in problem.strains])
    # EA L N Nᵀ, the extension of a beam carried by its flexible part: EA L² / l
    stretch = frame.axial * problem.lengths**2 / problem.flexible
    affine = np.einsum("m,mi,mj->ij", stretch, voigt, voigt)
    area = getattr(frame, "area", None)
    density = None if area is None else float(np.sum(area * problem.lengths) / problem.volume)
    return FrameStiffness(
        stiffness=sum(parts.values()),
        stretching=parts["axial"],
        bending=parts["bending"],
        torsion=parts.get("torsion", np.zeros_like(parts["axial"])),
        affine=affine / problem.volume,
        gram=gram,
        fluctuations=w,
        mean_length=float(problem.lengths.mean()),
        density=density,
        dimension=dim,
        shear=parts.get("shear"),
    )


#: The parts of the energy of the beams (:func:`beam_forces`), as named by
#: :meth:`FrameStiffness.split`, from the modes of the element stiffness
PARTS = {"axial": "stretching", "bending": "bending", "torsion": "torsion", "shear": "shear"}


def beam_forces(frame, axis: int = 0, *, strain=None) -> dict:
    """The axial force of each beam of a frame and the energy stored in each part of each
    beam under a uniaxial stress of 1 along ``axis`` (the lateral stresses free) or under
    a macroscopic ``strain`` (Voigt components with engineering shears, or a tensor), by
    the sparse periodic problem: the ``strain`` (Voigt), the ``force`` ``N = EA δ / l``
    of each beam (positive in tension; ``δ`` the extension of its flexible part ``l``,
    which its rigid ends do not change) and its energy per unit volume (per unit area in
    the plane) stored in its ``stretching``, ``bending``, ``torsion`` (in space) and
    transverse ``shear`` (Timoshenko beams), whose sum over the beams and the parts is
    ``E · C E / 2``: the fractions of :meth:`FrameStiffness.split` beam by beam. They show
    the force chains of a network and the beams that bend, as
    :func:`morphogen.viz.figures.frame_figure` draws them."""
    problem = _Periodic(frame)
    modes = problem.of_frame()
    ke = sum(modes.values())
    u, _, _ = problem.solve(ke)
    c = problem.energy(u, ke)
    dim = problem.dim
    if strain is None:
        if not 0 <= int(axis) < dim:
            raise ValueError(f"axis: from 0 to {dim - 1}")
        e = np.linalg.inv(c)[:, int(axis)]  # Σ = 1 along the axis
    else:
        e = np.asarray(strain, dtype=float)
        if e.shape == (dim, dim):
            e = np.array([e[a, b] * (1.0 if a == b else 2.0) for a, b in problem.strains])
        if e.shape != (len(c),):
            raise ValueError("strain: Voigt components (engineering shears) or a tensor")
    uu = u @ e  # element displacements (m x n)
    per_node = 6 if problem.space else 3
    axes = problem.vectors / problem.lengths[:, None]
    extension = np.einsum("mi,mi->m", axes, uu[:, per_node : per_node + dim] - uu[:, :dim])
    force = np.asarray(frame.axial, dtype=float) * extension / problem.flexible
    out = {"strain": e, "force": force}
    for mode, k in modes.items():
        out[PARTS[mode]] = 0.5 * np.einsum("ma,mab,mb->m", uu, k, uu) / problem.volume
    return out


def network_stiffness(
    framework, radius, *, young=1.0, poisson=0.3, dimension: int = 3, shear_coefficient=None,
    rigid_joints=None,
):  # fmt: skip
    """:func:`frame_stiffness` of the frame of beams of circular sections of a periodic
    framework (or of its dictionary, as kept in the metadata of the network models):
    ``radius``, ``young`` and ``poisson`` one value or one per bar; Timoshenko beams with
    a ``shear_coefficient`` (``"cowper"``, a value or one per bar), Euler–Bernoulli beams
    without; with ``rigid_joints``, beams rigid over that many radii from each node (the
    struts merged in the joints of a continuum)."""
    options = {
        "young": young,
        "poisson": poisson,
        "shear_coefficient": shear_coefficient,
        "rigid_joints": rigid_joints,
    }
    if dimension == 2:
        return frame_stiffness(Frame.from_framework(framework, radius, **options))
    if dimension != 3:
        raise ValueError("dimension: 2 or 3")
    return frame_stiffness(SpaceFrame.from_framework(framework, radius, **options))


#: The radius of the fibres of a realization of a network model: the first of these keys
#: in its summary (``delaunay_network``, ``open_foam``: the radius solved for a volume
#: fraction), else of the parameters of its model (``random_lattice``)
RADII = ("radius", "strut_radius")


def network_frame(micro, *, young=1.0, poisson=0.3, shear_coefficient=None, rigid_joints=None):
    """The frame of beams of a realization of a network model, whose metadata keeps its
    framework (``delaunay_network``, ``open_foam``, ``random_lattice``: that of the
    periodic box, or of the larger periodic lattice of a window): beams of circular
    sections of the radius of its fibres (:data:`RADII`), with the options of
    :func:`network_stiffness`; a plane :class:`Frame` when the framework lies in a plane
    (its nodes at one height, no bar across the box along z: a planar Delaunay network),
    else a :class:`SpaceFrame`. A microstructure without a framework is refused."""
    from morphogen.design.frames import Framework

    metadata = getattr(micro, "metadata", None) or {}
    model = getattr(micro, "model", None)
    name = model.name if model is not None else type(micro).__name__
    framework = metadata.get("framework")
    if framework is None:
        raise ValueError(
            f"{name} has no framework of fibres or struts, hence no frame of beams: the "
            "realizations of delaunay_network, open_foam and random_lattice keep theirs"
        )
    if isinstance(framework, dict):
        framework = Framework.from_dict(framework)
    summary = metadata.get("summary") or {}
    radii = [summary.get(k) for k in RADII] + [getattr(model, k, None) for k in RADII]
    radius = next((float(r) for r in radii if r is not None), None)
    if radius is None:
        raise ValueError(f"{name}: no radius of the fibres in its summary or its model")
    options = {
        "young": young,
        "poisson": poisson,
        "shear_coefficient": shear_coefficient,
        "rigid_joints": rigid_joints,
    }
    nodes, shifts = np.asarray(framework.nodes), np.asarray(framework.shifts)
    if np.ptp(nodes[:, 2]) == 0 and not np.any(shifts[:, 2]):
        return Frame.from_framework(framework, radius, **options)
    return SpaceFrame.from_framework(framework, radius, **options)


# -- networks of non-identical beams (Ban et al. 2016) -----------------------------------


def fibre_scales(count: int, cv: float, law: str = "normal", seed: SeedLike = None):
    """``count`` factors of mean 1 and coefficient of variation ``cv``: ``normal`` (draws
    below 1e-3 raised to it), ``lognormal`` or ``bimodal`` (1 - cv or 1 + cv, equally
    likely)."""
    if law not in LAWS:
        raise ValueError(f"law: one of {LAWS}")
    if not 0.0 <= cv < (1.0 if law == "bimodal" else np.inf):
        raise ValueError("cv: non-negative (below 1 for the bimodal law)")
    rng, _ = make_rng(seed)
    if law == "normal":
        return np.maximum(1.0 + cv * rng.standard_normal(count), 1e-3)
    if law == "lognormal":
        sigma2 = np.log1p(cv * cv)
        return np.exp(np.sqrt(sigma2) * rng.standard_normal(count) - 0.5 * sigma2)
    return 1.0 + cv * rng.choice((-1.0, 1.0), size=count)


def _scaled(frame, scale, mode: str) -> dict:
    """The stiffnesses of the beams that a factor ``scale`` (one per beam) of ``mode``
    changes, scaled."""
    if mode not in MODES:
        raise ValueError(f"mode: one of {MODES}")
    s = np.broadcast_to(np.asarray(scale, dtype=float), (len(frame.beams),))
    shear = getattr(frame, "shear", None)
    if mode == "bending":
        return {"bending": frame.bending * s}
    changes = {"axial": frame.axial * s}
    if mode == "axial":
        return changes
    if shear is not None:
        changes["shear"] = shear * s
    if mode == "modulus":
        changes["bending"] = frame.bending * s
        if isinstance(frame, SpaceFrame):
            changes["torsion"] = frame.torsion * s
    return changes


def scaled_frame(frame, scale, mode: str = "modulus"):
    """The frame with the stiffness of a mode of its beams multiplied by ``scale`` (one
    factor per beam): ``axial`` (EA), ``bending`` (EI), ``area`` (the area of a general
    section: EA and the shear stiffness κGA of Timoshenko beams, EI and GJ kept) or
    ``modulus`` (all of them: the Young's modulus of each beam)."""
    return dataclasses.replace(frame, **_scaled(frame, scale, mode))


def softening_coefficients(frame, mode: str = "modulus", axis: int = 0) -> dict:
    """The Young's modulus ``E_0`` of a frame along ``axis`` and the coefficients of its
    softening and scatter when the stiffness of a mode of its beams varies independently
    from beam to beam, with mean the present one and coefficient of variation ``CV``:
    ``(E_0 - ⟨E⟩) / E_0 = s CV²`` and ``σ_E² / E_0² = v CV²`` to second order (Ban et al.
    2016, eqs. 22 and 25, computed exactly: the module docstring), with the derivatives
    ``dE`` and ``d2E`` per beam of ``E`` relative to its factor. The modes are those of
    :func:`scaled_frame`; the stiffness of a Timoshenko beam is not linear in its ``EI``
    or ``κGA`` (``bending``, ``area``), whose second derivative adds ``U · K_i'' U / V``."""
    if mode not in MODES:
        raise ValueError(f"mode: one of {MODES}")
    problem = _Periodic(frame)
    modes = problem.of_frame()
    ke = sum(modes.values())
    u, _, lu = problem.solve(ke)
    c = problem.energy(u, ke)
    s = np.linalg.inv(c)
    e0 = 1.0 / s[axis, axis]
    strain = s[:, axis] / s[axis, axis]  # uniaxial stress, unit strain along the axis
    uu = u @ strain  # element displacements of the uniaxial stress (m x n)
    km, curvature = problem.derivatives(modes, mode)
    force = np.einsum("mab,mb->ma", km, uu)  # the force of a unit factor on each beam
    first = np.einsum("ma,ma->m", uu, force) / problem.volume
    others = [j for j in range(len(c)) if j != axis]
    g = np.einsum("maj,ma->mj", u[:, :, others], force)
    relaxed = np.einsum("mj,jk,mk->m", g, np.linalg.inv(problem.volume * c[np.ix_(others, others)]),
                        g)  # fmt: skip
    fluct = np.empty(len(force))
    for start in range(0, len(force), BATCH):
        part = slice(start, start + BATCH)
        k = len(force[part])
        local = np.zeros((k, force.shape[1], k))
        local[np.arange(k), :, np.arange(k)] = force[part]
        rhs = np.zeros((problem.n_dofs, k))
        np.add.at(rhs, problem.dofs[part], local)
        rhs = rhs[problem.free]
        fluct[part] = np.einsum("ij,ij->j", rhs, lu.solve(rhs))
    second = -2.0 / problem.volume * (fluct + relaxed)
    if curvature is not None:  # the element stiffness itself concave in the factor
        second = second + np.einsum("ma,mab,mb->m", uu, curvature, uu) / problem.volume
    return {
        "young": float(e0),
        "mean": float(-second.sum() / (2.0 * e0)),
        "variance": float(np.sum(first**2) / e0**2),
        "dE": first,
        "d2E": second,
    }


@dataclass
class Softening:
    """Replicas of the stiffness of the beams of a frame (:func:`softening`): the
    coefficients of variation ``cv`` and, for each, the mean and standard deviation of the
    Young's modulus over the replicas relative to that of the identical beams ``young``,
    the mean corrected by its first-order part (a control variate of zero mean, which
    removes most of the scatter of the replicas) and its standard error, the
    coefficients ``s`` and ``v`` of :func:`softening_coefficients`, and the study: the
    ``mode`` varied, the ``law`` of its factors, the number of ``replicas`` per CV and the
    ``axis`` of the uniaxial stress."""

    young: float
    cv: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    corrected: np.ndarray
    error: np.ndarray
    coefficients: dict
    mode: str = "modulus"
    law: str = "normal"
    replicas: int = 0
    axis: int = 0

    def softening(self) -> np.ndarray:
        """``(E_0 - ⟨E⟩) / E_0`` of the corrected means."""
        return 1.0 - self.corrected

    def predicted(self) -> np.ndarray:
        """``s CV²``, the second-order softening."""
        return self.coefficients["mean"] * self.cv**2

    def scatter(self) -> np.ndarray:
        """``σ_E² / E_0²`` of the replicas."""
        return self.std**2

    def predicted_scatter(self) -> np.ndarray:
        """``v CV²``, the second-order scatter."""
        return self.coefficients["variance"] * self.cv**2

    def rows(self) -> list[dict]:
        """A row per coefficient of variation: the study (``mode``, ``law``, ``cv``,
        ``replicas``), the plain ``mean`` of ``E / E_0`` over the replicas and its standard
        error ``σ / √n`` (``mean_error``), the ``corrected`` mean and its standard error,
        the second-order mean ``1 - s CV²`` (``theory``), the standard deviation ``std``
        of ``E / E_0``, the ``scatter`` ``σ_E² / E_0²`` with its standard error for normal
        draws, ``σ² √(2 / (n - 1))``, and ``v CV²`` (``theory_scatter``)."""
        n = max(int(self.replicas), 2)
        rows = zip(self.cv, self.mean, self.std, self.corrected, self.error,
                   1.0 - self.predicted(), self.predicted_scatter(), strict=True)  # fmt: skip
        return [
            {
                "mode": self.mode,
                "law": self.law,
                "cv": float(cv),
                "replicas": int(self.replicas),
                "mean": float(mean),
                "mean_error": float(std / np.sqrt(n)),
                "corrected": float(corrected),
                "corrected_error": float(error),
                "theory": float(theory),
                "std": float(std),
                "scatter": float(std**2),
                "scatter_error": float(std**2 * np.sqrt(2.0 / (n - 1))),
                "theory_scatter": float(scatter),
            }
            for cv, mean, std, corrected, error, theory, scatter in rows
        ]


def softening(
    frame,
    mode: str = "modulus",
    cvs=(0.1, 0.2, 0.3),
    *,
    replicas: int = 100,
    law: str = "normal",
    axis: int = 0,
    seed: SeedLike = None,
    callback=None,
) -> Softening:
    """The softening of a network of non-identical beams (Ban et al. 2016): ``replicas``
    draws of the stiffness of a mode of the beams (:func:`fibre_scales` of each ``cv``
    and ``law``; :func:`scaled_frame`) on the same geometry, the Young's modulus along
    ``axis`` of each, against the second-order theory. ``callback(done, total)`` follows
    the replicas (one solve each)."""
    if replicas < 2:
        raise ValueError("replicas: at least 2")
    if law not in LAWS:
        raise ValueError(f"law: one of {LAWS}")
    cvs = np.atleast_1d(np.asarray(cvs, dtype=float))
    if cvs.size == 0:
        raise ValueError("cvs: at least one coefficient of variation")
    if np.any(cvs < 0) or (law == "bimodal" and np.any(cvs >= 1.0)):
        raise ValueError("cv: non-negative (below 1 for the bimodal law)")
    coefficients = softening_coefficients(frame, mode, axis)
    e0, slope = coefficients["young"], coefficients["dE"]
    rng, _ = make_rng(seed)
    problem = _Periodic(frame)
    base = problem.values()
    rows = []
    total, done = cvs.size * int(replicas), 0
    for cv in cvs:
        values, linear = [], []
        for _ in range(int(replicas)):
            scale = fibre_scales(len(frame.beams), float(cv), law, rng)
            ke = sum(problem.stiffness(**{**base, **_scaled(frame, scale, mode)}).values())
            u, _, _ = problem.solve(ke)
            s = np.linalg.inv(problem.energy(u, ke))
            values.append(1.0 / s[axis, axis])
            linear.append(float(slope @ (scale - 1.0)))
            done += 1
            if callback is not None:
                callback(done, total)
        values, linear = np.array(values) / e0, np.array(linear) / e0
        corrected = values - linear
        rows.append((cv, values.mean(), values.std(ddof=1), corrected.mean(),
                     corrected.std(ddof=1) / np.sqrt(len(values))))  # fmt: skip
    cv, mean, std, corrected, error = (np.array(col) for col in zip(*rows, strict=True))
    return Softening(e0, cv, mean, std, corrected, error, coefficients, mode, law,
                     int(replicas), int(axis))  # fmt: skip
