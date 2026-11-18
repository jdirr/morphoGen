"""Periodic frames of Euler–Bernoulli beams in the plane (:class:`Frame`: the honeycomb,
triangular, square and kagome lattices of architectured materials) and in space
(:class:`SpaceFrame`: the cubic, body-centred cubic, octet and Kelvin lattices of struts),
their linear stiffness, the axial forces of a macroscopic strain and the linear buckling
of a cell of several primitive cells (Azulay 2023).

In space, a node carries ``(u, v, w, θx, θy, θz)``; a strut of circular section has the
axial (``EA/L``), torsional (``GJ/L``) and bending (``EI``, the same about both axes of
the section) stiffness of a 3D Euler–Bernoulli beam and the consistent geometric
stiffness of its transverse motions in both planes, with the torsional term
``N I_p / (A L)``; rotations (pseudo-vectors) turn by ``det(R) R`` under a symmetry.

A node carries ``(u, v, θ)``. A beam of axis ``e`` and length ``L`` has the stiffness of
Euler–Bernoulli beams (axial ``EA/L``; cubic Hermite bending ``EI/L^3 [12, 6L, 4L^2,
...]``) and, under the axial force ``N`` (positive in tension), the consistent geometric
stiffness ``N/30L [36, 3L, 4L^2, -L^2, ...]`` of its transverse motion. The cell is
periodic: the displacement of a node is ``E X + w``, ``E`` the macroscopic strain and
``w`` (and ``θ``) periodic, so that a beam from node ``i`` to node ``j`` shifted by the
periods ``s`` of the cell (``x_j + A s - x_i`` its vector ``d``) stretches by
``e · (E d + w_j - w_i)``. The axial forces of a macroscopic strain come from the linear
periodic problem; the linear buckling loads ``λ`` of that strain are those of
``(K_e + λ K_g) φ = 0`` on the periodic fluctuations, translations excluded.

Both frames take, as an option, Timoshenko beams: a transverse shear stiffness ``κGA`` per
beam (the attribute ``shear``; ``shear_coefficient`` κ in ``from_framework`` and
``lattice_of``, Cowper's for solid sections by :func:`cowper_coefficient`) gives the exact
two-node Timoshenko stiffness, ``EI / L³(1 + Φ) [12, 6L, (4 + Φ)L², (2 − Φ)L², ...]`` with
``Φ = 12 EI / (κGA L²)`` in each bending plane (``θ`` the rotation of the section), and the
consistent geometric stiffness of its interpolation, ``N / 30L(1 + Φ)² [36 + 60Φ + 30Φ²,
3L, (4 + 5Φ + 2.5Φ²)L², −(1 + 5Φ + 2.5Φ²)L², ...]``; without it (``shear`` None) the beams
are those of Euler–Bernoulli, ``Φ = 0``.

The frames of random periodic frameworks (fibre networks, open foams, lattices with
defects) are built by ``from_framework``, beams of circular sections along their bars, and
their effective stiffness is solved on sparse matrices by
:mod:`morphogen.homogenization.networks` (``effective_stiffness``). As an option
(``rigid_joints``, the attribute ``rigid``), the beams of a framework have rigid ends at
their joints, the struts merged in the nodes of a continuum: a beam flexible over ``l - a₁
- a₂`` whose ends move with the rigid arms ``a₁ e`` and ``-a₂ e`` of its nodes, ``u + θ ×
a e``; only the effective stiffness of the sparse problem takes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.linalg import block_diag, eigh

#: Primitive cells (wall length 1): lattice vectors (columns), nodes, walls ``(i, j, s)``
#: from node ``i`` to node ``j`` shifted by the primitive periods ``s``, and the centre
#: and generators of the point group of the lattice.
_S3 = np.sqrt(3.0)


def _rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


LATTICES = {
    "honeycomb": {
        "lattice": np.array([[_S3, _S3 / 2], [0.0, 1.5]]),
        "nodes": np.array([[0.0, 1.0], [-_S3 / 2, 0.5]]),
        "walls": [(0, 1, (0, 0)), (0, 1, (1, 0)), (0, 1, (0, 1))],
        "centre": np.zeros(2),
        "rotation": np.pi / 3,
    },
    "triangular": {
        "lattice": np.array([[1.0, 0.5], [0.0, _S3 / 2]]),
        "nodes": np.zeros((1, 2)),
        "walls": [(0, 0, (1, 0)), (0, 0, (0, 1)), (0, 0, (-1, 1))],
        "centre": np.zeros(2),
        "rotation": np.pi / 3,
    },
    "square": {
        "lattice": np.eye(2),
        "nodes": np.zeros((1, 2)),
        "walls": [(0, 0, (1, 0)), (0, 0, (0, 1))],
        "centre": np.zeros(2),
        "rotation": np.pi / 2,
    },
    "kagome": {
        "lattice": np.array([[2.0, 1.0], [0.0, _S3]]),
        "nodes": np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _S3 / 2]]),
        "walls": [
            (0, 1, (0, 0)),
            (1, 2, (0, 0)),
            (2, 0, (0, 0)),
            (1, 0, (1, 0)),
            (2, 0, (0, 1)),
            (2, 1, (-1, 1)),
        ],
        "centre": np.array([1.5, _S3 / 2]),  # a hexagon
        "rotation": np.pi / 3,
    },
}


def _framework_arrays(framework, dimension: int):
    """Nodes, bars, shifts and lattice of a periodic framework (an object with the
    attributes of :class:`morphogen.design.frames.Framework`, or its dictionary) in the
    plane (its first two axes; no bar may cross the box along the third) or in space."""
    if isinstance(framework, dict):
        from morphogen.design.frames import Framework

        framework = Framework.from_dict(framework)
    nodes = np.asarray(framework.nodes, dtype=float)
    bars = np.asarray(framework.bars, dtype=np.int64).reshape(-1, 2)
    shifts = np.asarray(framework.shifts, dtype=np.int64).reshape(len(bars), -1)
    box = np.asarray(framework.box, dtype=float).ravel()
    if len(bars) == 0:
        raise ValueError("from_framework: a framework without bars")
    if np.any(np.bincount(bars.ravel(), minlength=len(nodes)) == 0):
        raise ValueError("from_framework: isolated nodes (without_isolated_nodes() first)")
    if dimension == 2:
        if shifts.shape[1] > 2 and np.any(shifts[:, 2:] != 0):
            raise ValueError("from_framework: a bar crosses the box along z (not a plane frame)")
        if nodes.shape[1] > 2 and np.ptp(nodes[:, 2]) > 1e-9 * float(np.max(box)):
            raise ValueError(
                "from_framework: the nodes lie at several heights (not a plane frame: "
                "SpaceFrame.from_framework, or network_stiffness with dimension 3)"
            )
        return nodes[:, :2], bars, shifts[:, :2], np.diag(box[:2])
    return nodes, bars, shifts, np.diag(box)


def _per_bar(value, count: int, name: str) -> np.ndarray:
    """One positive value per bar (a single value repeated)."""
    out = np.broadcast_to(np.asarray(value, dtype=float), (count,)).copy()
    if not np.all(np.isfinite(out)) or np.any(out <= 0):
        raise ValueError(f"from_framework: {name} positive, one value or one per bar")
    return out


#: The least flexible part of a beam with rigid ends, relative to its length: the rigid
#: ends of the shortest beams (edges shorter than two rigid lengths, inside their nodes)
#: are scaled down to leave it
FLEXIBLE_LIMIT = 0.1


def _rigid_ends(rigid_joints, radius: np.ndarray, lengths: np.ndarray):
    """The lengths of the rigid ends of each beam at its two nodes (m x 2), ``rigid_joints``
    radii of the beam (one value or one per bar; None or 0: no rigid ends), scaled down on
    the beams shorter than them so that :data:`FLEXIBLE_LIMIT` of each stays flexible."""
    if rigid_joints is None:
        return None
    c = np.broadcast_to(np.asarray(rigid_joints, dtype=float), radius.shape)
    if not np.all(np.isfinite(c)) or np.any(c < 0):
        raise ValueError("from_framework: rigid_joints non-negative, one value or one per bar")
    if not np.any(c > 0):  # ends of no length: the frame without rigid ends, every path open
        return None
    a = c * radius
    scale = np.minimum(1.0, 0.5 * (1.0 - FLEXIBLE_LIMIT) * lengths / np.maximum(a, 1e-300))
    return np.repeat((a * scale)[:, None], 2, axis=1)


def _check_rigid(frame) -> None:
    """The dense paths (stiffness, axial forces, buckling) take beams without rigid ends."""
    if frame.rigid is not None:
        raise ValueError(
            "rigid joints: the effective stiffness of the sparse periodic problem only "
            "(effective_stiffness, networks.frame_stiffness)"
        )


#: The solid sections of Cowper's shear coefficients
SECTIONS = ("circular", "rectangular")


def cowper_coefficient(poisson, section: str = "circular"):
    """Cowper's (1966) shear coefficient ``κ`` of a solid section of Poisson's ratio
    ``poisson``: ``6(1 + ν)/(7 + 6ν)`` for a circle, ``10(1 + ν)/(12 + 11ν)`` for a
    rectangle (the transverse shear stiffness of a Timoshenko beam is ``κGA``)."""
    nu = np.asarray(poisson, dtype=float)
    if np.any(nu <= -1.0) or np.any(nu >= 0.5):
        raise ValueError("cowper_coefficient: poisson in (-1, 0.5)")
    if section == "circular":
        return 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)
    if section == "rectangular":
        return 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu)
    raise ValueError(f"cowper_coefficient: section one of {SECTIONS}")


def _shear_stiffness(coefficient, young, poisson, area, section: str):
    """The transverse shear stiffness ``κGA`` of each beam, ``G = E / 2(1 + ν)``, for a
    shear ``coefficient`` κ (one value, one per beam, or ``"cowper"``: Cowper's for the
    section); None (Euler–Bernoulli beams) without coefficient."""
    if coefficient is None:
        return None
    count = len(area)
    nu = np.broadcast_to(np.asarray(poisson, dtype=float), (count,))
    if isinstance(coefficient, str):
        if coefficient != "cowper":
            raise ValueError("shear_coefficient: 'cowper', a value or one per beam")
        kappa = cowper_coefficient(nu, section)
    else:
        kappa = _per_bar(coefficient, count, "shear_coefficient")
        if np.any(nu <= -1.0) or np.any(nu >= 0.5):
            raise ValueError("from_framework: poisson in (-1, 0.5)")
    return kappa * young / (2.0 * (1.0 + nu)) * area


def _shear_parameter(bending, shear, length):
    """The shear parameter ``Φ = 12 EI / (κGA L²)`` of a Timoshenko beam: 0 for an
    Euler–Bernoulli beam (``shear`` None or infinite)."""
    if shear is None:
        return 0.0
    return 12.0 * bending / (shear * length**2)


def beam_matrices(d, axial: float, bending: float, force: float = 0.0, shear=None):
    """Elastic and geometric stiffness (6 x 6, global axes, dofs ``u1 v1 θ1 u2 v2 θ2``) of
    a beam of vector ``d``, ``EA``, ``EI`` and axial force ``force``: an Euler–Bernoulli
    beam, or with the transverse shear stiffness ``shear`` (``κGA``) the exact two-node
    Timoshenko beam of shear parameter ``Φ = 12 EI / (κGA L²)`` (``θ`` the rotation of
    the section; Przemieniecki 1968) and the consistent geometric stiffness ``N ∫ v'² dx``
    of its interpolation (Friedman & Kosmatka 1993), the Euler–Bernoulli ones at ``Φ =
    0``."""
    length = float(np.hypot(*d))
    c, s = d[0] / length, d[1] / length
    L = length
    phi = _shear_parameter(bending, shear, L)
    a, b = axial / length, bending / length**3 / (1.0 + phi)
    ke = np.array(
        [
            [a, 0, 0, -a, 0, 0],
            [0, 12 * b, 6 * b * L, 0, -12 * b, 6 * b * L],
            [0, 6 * b * L, (4 + phi) * b * L**2, 0, -6 * b * L, (2 - phi) * b * L**2],
            [-a, 0, 0, a, 0, 0],
            [0, -12 * b, -6 * b * L, 0, 12 * b, -6 * b * L],
            [0, 6 * b * L, (2 - phi) * b * L**2, 0, -6 * b * L, (4 + phi) * b * L**2],
        ]
    )
    g = force / (30.0 * L * (1.0 + phi) ** 2)
    p, q, r = _geometric_terms(phi, L)
    kg = g * np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, p, 3 * L, 0, -p, 3 * L],
            [0, 3 * L, q, 0, -3 * L, r],
            [0, 0, 0, 0, 0, 0],
            [0, -p, -3 * L, 0, p, -3 * L],
            [0, 3 * L, r, 0, -3 * L, q],
        ]
    )
    rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    t = np.kron(np.eye(2), rot)
    return t.T @ ke @ t, t.T @ kg @ t


def _geometric_terms(phi: float, length: float) -> tuple[float, float, float]:
    """The entries ``(v v, θ θ, θ1 θ2)`` of the geometric stiffness of a Timoshenko beam
    times ``30 L (1 + Φ)² / N``: ``36``, ``4L²`` and ``-L²`` at ``Φ = 0``."""
    return (
        36 + 60 * phi + 30 * phi**2,
        (4 + 5 * phi + 2.5 * phi**2) * length**2,
        -(1 + 5 * phi + 2.5 * phi**2) * length**2,
    )


@dataclass
class Frame:
    """A periodic frame: ``nodes`` (positions in the cell), ``beams`` (node pairs),
    ``shifts`` (periods between the ends: beam ``k`` ends at ``nodes[j] + lattice @
    shifts[k]``), the ``lattice`` of the cell (vectors in the columns), ``EA`` and ``EI`` of
    each beam, the centre and generators ``(R, t)`` of its symmetry group, the area of
    the section of each beam (for the fibre volume per unit area), the transverse shear
    stiffness ``κGA`` of each beam: Timoshenko beams (None, the default: Euler–Bernoulli
    beams, rigid in shear) and the lengths of the rigid ends of each beam at its first and
    second node (m x 2; None, the default: none), which the effective stiffness alone
    takes."""

    nodes: np.ndarray
    beams: np.ndarray
    shifts: np.ndarray
    lattice: np.ndarray
    axial: np.ndarray
    bending: np.ndarray
    centre: np.ndarray
    generators: list
    area: np.ndarray | None = None
    shear: np.ndarray | None = None
    rigid: np.ndarray | None = None

    @classmethod
    def from_framework(
        cls, framework, radius, *, young=1.0, poisson=0.3, shear_coefficient=None,
        rigid_joints=None,
    ) -> Frame:  # fmt: skip
        """The plane frame of a periodic framework of the plane (the first two axes of a
        :class:`morphogen.design.frames.Framework`, or of its dictionary as the metadata
        of the planar network models keep it): a beam per bar, of circular section of
        ``radius`` and Young's modulus ``young`` (one value, or one per bar), ``EA = E π
        r²`` and ``EI = E π r⁴ / 4`` (bending in the plane). The box is the cell; its
        symmetry, the translations alone. With a ``shear_coefficient`` κ (one value, one
        per bar, or ``"cowper"``: Cowper's ``6(1 + ν)/(7 + 6ν)``), Timoshenko beams of
        shear stiffness ``κGA``, ``G = E / 2(1 + ν)`` (``poisson`` one value or one per
        bar); Euler–Bernoulli beams without. With ``rigid_joints`` (one value or one per
        bar), the beams are rigid over that many of their radii from each node (the
        struts merged in the joints; :data:`FLEXIBLE_LIMIT` of the shortest left
        flexible)."""
        nodes, bars, shifts, lattice = _framework_arrays(framework, 2)
        r = _per_bar(radius, len(bars), "radius")
        e = _per_bar(young, len(bars), "young")
        area = np.pi * r**2
        frame = cls(
            nodes,
            bars,
            shifts,
            lattice,
            e * area,
            e * np.pi * r**4 / 4.0,
            np.zeros(2),
            [(np.eye(2), np.zeros(2))],
            area,
            _shear_stiffness(shear_coefficient, e, poisson, area, "circular"),
        )
        frame.rigid = _rigid_ends(rigid_joints, r, np.linalg.norm(frame.vectors(), axis=1))
        return frame

    @classmethod
    def lattice_of(
        cls,
        kind: str = "honeycomb",
        cells=(2, 2),
        *,
        length: float = 1.0,
        elements: int = 4,
        young: float = 1.0,
        thickness: float = 0.1,
        depth: float = 1.0,
        poisson: float = 0.3,
        shear_coefficient=None,
    ) -> Frame:
        """A cell of ``cells`` primitive cells of a lattice (``honeycomb``, ``triangular``,
        ``square`` or ``kagome``) of walls of ``length``, each cut in ``elements`` beams of
        Young's modulus ``young`` and rectangular section ``thickness x depth``:
        Euler–Bernoulli beams, or with a ``shear_coefficient`` κ (a value, or
        ``"cowper"``: Cowper's ``10(1 + ν)/(12 + 11ν)`` of rectangles) Timoshenko beams of
        shear stiffness ``κGA``, ``G = E / 2(1 + ν)``."""
        if kind not in LATTICES:
            raise ValueError(f"lattice_of: one of {sorted(LATTICES)}")
        spec = LATTICES[kind]
        n1, n2 = (int(c) for c in cells)
        a = spec["lattice"] * length
        lattice = a @ np.diag([n1, n2])
        inv = np.linalg.inv(lattice)
        base = spec["nodes"] * length
        npc = len(base)
        nodes = [base[p] + a @ (i, j) for i in range(n1) for j in range(n2) for p in range(npc)]

        def index(p, i, j):
            return (i % n1) * n2 * npc + (j % n2) * npc + p

        def wrap(x):
            return lattice @ ((inv @ x) % 1.0)

        nodes = [wrap(x) for x in nodes]
        beams, shifts = [], []
        for i in range(n1):
            for j in range(n2):
                for p, q, (s1, s2) in spec["walls"]:
                    start = index(p, i, j)
                    end = index(q, i + s1, j + s2)
                    x0 = nodes[start]
                    x1 = base[q] + a @ (i + s1, j + s2)
                    x1 = x0 + (x1 - (base[p] + a @ (i, j)))  # unwrapped from the start node
                    chain = [start]
                    for k in range(1, int(elements)):
                        nodes.append(wrap(x0 + (k / elements) * (x1 - x0)))
                        chain.append(len(nodes) - 1)
                    chain.append(end)
                    for k in range(int(elements)):
                        u, v = chain[k], chain[k + 1]
                        step = (x1 - x0) / elements
                        shift = np.rint(inv @ (nodes[u] + step - nodes[v])).astype(int)
                        beams.append((u, v))
                        shifts.append(shift)
        area, inertia = thickness * depth, depth * thickness**3 / 12.0
        m = len(beams)
        rot = _rotation(spec["rotation"])
        mirror = np.array([[-1.0, 0.0], [0.0, 1.0]])
        generators = [
            (rot, np.zeros(2)),
            (mirror, np.zeros(2)),
            (np.eye(2), a[:, 0]),
            (np.eye(2), a[:, 1]),
        ]
        return cls(
            np.array(nodes),
            np.array(beams, dtype=np.int64),
            np.array(shifts, dtype=np.int64),
            lattice,
            np.full(m, young * area),
            np.full(m, young * inertia),
            spec["centre"] * length,
            generators,
            np.full(m, area),
            _shear_stiffness(shear_coefficient, young, poisson, np.full(m, area), "rectangular"),
        )

    @property
    def n_dofs(self) -> int:
        return 3 * len(self.nodes)

    def _shears(self) -> list:
        """The transverse shear stiffness ``κGA`` of each beam (None: Euler–Bernoulli)."""
        return [None] * len(self.beams) if self.shear is None else list(self.shear)

    @staticmethod
    def block(rotation) -> np.ndarray:
        """The action of a point operation on the dofs ``(u, v, θ)`` of a node."""
        r = np.asarray(rotation, dtype=float)
        return block_diag(r, np.linalg.det(r))

    def vectors(self) -> np.ndarray:
        """The vector of each beam, from its first end to its second."""
        return (
            self.nodes[self.beams[:, 1]]
            + self.shifts @ self.lattice.T
            - self.nodes[self.beams[:, 0]]
        )

    def _dofs(self) -> np.ndarray:
        e = self.beams
        return np.column_stack(
            [
                3 * e[:, 0],
                3 * e[:, 0] + 1,
                3 * e[:, 0] + 2,
                3 * e[:, 1],
                3 * e[:, 1] + 1,
                3 * e[:, 1] + 2,
            ]
        )

    def _assemble(self, forces=None) -> tuple[np.ndarray, np.ndarray]:
        _check_rigid(self)
        n = self.n_dofs
        ke, kg = np.zeros((n, n)), np.zeros((n, n))
        forces = np.zeros(len(self.beams)) if forces is None else np.asarray(forces, dtype=float)
        for d, a, b, f, q, dofs in zip(
            self.vectors(), self.axial, self.bending, forces, self._shears(), self._dofs(),
            strict=True,
        ):  # fmt: skip
            e, g = beam_matrices(d, a, b, f, q)
            ke[np.ix_(dofs, dofs)] += e
            kg[np.ix_(dofs, dofs)] += g
        return ke, kg

    def stiffness(self) -> np.ndarray:
        """The elastic stiffness of the periodic fluctuations (dense)."""
        return self._assemble()[0]

    def translations(self) -> np.ndarray:
        """The two rigid translations of the cell (dofs x 2, orthonormal)."""
        t = np.zeros((self.n_dofs, 2))
        t[0::3, 0] = t[1::3, 1] = 1.0 / np.sqrt(len(self.nodes))
        return t

    def complement(self) -> np.ndarray:
        """An orthonormal basis of the fluctuations orthogonal to the translations."""
        t = self.translations()
        q, _ = np.linalg.qr(np.hstack([t, np.eye(self.n_dofs)]))
        return q[:, 2:]

    def axial_forces(self, strain) -> tuple[np.ndarray, np.ndarray]:
        """The axial forces of the beams under the macroscopic ``strain`` (2 x 2) and the
        periodic fluctuations (dofs) of the linear periodic problem."""
        e = np.asarray(strain, dtype=float)
        ke = self.stiffness()
        dofs = self._dofs()
        force = np.zeros(self.n_dofs)
        affine = []
        for d, a, b, q, idx in zip(
            self.vectors(), self.axial, self.bending, self._shears(), dofs, strict=True
        ):
            k, _ = beam_matrices(d, a, b, shear=q)
            u = np.zeros(6)
            u[3:5] = e @ d  # the affine motion of the second end relative to the first
            affine.append(u)
            force[idx] += k @ u
        z = self.complement()
        w = z @ np.linalg.solve(z.T @ ke @ z, -z.T @ force)
        forces = np.empty(len(self.beams))
        for k, (d, a, idx, u) in enumerate(
            zip(self.vectors(), self.axial, dofs, affine, strict=True)
        ):
            length = float(np.hypot(*d))
            relative = u[3:5] + w[idx[3:5]] - w[idx[0:2]]
            forces[k] = a / length * float(d @ relative) / length
        return forces, w

    def buckling(self, strain, modes: int = 6) -> tuple[np.ndarray, np.ndarray]:
        """The lowest ``modes`` positive load factors ``λ`` of the linear buckling of the
        cell under ``λ strain`` and their modes (dofs x modes)."""
        forces, _ = self.axial_forces(strain)
        ke, kg = self._assemble(forces)
        z = self.complement()
        return _buckling(z.T @ ke @ z, z.T @ kg @ z, z, modes)

    def effective_stiffness(self):
        """The effective stiffness of the periodic frame (3 x 3, Voigt order 11, 22, 12
        with the engineering shear, per unit area of the plane), its split into stretching
        and bending and the non-affinity of the fluctuations, by the sparse periodic
        problem (:func:`morphogen.homogenization.networks.frame_stiffness`)."""
        from morphogen.homogenization.networks import frame_stiffness

        return frame_stiffness(self)

    def reciprocal(self) -> np.ndarray:
        """The reciprocal vectors of the cell (columns): ``b_i · a_j = 2π δ_ij``."""
        return 2.0 * np.pi * np.linalg.inv(self.lattice).T

    def bloch_buckling(self, strain, wavevectors, modes: int = 1) -> np.ndarray:
        """The lowest ``modes`` load factors of the linear buckling of the infinite frame
        under ``λ strain`` for the Bloch modes ``u(x + A s) = u(x) exp(i k · A s)`` of each
        wave vector (rows: wave vectors): the smallest over the Brillouin zone gives the
        critical wavelength, hence the cell of the patterns (Geymonat, Müller &
        Triantafyllidis 1993); the cell at ``k`` of ``(m1 b1 + m2 b2) / n`` is that of
        ``n x n`` cells."""
        _check_rigid(self)
        forces, _ = self.axial_forces(strain)
        n = self.n_dofs
        mats = [
            beam_matrices(d, a, b, f, q)
            for d, a, b, f, q in zip(
                self.vectors(), self.axial, self.bending, forces, self._shears(), strict=True
            )
        ]
        periods = self.shifts @ self.lattice.T
        out = []
        for k in np.atleast_2d(np.asarray(wavevectors, dtype=float)):
            ke, kg = np.zeros((n, n), complex), np.zeros((n, n), complex)
            for (e, g), dofs, period in zip(mats, self._dofs(), periods, strict=True):
                p = np.ones(6, complex)
                p[3:] = np.exp(1j * (k @ period))
                ke[np.ix_(dofs, dofs)] += np.conj(p)[:, None] * e * p[None, :]
                kg[np.ix_(dofs, dofs)] += np.conj(p)[:, None] * g * p[None, :]
            frac = self.lattice.T @ k / (2.0 * np.pi)  # Γ or its images: translations
            basis = self.complement() if np.allclose(frac, np.round(frac), atol=1e-9) else np.eye(n)
            ke, kg = basis.T @ ke @ basis, basis.T @ kg @ basis

            def real(h):
                h = 0.5 * (h + h.conj().T)
                return np.block([[h.real, -h.imag], [h.imag, h.real]])

            lam, _ = _buckling(real(ke), real(kg), np.eye(2 * basis.shape[1]), 2 * int(modes))
            out.append(np.pad(lam[0::2], (0, int(modes) - len(lam[0::2])), constant_values=np.nan))
        return np.array(out)

    def plot(self, ax=None, displacement=None, *, scale: float = 0.0, tiles=(1, 1), **style):
        """Draw the beams, displaced by ``displacement`` (its largest drawn as ``scale``
        times the median length of the beams; 0.3 by default), the cell repeated ``tiles``
        times; returns the axes."""
        if ax is None:
            from morphogen.viz.figures import _figure

            ax = _figure(5.0, 5.0).subplots()
        x = self.nodes.copy()
        if displacement is not None:
            u = np.asarray(displacement, dtype=float).reshape(-1, 3)[:, :2]
            peak = max(float(np.abs(u).max()), 1e-300)
            size = float(np.median(np.hypot(*self.vectors().T)))
            x = x + u * (scale or 0.3) * size / peak
        d = x[self.beams[:, 1]] + self.shifts @ self.lattice.T - x[self.beams[:, 0]]
        segments = []
        for i in range(int(tiles[0])):
            for j in range(int(tiles[1])):
                offset = self.lattice @ (i, j)
                start = x[self.beams[:, 0]] + offset
                segments.extend(zip(start, start + d, strict=True))
        from matplotlib.collections import LineCollection

        ax.add_collection(
            LineCollection(
                segments, colors=style.get("color", "k"), linewidths=style.get("lw", 1.0)
            )
        )
        ax.set_aspect("equal")
        ax.autoscale()
        ax.set_axis_off()
        return ax


def positive_subspace(k: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Orthonormal basis of the subspace where a symmetric stiffness is positive: its
    mechanisms (eigenvalues below ``tol`` times the largest) left out."""
    w, v = eigh(0.5 * (k + k.T))
    return v[:, w > tol * max(float(np.abs(w).max()), 1e-300)]


def _buckling(ke: np.ndarray, kg: np.ndarray, basis: np.ndarray, modes: int, floor=None):
    """Lowest positive ``λ`` of ``(K_e + λ K_g) φ = 0`` (``K_e`` positive definite): the
    eigenvalues ``μ = 1/λ`` of ``-K_g φ = μ K_e φ`` above ``floor`` (by default 1e-10
    times the largest ``|μ|``; a subspace where ``K_g`` vanishes has none)."""
    if ke.shape[0] == 0:
        return np.zeros(0), np.zeros((basis.shape[0], 0))
    ke, kg = 0.5 * (ke + ke.T), 0.5 * (kg + kg.T)
    try:
        mu, phi = eigh(-kg, ke)
    except np.linalg.LinAlgError:  # mechanisms (zero-energy modes): left out
        v = positive_subspace(ke)
        mu, phi = eigh(-(v.T @ kg @ v), v.T @ ke @ v)
        phi = v @ phi
    if floor is None:
        floor = 1e-10 * max(float(np.abs(mu).max()), 1e-300)
    keep = mu > floor
    loads, phi = 1.0 / mu[keep], phi[:, keep]
    order = np.argsort(loads)[: int(modes)]
    return loads[order], basis @ phi[:, order]


# -- frames in space ---------------------------------------------------------------------

_C4Z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
_C3 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # about [111]
#: The point group of the cubic lattices (O_h, about a node): generators
CUBIC_GENERATORS = (_C4Z, _C3, -np.eye(3))


def _kelvin_nodes() -> np.ndarray:
    """The vertices of the truncated octahedra of the body-centred cubic packing (the
    Kelvin cell), modulo the conventional cell of edge 1: permutations of ``(0, ±1/4,
    ±1/2)``."""
    out = []
    for perm in {(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)}:
        for s1, s2 in product((-1, 1), repeat=2):
            x = np.array([0.0, 0.25 * s1, 0.5 * s2])[list(perm)] % 1.0
            if not any(np.allclose(x, y) for y in out):
                out.append(x)
    return np.array(sorted(out, key=tuple))


_FCC = [np.array(c) for c in ((0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5))]


def _face_centred(basis) -> np.ndarray:
    """The nodes of a face-centred cubic lattice with a ``basis``, in its cubic cell."""
    sites = [np.zeros(3), *_FCC]
    return np.array([(site + np.asarray(b, dtype=float)) % 1.0 for site in sites for b in basis])


#: Lattices of struts (conventional cubic cell of edge 1): nodes, length of the struts
#: (between nearest neighbours) and the centring translations of the lattice; point group
#: O_h about the origin, or the ``generators`` of the space group (a point operation and a
#: translation each) where some of its operations move the origin: the diamond lattice
#: (a node of one of its two face-centred lattices at the origin: its tetrahedral group,
#: and the inversion about the middle of a strut) and the pyrochlore lattice of
#: tetrahedra joined by their corners (a node at the origin: its inversion and its
#: threefold axis, and the fourfold rotation-reflection about a node of the diamond
#: lattice whose struts it bisects).
SPACE_LATTICES = {
    "cubic": {"nodes": np.zeros((1, 3)), "strut": 1.0, "centring": []},
    "bcc": {
        "nodes": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        "strut": np.sqrt(3.0) / 2,
        "centring": [np.array([0.5, 0.5, 0.5])],
    },
    "octet": {
        "nodes": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]),
        "strut": np.sqrt(2.0) / 2,
        "centring": [np.array(c) for c in ((0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5))],
    },
    "kelvin": {
        "nodes": _kelvin_nodes(),
        "strut": np.sqrt(2.0) / 4,
        "centring": [np.array([0.5, 0.5, 0.5])],
    },
    "diamond": {
        "nodes": _face_centred([(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]),
        "strut": np.sqrt(3.0) / 4,
        "centring": _FCC,
        "generators": [
            (-_C4Z, np.zeros(3)),
            (_C3, np.zeros(3)),
            (-np.eye(3), np.array([0.25, 0.25, 0.25])),
        ],
    },
    "pyrochlore": {
        "nodes": _face_centred(
            [(0.0, 0.0, 0.0), (0.0, 0.75, 0.75), (0.75, 0.0, 0.75), (0.75, 0.75, 0.0)]
        ),
        "strut": np.sqrt(2.0) / 4,
        "centring": _FCC,
        "generators": [
            (-_C4Z, np.array([0.0, -0.25, -0.25])),
            (_C3, np.zeros(3)),
            (-np.eye(3), np.zeros(3)),
        ],
    },
}


def _struts(nodes: np.ndarray, length: float, tol: float = 1e-6) -> list:
    """The struts ``(i, j, s)`` of length ``length`` between nodes of a periodic cell of
    edge 1 (node ``j`` shifted by the periods ``s``), each once."""
    out = []
    for i, xi in enumerate(nodes):
        for j, xj in enumerate(nodes):
            if j < i:
                continue
            for shift in product((-1, 0, 1), repeat=3):
                if j == i and shift <= (0, 0, 0):
                    continue
                if abs(np.linalg.norm(xj + np.array(shift) - xi) - length) < tol:
                    out.append((i, j, shift))
    return out


def space_beam_matrices(
    d, axial: float, bending: float, torsion: float, force=0.0, polar=0.0, shear=None
):
    """Elastic and geometric stiffness (12 x 12, global axes, dofs ``u v w θx θy θz`` of
    each end) of a 3D Euler–Bernoulli beam of vector ``d``: ``EA``, ``EI`` about both axes
    of its section (circular or square), ``GJ``, the axial force ``force`` (positive in
    tension) and ``polar`` = ``I_p / A`` of the section (the torsional term of the
    geometric stiffness). With the transverse shear stiffness ``shear`` (``κGA``, the same
    in both planes), the exact Timoshenko beam and its geometric stiffness in each bending
    plane, of shear parameter ``Φ = 12 EI / (κGA L²)`` (as :func:`beam_matrices`)."""
    d = np.asarray(d, dtype=float)
    length = float(np.linalg.norm(d))
    ex = d / length
    helper = np.array([0.0, 0.0, 1.0]) if abs(ex[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    ey = np.cross(helper, ex)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    L = length
    ke, kg = np.zeros((12, 12)), np.zeros((12, 12))
    for i, j, v in ((0, 6, axial / L), (3, 9, torsion / L)):
        ke[i, i] = ke[j, j] = v
        ke[i, j] = ke[j, i] = -v
    phi = _shear_parameter(bending, shear, L)
    b, g = bending / L**3 / (1.0 + phi), force / (30.0 * L * (1.0 + phi) ** 2)
    plane_y = np.array(  # (v, θz) of both ends
        [
            [12, 6 * L, -12, 6 * L],
            [6 * L, (4 + phi) * L**2, -6 * L, (2 - phi) * L**2],
            [-12, -6 * L, 12, -6 * L],
            [6 * L, (2 - phi) * L**2, -6 * L, (4 + phi) * L**2],
        ]
    )
    p, q, r = _geometric_terms(phi, L)
    geo_y = np.array(
        [
            [p, 3 * L, -p, 3 * L],
            [3 * L, q, -3 * L, r],
            [-p, -3 * L, p, -3 * L],
            [3 * L, r, -3 * L, q],
        ]
    )
    flip = np.diag([1.0, -1.0, 1.0, -1.0])  # (w, θy): θy = -dw/dx
    for dofs, sign in (([1, 5, 7, 11], np.eye(4)), ([2, 4, 8, 10], flip)):
        idx = np.ix_(dofs, dofs)
        ke[idx] += b * sign @ plane_y @ sign
        kg[idx] += g * sign @ geo_y @ sign
    t = force * polar / L
    kg[3, 3] = kg[9, 9] = t
    kg[3, 9] = kg[9, 3] = -t
    rot = np.kron(np.eye(4), np.array([ex, ey, ez]))
    return rot.T @ ke @ rot, rot.T @ kg @ rot


@dataclass
class SpaceFrame:
    """A periodic frame of struts in space: ``nodes`` (positions in the cell), ``beams``
    (node pairs), ``shifts`` (periods between the ends), the ``lattice`` of the cell
    (vectors in the columns), ``EA``, ``EI``, ``GJ`` and ``I_p / A`` of each strut, and the
    centre and generators ``(R, t)`` of its symmetry group, the area of the section of
    each strut (for the relative density), its transverse shear stiffness ``κGA``:
    Timoshenko beams (None, the default: Euler–Bernoulli beams) and the lengths of its
    rigid ends at its first and second node (m x 2; None, the default: none), which the
    effective stiffness alone takes."""

    nodes: np.ndarray
    beams: np.ndarray
    shifts: np.ndarray
    lattice: np.ndarray
    axial: np.ndarray
    bending: np.ndarray
    torsion: np.ndarray
    polar: np.ndarray
    centre: np.ndarray
    generators: list
    area: np.ndarray | None = None
    shear: np.ndarray | None = None
    rigid: np.ndarray | None = None

    @classmethod
    def lattice_of(
        cls,
        kind: str = "octet",
        cells=(1, 1, 1),
        *,
        size: float = 1.0,
        elements: int = 2,
        young: float = 1.0,
        poisson: float = 0.3,
        radius: float = 0.05,
        shear_coefficient=None,
    ) -> SpaceFrame:
        """A cell of ``cells`` conventional cubic cells of edge ``size`` of a lattice of
        struts (``cubic``, ``bcc``, ``octet``, ``kelvin``, ``diamond`` or ``pyrochlore``:
        :data:`SPACE_LATTICES`), each strut cut in
        ``elements`` beams of Young's modulus ``young``, Poisson's ratio ``poisson`` and
        circular section of ``radius`` (relative to ``size``): Euler–Bernoulli beams, or
        with a ``shear_coefficient`` κ (a value, or ``"cowper"``: Cowper's ``6(1 + ν)/(7 +
        6ν)``) Timoshenko beams of shear stiffness ``κGA``."""
        if kind not in SPACE_LATTICES:
            raise ValueError(f"lattice_of: one of {sorted(SPACE_LATTICES)}")
        spec = SPACE_LATTICES[kind]
        counts = tuple(int(c) for c in cells)
        a = np.eye(3) * size
        lattice = a @ np.diag(counts)
        inv = np.linalg.inv(lattice)
        base = spec["nodes"] * size
        npc = len(base)
        cell_ids = list(product(*(range(c) for c in counts)))
        position = {c: k for k, c in enumerate(cell_ids)}

        def index(p, c):
            return position[tuple(int(x) % n for x, n in zip(c, counts, strict=True))] * npc + p

        def wrap(x):
            return lattice @ ((inv @ x) % 1.0)

        nodes = [wrap(base[p] + a @ c) for c in cell_ids for p in range(npc)]
        beams, shifts = [], []
        for c in cell_ids:
            for p, q, s in _struts(spec["nodes"], spec["strut"]):
                start, end = index(p, c), index(q, np.add(c, s))
                x0 = nodes[start]
                x1 = x0 + a @ (np.asarray(s) + spec["nodes"][q] - spec["nodes"][p])
                chain = [start]
                for k in range(1, int(elements)):
                    nodes.append(wrap(x0 + (k / elements) * (x1 - x0)))
                    chain.append(len(nodes) - 1)
                chain.append(end)
                step = (x1 - x0) / elements
                for k in range(int(elements)):
                    u, v = chain[k], chain[k + 1]
                    beams.append((u, v))
                    shifts.append(np.rint(inv @ (nodes[u] + step - nodes[v])).astype(int))
        r = radius * size
        area, inertia = np.pi * r**2, np.pi * r**4 / 4.0
        shear = young / (2.0 * (1.0 + poisson))
        m = len(beams)
        point = spec.get("generators") or [(g, np.zeros(3)) for g in CUBIC_GENERATORS]
        generators = [(g, a @ t) for g, t in point]
        generators += [(np.eye(3), a[:, k]) for k in range(3)]
        generators += [(np.eye(3), a @ t) for t in spec["centring"]]
        return cls(
            np.array(nodes),
            np.array(beams, dtype=np.int64),
            np.array(shifts, dtype=np.int64),
            lattice,
            np.full(m, young * area),
            np.full(m, young * inertia),
            np.full(m, shear * 2.0 * inertia),
            np.full(m, 2.0 * inertia / area),
            np.zeros(3),
            generators,
            np.full(m, area),
            _shear_stiffness(shear_coefficient, young, poisson, np.full(m, area), "circular"),
        )

    @classmethod
    def from_framework(
        cls, framework, radius, *, young=1.0, poisson=0.3, shear_coefficient=None,
        rigid_joints=None,
    ) -> SpaceFrame:  # fmt: skip
        """The frame of the bars of a periodic framework in space
        (:class:`morphogen.design.frames.Framework`, or its dictionary as the metadata of
        ``delaunay_network``, ``open_foam`` and ``random_lattice`` keep it): a strut per
        bar, of circular section of ``radius``, Young's modulus ``young`` and Poisson's
        ratio ``poisson`` (each one value, or one per bar): ``EA = E π r²``, ``EI = E π r⁴ /
        4`` about both axes of the section, ``GJ = G π r⁴ / 2``. The box of the framework
        is the cell; its symmetry, the translations alone. With a ``shear_coefficient`` κ
        (one value, one per bar, or ``"cowper"``: Cowper's ``6(1 + ν)/(7 + 6ν)``),
        Timoshenko beams of shear stiffness ``κGA``; Euler–Bernoulli beams without. With
        ``rigid_joints`` (one value or one per bar), the struts are rigid over that many of
        their radii from each node (merged in the joints; :data:`FLEXIBLE_LIMIT` of the
        shortest left flexible)."""
        nodes, bars, shifts, lattice = _framework_arrays(framework, 3)
        m = len(bars)
        r = _per_bar(radius, m, "radius")
        e = _per_bar(young, m, "young")
        nu = np.broadcast_to(np.asarray(poisson, dtype=float), (m,))
        if np.any(nu <= -1.0) or np.any(nu >= 0.5):
            raise ValueError("from_framework: poisson in (-1, 0.5)")
        area, inertia = np.pi * r**2, np.pi * r**4 / 4.0
        frame = cls(
            nodes,
            bars,
            shifts,
            lattice,
            e * area,
            e * inertia,
            e / (2.0 * (1.0 + nu)) * 2.0 * inertia,
            2.0 * inertia / area,
            np.zeros(3),
            [(np.eye(3), np.zeros(3))],
            area,
            _shear_stiffness(shear_coefficient, e, nu, area, "circular"),
        )
        frame.rigid = _rigid_ends(rigid_joints, r, np.linalg.norm(frame.vectors(), axis=1))
        return frame

    @property
    def n_dofs(self) -> int:
        return 6 * len(self.nodes)

    def _shears(self) -> list:
        """The transverse shear stiffness ``κGA`` of each strut (None: Euler–Bernoulli)."""
        return [None] * len(self.beams) if self.shear is None else list(self.shear)

    @staticmethod
    def block(rotation) -> np.ndarray:
        """The action of a point operation on the dofs of a node: displacements turned by
        ``R``, rotations (pseudo-vectors) by ``det(R) R``."""
        r = np.asarray(rotation, dtype=float)
        return block_diag(r, np.linalg.det(r) * r)

    def vectors(self) -> np.ndarray:
        """The vector of each beam, from its first end to its second."""
        return (
            self.nodes[self.beams[:, 1]]
            + self.shifts @ self.lattice.T
            - self.nodes[self.beams[:, 0]]
        )

    def _dofs(self) -> np.ndarray:
        e = self.beams
        return np.hstack([6 * e[:, :1] + np.arange(6), 6 * e[:, 1:] + np.arange(6)])

    def _matrices(self, forces=None) -> list:
        _check_rigid(self)
        forces = np.zeros(len(self.beams)) if forces is None else np.asarray(forces, dtype=float)
        return [
            space_beam_matrices(d, a, b, t, f, p, q)
            for d, a, b, t, f, p, q in zip(
                self.vectors(),
                self.axial,
                self.bending,
                self.torsion,
                forces,
                self.polar,
                self._shears(),
                strict=True,
            )
        ]

    def _assemble(self, forces=None) -> tuple[np.ndarray, np.ndarray]:
        n = self.n_dofs
        ke, kg = np.zeros((n, n)), np.zeros((n, n))
        for (e, g), dofs in zip(self._matrices(forces), self._dofs(), strict=True):
            ke[np.ix_(dofs, dofs)] += e
            kg[np.ix_(dofs, dofs)] += g
        return ke, kg

    def stiffness(self) -> np.ndarray:
        """The elastic stiffness of the periodic fluctuations (dense)."""
        return self._assemble()[0]

    def translations(self) -> np.ndarray:
        """The three rigid translations of the cell (dofs x 3, orthonormal)."""
        t = np.zeros((self.n_dofs, 3))
        for c in range(3):
            t[c::6, c] = 1.0 / np.sqrt(len(self.nodes))
        return t

    def complement(self) -> np.ndarray:
        """An orthonormal basis of the fluctuations orthogonal to the translations."""
        t = self.translations()
        q, _ = np.linalg.qr(np.hstack([t, np.eye(self.n_dofs)]))
        return q[:, 3:]

    def axial_forces(self, strain) -> tuple[np.ndarray, np.ndarray]:
        """The axial forces of the struts under the macroscopic ``strain`` (3 x 3) and the
        periodic fluctuations (dofs) of the linear periodic problem."""
        e = np.asarray(strain, dtype=float)
        mats = self._matrices()
        dofs = self._dofs()
        n = self.n_dofs
        ke = np.zeros((n, n))
        force = np.zeros(n)
        affine = []
        for (k, _), d, idx in zip(mats, self.vectors(), dofs, strict=True):
            ke[np.ix_(idx, idx)] += k
            u = np.zeros(12)
            u[6:9] = e @ d  # the affine motion of the second end relative to the first
            affine.append(u)
            force[idx] += k @ u
        z = self.complement()
        w = z @ np.linalg.solve(z.T @ ke @ z, -z.T @ force)
        forces = np.empty(len(self.beams))
        for k, (d, a, idx, u) in enumerate(
            zip(self.vectors(), self.axial, dofs, affine, strict=True)
        ):
            length = float(np.linalg.norm(d))
            relative = u[6:9] + w[idx[6:9]] - w[idx[0:3]]
            forces[k] = a / length * float(d @ relative) / length
        return forces, w

    def buckling(self, strain, modes: int = 6) -> tuple[np.ndarray, np.ndarray]:
        """The lowest ``modes`` positive load factors ``λ`` of the linear buckling of the
        cell under ``λ strain`` and their modes (dofs x modes)."""
        forces, _ = self.axial_forces(strain)
        ke, kg = self._assemble(forces)
        z = self.complement()
        return _buckling(z.T @ ke @ z, z.T @ kg @ z, z, modes)

    def relative_density(self) -> float:
        """The volume of the struts over that of the cell (overlaps at the nodes counted)."""
        lengths = np.linalg.norm(self.vectors(), axis=1)
        return float(np.sum(self.area * lengths) / abs(np.linalg.det(self.lattice)))

    def macroscopic_stiffness(self) -> np.ndarray:
        """The apparent stiffness (6 x 6, Voigt order 11, 22, 33, 23, 31, 12 with
        engineering shear strains) of the periodic frame: the energy of the periodic
        solutions of the unit strains per unit volume."""
        from morphogen.homogenization.elements import VOIGT

        mats = self._matrices()
        dofs = self._dofs()
        z = self.complement()
        n = self.n_dofs
        ke = np.zeros((n, n))
        for (k, _), idx in zip(mats, dofs, strict=True):
            ke[np.ix_(idx, idx)] += k
        factor = np.linalg.cholesky(z.T @ ke @ z)
        states = []
        for a, b in VOIGT:
            e = np.zeros((3, 3))
            e[a, b] = e[b, a] = 1.0 if a == b else 0.5
            local = np.zeros((len(self.beams), 12))
            local[:, 6:9] = self.vectors() @ e.T
            force = np.zeros(n)
            for (k, _), idx, u in zip(mats, dofs, local, strict=True):
                force[idx] += k @ u
            y = np.linalg.solve(factor, -z.T @ force)
            w = z @ np.linalg.solve(factor.T, y)
            states.append(local + w[dofs])
        volume = abs(np.linalg.det(self.lattice))
        out = np.zeros((6, 6))
        for (k, _), b in zip(mats, range(len(self.beams)), strict=True):
            u = np.array([s[b] for s in states])  # 6 x 12
            out += u @ k @ u.T
        return 0.5 * (out + out.T) / volume

    def effective_stiffness(self):
        """:meth:`macroscopic_stiffness` by the sparse periodic problem, for large random
        frames, with its split into stretching, bending and torsion and the non-affinity
        of the fluctuations (:func:`morphogen.homogenization.networks.frame_stiffness`)."""
        from morphogen.homogenization.networks import frame_stiffness

        return frame_stiffness(self)

    def reciprocal(self) -> np.ndarray:
        """The reciprocal vectors of the cell (columns): ``b_i · a_j = 2π δ_ij``."""
        return 2.0 * np.pi * np.linalg.inv(self.lattice).T

    def bloch_buckling(self, strain, wavevectors, modes: int = 1) -> np.ndarray:
        """The lowest ``modes`` load factors of the linear buckling of the infinite frame
        under ``λ strain`` for the Bloch modes of each wave vector (rows), as
        :meth:`Frame.bloch_buckling`."""
        forces, _ = self.axial_forces(strain)
        n = self.n_dofs
        mats = self._matrices(forces)
        periods = self.shifts @ self.lattice.T
        out = []
        for k in np.atleast_2d(np.asarray(wavevectors, dtype=float)):
            ke, kg = np.zeros((n, n), complex), np.zeros((n, n), complex)
            for (e, g), dofs, period in zip(mats, self._dofs(), periods, strict=True):
                p = np.ones(12, complex)
                p[6:] = np.exp(1j * (k @ period))
                ke[np.ix_(dofs, dofs)] += np.conj(p)[:, None] * e * p[None, :]
                kg[np.ix_(dofs, dofs)] += np.conj(p)[:, None] * g * p[None, :]
            frac = self.lattice.T @ k / (2.0 * np.pi)
            basis = self.complement() if np.allclose(frac, np.round(frac), atol=1e-9) else np.eye(n)
            ke, kg = basis.T @ ke @ basis, basis.T @ kg @ basis

            def real(h):
                h = 0.5 * (h + h.conj().T)
                return np.block([[h.real, -h.imag], [h.imag, h.real]])

            lam, _ = _buckling(real(ke), real(kg), np.eye(2 * basis.shape[1]), 2 * int(modes))
            out.append(np.pad(lam[0::2], (0, int(modes) - len(lam[0::2])), constant_values=np.nan))
        return np.array(out)

    def plot(self, ax=None, displacement=None, *, scale: float = 0.0, tiles=(1, 1, 1), **style):
        """Draw the struts in 3D axes, displaced by ``displacement`` (its largest drawn as
        ``scale`` times the median length of the beams; 0.3 by default), the cell repeated
        ``tiles`` times; returns the axes."""
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        if ax is None:
            from morphogen.viz.figures import _figure

            ax = _figure(5.0, 5.0).add_subplot(projection="3d")
        x = self.nodes.copy()
        if displacement is not None:
            u = np.asarray(displacement, dtype=float).reshape(-1, 6)[:, :3]
            peak = max(float(np.abs(u).max()), 1e-300)
            size = float(np.median(np.linalg.norm(self.vectors(), axis=1)))
            x = x + u * (scale or 0.3) * size / peak
        d = x[self.beams[:, 1]] + self.shifts @ self.lattice.T - x[self.beams[:, 0]]
        segments = []
        for offset in product(*(range(int(t)) for t in tiles)):
            start = x[self.beams[:, 0]] + self.lattice @ np.asarray(offset, dtype=float)
            segments.extend(zip(start, start + d, strict=True))
        segments = np.array(segments)
        ax.add_collection3d(
            Line3DCollection(
                segments, colors=style.get("color", "k"), linewidths=style.get("lw", 0.8)
            )
        )
        lo, hi = segments.reshape(-1, 3).min(axis=0), segments.reshape(-1, 3).max(axis=0)
        for setter, a, b in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lo, hi, strict=True):
            setter(a, b)
        ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-9)))
        ax.set_axis_off()
        return ax
