"""Self-contact in the cells of the nonlinear homogenization: the walls of the pores of a
periodic cell that close on themselves along a loading path.

:class:`ContactCell` is the periodic element of :class:`~morphogen.homogenization.nonlinear.
NonlinearFEProblem` (the fluctuation and the macroscopic strain components as unknowns,
loading paths in strains, stresses or both, the behaviours of morphoGen's points) with the
forces of the contact of its free surface with itself (:mod:`morphogen.contact`).

*Images.* A wall of a pore may meet the image of another across a face of the cell. The
faces of the surface near the boundary of the cell are therefore given images, translated
by the periods of the cell: the image of a node of displacement ``u`` moved by a period
``L`` is displaced by ``u + H L``, ``H`` the macroscopic displacement gradient (the cell
deforms with it), and the forces that it receives go to the node itself.

*Macroscopic stress.* The positions of the nodes follow the macroscopic gradient, ``x = X +
H X + w``: the forces ``f`` of the contacts work in a change of ``H`` by ``Σ f ⊗ X`` over the
nodes (the images at their own places ``X + L``). This term — nothing for two walls that
touch inside the cell, whose forces are opposite at the same place; the force that crosses
a face of the cell times the period for a wall and an image — is added to the integral of
the stress of the points: in the equilibrium of the stress-controlled components, in the
macroscopic stress of the results, and its derivatives in the tangent of the macroscopic
unknowns.
"""

from __future__ import annotations

import numpy as np

from morphogen.contact.kernels import (
    S_FACE,
    S_GAP,
    S_LAMBDA,
    S_PRESSURE,
    S_SHARE,
    S_TX,
    S_WORK,
)
from morphogen.contact.problem import contact_forces, contact_pass, stiffest_modulus
from morphogen.contact.surfaces import Pair, Surface
from morphogen.homogenization.nonlinear import Extended, NonlinearFEProblem, _SurfacePaths
from morphogen.mesh.tetra import TetMesh

#: The curves that a cell records at each step (``PathResult.series``): the area in
#: contact (reference area, of one of the two sides), the largest pressure, the deepest
#: penetration and the energy dissipated by friction
SERIES = ("contact_area", "contact_pressure", "contact_penetration", "contact_dissipation")


def walls(mesh: TetMesh) -> Surface:
    """The free faces of a cell that are not on its boundary: the walls of its pores (the
    faces of the solid on the faces of the box, or on the outer boundary of whole periodic
    cells, go on in the neighbouring cells)."""
    skin = Surface.of(mesh)
    outer = np.sort(mesh.outer_faces()[:, :3], axis=1)
    keys = {tuple(face) for face in outer.tolist()}
    mine = np.sort(skin.faces[:, :3], axis=1)
    inside = np.array([tuple(face) not in keys for face in mine.tolist()], dtype=bool)
    if not mesh.whole:  # and the faces that lie on a face of the box
        centroids = skin.centroids(mesh.nodes)
        tol = mesh.tolerance()
        on_box = np.any(
            (np.abs(centroids - mesh.lower) <= tol) | (np.abs(centroids - mesh.upper) <= tol),
            axis=1,
        )
        inside &= ~on_box
    return skin.subset(inside)


class _Images(Extended):
    """A surface of a periodic cell and its images near the cell: the extended nodes (the
    nodes of the mesh, then the images of some: ``node`` the node of each, ``offset`` its
    translation), the faces of the surface and of its images on them, and the pass of the
    contact of the surface with all of these."""

    def __init__(self, mesh: TetMesh, surface: Surface, spec: Pair, stiffest: float, periodic):
        n = mesh.n_nodes
        size = np.asarray(mesh.size, dtype=float)
        # the images kept: those within twice the reach of the search of the cell (the
        # deepest penetration and two faces), half a period at most
        edge = surface.size(mesh.nodes)
        depth = 0.5 * edge if spec.depth is None else float(spec.depth)
        reach = np.minimum(0.5 * size, 2.0 * (depth + 2.0 * edge))
        faces = [surface.faces]
        node, offset = [np.zeros(0, dtype=np.int64)], [np.zeros((0, 3))]
        count = n
        centroids = surface.centroids(mesh.nodes)
        for shift in np.ndindex(3, 3, 3):
            shift = np.array(shift) - 1
            if not shift.any() or np.any((shift != 0) & ~np.asarray(periodic)):
                continue
            moved = centroids + shift * size
            near = np.all((moved >= mesh.lower - reach) & (moved <= mesh.upper + reach), axis=1)
            if not near.any():
                continue
            used, inverse = np.unique(surface.faces[near], return_inverse=True)
            faces.append(count + inverse.reshape(-1, surface.faces.shape[1]))
            node.append(used)
            offset.append(np.broadcast_to(shift * size, (len(used), 3)))
            count += len(used)
        super().__init__(mesh.nodes, np.concatenate(node), np.concatenate(offset))
        master = Surface(np.concatenate(faces))
        self.surface = surface
        self.contact = contact_pass(self.reference, surface, master, True, None, spec, stiffest, 2)
        if len(self.node) > n:
            # a node of the boundary of the cell and the image of the node across it are
            # one node: the edges that a face shares with the image of its neighbour are
            # not on the rim of the surface, and the two faces share their nodes
            linked, to, shift = mesh.periodic_links()
            first, moved = np.arange(n), np.zeros((n, 3), dtype=np.int64)
            first[linked], moved[linked] = to, shift
            periods = np.rint(self.offset / size).astype(np.int64)
            key = np.column_stack([first[self.node], moved[self.node] + periods])
            _, ident = np.unique(key, axis=0, return_inverse=True)
            ident = np.ascontiguousarray(ident.ravel(), dtype=np.int64)
            self.contact.ident = ident
            self.contact.rims = np.ascontiguousarray(Surface(ident[master.faces[:, :3]]).rims())


class ContactCell(NonlinearFEProblem):
    """A cell whose free surface may touch itself along a loading path (see the module):
    the ``mesh`` of its solid (the pores left out: ``conforming_mesh(..., phases="solid")``,
    or ``mesh.subset``), the ``materials`` of its phases and the boundary conditions
    (``"periodic"``: the walls also meet their images across the faces of the cell;
    ``"kubc"``).

    ``friction``, ``stiffness``, ``tangential``, ``depth`` and ``points``: those of a
    :class:`~morphogen.contact.surfaces.Pair`; ``surface``: the faces that may touch (by
    default the walls of the pores, :func:`walls`); ``augmentations``, ``tolerance`` and
    ``neighbours``: those of :class:`~morphogen.contact.problem.ContactProblem`. The other
    options are those of :class:`~morphogen.homogenization.nonlinear.NonlinearFEProblem`
    (``kinematics``, finite strains by default; ``integration``, ``newton_tol``...); the
    tangent is factorized (``solver="direct"``), and kept whole with friction or with the
    points at the nodes (the derivatives of their shares).

    :meth:`run` returns the response to a path with, in its ``series`` (columns of its
    curves, :data:`SERIES`), at each step the area in contact ``contact_area`` (reference
    area, of one of the two sides), the largest pressure ``contact_pressure``, the deepest
    penetration ``contact_penetration`` and the energy that friction dissipated
    ``contact_dissipation``."""

    def __init__(
        self,
        mesh: TetMesh,
        materials,
        bc: str = "periodic",
        *,
        friction: float = 0.0,
        stiffness: float | None = None,
        tangential: float | None = None,
        depth: float | None = None,
        points: str = "gauss",
        surface: Surface | None = None,
        augmentations: int = 0,
        tolerance: float | None = None,
        neighbours: int = 12,
        kinematics: str = "finite",
        **options,
    ):
        if options.pop("solver", "direct") not in ("direct", "auto"):
            raise ValueError(
                "ContactCell: the tangent of the contacts is factorized (solver 'direct')"
            )
        if friction > 0 or points == "nodes":  # tangents that are not symmetric
            options.setdefault("tangent", "full")
        super().__init__(mesh, materials, bc, kinematics=kinematics, solver="direct", **options)
        surface = walls(mesh) if surface is None else surface
        if surface.order != mesh.order:
            raise ValueError("ContactCell: a surface of the faces of the mesh (Surface.of)")
        if not surface.n_faces:
            raise ValueError("ContactCell: no free surface inside the cell (no pore to close)")
        if augmentations < 0 or neighbours < 1:
            raise ValueError("ContactCell: augmentations >= 0, neighbours >= 1")
        spec = Pair(surface, friction=friction, stiffness=stiffness, tangential=tangential,
                    depth=depth, points=points)  # fmt: skip
        periodic = (self.bc == "periodic",) * 3
        stiffest = stiffest_modulus(materials, mesh.phase)
        self._images = self._extended = _Images(mesh, surface, spec, stiffest, periodic)
        self.augmentations, self.neighbours = int(augmentations), int(neighbours)
        size = surface.size(mesh.nodes)
        self._floor = 1e-6 * stiffest * size**2  # the force of a strain of 1e-6 on a face
        self._floor_stress = 1e-6 * stiffest * self.box_volume  # and its stress in the cell
        # parts that float until they touch: a shift of the diagonal of the tangent
        self.regularization = max(self.regularization, 1e-12)
        self.tolerance = 0.01 * size if tolerance is None else float(tolerance)
        if not self.tolerance > 0:
            raise ValueError("ContactCell: a positive tolerance")
        self._x_old = self._images.reference
        self._x_now = self._images.reference
        self._virial = 0.0
        self._augmented = 0
        self._history: list[tuple] = []

    def _discretization(self) -> _SurfacePaths:
        return _SurfacePaths(self)

    def _surface_forces(self, x, tangent: bool, first: bool):
        """The forces of the contact of the surface with itself and its images at the
        positions ``x`` of the extended nodes, and their tangent."""
        self._x_now = x
        return contact_forces(
            [self._images.contact], x, self._x_old, self.neighbours, 0.0, tangent, first
        )

    def _extra_stress(self):
        """The part of the macroscopic stress that the forces of the contacts carry (see
        the module), at the last evaluation."""
        return self._virial / self.box_volume

    def _begin(self) -> None:
        """The state of the slave points before a path: the faces and the gaps that they
        see in the cell at rest, none pressing."""
        contact = self._images.contact
        contact.state = contact.fresh()
        contact.multiplier = np.zeros(len(contact.face))
        self._x_old = self._x_now = self._images.reference
        contact_forces([contact], self._x_old, self._x_old, self.neighbours, 0.0, False, False)
        contact.state = contact.trial.copy()
        contact.state[:, S_PRESSURE] = 0.0
        contact.state[:, S_TX : S_TX + 3] = 0.0
        contact.state[:, S_WORK] = 0.0
        self._virial, self._augmented = 0.0, 0
        self._history = [(0.0, 0.0, 0.0, 0.0)]

    def _settle(self, state) -> bool:
        """On a converged state: the multipliers are updated and the step solved again
        while a point penetrates deeper than the tolerance; otherwise the state of the
        slave points is kept and the step recorded."""
        contact = self._images.contact
        pressed = contact.pressed
        deepest = float(-contact.trial[pressed, S_GAP].min()) if pressed.any() else 0.0
        if self._augmented < self.augmentations and deepest > self.tolerance:
            contact.state[:, S_LAMBDA] = contact.trial[:, S_PRESSURE]
            self._augmented += 1
            return True
        self._augmented = 0
        contact.state = contact.trial.copy()
        contact.multiplier = contact.state[:, S_LAMBDA].copy()
        self._x_old = self._x_now
        area = float((contact.weight * contact.state[:, S_SHARE])[pressed].sum())
        area /= contact.shares
        most = contact.shares * float(contact.state[pressed, S_PRESSURE].max()) if area else 0.0
        rubbed = float(contact.weight @ contact.state[:, S_WORK])
        self._history.append((area, most, max(deepest, 0.0), rubbed))
        return False

    def _rollback(self) -> None:
        """A step is divided: the multipliers and the faces of the end of the last one."""
        contact = self._images.contact
        contact.state[:, S_LAMBDA] = contact.multiplier
        contact.last = contact.state[:, S_FACE].astype(np.int64)
        self._augmented = 0

    def run(self, path, **options):
        """The response to a loading path (:meth:`~morphogen.homogenization.nonlinear.
        NonlinearFEProblem.run`), with the records of the contact in its ``series`` (see
        the class)."""
        self._begin()
        result = super().run(path, **options)
        history = np.array(self._history[: len(result.times)])
        result.series.update({name: history[:, j] for j, name in enumerate(SERIES)})
        return result

    def slave_points(self) -> dict:
        """The slave points at the end of the last path: their ``pressure``, tangential
        ``traction``, ``gap``, ``weight`` (reference area; of the points at the nodes of
        the faces, the share that the other surface sees) and energy ``dissipated`` by
        friction per unit area, and their ``position``."""
        contact = self._images.contact
        return {
            "position": contact.surface.positions(
                self._x_old, contact.face, contact.xi, contact.eta
            ),
            "pressure": contact.shares * contact.state[:, S_PRESSURE],
            "traction": contact.shares * contact.state[:, S_TX : S_TX + 3],
            "gap": contact.state[:, S_GAP],
            "weight": contact.weight * contact.state[:, S_SHARE],
            "dissipated": contact.state[:, S_WORK],
        }
