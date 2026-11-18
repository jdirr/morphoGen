"""Ready-made tests of a sample between rigid tools (see :mod:`morphogen.contact`): its
compression between two platens, on which it rubs and under which its own walls may close
on themselves (a foam, a lattice, a porous solid that densifies), and its indentation by a
sphere or a cone.

Both build a :class:`~morphogen.contact.problem.ContactProblem` of the mesh of the sample
with rigid obstacles, follow it in steps and give the curves that the tests measure: the
nominal stress against the nominal strain of a compression, the force against the depth
of an indentation with the modulus that it reads (Hertz, Sneddon; the slope of the
unloading: Oliver and Pharr 1992).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from morphogen.contact.hertz import hertz_sphere, layer_correction, sneddon_cone
from morphogen.contact.problem import ContactProblem, ContactResult
from morphogen.contact.surfaces import Obstacle, Pair, Surface
from morphogen.mesh.tetra import TetMesh

#: The indenters of :func:`indentation`
INDENTERS = ("sphere", "cone")


def _plane_axes(axis: int) -> tuple[int, int]:
    if axis not in (0, 1, 2):
        raise ValueError("axis: 0, 1 or 2")
    return tuple(a for a in range(3) if a != axis)


def _pins(mesh: TetMesh, axis: int) -> list[tuple]:
    """Prescribed displacements that hold a sample against the three rigid motions that
    its platens leave it (two translations along them and the rotation about their
    normal), on two nodes of its base: the one nearest the middle of the base (both
    components along the platens) and the one furthest from it along the first of their
    axes (the component along the second)."""
    b, c = _plane_axes(axis)
    x = mesh.nodes
    height = float(mesh.size[axis])
    base = np.flatnonzero(x[:, axis] <= x[:, axis].min() + 0.02 * height)
    middle = x[base][:, [b, c]].mean(axis=0)
    first = base[np.argmin(np.linalg.norm(x[base][:, [b, c]] - middle, axis=1))]
    second = base[np.argmax(np.abs(x[base, b] - x[first, b]))]
    if second == first or abs(x[second, b] - x[first, b]) < 1e-9 * height:
        raise ValueError("the base of the sample is a point: nothing to hold it by")
    return [(np.array([first]), [b, c], 0.0), (np.array([second]), c, 0.0)]


@dataclass
class CompressionResult:
    """The compression of a sample between two platens. At each step: the nominal
    ``strain`` (the approach of the platens over the height of the sample: negative), the
    ``force`` that the sample exerts on the moving platen (positive), the nominal
    ``stress`` (minus this force over the section of the box of the sample: negative), the
    ``area`` in contact with the platens (steps x 2: the fixed one, the moving one;
    reference area), the area of the walls of the sample in contact with themselves
    (``self_area``, of one of the two sides) and the energy that friction dissipated
    (``friction``). ``section`` and ``height``: those of the box of the sample.
    ``result``: the steps of the problem (:class:`~morphogen.contact.problem.
    ContactResult`: reactions, works, the slave points at the end); ``problem``: the
    problem (:meth:`~morphogen.contact.problem.ContactProblem.write` for its fields)."""

    strain: np.ndarray
    stress: np.ndarray
    force: np.ndarray
    area: np.ndarray
    self_area: np.ndarray
    friction: np.ndarray
    section: float
    height: float
    result: ContactResult = field(repr=False)
    problem: ContactProblem = field(repr=False)

    @property
    def modulus(self) -> float:
        """The apparent modulus of the first step: its stress over its strain."""
        return float(self.stress[1] / self.strain[1])

    def tangent(self) -> np.ndarray:
        """The tangent modulus of each step: the change of the stress over that of the
        strain (one value per step; stiffening as the walls close)."""
        return np.diff(self.stress) / np.diff(self.strain)

    def rows(self) -> list[dict]:
        """One row per step: strain, stress, force, the areas in contact (with the fixed
        platen, with the moving one, of the sample with itself) and the energy dissipated
        by friction."""
        return [
            {
                "strain": float(self.strain[k]),
                "stress": float(self.stress[k]),
                "force": float(self.force[k]),
                "area_fixed": float(self.area[k, 0]),
                "area_moving": float(self.area[k, 1]),
                "self_area": float(self.self_area[k]),
                "friction": float(self.friction[k]),
            }
            for k in range(len(self.strain))
        ]


def compression(
    mesh: TetMesh,
    materials,
    strain: float,
    *,
    axis: int = 2,
    friction: float = 0.0,
    self_contact: bool = True,
    self_friction: float | None = None,
    steps: int = 20,
    kinematics: str = "finite",
    fields: bool = False,
    stiffness: float | None = None,
    points: str = "gauss",
    **options,
) -> CompressionResult:
    """Compress a sample between two rigid platens across ``axis``: the platens are the
    planes of the two faces of the box of the ``mesh`` (that of the solid of a porous
    sample: ``conforming_mesh(..., phases="solid")``), the lower one fixed, the upper one
    brought down by ``-strain`` times the height of the box (``strain``: the nominal
    strain, negative) in ``steps`` steps.

    The free surface of the sample presses on the platens with Coulomb's ``friction``
    (0: lubricated platens, the sample widens freely; with friction it barrels) and, with
    ``self_contact``, on itself with ``self_friction`` (by default that of the platens):
    the walls of the pores that close, the struts of a lattice that meet. ``stiffness``:
    the penalty of all these contacts (see :class:`~morphogen.contact.surfaces.Pair`);
    ``points``: their contact points, ``"gauss"`` or ``"nodes"`` (a sample that meets a
    platen, or itself, by corners or edges); ``kinematics``: ``"finite"`` strains (the
    default) or ``"small"`` ones; ``fields``: keep
    the displacements of each step. The other ``options``: those of
    :class:`~morphogen.contact.problem.ContactProblem` (``bulk``, ``augmentations``,
    ``tolerance``, ``neighbours``) and of its :meth:`run` (``tol``, ``max_newton``,
    ``max_cuts``).

    Two nodes of the base hold the sample against the rigid motions along the platens
    (the forces that they take are those that the platens would otherwise have to: none
    for a sample that widens evenly)."""
    if not (math.isfinite(strain) and -1.0 < strain < 0.0):
        raise ValueError("compression: a nominal strain between -1 and 0 (the platens close)")
    if steps < 1:
        raise ValueError("compression: at least one step")
    b, c = _plane_axes(axis)
    skin = Surface.of(mesh)
    if not skin.n_faces:
        raise ValueError("compression: the mesh has no free surface")
    normal = np.zeros(3)
    normal[axis] = 1.0
    height = float(mesh.size[axis])
    section = float(mesh.size[b] * mesh.size[c])
    contact = {"friction": friction, "stiffness": stiffness, "points": points}
    fixed = Obstacle.plane(skin, tuple(mesh.lower), tuple(normal), name="fixed", **contact)
    moving = Obstacle.plane(
        skin, tuple(mesh.upper), tuple(-normal), name="moving",
        motion=tuple(strain * height * normal), **contact,
    )  # fmt: skip
    pairs = []
    if self_contact:
        rub = friction if self_friction is None else self_friction
        pairs.append(Pair(skin, friction=rub, stiffness=stiffness, points=points))
    build = {k: options.pop(k) for k in ("bulk", "augmentations", "tolerance", "neighbours")
             if k in options}  # fmt: skip
    problem = ContactProblem(mesh, materials, pairs, [fixed, moving], kinematics=kinematics,
                             **build)  # fmt: skip
    res = problem.run(_pins(mesh, axis), steps=steps, fields=fields, **options)
    first = len(pairs)  # the contacts: the pair of the surface with itself, then the platens
    force = res.forces[:, first + 1, axis]
    return CompressionResult(
        strain=strain * res.loads,
        stress=-force / section,
        force=force,
        area=res.area[:, first : first + 2],
        self_area=res.area[:, 0] if pairs else np.zeros(len(res.loads)),
        friction=np.asarray(res.work["friction"]),
        section=section,
        height=height,
        result=res,
        problem=problem,
    )


@dataclass
class IndentationResult:
    """The indentation of a sample. At each step: the ``depth`` of the indenter under the
    surface, the ``force`` on it, the ``area`` of the contact (the reference area of the
    points that press), its ``radius`` (of the disc of the same area), the largest
    ``pressure`` and the ``hardness`` (the mean pressure, force over area); ``modulus``:
    the reduced modulus ``E / (1 - ν²)`` of the homogeneous half-space that would take the
    force at that depth (Hertz for a sphere, Sneddon for a cone: that of the sample while
    the contact is small against it and against its heterogeneities). With an unloading:
    ``stiffness``, the slope of the force at its start, and ``unloading_modulus``, ``S
    sqrt(π) / (2 sqrt(A))`` (Oliver and Pharr 1992: the area is that of the points, known
    to a ring of them). ``radius_of_indenter`` (a sphere), ``thickness`` (of the sample
    under its face) and ``base``: those of the test. ``result`` and ``problem``: as for a
    compression."""

    depth: np.ndarray
    force: np.ndarray
    area: np.ndarray
    pressure: np.ndarray
    modulus: np.ndarray
    indenter: str
    stiffness: float | None
    unloading_modulus: float | None
    radius_of_indenter: float | None
    thickness: float
    base: str
    result: ContactResult = field(repr=False)
    problem: ContactProblem = field(repr=False)

    def layer_modulus(self, poisson: float) -> np.ndarray:
        """The reduced modulus read from the force of a sphere with the base of the sample
        taken into account: ``modulus`` over the factor by which a rigid base under a
        layer of the thickness of the sample, of Poisson's ratio ``poisson``, multiplies
        Hertz's force (:func:`~morphogen.contact.hertz.layer_correction`: bonded for a
        fixed base, free to slide for a smooth one). A sample wide against its
        thickness."""
        if self.indenter != "sphere":
            raise ValueError("layer_modulus: the correction is that of a sphere")
        bonded = self.base == "fixed"
        factor = np.array(
            [
                layer_correction(self.radius_of_indenter, h, self.thickness, poisson, bonded)
                for h in self.depth
            ]
        )
        return self.modulus / factor

    @property
    def radius(self) -> np.ndarray:
        return np.sqrt(self.area / math.pi)

    @property
    def hardness(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.area > 0, self.force / self.area, 0.0)

    def rows(self) -> list[dict]:
        """One row per step: depth, force, area and radius of the contact, largest and
        mean pressures, and the reduced modulus read from the force."""
        radius, hardness = self.radius, self.hardness
        return [
            {
                "depth": float(self.depth[k]),
                "force": float(self.force[k]),
                "area": float(self.area[k]),
                "radius": float(radius[k]),
                "pressure": float(self.pressure[k]),
                "hardness": float(hardness[k]),
                "modulus": float(self.modulus[k]),
            }
            for k in range(len(self.depth))
        ]


def indentation(
    mesh: TetMesh,
    materials,
    depth: float,
    *,
    indenter: str = "sphere",
    radius: float | None = None,
    half_angle: float = 70.3,
    point=None,
    axis: int = 2,
    friction: float = 0.0,
    steps: int = 10,
    unload: bool = False,
    base: str = "fixed",
    kinematics: str = "finite",
    fields: bool = False,
    stiffness: float | None = None,
    **options,
) -> IndentationResult:
    """Indent the upper face of a sample (across ``axis``) to a ``depth`` by a rigid
    ``indenter``: a ``"sphere"`` of ``radius``, or a ``"cone"`` of ``half_angle`` (degrees
    from its axis; 70.3: the cone of the same area as the pyramids of Berkovich and
    Vickers), at ``point`` of the face (two coordinates along it; by default its middle),
    in ``steps`` steps, then out again in as many with ``unload``.

    ``base``: the lower face is ``"fixed"`` (glued) or ``"smooth"`` (it slides on its
    support); the other faces are free. ``friction``: that of the indenter. ``stiffness``,
    ``kinematics``, ``fields`` and the other ``options``: as for :func:`compression`."""
    if indenter not in INDENTERS:
        raise ValueError(f"indentation: an indenter among {', '.join(INDENTERS)}")
    if not (math.isfinite(depth) and depth > 0):
        raise ValueError("indentation: a positive depth")
    if indenter == "sphere" and not (radius is not None and radius > depth):
        raise ValueError("indentation: the radius of the sphere, larger than the depth")
    if base not in ("fixed", "smooth"):
        raise ValueError("indentation: a base 'fixed' or 'smooth'")
    if steps < 1:
        raise ValueError("indentation: at least one step")
    b, c = _plane_axes(axis)
    x = mesh.nodes
    height = float(mesh.size[axis])
    top = float(mesh.upper[axis])
    faces = Surface.of(mesh, lambda at: np.abs(at[:, axis] - top) < 1e-6 * height)
    if not faces.n_faces:
        raise ValueError("indentation: no face of the sample on the upper face of its box")
    where = np.array(mesh.lower + 0.5 * mesh.size, dtype=float)
    if point is not None:
        where[[b, c]] = np.asarray(point, dtype=float).reshape(2)
    where[axis] = top
    normal = np.zeros(3)
    normal[axis] = 1.0

    def motion(t, down=-depth * normal):
        return (t if t <= 1.0 else 2.0 - t) * down

    contact = {"friction": friction, "stiffness": stiffness, "motion": motion}
    if indenter == "sphere":
        tool = Obstacle.sphere(faces, tuple(where + radius * normal), radius, **contact)
    else:
        tool = Obstacle.cone(faces, tuple(where), tuple(normal), half_angle, **contact)
    under = np.flatnonzero(x[:, axis] < float(mesh.lower[axis]) + 1e-6 * height)
    if not under.size:
        raise ValueError("indentation: no node of the sample on the lower face of its box")
    if base == "fixed":
        prescribed = [(under, [0, 1, 2], 0.0)]
    else:
        prescribed = [(under, axis, 0.0), *_pins(mesh, axis)]
    build = {k: options.pop(k) for k in ("bulk", "augmentations", "tolerance", "neighbours")
             if k in options}  # fmt: skip
    problem = ContactProblem(mesh, materials, obstacles=[tool], kinematics=kinematics, **build)
    load, count = (2.0, 2 * steps) if unload else (1.0, steps)
    res = problem.run(prescribed, load=load, steps=count, fields=fields, **options)
    t = res.loads
    sunk = depth * np.where(t <= 1.0, t, 2.0 - t)
    force = res.forces[:, 0, axis]  # the sample pushes the indenter up
    unit = np.array(
        [
            0.0 if h <= 0 else (
                hertz_sphere(radius, 1.0, approach=h)["force"] if indenter == "sphere"
                else sneddon_cone(half_angle, 1.0, h)["force"]
            )
            for h in sunk
        ]
    )  # fmt: skip
    with np.errstate(divide="ignore", invalid="ignore"):
        modulus = np.where(unit > 0, force / unit, 0.0)
    slope = reduced = None
    if unload:
        peak = int(np.argmin(np.abs(t - 1.0)))
        if peak + 1 < len(t) and sunk[peak] > sunk[peak + 1]:
            slope = float((force[peak] - force[peak + 1]) / (sunk[peak] - sunk[peak + 1]))
            if res.area[peak, 0] > 0:
                reduced = slope * math.sqrt(math.pi) / (2.0 * math.sqrt(res.area[peak, 0]))
    return IndentationResult(
        depth=sunk,
        force=force,
        area=res.area[:, 0],
        pressure=res.pressure[:, 0],
        modulus=modulus,
        indenter=indenter,
        stiffness=slope,
        unloading_modulus=reduced,
        radius_of_indenter=float(radius) if indenter == "sphere" else None,
        thickness=height,
        base=base,
        result=res,
        problem=problem,
    )
