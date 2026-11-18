"""Contact, self-contact and friction of the surfaces of a mesh, between themselves and
with rigid obstacles, in small and finite strains.

*Surfaces.* A :class:`Surface` is a set of faces of a tetrahedral mesh (three nodes, or
six on 10-node elements, which may be curved), their normals out of the body: the free
faces of the mesh (:meth:`Surface.of`), all or some. A :class:`Pair` is a slave surface
and a master surface that may touch, or one surface that may touch itself (the walls of
pores, the struts of a lattice); an :class:`Obstacle` a rigid plane, sphere, cylinder or
cone that moves with the load factor. Bodies meshed apart are brought into one mesh by
:func:`assemble`.

*Discretization.* The contact is enforced at the integration points of the faces of the
slave surface (three per face of three nodes, six per face of six), each with its share of
the reference area: its position ``x_s``, the nearest point ``x_m`` of the master surface
(the nearest point of a face among the master faces nearest by their centroids: Ericson's
regions on a triangle, Newton's method on a curved face of six nodes and on its edges),
the outward normal ``n`` of the master there and the gap ``g = (x_s - x_m) · n``. A point
presses when ``p = <λ - ε g>`` is positive (``ε`` the penalty, ``λ`` the multiplier of the
augmented Lagrangian, zero for the penalty alone); the force ``w (p n - t)`` goes to the
nodes of its face by their functions, and its opposite to those of the master face by
theirs at ``x_m``. Only master faces that face the slave point (opposite normals) and
that it has not passed by more than a given depth are looked at: a point further behind
a face is across a wall, not through it; a surface against itself also leaves out the
faces that share a node with that of the point. The tangent of the forces of a point is
taken by central differences on the nodes of its two faces.

*Corners and edges.* The points inside the faces leave the corners and the edges of a
body without a point: a corner that lands on a face enters it until the points next to it
press. With ``points="nodes"`` (of a :class:`Pair`, an :class:`Obstacle`, a
:class:`ContactCell`, :func:`compression`) the slave points are at the nodes of the faces
instead (their vertices; the middles of their edges and their centroids as well on faces
of six nodes). The weight of such a point is the share of its face that the other surface
sees (the cosine of the angle of their normals): the sides of a block that stands on a
plane, which share the nodes of the edges of its base, carry nothing.

*Friction* (Coulomb; Laursen & Simo 1993 for the frame-indifferent slip). The slip of a
point over a step is the move of ``x_m`` from the place, now, of the point of the master
that faced it at the end of the last step (the same material point of the master: no
slip under a rigid motion of the two bodies), in the tangent plane. The traction of the
last step, turned into that plane with its norm, plus the stick ``ε_t`` times the slip,
is returned radially to Coulomb's cone ``|t| <= μ p``; what is cut off is the slip that
dissipates. A point rubs from the step after that in which it came into contact.

*Solution.* :class:`ContactProblem` is the quasi-static solver of the cohesive zones
(:class:`~morphogen.fracture.cohesive.CohesiveProblem`: prescribed displacements, dead
loads, loadings in stages, an elastic or neo-Hookean bulk or the behaviours of
morphoGen's points, cohesive interfaces as well) with these forces. The faces that each
slave point sees are searched again at each evaluation. The first tangent of a step is
that of a predictor in which the points that touched at its start stay closed and stick.
With ``augmentations``, the multipliers are updated (Uzawa: ``λ <- <λ - ε g>``) and the
step solved again until no point penetrates deeper than a tolerance.

*Periodic cells.* :class:`ContactCell` is the periodic element of the nonlinear
homogenization with the contact of the walls of its pores, between themselves and with
their images across the faces of the cell (:mod:`morphogen.contact.periodic`): loading
paths in strains and stresses, the behaviours of morphoGen's points, the macroscopic
stress with the part that the contacts carry through the faces.

*Ready-made tests* (:mod:`morphogen.contact.experiments`). :func:`compression`: a sample
between two rigid platens, on which it rubs, its walls free to close on themselves (the
nominal stress against the nominal strain, the areas in contact); :func:`indentation`:
a sphere or a cone pressed into its upper face, and out again (the force against the
depth, the area of the contact, and the reduced modulus that the force reads by Hertz's
or Sneddon's law, or by the slope of the unloading).

The results (:class:`ContactResult`): at each step, the area in contact, the largest
pressure and the force of each contact, the works (those of the bulk and of the contact
add up to that of the grips), the energy that friction dissipated and the work of the
obstacles; at the end the pressure, traction and gap of each slave point. The closed
forms of :mod:`morphogen.contact.hertz` (Hertz, Sneddon, Cattaneo-Mindlin) are those of
the checks.
"""

from morphogen.contact.experiments import (
    CompressionResult,
    IndentationResult,
    compression,
    indentation,
)
from morphogen.contact.hertz import (
    effective_modulus,
    flat_punch,
    hertz_cylinder,
    hertz_sphere,
    layer_correction,
    mindlin_stick_radius,
    sneddon_cone,
)
from morphogen.contact.periodic import ContactCell, walls
from morphogen.contact.problem import ContactProblem, ContactResult
from morphogen.contact.surfaces import Obstacle, Pair, Surface, assemble

__all__ = [
    "CompressionResult",
    "ContactCell",
    "ContactProblem",
    "ContactResult",
    "IndentationResult",
    "Obstacle",
    "Pair",
    "Surface",
    "assemble",
    "compression",
    "effective_modulus",
    "flat_punch",
    "hertz_cylinder",
    "hertz_sphere",
    "indentation",
    "layer_correction",
    "mindlin_stick_radius",
    "sneddon_cone",
    "walls",
]
