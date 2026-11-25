"""Open-cell foams and trabecular bone: struts along the edges of a tessellation; fibre
networks along the edges of a Delaunay triangulation.

The struts of open-cell foams (polymer and metal foams, sponges) and of trabecular bone
(the cancellous bone of vertebrae and of the ends of long bones) lie along the edges of
cells close to those of a Voronoi tessellation; their stiffness scales as the square of
their relative density, their walls being open (Gibson and Ashby 1997). The model takes
the edges of the exact periodic Voronoi tessellation of germs (see
:func:`morphogen.mesh.polyhedra.periodic_cells`) as spherocylinders of radius ``r``,
with spheres of ``node_ratio r`` at the vertices (the thickening of the nodes, the
Plateau borders of liquid foams). The germs are

* Poisson points (``regularity`` 0): the Poisson-Voronoi foam;
* points at least ``regularity`` jamming distance of their random sequential addition
  apart, for more even cells;
* moved ``relaxation`` times to the centroids of their cells (Lloyd's algorithm, on
  sample points): toward a centroidal tessellation, whose regular cells are close to
  those of real foams.

Trabecular bone aligns its struts with the main loads (Wolff's law): the cells are
elongated by ``anisotropy`` along ``axis`` (the tessellation of the germs is computed
in the box shortened by that factor along the axis, then stretched back), or by the
stretches of an affine anisotropy along the box axes (:mod:`morphogen.models.affine`: the
schema of the other models, here with struts of circular sections). The strut
radius is given, or solved on each realization for a relative density
(``volume_fraction``, measured on voxels a few times finer than the struts). A
distribution of radii makes the struts polydisperse, a radius drawn for each; with a
``volume_fraction``, its law keeps its shape and its scale is solved (the radii ``s X``,
``X`` drawn from the law); the spheres of the nodes take ``node_ratio`` times the largest
radius of the struts that meet there.

The edges of a Poisson-Voronoi tessellation of ``lambda`` germs per unit volume are
``L_V = 5.832 lambda^(2/3)`` long per unit volume (Meijering 1953) and meet at ``N_V =
24 pi^2 / 35 lambda = 6.768 lambda`` vertices; stretched by ``a`` along an axis, a length
of isotropic directions grows by ``F(a) = (a + acosh(a) / sqrt(a^2 - 1)) / 2`` (``(a +
acos(a) / sqrt(1 - a^2)) / 2`` for ``a < 1``), and thin struts take the volume fraction
``pi r^2 L_V`` to first order.

The fibre networks of Ban et al. (2016) lie along the edges of the dual triangulation,
the Delaunay triangulation of the germs (:class:`DelaunayNetwork`): two germs are joined
when their Voronoi cells share a face, ``2 + 48 pi^2 / 35 = 15.54`` neighbours on average
for Poisson germs in space (Meijering's mean number of faces), 6 in the plane.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy import optimize
from scipy.spatial import Delaunay, cKDTree

from morphogen import settings, theory
from morphogen._utils import check_fraction, check_positive, exactly_one
from morphogen.design.frames import Framework
from morphogen.distributions import Distribution
from morphogen.domain import Domain, Grid
from morphogen.kernels.rsa import random_sequential_addition
from morphogen.models import _sizes as sizes
from morphogen.models.affine import Anisotropy
from morphogen.models.base import Model, param, register
from morphogen.models.planar import hardcore_germs, jamming_distance
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Segments, Spheres, with_periodic_images
from morphogen.seeding import SeedLike, make_rng

#: Volume fraction of spheres at the jamming limit of their random sequential addition.
RSA_JAMMING_3D = 0.384
#: Edge length per unit volume of Poisson-Voronoi tessellations, per lambda^(2/3).
VORONOI_EDGE_COEFF = 5.832
#: Vertices per unit volume of Poisson-Voronoi tessellations, per germ.
VORONOI_VERTEX_COEFF = 24.0 * math.pi**2 / 35.0


def stretch_factor(a: float) -> float:
    """Mean growth of the lengths of isotropic segments stretched by ``a`` along an axis:
    ``E[sqrt(1 + (a^2 - 1) cos^2 theta)]``."""
    if abs(a - 1.0) < 1e-12:
        return 1.0
    if a > 1.0:
        return 0.5 * (a + math.acosh(a) / math.sqrt(a * a - 1.0))
    return 0.5 * (a + math.acos(a) / math.sqrt(1.0 - a * a))


def lloyd_step(germs: np.ndarray, size: np.ndarray, samples: int, rng) -> np.ndarray:
    """Germs moved to the centroids of their periodic Voronoi cells, estimated on
    ``samples`` random points of the box."""
    n = len(germs)
    tree = cKDTree(germs, boxsize=size)
    pts = rng.random((samples, 3)) * size
    _, label = tree.query(pts, k=1, workers=settings.workers())
    d = pts - germs[label]
    d -= size * np.round(d / size)  # from the germ to the point, on the torus
    count = np.bincount(label, minlength=n)
    shift = np.column_stack([np.bincount(label, weights=d[:, i], minlength=n) for i in range(3)])
    moved = germs + shift / np.maximum(count, 1)[:, None]
    out = np.mod(moved, size)
    out[out >= size] = 0.0
    return out


def _voronoi_edges(germs: np.ndarray, size: np.ndarray):
    """The vertices of the periodic Voronoi complex of ``germs``, its edges each once (not
    their translates), and the class of each vertex on the torus with its translation."""
    from morphogen.mesh.polyhedra import periodic_cells, torus_classes

    cx = periodic_cells(germs, None, np.zeros(3), size)
    tol = 1e-9 * float(size.max())
    label, shift = torus_classes(cx.vertices, np.zeros(3), size, tol)
    e = cx.edges()
    a, b = label[e[:, 0]], label[e[:, 1]]
    rel = shift[e[:, 1]] - shift[e[:, 0]]  # the translates of an edge share (a, b, rel)
    swap = a > b  # one orientation: (a, b, rel) is (b, a, -rel)
    rel = np.where(swap[:, None], -rel, rel)
    lead = rel[np.arange(len(rel)), np.argmax(rel != 0, axis=1)]
    rel[(a == b) & (lead < 0)] *= -1  # from a vertex to its own translate
    keys = np.column_stack([np.where(swap, b, a), np.where(swap, a, b), rel])
    _, first = np.unique(keys, axis=0, return_index=True)
    return cx.vertices, e[np.sort(first)], label, shift


def _edge_ends(points, e, label, size) -> tuple[np.ndarray, np.ndarray]:
    start, end = points[e[:, 0]], points[e[:, 1]]
    # each edge from the end inside the box
    offset = np.floor(start / size) * size
    start, end = start - offset, end - offset
    _, keep = np.unique(label, return_index=True)
    vertices = np.mod(points[keep], size)
    return np.stack([start, end], axis=1), vertices


def _edge_framework(points, e, label, shift, size) -> Framework:
    _, keep = np.unique(label, return_index=True)  # the first point of each class
    wrap = np.floor(points[keep] / size).astype(np.int64)  # its cell, brought to the box
    a, b = label[e[:, 0]], label[e[:, 1]]
    shifts = shift[e[:, 1]] - shift[e[:, 0]] + wrap[b] - wrap[a]
    return Framework(points[keep] - wrap * size, np.column_stack([a, b]), shifts, size)


def periodic_edges(germs: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Edges of the periodic Voronoi tessellation of ``germs`` in the box ``[0, size)``,
    each once (not its translates): their ends, and the vertices, each once."""
    points, e, label, _ = _voronoi_edges(germs, size)
    return _edge_ends(points, e, label, size)


def periodic_framework(germs: np.ndarray, size: np.ndarray) -> Framework:
    """The edges of the periodic Voronoi tessellation of ``germs`` in the box ``[0, size)``
    as a framework: its vertices (in the box) joined by its edges, each once, from a vertex
    to an image of the other (the frame of beams of the open foams,
    :meth:`morphogen.homogenization.frames.SpaceFrame.from_framework`)."""
    size = np.asarray(size, dtype=float)
    points, e, label, shift = _voronoi_edges(germs, size)
    return _edge_framework(points, e, label, shift, size)


def frame_summary(framework: Framework, radius: float, dimension: int = 3) -> dict:
    """The stiffness of the frame of beams of a framework (fibres of Young's modulus 1 and
    Poisson's ratio 0.3), for the summaries of the network models
    (:meth:`morphogen.homogenization.networks.FrameStiffness.summary`), with the upper
    triangles of its stiffness ``frame_C11`` ... and of its compliance ``frame_S11`` ...
    (``S = C^-1``; Voigt order 11, 22, 33, 23, 31, 12 with engineering shears, 11, 22, 12
    per unit area in the plane; in units of the modulus of the fibres)."""
    from morphogen.homogenization.networks import network_stiffness
    from morphogen.homogenization.postprocess import matrix_entries

    stiffness = network_stiffness(framework, radius, dimension=dimension)
    out = stiffness.summary()
    out.update({f"frame_{k}": v for k, v in matrix_entries(stiffness.stiffness, "C").items()})
    out.update({f"frame_{k}": v for k, v in matrix_entries(stiffness.compliance(), "S").items()})
    return {k: float(f"{v:.8g}") for k, v in out.items()}


#: The doc of the parameter of the network models that adds the stiffness of their frame
FRAME_STIFFNESS = (
    "Add the stiffness of the frame of beams of the fibres (Young's modulus 1, Poisson's "
    "ratio 0.3) to the summary: E, Poisson's ratio, ratio to the affine estimate, fraction "
    "of stretching energy and non-affinity under uniaxial stress, and its stiffness C_ij and "
    "compliance S_ij (Voigt 11, 22, 33, 23, 31, 12, engineering shears)"
)


@register
@dataclass(frozen=True)
class OpenFoam(Model):
    """Open-cell foam or trabecular bone: struts along the edges of the periodic Voronoi
    tessellation of (regularized) germs, cells elongated by ``anisotropy`` along
    ``axis``, with a strut radius or a relative density. The framework of the struts is
    kept in the metadata (:func:`periodic_framework`), and with ``frame_stiffness`` the
    stiffness of its frame of beams in the summary."""

    name = "open_foam"
    phases: ClassVar[dict[int, str]] = {0: "pores", 1: "struts"}
    count_label = "cells"

    n_cells: int | None = param(
        None, doc="Number of cells: exactly in a periodic box, on average otherwise", kind="int"
    )
    intensity: float | None = param(None, doc="Cells per unit volume", unit="1/length^3")
    strut_radius: Distribution | None = param(
        None,
        doc="Radius of the struts: a number (monodisperse) or a distribution (polydisperse, "
        "one radius per strut; with volume_fraction, the shape of the law whose scale is "
        "solved)",
        kind="distribution",
        unit="length",
    )
    volume_fraction: float | None = param(
        None, doc="Relative density, reached on each realization", kind="fraction"
    )
    node_ratio: float = param(1.0, doc="Radius of the spheres at the vertices, in strut radii")
    regularity: float = param(
        0.0, doc="Least distance of the germs, in jamming distances (0: Poisson)", kind="fraction"
    )
    relaxation: int = param(0, doc="Lloyd iterations toward a centroidal tessellation", kind="int")
    anisotropy: Anisotropy = param(
        1.0,
        doc="Elongation of the cells along the axis, or an affine anisotropy {stretch: "
        "[sx, sy, sz], euler: [phi1, Phi, phi2]} whose principal axes are the box axes",
        kind="anisotropy",
    )
    axis: int = param(2, doc="Axis of the elongation (0: x, 1: y, 2: z)", kind="int")
    frame_stiffness: bool = param(False, doc=FRAME_STIFFNESS, kind="bool", kw_only=True)

    def _validate(self) -> None:
        exactly_one("n_cells", self.n_cells, "intensity", self.intensity)
        sizes.check_scale(self.strut_radius, self.volume_fraction, "strut_radius")
        if self.n_cells is not None and self.n_cells < 1:
            raise ValueError("n_cells: at least one cell")
        if self.intensity is not None:
            check_positive(self.intensity, "intensity")
        if self.volume_fraction is not None:
            check_fraction(self.volume_fraction, "volume_fraction")
        if not 1.0 <= self.node_ratio < math.inf:
            raise ValueError("node_ratio: at least 1 (nodes as thick as the struts), finite")
        if not 0.0 <= self.regularity <= 0.9:
            raise ValueError("regularity: from 0 (Poisson germs) to 0.9 (near jamming)")
        if not 0 <= self.relaxation <= 200:
            raise ValueError("relaxation: from 0 to 200 iterations")
        if self.anisotropy is None:
            raise ValueError("anisotropy: an elongation or an affine anisotropy (1: isotropic)")
        if self.anisotropy.box_stretch() is None:
            raise ValueError(
                "anisotropy: principal axes along the box axes (the tessellation is periodic)"
            )
        if self.axis not in (0, 1, 2):
            raise ValueError("axis: 0, 1 or 2")

    def theta(self, domain: Domain | None) -> float:
        """Cells per unit volume."""
        if self.intensity is not None:
            return float(self.intensity)
        if domain is None:
            raise ValueError("open_foam: the intensity of n_cells depends on the domain")
        return self.n_cells / domain.volume

    def _germs(self, rng, n: int, size: np.ndarray) -> np.ndarray:
        if self.regularity > 0:
            jamming = (6.0 * RSA_JAMMING_3D * float(np.prod(size)) / (math.pi * n)) ** (1 / 3)
            distance = self.regularity * jamming
            germs = random_sequential_addition(
                np.full(n, 0.5 * distance), size, 0.0, rng, max_attempts=4000 * n + 100_000
            )
        else:
            germs = rng.random((n, 3)) * size
        samples = int(min(4_000_000, 300 * n))
        for _ in range(self.relaxation):
            germs = lloyd_step(germs, size, samples, rng)
        return germs

    def _union(self, domain, ends, vertices, r, bars=None, **kwargs) -> PrimitiveUnion:
        """The struts of radius ``r`` (one value, or one per strut) and the spheres of the
        nodes, ``node_ratio`` times the radius (the largest of the struts ``bars`` that
        meet at a node, for struts of several radii)."""
        if np.ndim(r) == 0:
            radii, nodes = np.full(len(ends), r), np.full(len(vertices), self.node_ratio * r)
        else:
            radii = np.asarray(r, dtype=float).reshape(-1)
            nodes = sizes.node_radii(bars, radii, len(vertices), self.node_ratio)
        segments = Segments(ends[:, 0], ends[:, 1], radii, round_caps=True)
        spheres = None
        if self.node_ratio > 1.0:
            spheres = Spheres(vertices, nodes)
        return PrimitiveUnion(domain, spheres=spheres, segments=segments, **kwargs)

    def _solve_radius(self, domain, ends, vertices, length: float, unit=None, bars=None) -> float:
        """Radius of the struts of a realization for the relative density (voxels about
        a quarter of the radius wide); with ``unit`` radii (one per strut, drawn from a
        law), the scale ``s`` of the radii ``s unit``."""
        target = float(self.volume_fraction)
        if unit is not None:
            unit = np.asarray(unit, dtype=float)
            lengths = np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1)
            guess = math.sqrt(target * domain.volume / (math.pi * float(np.sum(unit**2 * lengths))))
            h = guess * float(unit.mean()) / 4.0
            shape = np.clip(np.ceil(domain.lengths / h), 16, 256).astype(int)
            grid = Grid(domain, tuple(int(s) for s in shape))

            def fraction(s):
                union = self._union(domain, ends, vertices, s * unit, bars=bars)
                return float(union.voxelize(grid=grid).data.mean())

            limit = 0.5 * float(domain.lengths.min()) / float(unit.max())
            return sizes.solve_scale(fraction, target, guess, limit, 1e-3 * guess)
        guess = math.sqrt(target * domain.volume / (math.pi * length))
        h = guess / 4.0
        shape = np.clip(np.ceil(domain.lengths / h), 16, 256).astype(int)
        grid = Grid(domain, tuple(int(s) for s in shape))

        def excess(r):
            union = self._union(domain, ends, vertices, r)
            return float(union.voxelize(grid=grid).data.mean()) - target

        lo, hi = 0.25 * guess, 2.0 * guess
        while excess(hi) < 0 and hi < 0.5 * float(domain.lengths.min()):
            lo, hi = hi, 2.0 * hi
        return float(optimize.brentq(excess, lo, hi, xtol=1e-3 * guess))

    def _radii(self, law, rng, domain, ends, vertices, length, bars) -> tuple:
        """The radius of the struts of a realization: one value (a number, or solved for
        the volume fraction), or one per strut (a distribution, its scale solved for the
        volume fraction), and the summary of the radii."""
        if self.volume_fraction is not None and law is None:
            r = self._solve_radius(domain, ends, vertices, length)
            return r, {}
        if not sizes.polydisperse(law):
            return float(law.moment(1)), {}
        radii = sizes.draw(law, rng, len(ends))
        out = {}
        if self.volume_fraction is not None:
            s = self._solve_radius(domain, ends, vertices, length, unit=radii, bars=bars)
            radii, out = s * radii, {"radius_scale": s}
        return radii, {**out, **sizes.size_summary("radius", radii, law)}

    @property
    def stretch(self) -> np.ndarray:
        """The stretches of the cells along x, y and z."""
        a = self.anisotropy
        if a.elongation is None:
            return a.box_stretch()
        stretch = np.ones(3)
        stretch[self.axis] = a.elongation
        return stretch

    def _elongation(self) -> float | None:
        """The elongation of uniaxial stretches (two equal ones), None otherwise."""
        t = np.sort(self.stretch)
        if abs(t[0] - t[1]) <= 1e-12 * t[2]:
            return float(t[2] / t[0])
        if abs(t[1] - t[2]) <= 1e-12 * t[2]:
            return float(t[0] / t[2])
        return None

    def generate(self, domain: Domain, seed: SeedLike = None) -> PrimitiveUnion:
        rng, seed_info = make_rng(seed)
        theta = self.theta(domain)
        if domain.periodic and self.n_cells is not None:
            n = int(self.n_cells)
        else:
            n = max(1, int(rng.poisson(theta * domain.volume)))
        stretch = self.stretch
        size = domain.lengths / stretch  # the box of the isotropic tessellation
        germs = self._germs(rng, n, size)
        points, e, label, shift = _voronoi_edges(germs, size)
        ends, vertices = _edge_ends(points, e, label, size)
        frame = _edge_framework(points, e, label, shift, size)
        frame = Framework(frame.nodes * stretch, frame.bars, frame.shifts, domain.lengths)
        ends = ends * stretch + domain.lower
        vertices = vertices * stretch + domain.lower
        length = float(np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1).sum())
        r, spread = self._radii(self.strut_radius, rng, domain, ends, vertices, length,
                                frame.bars)  # fmt: skip
        mean = float(np.mean(r))
        summary = {
            "n_cells": n,
            "strut_radius": mean,
            "edge_length_density": length / domain.volume,
        }
        summary.update({f"strut_{k}": v for k, v in spread.items()})
        if self.frame_stiffness:
            summary.update(frame_summary(frame, r))
        metadata = {"summary": summary, "framework": frame.to_dict()}
        if np.ndim(r):  # the radius of each strut, in the order of the bars of the framework
            metadata["radii"] = np.asarray(r).tolist()
        return self._union(domain, ends, vertices, r, bars=frame.bars, model=self,
                           seed=seed_info, metadata=metadata)  # fmt: skip

    # -- theory ----------------------------------------------------------------------------

    def expected_count(self, domain: Domain) -> float:
        return self.theta(domain) * domain.volume

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.intensity is None and domain is None:
            return out
        theta = self.theta(domain)
        out["intensity"] = theta
        a = self._elongation()
        if self.regularity == 0 and self.relaxation == 0:  # Poisson-Voronoi edges
            out["vertex_density"] = VORONOI_VERTEX_COEFF * theta
        if self.regularity == 0 and self.relaxation == 0 and a is not None:
            lv = VORONOI_EDGE_COEFF * (a * theta) ** (2 / 3) * stretch_factor(a) / a
            out["edge_length_density"] = lv
            if self.strut_radius is not None and self.volume_fraction is None:
                out["volume_fraction_first_order"] = sizes.first_order_fraction(
                    self.strut_radius, lv
                )
        return out


def _circumspheres(simplices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centres and radii of the circumspheres of simplices (m x (d + 1) x d vertices)."""
    p0 = simplices[:, 0]
    a = 2.0 * (simplices[:, 1:] - p0[:, None])
    b = np.sum(simplices[:, 1:] ** 2, axis=2) - np.sum(p0**2, axis=1)[:, None]
    centre = np.linalg.solve(a, b[..., None])[..., 0]
    return centre, np.linalg.norm(centre - p0, axis=1)


def periodic_delaunay(points, size) -> tuple[np.ndarray, np.ndarray]:
    """The edges of the Delaunay triangulation of the periodic box ``[0, size)`` of the
    ``points`` (2 or 3 coordinates), each once (not its translates): the pairs ``(i, j)``
    and the shifts ``s`` of the edges from ``points[i]`` to ``points[j] + s * size``. The
    points and their images within a margin of the box (three mean spacings, then doubled)
    are triangulated (SciPy/Qhull) until the circumsphere of every simplex at a point of
    the box lies in the margin, so that it is empty of every image: the simplices at the
    points of the box are those of the periodic triangulation. The images are those of as
    many periods along each axis as the margin spans (thin boxes: several across)."""
    pts = np.asarray(points, dtype=float)
    n, dim = pts.shape
    size = np.asarray(size, dtype=float).reshape(dim)
    if n < dim + 2:
        raise ValueError(f"delaunay_network: at least {dim + 2} points, got {n}")
    margin = 3.0 * (float(np.prod(size)) / n) ** (1.0 / dim)
    while True:
        reach = np.ceil(margin / size).astype(int)  # the periods within the margin
        shifts = np.array(list(itertools.product(*(range(-k, k + 1) for k in reach))))
        shifts = shifts[np.argsort(np.abs(shifts).sum(axis=1), kind="stable")]  # the box first
        image, index, shift = [], [], []
        for s in shifts:
            q = pts + s * size
            keep = np.all((q >= -margin) & (q < size + margin), axis=1)
            image.append(q[keep])
            index.append(np.flatnonzero(keep))
            shift.append(np.repeat(s[None], int(keep.sum()), axis=0))
        allp, index, shift = np.concatenate(image), np.concatenate(index), np.concatenate(shift)
        tri = Delaunay(allp)
        at_box = tri.simplices[np.any(tri.simplices < n, axis=1)]
        centre, radius = _circumspheres(allp[at_box])
        inside = np.all(centre - radius[:, None] >= -margin) and np.all(
            centre + radius[:, None] <= size + margin
        )
        if inside:
            break
        if margin > 4.0 * float(size.max()):
            raise ValueError("delaunay_network: too few points for a periodic triangulation")
        margin *= 2.0
    indptr, neighbours = tri.vertex_neighbor_vertices
    i = np.repeat(np.arange(n), np.diff(indptr)[:n])
    j = neighbours[: indptr[n]]
    a, b, s = i, index[j], shift[j]  # from the point i of the box to the image j
    first = s[np.arange(len(s)), np.argmax(s != 0, axis=1)]
    flip = (a > b) | ((a == b) & (first < 0))  # one orientation: (a, b, s) is (b, a, -s)
    key = np.column_stack(
        [np.where(flip, b, a), np.where(flip, a, b), np.where(flip[:, None], -s, s)]
    )
    key = np.unique(key, axis=0)
    return key[:, :2], key[:, 2:]


DIMENSIONS = (2, 3)


@register
@dataclass(frozen=True)
class DelaunayNetwork(Model):
    """Fibre networks along the edges of the Delaunay triangulation of random points (Ban,
    Barocas, Shephard & Picu 2016), in space or in a plane.

    As in Ban et al. (J. Mech. Phys. Solids 87, 38, 2016), points are drawn in the box and
    fibres join the points whose Voronoi cells share a face: the edges of their Delaunay
    triangulation (periodic: :func:`periodic_delaunay`), the struts of the dual of the open
    foams of :class:`OpenFoam` (spherocylinders of ``radius``, spheres of ``node_ratio``
    radii at the nodes), cross-linked at every node. The points are Poisson points, at
    least ``regularity`` jamming distance of their random sequential addition apart
    (hard-core points), or moved ``relaxation`` times to the centroids of their Voronoi
    cells (space only). ``dimension: 2``: a planar network of points of the plane ``z`` in
    the middle of the domain (the planar Delaunay graph; ``intensity`` per unit area).
    The radius is given, or solved on each realization for a ``volume_fraction``; a
    distribution of radii makes the fibres polydisperse, a radius drawn for each (with a
    ``volume_fraction``, its law scaled to reach it, as for the struts of
    :class:`OpenFoam`). The framework of the nodes and fibres
    (:class:`morphogen.design.frames.Framework`) is kept in the metadata, its coordination
    and fibre lengths in the summary, and with ``frame_stiffness`` the stiffness of its
    frame of beams (:mod:`morphogen.homogenization.networks`).
    """

    name = "delaunay_network"
    phases: ClassVar[dict[int, str]] = {0: "pores", 1: "fibres"}
    count_label = "nodes"

    n_points: int | None = param(
        None, doc="Number of nodes: exactly in a periodic box, on average otherwise", kind="int"
    )
    intensity: float | None = param(
        None, doc="Nodes per unit volume (per unit area in a plane)", unit="1/length^3"
    )
    radius: Distribution | None = param(
        None,
        doc="Radius of the fibres: a number (monodisperse) or a distribution (polydisperse, "
        "one radius per fibre; with volume_fraction, the shape of the law whose scale is "
        "solved)",
        kind="distribution",
        unit="length",
    )
    volume_fraction: float | None = param(
        None, doc="Volume fraction of the fibres, reached on each realization", kind="fraction"
    )
    node_ratio: float = param(1.0, doc="Radius of the spheres at the nodes, in fibre radii")
    regularity: float = param(
        0.0, doc="Least distance of the points, in jamming distances (0: Poisson)", kind="fraction"
    )
    relaxation: int = param(
        0, doc="Lloyd iterations toward a centroidal tessellation (space only)", kind="int"
    )
    dimension: int = param(3, doc="3, or 2: a planar network in the middle plane z", kind="int")
    frame_stiffness: bool = param(False, doc=FRAME_STIFFNESS, kind="bool", kw_only=True)

    # the germs and the radius of a fraction of the open foams
    _germs = OpenFoam._germs
    _solve_radius = OpenFoam._solve_radius
    _radii = OpenFoam._radii

    def _union(self, domain, ends, vertices, r, bars=None, **kwargs) -> PrimitiveUnion:
        """The struts of the open foams; in a box that is not periodic, with their periodic
        images: the periodic network cut by the box, the fibres leaving it by a face
        entering it again by the opposite one."""
        union = OpenFoam._union(self, domain, ends, vertices, r, bars, **kwargs)
        if domain.periodic:
            return union
        spheres = with_periodic_images(union.spheres, domain)
        segments = with_periodic_images(union.segments, domain)
        return PrimitiveUnion(domain, spheres=spheres, segments=segments, **kwargs)

    def _validate(self) -> None:
        exactly_one("n_points", self.n_points, "intensity", self.intensity)
        sizes.check_scale(self.radius, self.volume_fraction, "radius")
        if self.dimension not in DIMENSIONS:
            raise ValueError("dimension: 2 or 3")
        if self.n_points is not None and self.n_points < self.dimension + 2:
            raise ValueError(f"n_points: at least {self.dimension + 2}")
        if self.intensity is not None:
            check_positive(self.intensity, "intensity")
        if self.volume_fraction is not None:
            check_fraction(self.volume_fraction, "volume_fraction")
        if not 1.0 <= self.node_ratio < math.inf:
            raise ValueError("node_ratio: at least 1 (nodes as thick as the fibres), finite")
        if not 0.0 <= self.regularity <= 0.9:
            raise ValueError("regularity: from 0 (Poisson points) to 0.9 (near jamming)")
        if not 0 <= self.relaxation <= 200:
            raise ValueError("relaxation: from 0 to 200 iterations")
        if self.relaxation and self.dimension == 2:
            raise ValueError("relaxation: networks in space only (dimension: 3)")

    def _measure(self, domain: Domain) -> float:
        """The volume of the domain, or the area of its plane for a planar network."""
        lengths = domain.lengths
        return float(np.prod(lengths[:2]) if self.dimension == 2 else np.prod(lengths))

    def theta(self, domain: Domain | None) -> float:
        """Nodes per unit volume (per unit area for a planar network)."""
        if self.intensity is not None:
            return float(self.intensity)
        if domain is None:
            raise ValueError("delaunay_network: the intensity of n_points depends on the domain")
        return self.n_points / self._measure(domain)

    def framework(self, domain: Domain, rng) -> Framework:
        """The nodes (from the lower corner of the domain) and fibres of a realization."""
        lengths = np.asarray(domain.lengths, dtype=float)
        planar = self.dimension == 2
        size = lengths[:2] if planar else lengths
        if domain.periodic and self.n_points is not None:
            n = int(self.n_points)
        else:
            n = max(
                self.dimension + 2, int(rng.poisson(self.theta(domain) * self._measure(domain)))
            )
        if planar:
            least = self.regularity * jamming_distance(n, float(np.prod(size)))
            points = hardcore_germs(n, size, least, rng)
            points = np.column_stack([points, np.full(n, 0.5 * lengths[2])])
        else:
            points = self._germs(rng, n, size)
        pairs, shifts = periodic_delaunay(points[:, : self.dimension], size)
        if planar:
            shifts = np.column_stack([shifts, np.zeros(len(shifts), dtype=np.int64)])
        return Framework(points, pairs, shifts, lengths)

    def generate(self, domain: Domain, seed: SeedLike = None) -> PrimitiveUnion:
        rng, seed_info = make_rng(seed)
        frame = self.framework(domain, rng)
        vectors = frame.vectors()
        start = frame.nodes[frame.bars[:, 0]] + domain.lower
        ends = np.stack([start, start + vectors], axis=1)
        vertices = frame.nodes + domain.lower
        lengths = np.linalg.norm(vectors, axis=1)
        total = float(lengths.sum())
        r, spread = self._radii(self.radius, rng, domain, ends, vertices, total, frame.bars)
        summary = {
            "nodes": frame.n_nodes,
            "fibres": frame.n_bars,
            "coordination": round(frame.coordination(), 6),
            "mean_fibre_length": float(lengths.mean()),
            "fibre_length_cv": float(lengths.std() / lengths.mean()),
            "fibre_length_density": total / domain.volume,
            "radius": float(np.mean(r)),
            **spread,
        }
        if self.dimension == 2:
            summary["fibre_length_per_area"] = total / self._measure(domain)
        if self.frame_stiffness:
            summary.update(frame_summary(frame, r, self.dimension))
        metadata = {"summary": summary, "framework": frame.to_dict()}
        if np.ndim(r):  # the radius of each fibre, in the order of the bars of the framework
            metadata["radii"] = np.asarray(r).tolist()
        return self._union(domain, ends, vertices, r, bars=frame.bars, model=self,
                           seed=seed_info, metadata=metadata)  # fmt: skip

    def expected_count(self, domain: Domain) -> float:
        return self.theta(domain) * self._measure(domain)

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        """The coordination of the Poisson-Delaunay graph, ``2 + 48 pi^2 / 35`` in space (the
        mean number of faces of the Poisson-Voronoi cell, Meijering 1953) and 6 in the plane
        (Euler), and in the plane the mean fibre length ``32 / (9 pi) lambda^(-1/2)`` (Miles
        1970) and the fibre length per unit area ``3 lambda`` times it. In space the mean
        length of a fibre has no closed form: ``1.237 lambda^(-1/3)`` (Miles 1974) is that
        of the edges of the typical tetrahedron, which counts each fibre once per
        tetrahedron around it, and the fibres themselves are longer (1.285 on 20,000
        points)."""
        out: dict[str, float] = {}
        if self.intensity is None and domain is None:
            return out
        theta = self.theta(domain)
        out["intensity"] = theta
        if self.regularity == 0 and self.relaxation == 0:  # Poisson-Delaunay
            if self.dimension == 3:
                out["coordination"] = theory.VORONOI_MEAN_FACES
            else:
                mean = 32.0 / (9.0 * math.pi) / math.sqrt(theta)
                out["coordination"] = 6.0
                out["mean_fibre_length"] = mean
                out["fibre_length_per_area"] = 3.0 * theta * mean
        return out
