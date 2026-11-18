"""Meshes that follow the sharp edges of the phases: the smooth pieces of the unions of
spheres and segments (:mod:`morphogen.models.pieces`: overlaps, colourings), the mesher
that cuts the box piece by piece (the groove of two spheres against its circle and the
exact volume of their union, the points where three spheres meet, matching faces, coated
spheres against the exact volumes of their phases), the curved elements along the sharp
edges, the options and their errors. The meshes here are those of the background grid
(``method="grid"``, docs/theory.md §80), whose figures they quote; the exact route
follows the same edges from an octree (tests/test_exact_phases.py)."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

import morphogen as mg
from morphogen.config import ConfigError, load_config
from morphogen.io import medit
from morphogen.mesh import conforming
from morphogen.mesh.conforming import has_mmg
from morphogen.mesh.curved import curved_mesh
from morphogen.mesh.tetra import TET_EDGES, TetMesh
from morphogen.models import model_from_config
from morphogen.models.pieces import Piece, colours, overlaps, union_pieces
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Segments, Spheres

needs_mmg = pytest.mark.skipif(not has_mmg(), reason="MMG (pymmg) is not installed")

CENTRES = np.array([[1.37, 1.52, 1.47], [2.61, 1.45, 1.56]])
RADII = np.array([1.0, 0.9])
BOX = (4.0, 3.0, 3.0)


def two_spheres() -> PrimitiveUnion:
    """Two spheres that overlap, in a general position in a box."""
    return PrimitiveUnion(mg.Domain(size=BOX, periodic=False), Spheres(CENTRES, RADII), None)


def union_volume() -> float:
    """The volume of the two spheres: their volumes less that of their lens."""
    r1, r2 = RADII
    d = float(np.linalg.norm(CENTRES[1] - CENTRES[0]))
    lens = math.pi * (r1 + r2 - d) ** 2 * (d * d + 2 * d * (r1 + r2) - 3 * (r1 - r2) ** 2)
    return 4 * math.pi / 3 * (r1**3 + r2**3) - lens / (12 * d)


def to_the_circle(points: np.ndarray) -> np.ndarray:
    """The distances of points to the circle where the two spheres meet."""
    r1, r2 = RADII
    axis = CENTRES[1] - CENTRES[0]
    d = float(np.linalg.norm(axis))
    along = (d * d + r1 * r1 - r2 * r2) / (2 * d)
    radius = math.sqrt(r1 * r1 - along * along)
    rel = np.asarray(points) - CENTRES[0]
    s = rel @ axis / d
    return np.hypot(s - along, np.linalg.norm(rel - s[:, None] * axis / d, axis=1) - radius)


def circle_length() -> float:
    r1, r2 = RADII
    d = float(np.linalg.norm(CENTRES[1] - CENTRES[0]))
    return 2 * math.pi * math.sqrt(r1 * r1 - ((d * d + r1 * r1 - r2 * r2) / (2 * d)) ** 2)


# -- the pieces --------------------------------------------------------------------------


def test_overlaps_of_primitives():
    """The pairs of primitives that enter each other: two spheres, by more than a depth
    or not; with radii increased; through the faces of a periodic box, against the
    search of every pair; spheres and segments."""
    micro = two_spheres()
    assert overlaps(micro).tolist() == [[0, 1]]
    deep = RADII.sum() - float(np.linalg.norm(CENTRES[1] - CENTRES[0]))
    assert overlaps(micro, depth=deep - 1e-9).tolist() == [[0, 1]]
    assert overlaps(micro, depth=deep + 1e-9).shape == (0, 2)
    apart = PrimitiveUnion(
        mg.Domain(size=BOX, periodic=False),
        Spheres(np.array([[1.0, 1.5, 1.5], [3.2, 1.5, 1.5]]), np.array([1.0, 1.0])),
        None,
    )
    assert not len(overlaps(apart)) and not len(overlaps(apart, offset=0.099))
    assert overlaps(apart, offset=0.101).tolist() == [[0, 1]]  # thicker by 0.1 each: they meet
    # a Boolean model in a periodic box against the search of every pair
    model = model_from_config({"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.4})
    boolean = model.generate(mg.Domain.cube(6.0, periodic=True), seed=3)
    c, r = boolean.spheres.centers, boolean.spheres.radii
    every = set()
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            delta = c[i] - c[j]
            delta -= 6.0 * np.round(delta / 6.0)
            if np.linalg.norm(delta) < r[i] + r[j]:
                every.add((i, j))
    found = set(map(tuple, overlaps(boolean).tolist()))
    assert found == every and len(found) > 20
    through = {(i, j) for i, j in every if np.linalg.norm(c[i] - c[j]) >= r[i] + r[j]}
    assert len(through) > 3  # pairs that meet through a face only
    open_box = PrimitiveUnion(mg.Domain.cube(6.0, periodic=False), boolean.spheres, None)
    assert set(map(tuple, overlaps(open_box).tolist())) == every - through
    # segments: two that cross, one apart, and a sphere on the first (primitive 0)
    segments = Segments(
        np.array([[0.5, 1.0, 1.0], [1.5, 0.2, 1.3], [0.5, 2.5, 2.5]]),
        np.array([[2.5, 1.0, 1.0], [1.5, 1.8, 1.3], [2.5, 2.5, 2.5]]),
        np.array([0.2, 0.2, 0.2]),
    )
    crossing = PrimitiveUnion(
        mg.Domain.cube(3.0, periodic=False),
        Spheres(np.array([[2.0, 1.0, 1.35]]), np.array([0.2])),
        segments,
    )
    assert overlaps(crossing).tolist() == [[0, 1], [1, 2]]
    assert overlaps(crossing, depth=0.06).tolist() == [[1, 2]]  # the sphere enters by 0.05


def test_colours_and_pieces():
    """A colouring gives different colours to the two vertices of each edge, with few
    colours; the pieces of a union are unions of primitives that do not overlap, and the
    smallest of their distances is the distance to the union."""
    rng = np.random.default_rng(0)
    pairs = np.unique(np.sort(rng.integers(0, 60, (150, 2)), axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    colour = colours(60, pairs)
    assert np.all(colour[pairs[:, 0]] != colour[pairs[:, 1]])
    degree = np.bincount(pairs.ravel(), minlength=60)
    assert colour.min() == 0 and colour.max() <= degree.max() and colour.max() < 8
    assert colours(4, np.zeros((0, 2), dtype=int)).tolist() == [0, 0, 0, 0]
    model = model_from_config({"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.4})
    boolean = model.generate(mg.Domain.cube(6.0, periodic=True), seed=3)
    parts = union_pieces(boolean)
    assert 3 <= len(parts) <= 7 and sum(len(p.spheres) for p in parts) == len(boolean.spheres)
    assert all(not len(overlaps(part)) for part in parts)
    points = rng.uniform(0.0, 6.0, (4000, 3))
    smallest = np.minimum.reduce([part.signed_distance(points, 3.0) for part in parts])
    np.testing.assert_allclose(smallest, boolean.signed_distance(points, 3.0), atol=1e-12)
    pieces, rest = boolean.implicit_pieces()
    assert rest == 0 and all(isinstance(p, Piece) and p.phase == 1 for p in pieces)
    np.testing.assert_array_equal(pieces[0].at(points, 3.0), parts[0].signed_distance(points, 3.0))
    # the shallow overlaps are left together when a depth is asked
    assert len(union_pieces(boolean, depth=0.5)) < len(parts)


def test_pieces_of_coatings_and_shells():
    """A coating: the primitives, then the same thicker by its thickness, in more pieces
    since they overlap more; hollow members: their bores, then the primitives. The
    structures whose functions are not those of primitives give no pieces."""
    model = model_from_config(
        {
            "type": "coating",
            "model": {"type": "hardcore_spheres", "radius": 1.0, "volume_fraction": 0.2},
            "thickness": 0.3,
        }
    )
    coated = model.generate(mg.Domain.cube(5.0, periodic=True), seed=2)
    pieces, rest = coated.implicit_pieces(0.0)
    assert rest == 0 and [p.phase for p in pieces] == [1, 2, 2, 2]
    assert len(pieces[0].union.spheres) == 6 and np.all(pieces[0].union.spheres.radii == 1.0)
    assert sum(len(p.union.spheres) for p in pieces[1:]) == 6
    assert all(np.allclose(p.union.spheres.radii, 1.3) for p in pieces[1:])
    # the phase of a point: that of the first piece that holds it
    points = np.random.default_rng(1).uniform(0.0, 5.0, (3000, 3))
    phase = np.zeros(len(points), dtype=int)
    for piece in reversed(pieces):
        phase[piece.at(points, 2.0) < 0] = piece.phase
    exact = np.argmin([coated.implicit_at(k, points, 2.0) for k in (0, 1, 2)], axis=0)
    np.testing.assert_array_equal(phase, exact)
    hollow = model_from_config(
        {
            "type": "coating",
            "model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.3},
            "thickness": 0.25,
            "side": "inside",
        }
    ).generate(mg.Domain.cube(5.0, periodic=True), seed=1)
    pieces, rest = hollow.implicit_pieces(0.0)
    phases = [p.phase for p in pieces]
    assert rest == 0 and phases == sorted(phases) and set(phases) == {0, 1}
    assert np.allclose(pieces[0].union.spheres.radii, 0.75)  # the bores first
    gyroid = model_from_config(
        {"type": "tpms", "surface": "gyroid", "cell_size": 2.0, "volume_fraction": 0.3}
    )
    assert gyroid.generate(mg.Domain.cube(2.0, periodic=True), seed=0).implicit_pieces() is None


def test_ridges_of_a_mesh_file(tmp_path):
    """The ridges, corners and required entities of a mesh file come back as indices."""
    nodes = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    medit.write_mesh(
        tmp_path / "one.mesh",
        nodes,
        np.array([[0, 1, 2, 3]]),
        np.array([[0, 2, 1], [0, 1, 3]]),
        edges=np.array([[0, 1], [1, 2], [0, 3]]),
        edge_refs=np.array([0, 1007, 5]),
        ridges=np.array([0, 1]),
        corners=np.array([1]),
        required_vertices=np.array([0, 3]),
        required_edges=np.array([2]),
        required_triangles=np.array([1]),
    )
    out = medit.read_mesh(tmp_path / "one.mesh")
    assert out["Edges"].tolist() == [[0, 1], [1, 2], [0, 3]]
    assert out["Edges_refs"].tolist() == [0, 1007, 5] and out["Ridges"].tolist() == [0, 1]
    assert out["Corners"].tolist() == [1] and out["RequiredVertices"].tolist() == [0, 3]
    assert out["RequiredEdges"].tolist() == [2] and out["RequiredTriangles"].tolist() == [1]


def test_sharp_edges_follow_the_meshes_derived():
    """The sharp edges of a mesh go with the meshes made of it: 10-node elements, the
    4-node ones again, a part of the elements (with the new numbers of its nodes)."""
    nodes = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    elements = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    mesh = TetMesh(nodes, elements, np.array([0, 1]), np.zeros(3), np.ones(3),
                   sharp=np.array([[1, 2], [2, 3], [0, 1]]),
                   sharp_pieces=np.array([[0, 1], [0, 2], [1, 2]]))  # fmt: skip
    quad = mesh.quadratic()
    assert quad.sharp.tolist() == mesh.sharp.tolist() and quad.n_nodes == 14
    assert quad.linear().sharp.tolist() == mesh.sharp.tolist()
    part = mesh.select(np.array([False, True]))  # the nodes 1, 2, 3, 4 become 0, 1, 2, 3
    assert part.sharp.tolist() == [[0, 1], [1, 2]] and part.sharp_pieces.tolist() == [
        [0, 1],
        [0, 2],
    ]
    assert quad.subset([1]).sharp.tolist() == [[0, 1], [1, 2]]
    with pytest.raises(ValueError, match="sharp_pieces"):
        TetMesh(nodes, elements, np.array([0, 1]), np.zeros(3), np.ones(3),
                sharp=np.array([[1, 2]]), sharp_pieces=np.array([[0, 1], [0, 2]]))  # fmt: skip


def test_configuration_of_the_sharp_edges(tmp_path):
    path = tmp_path / "study.yaml"
    text = (
        "model: {type: boolean_spheres, radius: 1.0, volume_fraction: 0.3}\n"
        "domain: {size: 5.0}\nmesh: {size: 0.5, sharp_edges: KEY}\n"
    )
    path.write_text(text.replace("KEY", "false"), encoding="utf-8")
    spec = load_config(path).mesh
    assert spec.sharp_edges is False and spec.kwargs()["sharp_edges"] is False
    path.write_text(text.replace(", sharp_edges: KEY", ""), encoding="utf-8")
    assert "sharp_edges" not in load_config(path).mesh.kwargs()
    path.write_text(text.replace("KEY", "2"), encoding="utf-8")
    with pytest.raises(ConfigError, match="sharp_edges"):
        load_config(path)


# -- the mesher --------------------------------------------------------------------------


@needs_mmg
def test_curved_elements_along_the_groove_of_two_spheres():
    """Two overlapping spheres meshed with 10-node elements of size 0.3: the mesh has
    edges along the circle where the spheres meet, their nodes are on it, and the volume
    of the union is exact within 0.003 % (0.05 % when the groove is rounded)."""
    micro = two_spheres()
    mesh = mg.mesh.conforming_mesh(micro, size=0.3, order=2, method="grid")
    info = mesh.metadata["sharp_edges"]
    assert info["pieces"] == 2 and info["edges"] == len(mesh.sharp) > 40
    assert info["depth"] == pytest.approx(0.075)
    assert np.all(mesh.sharp_pieces == [0, 1])
    corners = np.unique(mesh.sharp)
    off = to_the_circle(mesh.nodes[corners])
    assert off.max() < 0.015 and np.count_nonzero(off < 1e-9) >= 0.95 * len(corners)
    # the edges make the whole circle, once: each of their nodes belongs to two of them
    assert np.all(np.bincount(mesh.sharp.ravel())[corners] == 2)
    chords = np.linalg.norm(mesh.nodes[mesh.sharp[:, 0]] - mesh.nodes[mesh.sharp[:, 1]], axis=1)
    assert chords.sum() == pytest.approx(circle_length(), rel=5e-3)
    assert 0.03 < chords.min() and chords.max() < 0.3
    # the nodes of the middles of these edges, on the circle too
    ends = np.sort(mesh.elements[:, TET_EDGES].reshape(-1, 2), axis=1)
    middles = mesh.elements[:, 4:].reshape(-1)
    key = {tuple(e): m for e, m in zip(ends.tolist(), middles.tolist(), strict=True)}
    between = np.array([key[tuple(sorted(edge))] for edge in mesh.sharp.tolist()])
    assert np.count_nonzero(to_the_circle(mesh.nodes[between]) < 1e-9) >= 0.95 * len(between)
    curved = mesh.metadata["curved"]
    assert curved["sharp_nodes"] == len(corners) + len(between)
    assert curved["nodes"] >= curved["interface_nodes"] - 2
    volume = mesh.phase_fractions()[1] * float(np.prod(BOX))
    assert volume == pytest.approx(union_volume(), rel=3e-5)
    assert mesh.quality().min() > 0.15 and mesh.min_jacobian() >= 0.3


@needs_mmg
def test_linear_mesh_with_and_without_the_groove():
    """The 4-node mesh of the two spheres: its sharp edges follow the circle within the
    Hausdorff distance of the mesher; without them (``sharp_edges=False``) no node of
    the interface comes as near the circle, and the mesh has no sharp edges."""
    micro = two_spheres()
    sharp = mg.mesh.conforming_mesh(micro, size=0.4, order=1, method="grid")
    assert sharp.order == 1 and "curved" not in sharp.metadata
    off = to_the_circle(sharp.nodes[np.unique(sharp.sharp)])
    assert off.max() < 0.02 and np.median(off) < 0.008  # the Hausdorff distance: 0.02
    chords = np.linalg.norm(sharp.nodes[sharp.sharp[:, 0]] - sharp.nodes[sharp.sharp[:, 1]], axis=1)
    assert chords.sum() == pytest.approx(circle_length(), rel=0.01)
    rounded = mg.mesh.conforming_mesh(micro, size=0.4, order=1, sharp_edges=False, method="grid")
    assert rounded.sharp is None and "sharp_edges" not in rounded.metadata
    # a polyhedron inscribed in the spheres in both cases: the same loss of volume
    assert sharp.phase_fractions()[1] == pytest.approx(rounded.phase_fractions()[1], rel=0.02)
    # the curved elements of the mesh without sharp edges: as before
    quad = curved_mesh(rounded, micro)
    assert quad.metadata["curved"]["sharp_nodes"] == 0
    # and those of the mesh with them, made afterwards
    quad = curved_mesh(sharp, micro)
    assert quad.metadata["curved"]["sharp_nodes"] > 2 * len(np.unique(sharp.sharp)) - 2
    volume = quad.phase_fractions()[1] * float(np.prod(BOX))
    assert volume == pytest.approx(union_volume(), rel=1e-4)


@needs_mmg
def test_options_and_errors_of_the_sharp_edges(monkeypatch, caplog):
    """A structure without pieces cannot be asked for its sharp edges; one without
    overlaps is meshed as before by default; one phase alone keeps its sharp edges; a
    failure of the cuts falls back on the rounded edges with a warning, unless the sharp
    edges were asked."""
    gyroid = model_from_config(
        {"type": "tpms", "surface": "gyroid", "cell_size": 2.0, "volume_fraction": 0.3}
    ).generate(mg.Domain.cube(2.0, periodic=True), seed=0)
    with pytest.raises(ValueError, match="smooth pieces"):
        mg.mesh.conforming_mesh(gyroid, size=0.5, sharp_edges=True, method="grid")
    one = PrimitiveUnion(mg.Domain.cube(3.0, periodic=False),
                         Spheres(np.array([[1.5, 1.4, 1.6]]), np.array([1.0])), None)  # fmt: skip
    assert (
        mg.mesh.conforming_mesh(one, size=0.6, method="grid").sharp is None
    )  # one piece: nothing to follow
    asked = mg.mesh.conforming_mesh(one, size=0.6, sharp_edges=True, method="grid")
    assert asked.metadata["sharp_edges"] == {"pieces": 1, "edges": 0, "depth": 0.15}

    def broken(*args, **kwargs):
        raise RuntimeError("mmg3d failed (exit code 1):\\nno luck")

    micro = two_spheres()
    # one phase alone: the spheres, by name or by number, with their groove
    for phases in ("solid", [1]):
        solid = mg.mesh.conforming_mesh(micro, size=0.6, order=1, phases=phases, method="grid")
        assert set(np.unique(solid.phase)) == {1} and len(solid.sharp) > 15
        assert solid.metadata["phases"] == "solid"
        assert to_the_circle(solid.nodes[np.unique(solid.sharp)]).max() < 0.03
    monkeypatch.setattr(conforming, "_mmg_pieces", broken)
    with caplog.at_level(logging.WARNING, logger="morphogen"):
        mesh = mg.mesh.conforming_mesh(micro, size=0.6, order=1, method="grid")
    assert mesh.sharp is None and "sharp edges could not be followed" in caplog.text
    with pytest.raises(RuntimeError, match="mmg3d failed"):
        mg.mesh.conforming_mesh(micro, size=0.6, order=1, sharp_edges=True, method="grid")


@pytest.mark.slow
@needs_mmg
def test_three_spheres_meet_at_two_points():
    """Three spheres that overlap two by two: three sharp edges, which join at the two
    points where the three surfaces meet; the mesh has a node at each, on the three
    spheres to round-off."""
    centres = np.array([[1.45, 1.52, 1.47], [2.55, 1.45, 1.56], [2.03, 2.45, 1.4]])
    radii = np.array([1.0, 0.9, 0.95])
    micro = PrimitiveUnion(mg.Domain(size=(4.0, 4.0, 3.0), periodic=False),
                           Spheres(centres, radii), None)  # fmt: skip
    mesh = mg.mesh.conforming_mesh(micro, size=0.3, order=2, method="grid")
    assert mesh.metadata["sharp_edges"]["pieces"] == 3
    assert np.unique(mesh.sharp_pieces, axis=0).tolist() == [[0, 1], [0, 2], [1, 2]]
    meeting: dict[int, set] = {}
    for (a, b), pair in zip(mesh.sharp.tolist(), mesh.sharp_pieces.tolist(), strict=True):
        meeting.setdefault(a, set()).update(pair)
        meeting.setdefault(b, set()).update(pair)
    triple = [node for node, group in meeting.items() if len(group) == 3]
    assert len(triple) == 2
    for node in triple:
        distances = np.linalg.norm(mesh.nodes[node] - centres, axis=1) - radii
        assert np.abs(distances).max() < 1e-12
        assert np.count_nonzero(mesh.sharp == node) == 3  # three sharp edges end there
    curved = mesh.metadata["curved"]
    assert curved["nodes"] >= curved["interface_nodes"] - 5
    assert mesh.quality().min() > 0.1


@pytest.mark.slow
@needs_mmg
def test_sharp_edges_through_the_faces_of_a_periodic_box():
    """Two spheres that overlap through a face of a periodic box, their groove across it:
    the opposite faces of the mesh match, the sharp edges that end on a face have their
    image on the opposite one, and the volume of the union is exact within 0.001 %."""
    domain = mg.Domain.cube(3.0, periodic=True)
    centres = np.array([[0.1, 1.1, 1.5], [2.95, 2.0, 1.45]])
    micro = PrimitiveUnion(domain, Spheres(centres, np.array([1.0, 0.9])), None)
    assert overlaps(micro).tolist() == [[0, 1]]
    assert not len(overlaps(PrimitiveUnion(mg.Domain.cube(3.0, periodic=False),
                                           micro.spheres, None)))  # fmt: skip
    mesh = mg.mesh.conforming_mesh(micro, size=0.3, order=2, method="grid")
    assert mesh.periodic and mesh.matching_faces()
    assert mesh.metadata["sharp_edges"]["edges"] == len(mesh.sharp) > 40
    ends = mesh.nodes[np.unique(mesh.sharp)]
    on_low = ends[np.abs(ends[:, 0]) < 1e-9]
    on_high = ends[np.abs(ends[:, 0] - 3.0) < 1e-9]
    assert len(on_low) == len(on_high) == 2  # the circle crosses the face twice
    for point in on_low:
        assert np.abs(on_high - (point + np.array([3.0, 0.0, 0.0]))).sum(axis=1).min() < 1e-9
    delta = centres[0] - centres[1]
    d = float(np.linalg.norm(delta - 3.0 * np.round(delta / 3.0)))
    lens = math.pi * (1.9 - d) ** 2 * (d * d + 3.8 * d - 0.03) / (12 * d)
    exact = (4 * math.pi / 3 * (1.0 + 0.729) - lens) / 27.0
    assert mesh.phase_fractions()[1] == pytest.approx(exact, rel=1e-5)
    curved = mesh.metadata["curved"]
    assert curved["nodes"] > 0.93 * curved["interface_nodes"]


@pytest.mark.slow
@needs_mmg
def test_coated_spheres_keep_the_volume_of_their_interphase():
    """Spheres with coatings that overlap, in a periodic box (three phases): the mesh
    follows the grooves of the coatings, and the volume of the interphase — the shells
    less the lenses that two coated spheres share — is exact within 0.2 %, where the
    rounded grooves give 1 %."""
    model = model_from_config(
        {
            "type": "coating",
            "model": {"type": "hardcore_spheres", "radius": 1.0, "volume_fraction": 0.2},
            "thickness": 0.3,
        }
    )
    micro = model.generate(mg.Domain.cube(5.0, periodic=True), seed=2)
    centres = micro.inner.spheres.centers
    shells = len(centres) * 4 * math.pi / 3 * 1.3**3
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            delta = centres[i] - centres[j]
            d = float(np.linalg.norm(delta - 5.0 * np.round(delta / 5.0)))
            if d < 2.6:
                shells -= math.pi * (2.6 - d) ** 2 * (d * d + 5.2 * d) / (12 * d)
    spheres = len(centres) * 4 * math.pi / 3
    exact = {1: spheres / 125.0, 2: (shells - spheres) / 125.0}
    mesh = mg.mesh.conforming_mesh(micro, size=0.5, order=2, method="grid")
    assert mesh.matching_faces() and set(np.unique(mesh.phase)) == {0, 1, 2}
    assert mesh.metadata["sharp_edges"]["pieces"] == 4 and len(mesh.sharp) > 100
    assert set(map(tuple, mesh.sharp_pieces.tolist())) <= {(1, 2), (1, 3), (2, 3)}
    fractions = mesh.phase_fractions()
    assert fractions[2] == pytest.approx(exact[2], rel=2e-3)
    assert fractions[1] == pytest.approx(exact[1], rel=2e-3)
    rounded = mg.mesh.conforming_mesh(micro, size=0.5, order=2, sharp_edges=False, method="grid")
    assert abs(rounded.phase_fractions()[2] / exact[2] - 1) > 3 * abs(fractions[2] / exact[2] - 1)
    on, all_ = mesh.metadata["curved"]["nodes"], mesh.metadata["curved"]["interface_nodes"]
    was = rounded.metadata["curved"]["nodes"] / rounded.metadata["curved"]["interface_nodes"]
    assert on / all_ > was


@pytest.mark.slow
@needs_mmg
def test_fibres_that_cross():
    """Fibres with round caps that cross, in a periodic box: the lines where two fibres
    meet are sharp edges of the mesh, between different pieces, and the faces match."""
    model = model_from_config(
        {"type": "boolean_cylinders", "radius": 0.4, "length": 3.0, "volume_fraction": 0.3,
         "caps": "round"}
    )  # fmt: skip
    micro = model.generate(mg.Domain.cube(4.0, periodic=True), seed=1)
    assert micro.segments.round_caps and len(overlaps(micro)) > 10
    mesh = mg.mesh.conforming_mesh(micro, size=0.4, order=2, method="grid")
    assert mesh.matching_faces() and len(mesh.sharp) > 30
    assert np.all(mesh.sharp_pieces[:, 0] < mesh.sharp_pieces[:, 1])
    pieces, _ = micro.implicit_pieces(mesh.metadata["sharp_edges"]["depth"])
    assert mesh.sharp_pieces.max() < len(pieces) == mesh.metadata["sharp_edges"]["pieces"]
    # the nodes of the sharp edges lie near the surfaces of their two pieces
    worst = 0.0
    for (a, b), (p, q) in zip(mesh.sharp[:60].tolist(), mesh.sharp_pieces[:60].tolist(),
                              strict=True):  # fmt: skip
        both = [abs(float(pieces[k].at(mesh.nodes[[a, b]], 1.0).max())) for k in (p, q)]
        worst = max(worst, *both)
    assert worst < 0.05
    curved = mesh.metadata["curved"]
    assert curved["nodes"] > 0.85 * curved["interface_nodes"]
