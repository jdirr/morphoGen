"""Woven textile composites (morphogen.models.textile): the interlacing matrices of the 2D
weaves, the paths of the yarns against the half cosines and arcs they are made of (their
arc lengths in closed form against quadratures), the volumes of the swept sections
against Monte Carlo, the crossings, the binders of the 3D weaves, periodicity and windows,
the labels and orientations of the segments (straight tows exactly along their axes, the
FFT stiffness with orientations equal to that of the tows given in the axes of the
sample), the conforming meshes of the exact geometry with a label and an orientation per
element (the finite elements of oriented tows against the same tows given in the axes of
the sample, and against the FFT solver), and the round trips; the variability of the yarns
(the processes along them against their prescribed standard deviations and correlations,
neighbouring yarns, varying sections against points, the mean areas and lengths of the
theory), the binders of any depth and the angle interlocks without stuffers, the tows as
composite cylinders."""

from __future__ import annotations

import dataclasses
import itertools
import json
import math

import numpy as np
import pytest
from scipy.integrate import quad

import morphogen as mg
from morphogen.homogenization import Material, homogenize
from morphogen.homogenization.unidirectional import composite_cylinders
from morphogen.mesh.conforming import conforming_mesh, has_mmg
from morphogen.mesh.exact import exact_route
from morphogen.mesh.tetra import TET_FACES
from morphogen.models.base import Microstructure
from morphogen.models.textile import (
    Piece,
    TextileStructure,
    WovenTextile,
    Yarn,
    interlacing,
    neighbour_correlations,
    path_length,
    process_covariance,
    process_spectrum,
    process_values,
    profile,
    section_area,
)
from morphogen.orientations import euler_to_matrix

#: Transversely isotropic tows (carbon/epoxy: E_L 140, E_T 10, G_LT 5 GPa, nu_LT 0.3,
#: nu_TT 0.4), the axis 3 along the fibres
HEXAGONAL = (12.087, 4.944, 5.109, 143.066, 5.0)
ORTHOGONAL = dict(weave="orthogonal", layers=2, warp_spacing=2.0, yarn_width=1.4,
                  yarn_thickness=0.3, binder_width=0.5, binder_thickness=0.2)  # fmt: skip
#: An orthogonal weave of round tows, quick to mesh from its exact geometry
ROUND = dict(weave="orthogonal", layers=1, warp_spacing=2.0, yarn_width=1.0, yarn_thickness=0.6,
             binder_width=0.6, binder_thickness=0.4)  # fmt: skip


def along(axis):
    """The tows of :data:`HEXAGONAL` with their fibres along x (0) or y (1), in the axes
    of the sample (Voigt order 11 22 33 23 31 12)."""
    c11, c12, c13, c33, c44 = HEXAGONAL
    c66 = 0.5 * (c11 - c12)
    c = np.zeros((6, 6))
    trans = [a for a in range(3) if a != axis]
    c[axis, axis] = c33
    for a in trans:
        c[a, a] = c11
        c[a, axis] = c[axis, a] = c13
    c[trans[0], trans[1]] = c[trans[1], trans[0]] = c12
    shear = {frozenset((1, 2)): 3, frozenset((2, 0)): 4, frozenset((0, 1)): 5}
    for pair, k in shear.items():
        c[k, k] = c44 if axis in pair else c66
    return Material(stiffness=c)


def monte_carlo(micro, n=400_000, seed=0):
    """The fraction of each phase at uniform points of the domain, and their standard
    errors."""
    d = micro.domain
    pts = d.lower + np.random.default_rng(seed).random((n, 3)) * d.lengths
    phase = micro.phase_of(pts)
    p = np.array([np.mean(phase == k) for k in range(4)])
    return p, np.sqrt(np.maximum(p * (1 - p), 1e-12) / n)


def test_interlacing_matrices():
    """The repeats of the 2D weaves: plain, twill (each warp over n, under m, the next
    shifted by one weft), satin (one crossing per row and per column, at the move) and
    basket (the plain weave of groups)."""
    assert np.array_equal(interlacing("plain"), [[True, False], [False, True]])
    twill = interlacing("twill", [2, 1])
    assert twill.shape == (3, 3)
    assert np.array_equal(twill, [[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    for i in range(3):
        assert np.array_equal(np.roll(twill[0], i), twill[i])
    satin = interlacing("satin", [8, 3])
    under = ~satin
    assert under.sum(axis=0).tolist() == [1] * 8 and under.sum(axis=1).tolist() == [1] * 8
    assert [int(np.flatnonzero(row)[0]) for row in under] == [(3 * i) % 8 for i in range(8)]
    basket = interlacing("basket", [3])
    assert basket.shape == (6, 6)
    assert np.array_equal(basket[::3, ::3], interlacing("plain"))
    assert np.all(basket[:3, :3]) and not np.any(basket[:3, 3:])
    for weave, pattern in (("twill", [2]), ("satin", [5, 1]), ("satin", [6, 3]),
                           ("basket", [0]), ("plain", [1])):  # fmt: skip
        with pytest.raises(ValueError, match="pattern"):
            interlacing(weave, pattern)


def test_pieces_and_their_lengths():
    """The half cosine and its arc length (2 L / pi) E(-(pi dz / 2L)^2), the arcs and
    lines of a rounded rise (tangent at both ends, through its end points), and the paths
    through stations: their lengths against quadratures."""
    cos = Piece("cosine", 1.0, 0.2, 3.0, -0.3)
    span, rise = 2.0, -0.5
    slope = rise * math.pi / (2 * span)
    exact = quad(lambda x: math.hypot(1.0, slope * math.sin(math.pi * x / span)), 0.0, span)[0]
    assert cos.length == pytest.approx(exact, rel=1e-12)
    s, z, ts, tz = cos.sample(64)
    assert (s[0], z[0], s[-1], z[-1]) == pytest.approx((1.0, 0.2, 3.0, -0.3))
    assert np.allclose(ts**2 + tz**2, 1.0)
    chord = np.sum(np.hypot(np.diff(s), np.diff(z)))
    assert chord == pytest.approx(exact, rel=2e-4)
    # a rise by arcs and a line: continuous, tangent, horizontal at both ends
    pieces = profile([0.0, 1.5, 1.5, 0.0], 0.5, 2.0, transition=0.3, radius=0.3)
    kinds = [p.kind for p in pieces]
    assert "arc" in kinds and kinds.count("line") >= 3
    for a, b in itertools.pairwise(pieces):
        assert (a.s1, a.z1) == pytest.approx((b.s0, b.z0), abs=1e-12)
        ta = a.sample(1)[2:]
        tb = b.sample(1)[2:]
        assert (ta[0][-1], ta[1][-1]) == pytest.approx((tb[0][0], tb[1][0]), abs=1e-9)
    for p in pieces:
        s, z, ts, tz = p.sample(400)
        assert np.sum(np.hypot(np.diff(s), np.diff(z))) == pytest.approx(p.length, rel=1e-5)
    # the vertical part of a rise through a gap of 0.6 with corners of radius 0.3
    vertical = [p for p in pieces if p.kind == "line" and p.s0 == pytest.approx(p.s1)]
    assert len(vertical) == 2 and vertical[0].length == pytest.approx(1.5 - 0.6)
    # level stations merged, and the total length of a plain profile
    flat = profile([0.0, 0.0, 0.0], 0.0, 1.0)
    assert len(flat) == 1 and path_length(flat) == pytest.approx(2.0)


def test_section_areas():
    """The areas of the sections against the fractions of points in their bounding
    rectangles (the lens of a circle when as thick as wide: the disc)."""
    from morphogen.models.textile import _inside_section

    rng = np.random.default_rng(3)
    n = 1_000_000
    a, c = 1.0, 0.3
    u, v = rng.uniform(-a, a, n), rng.uniform(-c, c, n)
    for section, power in (("ellipse", 2.0), ("lenticular", 2.0), ("power", 3.0),
                           ("power", 1.0)):  # fmt: skip
        p = np.mean(_inside_section(u, v, a, c, section, power))
        se = math.sqrt(p * (1 - p) / n)
        assert section_area(a, c, section, power) / (4 * a * c) == pytest.approx(p, abs=4 * se)
    assert section_area(1.0, 1.0, "lenticular") == pytest.approx(math.pi)
    assert section_area(1.0, 0.5, "power", 2.0) == pytest.approx(section_area(1.0, 0.5))
    assert section_area(1.0, 0.5, "power", 1.0) == pytest.approx(1.0)  # a rhombus


def test_plain_weave_crossings_crimp_and_volumes():
    """A plain weave of thin yarns: at each crossing the warp is above the weft where the
    interlacing matrix says so; the crimp is that of half cosines between the levels
    +-(c_w + c_f)/2; the yarns do not interpenetrate and their fractions are the areas of
    their sections times their lengths."""
    model = WovenTextile(weave="plain", warp_spacing=2.0, yarn_width=1.0, yarn_thickness=0.2)
    domain = mg.Domain(size=(4.0, 4.0, 0.8), periodic=True)
    micro = model.generate(domain, seed=1)
    assert micro.n_yarns == 4 and model.thickness() == pytest.approx(0.4)
    w = interlacing("plain")
    middle, h = 0.4, 0.1  # the mid-plane of the ply, half the gap of the centrelines
    for i in range(2):
        for j in range(2):
            x, y = (j + 0.5) * 2.0, (i + 0.5) * 2.0
            upper, lower = micro.phase_of([[x, y, middle + h], [x, y, middle - h]])
            assert (upper, lower) == ((1, 2) if w[i, j] else (2, 1))
    # crimp: two half cosines of rise 2h = 0.2 over a spacing of 2 per period of 4
    k = math.pi * 0.2 / (2 * 2.0)
    arc = quad(lambda x: math.sqrt(1 + (k * math.sin(math.pi * x / 2.0)) ** 2), 0, 2.0)[0]
    crimp = arc / 2.0 - 1.0
    theory = model.theory(domain)
    assert theory["warp_crimp"] == pytest.approx(crimp, rel=1e-9)
    assert micro.summary()["warp_crimp"] == pytest.approx(crimp, rel=1e-9)
    for yarn in micro.yarns:  # the polylines, slightly shorter than the paths
        assert yarn.length == pytest.approx(4.0 * (1.0 + crimp), rel=1e-4)
    assert micro.summary()["interpenetration"] < 1e-3
    exact = math.pi * 0.5 * 0.1 * 4.0 * (1 + crimp) * 2 / domain.volume  # 2 warps
    assert theory["warp_fraction"] == pytest.approx(exact, rel=1e-9)
    assert theory["volume_fraction"] == theory["warp_fraction"]
    p, se = monte_carlo(micro)
    for k, name in ((1, "warp"), (2, "weft")):
        assert p[k] == pytest.approx(theory[f"{name}_fraction"], abs=4 * se[k])
    assert p[3] == 0.0


def test_interpenetrating_yarns_partition_their_union():
    """Wide yarns of a plain weave overlap near the crossings: a point inside two belongs
    to the one it is deeper in (one phase per point), and the union of the yarns is their
    nominal fraction times one minus the share where they overlap."""
    model = WovenTextile(weave="plain", warp_spacing=2.0, yarn_width=1.8, yarn_thickness=0.3)
    micro = model.generate(mg.Domain(size=(4.0, 4.0, 0.6), periodic=True), seed=1)
    s = micro.summary()
    assert 0.03 < s["interpenetration"] < 0.1
    p, _ = monte_carlo(micro, n=1_000_000)
    union = 1.0 - p[0]
    assert union == pytest.approx(s["yarn_fraction"] * (1 - s["interpenetration"]), abs=0.003)
    img = micro.voxelize((32, 32, 12))
    count = micro.yarn_count(grid=img.grid()).data
    assert count.max() >= 2 and np.all((count > 0) == (img.data > 0))


def test_orthogonal_weave():
    """The 3D orthogonal weave: straight warps and wefts in layers, binders over the top
    wefts and under the bottom ones, through the thickness in the gaps (no overlap); the
    exact fractions, the crimp of the binders (arcs and vertical lines); the fabric
    centred in the box, nothing beyond its thickness."""
    model = WovenTextile(**ORTHOGONAL)
    domain = mg.Domain(size=(4.0, 4.0, 2.0), periodic=True)
    micro = model.generate(domain, seed=1)
    s = micro.summary()
    assert (s["n_warp"], s["n_weft"], s["n_binder"]) == (4, 6, 2)
    assert model.thickness() == pytest.approx(2 * 0.6 + 0.3 + 2 * 0.2)
    assert s["interpenetration"] == 0.0
    binders = [y for y in micro.yarns if y.family == "binder"]
    z = np.concatenate([y.nodes[:, 2] for y in binders])
    base = 0.5 * (2.0 - 1.9)
    assert z.min() == pytest.approx(base + 0.1) and z.max() == pytest.approx(base + 1.8)
    # the crimp: per period of 4, two plateaus of 1.4 and two rises of 1.7 through gaps of
    # 0.6 (quarter circles of radius 0.3 and vertical lines of 1.1)
    rise = 2 * (math.pi / 2 * 0.3) + 1.1
    assert s["binder_crimp"] == pytest.approx((2 * 1.4 + 2 * rise) / 4.0 - 1.0, rel=1e-12)
    assert s["warp_crimp"] == s["weft_crimp"] == 0.0
    theory = model.theory(domain)
    p, se = monte_carlo(micro)
    for k, name in ((1, "warp"), (2, "weft"), (3, "binder")):
        assert p[k] == pytest.approx(theory[f"{name}_fraction"], abs=4 * se[k])
    assert theory["warp_fraction"] == pytest.approx(4 * math.pi * 0.7 * 0.15 * 4.0 / 32.0)
    img = micro.voxelize((40, 40, 40))
    assert not np.any(img.data[:, :, 0]) and not np.any(img.data[:, :, -1])  # resin films
    # labels: a segment per straight tow, the matrix last; its orientation along its axis
    labels = micro.labels(grid=img.grid()).data
    assert np.array_equal(micro.cell_phases[labels], img.data)
    assert np.all(labels[img.data == 0] == micro.matrix_label)
    axes3 = euler_to_matrix(micro.orientations)[:, 2]
    k = 0
    for yarn in micro.yarns:
        along = {"warp": [1, 0, 0], "weft": [0, 1, 0]}.get(yarn.family)
        if along is not None:
            assert yarn.n_segments == 1
            assert np.allclose(np.abs(axes3[k]), along)
        k += yarn.n_segments
    assert np.allclose(micro.orientations[-1], 0.0)


def test_angle_interlocks():
    """Layer-to-layer: a binder per warp layer, each between a weft of its layer and one
    of the layer below; through the thickness: a binder from the top to the bottom, one
    weft layer per column; the axes of their segments along the slopes of their paths."""
    common = dict(layers=3, warp_spacing=2.0, yarn_width=1.4, yarn_thickness=0.3,
                  binder_width=0.5, binder_thickness=0.2)  # fmt: skip
    l2l = WovenTextile(weave="layer_to_layer", **common)
    micro = l2l.generate(mg.Domain(size=(4.0, 4.0, 2.6), periodic=True), seed=1)
    binders = [y for y in micro.yarns if y.family == "binder"]
    assert len(binders) == 2 * 3 and l2l.repeat() == (2, 2)
    # the slots of the binders: 0.1, the three warp layers 0.65, 1.25, 1.85, and 2.4 (above
    # the bottom of the fabric); the binder of level l between the slots l and l + 2
    spans = sorted(round(float(np.ptp(y.nodes[:, 2])), 9) for y in binders)
    assert spans == pytest.approx([1.15] * 4 + [1.2] * 2)
    att = WovenTextile(weave="through_thickness", **common)
    assert att.repeat() == (8, 2)
    micro = att.generate(mg.Domain(size=(16.0, 4.0, 2.6), periodic=True), seed=1)
    index = next(k for k, y in enumerate(micro.yarns) if y.family == "binder")
    binder = micro.yarns[index]
    base = 0.5 * (2.6 - att.thickness())
    assert binder.nodes[:, 2].min() == pytest.approx(base + 0.1)
    assert binder.nodes[:, 2].max() == pytest.approx(att.thickness() + base - 0.1)
    # the material axis 3 of each segment along its chord
    start, end = binder.ends()[:2]
    chords = (end - start) / np.linalg.norm(end - start, axis=1, keepdims=True)
    k = sum(y.n_segments for y in micro.yarns[:index])
    axes3 = euler_to_matrix(micro.orientations[k : k + binder.n_segments])[:, 2]
    assert np.allclose(axes3, chords, atol=1e-9)
    assert np.max(np.abs(chords[:, 2])) > 0.1  # the binder climbs


def test_periodicity_windows_and_axes():
    """A box of two repeats is two copies of one; a window of the same box is the
    periodic fabric there; the fabric stacked along x or y is the same, its coordinates
    permuted; plies shifted from one another."""
    model = WovenTextile(weave="twill", pattern=[2, 1], warp_spacing=2.0, weft_spacing=1.5,
                         yarn_width=1.6, weft_width=1.2, yarn_thickness=0.3, layers=2,
                         ply_shift=[0.5, 0.25], ply_gap=0.1)  # fmt: skip
    size = (9.0, 6.0, 1.5)
    periodic = model.generate(mg.Domain(size=size, periodic=True), seed=1)
    img = periodic.voxelize((72, 24, 12))
    assert np.mean(img.data[:36] == img.data[36:]) > 0.9995
    window = model.generate(mg.Domain(size=size, periodic=False), seed=1)
    pts = np.random.default_rng(2).random((100_000, 3)) * np.array(size)
    phases = periodic.phase_of(pts)
    assert np.array_equal(window.phase_of(pts), phases)
    for axis, order, sizes in ((0, [2, 0, 1], (1.5, 9.0, 6.0)), (1, [0, 2, 1], (9.0, 1.5, 6.0))):
        turned = WovenTextile.from_config(dict(model.to_config(), axis=axis))
        other = turned.generate(mg.Domain(size=sizes, periodic=True), seed=1)
        assert np.array_equal(other.phase_of(pts[:, order]), phases)
    wrong = r"multiples of 3 x weft_spacing = 4\.5.*nearest: 4\.5 or 9"
    with pytest.raises(ValueError, match=wrong):
        model.generate(mg.Domain(size=(6.0, 6.0, 1.5), periodic=True))
    with pytest.raises(ValueError, match="at least that"):
        model.generate(mg.Domain(size=(9.0, 6.0, 1.0), periodic=True))


def test_point_and_grid_functions():
    """The functions of the phases at points are those of the grids at the centres of
    their voxels; their signs are the phases; the level set of merged yarns."""
    from morphogen.models.implicit import band_of

    micro = WovenTextile(**ORTHOGONAL).generate(mg.Domain(size=(4.0, 4.0, 2.0)), seed=1)
    img = micro.voxelize((24, 24, 12))
    pts = img.grid().center_points()
    assert np.array_equal(micro.phase_of(pts).reshape(img.shape), img.data)
    band = band_of(img.grid())
    at = micro.phase_level_sets_at(pts, band)
    assert micro.implicit_phases() == [0, 1, 2, 3]
    for k in micro.implicit_phases():
        grid = micro.phase_level_set(k, grid=img.grid()).data
        assert np.allclose(at[k].reshape(img.shape), grid, atol=1e-6)
        assert np.all((grid < 0) <= (img.data == k)) and np.all((grid > 0) <= (img.data != k))
    merged = WovenTextile(**ORTHOGONAL, merge_yarns=True)
    assert merged.has_level_set and not WovenTextile(**ORTHOGONAL).has_level_set
    assert merged.active_phases() == {0: "matrix", 1: "yarns"}
    m = merged.generate(mg.Domain(size=(4.0, 4.0, 2.0)), seed=1)
    assert m.implicit_phases() == [0, 1] and m.phase_names == {0: "matrix", 1: "yarns"}
    assert micro.phase_names == {0: "matrix", 1: "warp", 2: "weft", 3: "binder"}
    level = m.level_set(grid=img.grid()).data
    assert np.array_equal(level <= 0, img.data > 0)
    frac = m.voxelize((24, 24, 12), supersampling=2).data
    assert frac.dtype == np.float32 and 0.3 < float(frac.mean()) < 0.6


def test_polydisperse_yarns_and_errors():
    """Sizes drawn per yarn from their laws, bounded by the spacings; the parameters
    checked."""
    model = WovenTextile(weave="plain", warp_spacing=2.0,
                         yarn_width={"type": "uniform", "low": 1.2, "high": 1.8},
                         yarn_thickness={"type": "uniform", "low": 0.2, "high": 0.3})  # fmt: skip
    micro = model.generate(mg.Domain(size=(8.0, 8.0, 0.8), periodic=True), seed=3)
    widths = np.array([2 * y.half_width for y in micro.yarns])
    assert len(np.unique(widths)) == micro.n_yarns and np.all((widths >= 1.2) & (widths <= 1.8))
    again = model.generate(mg.Domain(size=(8.0, 8.0, 0.8), periodic=True), seed=3)
    assert np.array_equal(widths, [2 * y.half_width for y in again.yarns])
    assert model.thickness() == pytest.approx(0.5)
    for bad, match in (
        ({"weave": "jacquard"}, "weave"),
        ({"weave": "plain", "yarn_width": 2.5}, "warp_spacing"),
        ({"weave": "orthogonal", "pattern": [2, 2]}, "pattern"),
        ({"weave": "orthogonal", "binder_width": 1.0, "yarn_width": 1.5}, "binders between"),
        ({"weave": "orthogonal", "binder_thickness": 0.5}, "pass between the wefts"),
        ({"weave": "plain", "section": "square"}, "section"),
        ({"weave": "plain", "section_power": 0.5}, "section_power"),
        ({"weave": "plain", "transition": 0.0}, "transition"),
        ({"weave": "orthogonal", "ply_gap": 0.1}, "plies of 2D weaves"),
        ({"weave": "plain", "section": "lenticular", "yarn_thickness": 1.7}, "no thicker"),
        ({"weave": "plain", "samples": 1}, "samples"),
    ):
        with pytest.raises(ValueError, match=match):
            WovenTextile(**bad)


def test_round_trips():
    """The configuration and the JSON geometry give the same textile."""
    model = WovenTextile(weave="satin", pattern=[5, 2], warp_spacing=2.0, yarn_width=1.9,
                         yarn_thickness=0.4, section="lenticular",
                         fibre_fraction_in_yarns=0.6)  # fmt: skip
    assert mg.model_from_config(model.to_config()) == model
    micro = model.generate(mg.Domain(size=(10.0, 10.0, 0.8), periodic=True), seed=1)
    data = json.loads(json.dumps(micro.to_dict()))
    back = Microstructure.from_dict(data)
    img = micro.voxelize((40, 40, 8))
    assert np.array_equal(back.voxelize(grid=img.grid()).data, img.data)
    assert back.summary() == micro.summary()
    theory = model.theory()
    assert theory["fibre_volume_fraction"] == pytest.approx(0.6 * theory["yarn_fraction"])
    assert model.expected_count(mg.Domain(size=(10.0, 10.0, 0.8))) == pytest.approx(10.0)


def test_oriented_tows_homogenized():
    """The FFT stiffness of an orthogonal composite whose straight tows are turned by the
    orientations of their segments equals that of the same tows given in the axes of the
    sample (fibres along x for the warps, along y for the wefts), the binders as the
    matrix."""
    micro = WovenTextile(**ORTHOGONAL).generate(mg.Domain(size=(4.0, 4.0, 2.0)), seed=1)
    img = micro.voxelize((16, 16, 8))
    labels = micro.labels(grid=img.grid()).data
    resin = Material(young=3.5, poisson=0.35)
    tow = Material.from_config({"hexagonal": list(HEXAGONAL)})
    oriented = homogenize(img, {0: resin, 1: tow, 2: tow, 3: resin}, properties=["stiffness"],
                          grains=labels, orientations=micro.orientations)  # fmt: skip
    given = homogenize(img, {0: resin, 1: along(0), 2: along(1), 3: resin},
                       properties=["stiffness"])  # fmt: skip
    a = oriented["stiffness.periodic"]["tensor"]
    b = given["stiffness.periodic"]["tensor"]
    assert np.allclose(a, b, rtol=1e-5, atol=1e-6 * np.abs(b).max())
    assert a[0, 0] > 3 * a[2, 2]  # stiff in the plane of the fabric


@pytest.fixture(scope="module")
def round_mesh():
    """A woven textile of round tows and its conforming mesh, from its exact geometry."""
    if not has_mmg():
        pytest.skip("MMG (pymmg) is not installed")
    micro = WovenTextile(**ROUND).generate(mg.Domain(size=(4.0, 4.0, 2.8)), seed=1)
    return micro, conforming_mesh(micro, size=0.8, hausdorff=0.08, min_size=0.08)


def interface_areas(mesh) -> np.ndarray:
    """The area of the interfaces of each phase with the others (the faces between two
    elements of different phases)."""
    faces = mesh.elements[:, TET_FACES].reshape(-1, 3)
    owner = np.repeat(np.arange(mesh.n_elements), 4)
    _, first, inverse = np.unique(np.sort(faces, axis=1), axis=0, return_index=True,
                                  return_inverse=True)  # fmt: skip
    twin = np.full(len(first), -1)
    seen = np.full(len(first), -1)
    for i, f in enumerate(inverse.ravel()):  # the two elements of each inner face
        if seen[f] < 0:
            seen[f] = owner[i]
        else:
            twin[f] = owner[i]
    a, b = seen[twin >= 0], twin[twin >= 0]
    between = mesh.phase[a] != mesh.phase[b]
    v = mesh.nodes[faces[first[twin >= 0][between]]]
    area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    out = np.zeros(int(mesh.phase.max()) + 1)
    np.add.at(out, mesh.phase[a][between], area)
    np.add.at(out, mesh.phase[b][between], area)
    return out


def test_exact_conforming_mesh_of_a_textile(round_mesh):
    """Woven textiles are meshed from the exact functions of their phases (no background
    grid): periodic faces that match, the volume of each phase that of the yarns (Monte
    Carlo) within the Hausdorff distance times the area of its interfaces; each element
    labelled by the segment of its yarn (or the matrix), of its phase, its orientation
    along the path of the segment."""
    micro, mesh = round_mesh
    assert exact_route(micro) == "level_set"
    assert mesh.metadata["mesher"].startswith("adaptive level set")
    assert mesh.periodic and mesh.matching_faces()
    volumes = np.array([mesh.volumes()[mesh.phase == k].sum() for k in range(4)])
    p, se = monte_carlo(micro)
    exact = p * micro.domain.volume
    bound = mesh.metadata["hausdorff"] * interface_areas(mesh) + 4 * se * micro.domain.volume
    assert np.all(np.abs(volumes - exact) <= bound)
    assert np.array_equal(micro.cell_phases[mesh.cell], mesh.phase)
    assert np.all(mesh.cell[mesh.phase == 0] == micro.matrix_label)
    assert np.array_equal(mesh.orientations, micro.orientations)
    assert mesh.metadata["labelled_elements"] < 0.05 * mesh.n_elements
    axis3 = euler_to_matrix(mesh.orientations[mesh.cell])[:, 2]
    family = {1: [1.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0]}  # the straight warps and wefts
    for phase, direction in family.items():
        assert np.allclose(np.abs(axis3[mesh.phase == phase]), direction, atol=1e-9)
    binder = axis3[mesh.phase == 3]  # in the planes xz of their paths
    assert np.abs(binder[:, 1]).max() < 1e-9 and np.abs(binder[:, 2]).max() > 0.5


def test_finite_elements_of_oriented_tows(round_mesh):
    """The finite-element stiffness of the tows turned by the orientations of the elements
    equals that of the same tows given in the axes of the sample (the straight warps and
    wefts, the binders of resin), and the FFT stiffness of the same textile on voxels
    within a few percent."""
    micro, mesh = round_mesh
    resin = Material(young=3.5, poisson=0.35)
    tow = Material.from_config({"hexagonal": list(HEXAGONAL)})
    oriented = homogenize(mesh, {0: resin, 1: tow, 2: tow, 3: resin},
                          properties=["stiffness"], solver="direct")  # fmt: skip
    plain = dataclasses.replace(mesh, cell=None, orientations=None)
    given = homogenize(plain, {0: resin, 1: along(0), 2: along(1), 3: resin},
                       properties=["stiffness"], solver="direct")  # fmt: skip
    a = oriented["stiffness.periodic"]["tensor"]
    b = given["stiffness.periodic"]["tensor"]
    assert np.allclose(a, b, rtol=1e-8, atol=1e-8 * np.abs(b).max())
    img = micro.voxelize((40, 40, 28))
    labels = micro.labels(grid=img.grid()).data
    fft = homogenize(img, {0: resin, 1: tow, 2: tow, 3: resin}, properties=["stiffness"],
                     grains=labels, orientations=micro.orientations)  # fmt: skip
    c = fft["stiffness.periodic"]["tensor"]
    assert np.diag(a)[:3] == pytest.approx(np.diag(c)[:3], rel=0.06)


# -- the variability of the yarns, the binders, the stuffers, the tows -----------------------

#: An orthogonal weave of straight warps and wefts, whose yarns vary along their paths
VARYING = dict(ORTHOGONAL, deviation_in_plane=0.06, deviation_through=0.03,
               width_variation=0.1, thickness_variation=0.12, size_correlation=-0.5,
               correlation_length=[4.0, 3.0])  # fmt: skip


def test_processes_of_the_variations():
    """The spectra of the processes: unit variance, the gaussian covariance exact on a
    circle much longer than it, the exponential one smoothed at its cusp by the
    truncation; the sample variance, covariances and slopes of drawn processes; the
    correlations of yarns side by side, positive definite and periodized on a ring."""
    period, length = 40.0, 4.0
    d = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    gauss = process_spectrum(period, length)
    assert gauss[1].sum() == pytest.approx(1.0)
    assert process_covariance(gauss, d) == pytest.approx(np.exp(-(d**2) / (2 * length**2)),
                                                         abs=1e-9)  # fmt: skip
    assert float(np.sum(gauss[1] * gauss[0] ** 2)) == pytest.approx(1 / length**2, rel=1e-9)
    expo = process_spectrum(period, length, "exponential")
    assert process_covariance(expo, d) == pytest.approx(np.exp(-d / length), abs=0.045)
    assert process_covariance(expo, d[3:]) == pytest.approx(np.exp(-d[3:] / length), abs=0.025)
    rng = np.random.default_rng(0)
    s = np.linspace(0.0, period, 400, endpoint=False)
    draws = [process_values(gauss, rng.standard_normal((len(gauss[0]), 2)), s)
             for _ in range(2000)]  # fmt: skip
    values = np.array([v for v, _ in draws])
    slopes = np.array([g for _, g in draws])
    assert values.var() == pytest.approx(1.0, abs=0.03)
    for lag in (20, 40, 80):  # 2, 4 and 8 along s
        assert np.mean(values[:, :-lag] * values[:, lag:]) == pytest.approx(
            float(process_covariance(gauss, lag * 0.1)), abs=0.03
        )
    assert slopes.var() == pytest.approx(1 / length**2, rel=0.05)
    with pytest.raises(ValueError, match="covariance"):
        process_spectrum(period, length, "matern")
    r = neighbour_correlations(6, 0.5, periodic=True)
    assert r[0, 1] == r[0, 5] == pytest.approx((0.5 + 0.5**5) / (1 + 0.5**6))
    assert r[0, 3] == pytest.approx(2 * 0.5**3 / (1 + 0.5**6))
    assert np.all(np.linalg.eigvalsh(r) > 0)
    assert np.allclose(neighbour_correlations(4, 0.5, periodic=False)[0], 0.5 ** np.arange(4))
    assert np.array_equal(neighbour_correlations(3, 0.0, periodic=True), np.eye(3))


def straight_deviations(micro, nominal, family="warp"):
    """The deviations across and through the thickness of the straight yarns of a family
    (their nodes against the same weave without variability, the same seed) and the factors
    of their widths and thicknesses, one row per yarn in the order of the realization."""
    pairs = [(y, y0) for y, y0 in zip(micro.yarns, nominal.yarns, strict=True)
             if y.family == family]  # fmt: skip
    across = 1 if family == "warp" else 0
    da = np.array([y.nodes[:, across] - y0.nodes[0, across] for y, y0 in pairs])
    dz = np.array([y.nodes[:, 2] - y0.nodes[0, 2] for y, y0 in pairs])
    fw = np.array([y.half_widths / y.half_width for y, _ in pairs])
    ft = np.array([y.half_thicknesses / y.half_thickness for y, _ in pairs])
    return da, dz, fw, ft, [y0 for _, y0 in pairs]


def test_variability_of_the_yarns():
    """Deviations of the centrelines and lognormal variations of the sections along the
    yarns: the standard deviations and correlations along the straight warps of an
    orthogonal weave (their nodes against the same weave without variability) against those
    prescribed, the coefficients of variation and the correlation of the sizes, their
    statistics in the summary, the yarns closed across the period without a jump and their
    frames orthonormal, the fractions against the theory and against Monte Carlo, the
    seeds."""
    domain = mg.Domain(size=(40.0, 40.0, 2.4), periodic=True)
    model = WovenTextile(**VARYING)
    micro = model.generate(domain, seed=5)
    nominal = WovenTextile(**ORTHOGONAL).generate(domain, seed=5)
    assert [y.family for y in micro.yarns] == [y.family for y in nominal.yarns]
    dy, dz, fw, ft, _ = straight_deviations(micro, nominal)
    step = 40.0 / dy.shape[1]
    assert step <= 3.0 / 6 + 1e-12  # a sixth of the shorter correlation length
    assert dy.std() == pytest.approx(0.06, rel=0.08) and abs(dy.mean()) < 0.01
    assert dz.std() == pytest.approx(0.03, rel=0.08)
    centre = process_spectrum(40.0, 4.0)
    for lag in (4, 8, 16):
        rho = np.mean(dy[:, :-lag] * dy[:, lag:]) / dy.var()
        assert rho == pytest.approx(float(process_covariance(centre, lag * step)), abs=0.08)
    assert np.std(fw) / np.mean(fw) == pytest.approx(0.1, rel=0.1)
    assert np.std(ft) / np.mean(ft) == pytest.approx(0.12, rel=0.1)
    assert np.mean(fw) == pytest.approx(1.0, abs=0.01)
    logs = np.corrcoef(np.log(fw).ravel(), np.log(ft).ravel())[0, 1]
    assert logs == pytest.approx(-0.5, abs=0.06)
    s = micro.summary()
    assert s["deviation_in_plane"] == pytest.approx(0.06, rel=0.08)
    assert s["width_variation"] == pytest.approx(0.1, rel=0.1)
    assert s["size_correlation"] == pytest.approx(-0.5, abs=0.06)
    for yarn in micro.yarns:  # closed across the period: no segment longer than the others
        start, end = yarn.ends()[:2]
        lengths = np.linalg.norm(end - start, axis=1)
        if yarn.family != "binder":
            assert lengths.max() < 1.02 * np.median(lengths)
        chords = (end - start) / lengths[:, None]
        assert np.abs(np.einsum("ij,ij->i", yarn.normals, yarn.binormals)).max() < 1e-12
        assert np.abs(np.einsum("ij,ij->i", chords, yarn.normals[: len(chords)])).max() < 0.05
    theory = model.theory(domain)
    volumes = micro.volumes()
    for name in ("warp", "weft"):
        chosen = [k for k, y in enumerate(micro.yarns) if y.family == name]
        assert volumes[chosen].sum() / domain.volume == pytest.approx(
            theory[f"{name}_fraction"], rel=0.02
        )
    p, se = monte_carlo(micro, n=600_000)
    union = 1.0 - p[0]
    assert union == pytest.approx(s["yarn_fraction"] * (1 - s["interpenetration"]),
                                  abs=4 * se[0] + 1e-3)  # fmt: skip
    again = model.generate(domain, seed=5)
    assert all(np.array_equal(a.nodes, b.nodes) for a, b in zip(again.yarns, micro.yarns,
                                                                 strict=True))  # fmt: skip
    other = model.generate(domain, seed=6)
    assert not np.array_equal(other.yarns[0].nodes, micro.yarns[0].nodes)


def test_neighbouring_yarns_vary_together():
    """The deviations of the warps of a layer correlate as ``rho^d`` between yarns ``d``
    apart, periodized over the ring of the yarns of a periodic box; those of the two warp
    layers stay independent."""
    domain = mg.Domain(size=(40.0, 40.0, 2.4), periodic=True)
    params = dict(ORTHOGONAL, deviation_in_plane=0.06, correlation_length=4.0,
                  neighbour_correlation=0.7)  # fmt: skip
    model = WovenTextile(**params)
    layers = {"lower": [], "upper": []}
    for seed in range(4):
        micro = model.generate(domain, seed=seed)
        nominal = WovenTextile(**ORTHOGONAL).generate(domain, seed=seed)
        dy, _, _, _, yarns = straight_deviations(micro, nominal)
        height = np.array([y.nodes[0, 2] for y in yarns])
        across = np.array([y.nodes[0, 1] for y in yarns])
        for name, chosen in (("lower", height < 1.2), ("upper", height > 1.2)):
            order = np.argsort(across[chosen])
            layers[name].append(dy[chosen][order])
    lower, upper = np.array(layers["lower"]), np.array(layers["upper"])  # seeds x yarns x nodes
    expected = neighbour_correlations(lower.shape[1], 0.7, periodic=True)
    for gap in (1, 2, 5):
        pairs = lower * np.roll(lower, -gap, axis=1)
        assert pairs.mean() / lower.var() == pytest.approx(expected[0, gap], abs=0.12)
    assert abs(np.mean(lower * upper) / lower.std() / upper.std()) < 0.15


def test_varying_sections_against_points():
    """A straight yarn whose half width grows linearly from 0.4 to 0.8 along it: the points
    across it just inside and just outside the half width there, its function there; its
    volume (four Gauss points per segment) against the closed form; the round trip of its
    sizes."""
    domain = mg.Domain(size=(4.0, 2.0, 1.0), periodic=False)
    x = np.linspace(0.5, 3.5, 7)
    nodes = np.stack([x, np.full(7, 1.0), np.full(7, 0.5)], axis=1)
    widths = 0.4 + 0.4 * (x - 0.5) / 3.0
    yarn = Yarn("warp", 1, nodes, np.tile([0.0, 1.0, 0.0], (7, 1)),
                np.tile([0.0, 0.0, 1.0], (7, 1)), np.zeros(3), 0.6, 0.2, widths=widths)  # fmt: skip
    micro = TextileStructure(domain, [yarn])
    xs = np.linspace(0.6, 3.4, 15)
    half = 0.4 + 0.4 * (xs - 0.5) / 3.0
    inside = np.stack([xs, 1.0 + 0.999 * half, np.full(15, 0.5)], axis=1)
    outside = np.stack([xs, 1.0 + 1.001 * half, np.full(15, 0.5)], axis=1)
    assert np.all(micro.phase_of(inside) == 1) and np.all(micro.phase_of(outside) == 0)
    assert micro.phase_level_sets_at(inside, 0.3)[1] == pytest.approx(-0.001 * half, rel=1e-6)
    exact = math.pi * 0.2 * 3.0 * 0.6  # pi c times the integral of a(x) along the yarn
    assert micro.volumes()[0] == pytest.approx(exact, rel=1e-12)
    back = Microstructure.from_dict(json.loads(json.dumps(micro.to_dict())))
    assert np.array_equal(back.yarns[0].widths, widths) and back.yarns[0].thicknesses is None
    inside_share, shared = micro.sampled(points=20_000)
    assert inside_share[0] == 1.0 and shared[0] == 0.0
    with pytest.raises(ValueError, match="one positive half size per node"):
        Yarn("warp", 1, nodes, yarn.normals, yarn.binormals, np.zeros(3), 0.6, 0.2,
             widths=widths[:3])  # fmt: skip


def test_mean_areas_and_lengths_of_varying_yarns():
    """The mean area of varying sections: ``exp(rho s_w s_t)`` times the area at the mean
    sizes for the ellipse, the Gauss–Hermite quadrature for the lens against Monte Carlo;
    straight yarns lengthened by the deviations of their centrelines, ``1 + lambda_2 (s_n^2
    + s_z^2) / 2``, in the theory and in their realizations."""
    lens = WovenTextile(weave="plain", warp_spacing=2.0, yarn_width=1.6, yarn_thickness=0.4,
                        section="lenticular", width_variation=0.2, thickness_variation=0.25,
                        size_correlation=0.4)  # fmt: skip
    rng = np.random.default_rng(1)
    g1, g2 = rng.standard_normal((2, 1_000_000))
    sw, st = math.sqrt(math.log(1 + 0.2**2)), math.sqrt(math.log(1 + 0.25**2))
    a = 0.8 * np.exp(sw * g1 - sw**2 / 2)
    c = 0.2 * np.exp(st * (0.4 * g1 + math.sqrt(1 - 0.16) * g2) - st**2 / 2)
    areas = section_area(a, c, "lenticular")
    assert lens._mean_area("warp") == pytest.approx(areas.mean(), abs=4 * areas.std() / 1000)
    ellipse = WovenTextile(weave="plain", warp_spacing=2.0, yarn_width=1.6, yarn_thickness=0.4,
                           width_variation=0.2, thickness_variation=0.25,
                           size_correlation=0.4)  # fmt: skip
    assert ellipse._mean_area("warp") == pytest.approx(
        math.pi * 0.8 * 0.2 * math.exp(0.4 * sw * st), rel=1e-12
    )
    straight = WovenTextile(**dict(ORTHOGONAL, deviation_in_plane=0.3, deviation_through=0.2,
                                   correlation_length=2.0))  # fmt: skip
    domain = mg.Domain(size=(16.0, 16.0, 2.4), periodic=True)
    plain = WovenTextile(**ORTHOGONAL).theory(domain)
    longer = straight.theory(domain)
    assert longer["warp_fraction"] / plain["warp_fraction"] == pytest.approx(
        1 + (0.3**2 + 0.2**2) / (2 * 2.0**2), rel=1e-9
    )
    ratios = []
    for seed in range(3):
        m = straight.generate(domain, seed=seed)
        warps = sum(v for v, y in zip(m.volumes(), m.yarns, strict=True) if y.family == "warp")
        ratios.append(warps / domain.volume / longer["warp_fraction"])
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.004)


def test_binder_depths_and_stuffers():
    """Layer-to-layer binders tying three weft layers (two per column, each across two
    warp layers) and all four; orthogonal binders tying two, the columns in turn between
    the layers (a repeat of six rows); an angle interlock without stuffers: the crimps in
    closed form, the fractions against Monte Carlo, the overlaps, the errors."""
    common = dict(layers=3, warp_spacing=2.0, yarn_width=1.4, yarn_thickness=0.3,
                  binder_width=0.5, binder_thickness=0.2)  # fmt: skip
    deep = WovenTextile(weave="layer_to_layer", binder_depth=3, **common)
    domain = mg.Domain(size=(4.0, 4.0, 2.6), periodic=True)
    micro = deep.generate(domain, seed=1)
    slots = np.array(deep._levels()["slots"])  # 0.1, 0.65, 1.25, 1.85 and 2.4
    binders = [y for y in micro.yarns if y.family == "binder"]
    assert len(binders) == 2 * 2
    assert deep.expected_count(domain) == pytest.approx(len(micro.yarns))
    spans = sorted(round(float(np.ptp(y.nodes[:, 2])), 9) for y in binders)
    assert spans == pytest.approx(sorted([slots[3] - slots[0]] * 2 + [slots[4] - slots[1]] * 2))

    def half_cosine_crimp(rise, span=2.0):
        k = math.pi * rise / (2 * span)
        arc = quad(lambda x: math.sqrt(1 + (k * math.sin(math.pi * x / span)) ** 2), 0, span)[0]
        return arc / span - 1.0

    crimp = 0.5 * (half_cosine_crimp(slots[3] - slots[0]) + half_cosine_crimp(slots[4] - slots[1]))
    theory = deep.theory(domain)
    assert theory["binder_crimp"] == pytest.approx(crimp, rel=1e-9)
    s = micro.summary()  # the steep half cosines graze the wefts they pass between
    assert 0.0 < s["interpenetration"] < 0.03
    assert s["yarn_fraction"] == pytest.approx(theory["yarn_fraction"], rel=1e-3)
    p, se = monte_carlo(micro)
    assert 1.0 - p[0] == pytest.approx(s["yarn_fraction"] * (1 - s["interpenetration"]),
                                       abs=4 * se[0] + 1e-3)  # fmt: skip
    full = WovenTextile(weave="layer_to_layer", binder_depth=4, **common)
    binders = [y for y in full.generate(domain, seed=1).yarns if y.family == "binder"]
    assert len(binders) == 2
    assert np.ptp(binders[0].nodes[:, 2]) == pytest.approx(slots[4] - slots[0])
    default = WovenTextile(weave="layer_to_layer", **common)
    assert default.theory(domain) == WovenTextile(weave="layer_to_layer", binder_depth=2,
                                                  **common).theory(domain)  # fmt: skip
    shallow = WovenTextile(weave="orthogonal", binder_depth=2, **common)
    assert shallow.repeat() == (2, 6)
    box = mg.Domain(size=(4.0, 12.0, 2.6), periodic=True)
    micro = shallow.generate(box, seed=1)
    binders = sorted((y for y in micro.yarns if y.family == "binder"),
                     key=lambda y: y.nodes[0, 1])  # fmt: skip
    base = 0.5 * (2.6 - shallow.thickness())
    lows = [float(y.nodes[:, 2].min()) for y in binders]
    assert lows == pytest.approx([base + slots[i % 3] for i in range(6)])
    assert micro.summary()["interpenetration"] == 0.0
    p, se = monte_carlo(micro)
    assert p[3] == pytest.approx(shallow.theory(box)["binder_fraction"], abs=4 * se[3])
    bare = WovenTextile(weave="through_thickness", stuffers=False, layers=3, warp_spacing=2.0,
                        yarn_width=1.4, yarn_thickness=0.3)  # fmt: skip
    assert bare.active_phases() == {0: "matrix", 2: "weft", 3: "binder"}
    assert bare.thickness() == pytest.approx(3 * (0.3 + 0.3) + 0.3 + 2 * 0.3)
    micro = bare.generate(mg.Domain(size=(16.0, 4.0, 2.7), periodic=True), seed=1)
    assert "n_warp" not in micro.summary() and micro.implicit_phases() == [0, 2, 3]
    assert all(2 * y.half_width == pytest.approx(1.4) for y in micro.yarns)
    theory = bare.theory(micro.domain)
    assert theory["volume_fraction"] == 0.0 and "warp_fraction" not in theory
    p, se = monte_carlo(micro)
    assert p[1] == 0.0
    union = 1.0 - p[0]
    s = micro.summary()
    assert union == pytest.approx(s["yarn_fraction"] * (1 - s["interpenetration"]),
                                  abs=4 * se[0] + 1e-3)  # fmt: skip
    for bad, match in (
        (dict(weave="through_thickness", binder_depth=2, **common), "binder_depth"),
        (dict(weave="layer_to_layer", binder_depth=5, **common), "from 2 to layers"),
        (dict(weave="orthogonal", binder_depth=1, **common), "from 2 to layers"),
        (dict(weave="orthogonal", stuffers=False, **common), "angle interlocks only"),
        (dict(weave="plain", deviation_in_plane=-0.1), "deviation_in_plane"),
        (dict(weave="plain", size_correlation=1.0), "size_correlation"),
        (dict(weave="plain", neighbour_correlation=1.0), "neighbour_correlation"),
        (dict(weave="plain", correlation_length=[1.0, 2.0, 3.0]), "correlation_length"),
        (dict(weave="plain", variation_covariance="matern"), "variation_covariance"),
    ):
        with pytest.raises(ValueError, match=match):
            WovenTextile(**bad)


def test_tows_of_the_model():
    """The tows as the composite cylinders of their fibres in the matrix, at the fibre
    fraction of the model."""
    model = WovenTextile(**dict(ORTHOGONAL, fibre_fraction_in_yarns=0.6))
    fibre = {"hexagonal": [18.08, 6.54, 4.97, 232.05, 15.0]}
    tow = model.tow_material(fibre, {"young": 3.5, "poisson": 0.35})
    expected = composite_cylinders(Material.from_config(fibre),
                                   Material(young=3.5, poisson=0.35), 0.6).stiffness  # fmt: skip
    assert np.allclose(tow.stiffness, expected) and tow.symmetry == "hexagonal"
    with pytest.raises(ValueError, match="fibre_fraction_in_yarns"):
        WovenTextile(**ORTHOGONAL).tow_material(fibre, {"young": 3.5, "poisson": 0.35})
