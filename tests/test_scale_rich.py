"""Scale-rich networks (Both et al. 2026): the convex polygons of the growth, the exact
partition of the box and the topology of the thin network (Euler: N + 1 voids of four
vertices on average), the ends of the ligaments on their neighbours, the laws of the
thicknesses (exact), of the first length (closed form against quadrature and Monte
Carlo), of the relative density (mean field) and of the jamming threshold, the tails of
the lengths and degrees, the orientations and gradients of nucleation, the frame, the
voxels, level set and point queries, the solved volume fraction, the template and the
interface."""

import itertools
import math

import numpy as np
import pytest
from scipy import integrate

import morphogen as mg
from morphogen.models import Microstructure, ScaleRichNetwork
from morphogen.models.scale_rich import (
    BOX,
    LENGTH_CONSTANT,
    chord,
    clip_polygon,
    density_estimate,
    first_length,
    fits,
    grow,
    inside_polygon,
    length_sum,
    polygon_area,
)

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
LABELS = np.array([-1, -2, -3, -4], dtype=np.int64)
PLANE = mg.Domain(size=(1.0, 1.0, 0.004), periodic=False)


def _thin(n, seed):
    """A network of thin ligaments (the T-tessellation of the growth) in the unit square."""
    return grow((0.0, 0.0), (1.0, 1.0), n, lambda t: 1e-9, seed)


def _phi(polygon, points):
    """``max_e (n_e . x - c_e)`` of a convex polygon: zero on its boundary."""
    p = np.atleast_2d(points)
    e = np.roll(polygon, -1, axis=0) - polygon
    n = np.column_stack([e[:, 1], -e[:, 0]]) / np.linalg.norm(e, axis=1)[:, None]
    return np.max(p @ n.T - np.sum(n * polygon, axis=1), axis=1)


def _distance(polygon, points):
    """The distance of points outside a polygon to it: to the nearest of its edges."""
    p = np.atleast_2d(points)[:, None, :]
    a, b = polygon[None], np.roll(polygon, -1, axis=0)[None]
    t = np.clip(np.sum((p - a) * (b - a), axis=2) / np.sum((b - a) ** 2, axis=2), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * (b - a)), axis=2).min(axis=1)


def test_convex_polygons():
    # a line across the square: the two parts, their areas and the faces of their edges
    below, faces = clip_polygon(SQUARE, LABELS, 0.0, 1.0, 0.25, 7, 1e-12)
    assert polygon_area(below) == pytest.approx(0.25)
    assert sorted(faces.tolist()) == [-4, -2, -1, 7]
    above, faces = clip_polygon(SQUARE, LABELS, 1.0, 1.0, 1.0, 7, 1e-12)  # x + y <= 1
    assert polygon_area(above) == pytest.approx(0.5) and len(above) == 3
    # through a vertex: no edge of zero length, the new face kept
    corner, faces = clip_polygon(SQUARE, LABELS, 1.0, -1.0, 0.0, 9, 1e-12)  # x <= y
    assert len(corner) == 3 and 9 in faces and polygon_area(corner) == pytest.approx(0.5)
    # the chord through a point along a direction, and the faces it leaves by
    s0, s1, f0, f1 = chord(SQUARE, LABELS, 0.25, 0.5, 1.0, 0.0)
    assert (s0, s1, f0, f1) == (pytest.approx(-0.25), pytest.approx(0.75), -4, -2)
    c = math.sqrt(0.5)
    s0, s1, _, _ = chord(SQUARE, LABELS, 0.5, 0.5, c, c)
    assert s1 - s0 == pytest.approx(math.sqrt(2.0))
    # a strip of half width h fits where both its sides lie in the polygon
    assert fits(SQUARE, 0.5, 0.2, 0.0, 1.0, 0.19)
    assert not fits(SQUARE, 0.5, 0.2, 0.0, 1.0, 0.21)
    assert inside_polygon(SQUARE, 0.3, 0.9) and not inside_polygon(SQUARE, 1.2, 0.5)


def test_first_length():
    """The mean chord through a uniform point of a rectangle along a uniform direction:
    the closed form against the integral of l^2 over the lines (offsets and angles), and
    the first ligaments of thin networks."""

    def by_quadrature(a, b):
        """``int l^2 dp dtheta / (pi a b)``: along each direction the chord is linear in
        the offset between the projections of the vertices (Simpson's rule exact there),
        the angles by Gauss-Kronrod between the kinks."""
        verts = np.array([[0.0, 0.0], [a, 0.0], [a, b], [0.0, b]])
        centre = 0.5 * np.array([a, b])

        def chord_length(p, n, d):
            s0, s1, _, _ = chord(verts, LABELS, *(centre + p * n), *d)
            return max(0.0, s1 - s0)

        def squared(theta):
            d = np.array([math.cos(theta), math.sin(theta)])
            n = np.array([-d[1], d[0]])
            cuts = np.unique((verts - centre) @ n)
            total = 0.0
            for lo, hi in itertools.pairwise(cuts):
                eps = 1e-13 * (hi - lo)
                ends = [chord_length(p, n, d) for p in (lo + eps, 0.5 * (lo + hi), hi - eps)]
                total += (hi - lo) / 6.0 * (ends[0] ** 2 + 4.0 * ends[1] ** 2 + ends[2] ** 2)
            return total

        kink = math.atan2(b, a)
        points = [0.0, kink, 0.5 * math.pi, math.pi - kink, math.pi]
        total = sum(integrate.quad(squared, lo, hi, epsabs=1e-13, epsrel=1e-12, limit=200)[0]
                    for lo, hi in itertools.pairwise(points))  # fmt: skip
        return total / (math.pi * a * b)

    assert first_length(1.0, 1.0) == pytest.approx(0.946402, abs=1e-6)  # Kellerer 1971
    for a, b in ((1.0, 1.0), (2.0, 1.0), (0.4, 3.0)):
        assert first_length(a, b) == pytest.approx(by_quadrature(a, b), rel=1e-7)
    assert first_length(2.0, 6.0) == pytest.approx(2.0 * first_length(1.0, 3.0))
    lengths = [_thin(1, seed).lengths[0] for seed in range(4000)]
    error = np.std(lengths) / math.sqrt(len(lengths))
    assert abs(np.mean(lengths) - first_length(1.0, 1.0)) < 3.5 * error


def test_partition_and_topology():
    """Ligaments and voids partition the box; the thin network is a T-tessellation: one
    void more than ligaments, four vertices per void on average (each end of a ligament a
    corner of the two voids beside it), every end on a face of its neighbour."""
    g = _thin(3000, 1)
    assert g.n == 3000 and g.n_cells == 3001 and not g.jammed
    assert g.cell_areas().sum() + g.body_areas().sum() == pytest.approx(1.0, abs=1e-12)
    assert g.body_areas().sum() == pytest.approx(1e-9 * g.lengths.sum(), rel=1e-4)
    counts = np.diff(g.cell_offsets)
    assert counts.mean() == pytest.approx(4.0, abs=1e-12) and counts.min() >= 3
    # thick ligaments: the areas of the bodies and voids still sum to the box, exactly
    thick = grow((0.0, 0.0), (2.0, 1.0), 800, lambda t: 0.05 * t**-0.4, 2)
    assert thick.body_areas().sum() + thick.cell_areas().sum() == pytest.approx(2.0, abs=1e-11)
    assert np.all(thick.body_areas() > 0) and np.all(thick.cell_areas() > 0)
    bodies = thick.bodies()
    for i in range(thick.n):  # each end on the boundary of the box or on its neighbour
        for end, j in zip(thick.ends[i], thick.neighbours[i], strict=True):
            if j == BOX:
                assert min(*end, 2.0 - end[0], 1.0 - end[1]) == pytest.approx(0.0, abs=1e-12)
            else:
                assert j < i and abs(_phi(bodies[j], end)[0]) < 1e-10
    # the degrees of the dual graph: two per pair of ligaments that meet
    links = thick.links()
    degrees = thick.degrees()
    assert degrees.sum() == 2 * len(links) == 2 * (2 * thick.n - np.sum(thick.neighbours == BOX))


def test_thickness_and_density():
    """The thicknesses lambda_0 t^-alpha exactly; the mean-field relative density within 3 %
    of the realizations (the transient of the first lengths, below c sqrt(A/t), and the
    second order of the voids)."""
    model = ScaleRichNetwork(n_ligaments=500, decay=0.5, initial_thickness=0.06)
    micro = model.generate(PLANE, seed=1)
    t = np.arange(1, 501)
    np.testing.assert_allclose(micro.thicknesses, 0.06 * t**-0.5, rtol=1e-15)
    assert model.theory()["thickness_exponent"] == 3.0
    for decay, thickness, n in ((0.5, 0.06, 500), (0.9, 0.06, 3000), (0.3, 0.02, 2000)):
        model = ScaleRichNetwork(n_ligaments=n, decay=decay, initial_thickness=thickness)
        rho = [model.generate(PLANE, seed=s).volume_fraction for s in range(10)]
        expected = model.theory(PLANE)["volume_fraction"]
        error = np.std(rho) / math.sqrt(len(rho))
        assert abs(np.mean(rho) - expected) < 0.03 * expected + 3.0 * error
    # the first order and its closure
    s = length_sum(0.5, 500, 1.0, 1.0)
    t = np.arange(2, 501)
    assert s == pytest.approx(first_length(1, 1) + LENGTH_CONSTANT * np.sum(t**-1.0))
    assert density_estimate(0.5) == pytest.approx(0.4375) and density_estimate(2.5) == 1.0


def test_jamming():
    """Below the thickness at which the mean field fills the box (rho_1 = 2) the network
    grows its ligaments, above it it jams near a density of 0.93 to 0.99."""
    for decay, n in ((0.5, 1000), (0.0, 400)):
        star = ScaleRichNetwork(n_ligaments=n, decay=decay, initial_thickness=1.0).theory(PLANE)
        star = star["jamming_thickness"]
        for factor, jammed in ((0.75, False), (1.25, True)):
            model = ScaleRichNetwork(n_ligaments=n, decay=decay, initial_thickness=factor * star)
            for seed in range(3):
                summary = model.generate(PLANE, seed=seed).metadata["summary"]
                assert bool(summary["jammed"]) is jammed
                if jammed:
                    assert summary["ligaments"] < n and 0.9 < summary["volume_fraction"] < 1.0
    # an endless growth (alpha > 1/2) jams above the thickness of zeta(alpha + 1/2)
    theory = ScaleRichNetwork(n_ligaments=10, decay=1.0, initial_thickness=0.1).theory(PLANE)
    endless = first_length(1, 1) + LENGTH_CONSTANT * (2.6123753486854883 - 1.0)
    assert theory["endless_jamming_thickness"] == pytest.approx(2.0 / endless)


def test_tails_of_lengths_and_degrees():
    """The lengths and degrees of 30,000 ligaments: power-law tails of exponent 3 (Both et
    al.: 3 and 2.9), the thicknesses exactly 1 + 1/alpha."""
    model = ScaleRichNetwork(n_ligaments=30000, decay=0.9, initial_thickness=0.1)
    stats = model.generate(PLANE, seed=1).statistics(min_tail=100)
    assert stats["thickness"].exponent == pytest.approx(1.0 + 1.0 / 0.9, abs=0.03)
    assert stats["length"].exponent == pytest.approx(3.0, abs=0.15)
    assert stats["degree"].exponent == pytest.approx(3.0, abs=0.2)
    assert stats["degree"].discrete and stats["degree"].x_min >= 2


def test_orientations_and_nucleation():
    # aligned: the angles of the ligaments among those given; one angle: parallel ligaments
    micro = ScaleRichNetwork(n_ligaments=300, initial_thickness=0.02,
                             angles=[0, 90]).generate(PLANE, seed=1)  # fmt: skip
    a = np.degrees(micro.growth.angle)
    assert np.all(np.isclose(a, 0.0) | np.isclose(a, 90.0) | np.isclose(a, 180.0))
    micro = ScaleRichNetwork(n_ligaments=200, initial_thickness=0.02,
                             angles=[30]).generate(PLANE, seed=1)  # fmt: skip
    summary = micro.metadata["summary"]
    assert summary["geometric_anisotropy"] < 1e-12
    assert summary["principal_angle"] == pytest.approx(30.0)
    # the spread of thin ligaments, which always fit (thick ones fit less often along the
    # narrow voids of an aligned network: their angles spread more)
    spread = ScaleRichNetwork(n_ligaments=1000, initial_thickness=1e-7, angles=[30],
                              angle_spread=10).generate(PLANE, seed=1)  # fmt: skip
    assert np.degrees(spread.growth.angle).std() == pytest.approx(10.0, rel=0.1)
    # polarized: the lines through the pole (radial), or across its direction
    pole = np.array([0.5, 0.4])
    for turn in (0.0, 90.0):
        model = ScaleRichNetwork(n_ligaments=100, initial_thickness=0.01, pole=list(pole),
                                 pole_angle=turn, polarized_fraction=0.3)  # fmt: skip
        g = model.generate(PLANE, seed=2).growth
        d = np.column_stack([np.cos(g.angle), np.sin(g.angle)])
        n = np.column_stack([-d[:, 1], d[:, 0]])
        r = pole - g.nucleation
        off = np.abs(np.sum((n if turn == 0 else d) * r, axis=1))
        assert np.all(off[:30] < 1e-9) and np.median(off[30:]) > 1e-3
    # the nucleation densities: uniform, linear along x (mean 2/3 of a density 2x), radial
    for gradient, expected in ((None, 0.5), ([0.0, 1.0], 2.0 / 3.0)):
        model = ScaleRichNetwork(n_ligaments=4000, initial_thickness=1e-9,
                                 nucleation_gradient=gradient)  # fmt: skip
        x = model.generate(PLANE, seed=3).growth.nucleation[:, 0]
        sd = math.sqrt(1.0 / 12.0 if gradient is None else 1.0 / 18.0)
        assert abs(x.mean() - expected) < 3.5 * sd / math.sqrt(len(x))
    model = ScaleRichNetwork(n_ligaments=4000, initial_thickness=1e-9, pole=[0.5, 0.5],
                             polarized_fraction=0.0, nucleation_gradient=[0.0, 1.0],
                             radial_gradient=True)  # fmt: skip
    r = np.linalg.norm(model.generate(PLANE, seed=4).growth.nucleation - 0.5, axis=1)
    u = (np.arange(400) + 0.5) / 400  # the moments of r for a density r, by midpoints
    w = np.hypot(*np.meshgrid(u - 0.5, u - 0.5))
    m1, m2 = np.sum(w**2) / np.sum(w), np.sum(w**3) / np.sum(w)
    assert abs(r.mean() - m1) < 3.5 * math.sqrt(m2 - m1**2) / math.sqrt(len(r))


def test_structure_voxels_and_frame():
    model = ScaleRichNetwork(n_ligaments=400, decay=0.6, initial_thickness=0.08)
    domain = mg.Domain(size=(2.0, 1.0, 0.01), periodic=False)
    micro = model.generate(domain, seed=5)
    img = micro.voxelize(voxel_size=(0.004, 0.004, 0.005))
    assert img.shape == (500, 250, 2) and np.array_equal(img.data[:, :, 0], img.data[:, :, 1])
    assert img.data.mean() == pytest.approx(micro.volume_fraction, abs=0.01)
    inside = micro.contains(img.grid().center_points()).reshape(img.shape)
    assert np.array_equal(inside, img.data == 1)
    fine = micro.voxelize(voxel_size=0.01, supersampling=5)
    assert fine.data.dtype == np.float32
    assert fine.data.mean() == pytest.approx(micro.volume_fraction, abs=0.005)
    ls = micro.level_set(voxel_size=(0.004, 0.004, 0.005))
    assert np.array_equal(ls.data < 0, img.data == 1)
    # in the voids, the level set is the distance to the ligaments near them, or less
    # (their corners); the distance exactly where the nearest point lies on a side
    plane = img.grid().center_points()[: 500 * 250 * 2 : 2, :2]  # the first layer
    field = ls.data[:, :, 0].ravel()
    void = np.flatnonzero(field > 0)[::97]
    exact = np.min([_distance(b, plane[void]) for b in micro.growth.bodies()], axis=0)
    reach = 0.5 * micro.thicknesses.max() + 4 * 0.005
    near = exact < reach - 1e-6
    assert np.all(field[void][near] <= exact[near] + 1e-6)
    assert np.all(field[void][~near] >= reach - 1e-6)
    assert np.mean(np.isclose(field[void][near], exact[near], atol=1e-6)) > 0.8
    # the explicit description round trips, the realization repeats with its seed
    back = Microstructure.from_dict(micro.to_dict())
    assert back.model == model and np.array_equal(back.voxelize(voxel_size=0.02).data,
                                                  micro.voxelize(voxel_size=0.02).data)  # fmt: skip
    again = model.generate(domain, seed=5)
    assert np.array_equal(again.growth.ends, micro.growth.ends)
    # prisms along x: the network in the y-z plane
    micro = ScaleRichNetwork(n_ligaments=50, initial_thickness=0.1, axis=0).generate(
        mg.Domain(size=(0.02, 1.0, 1.0), periodic=False), seed=1
    )
    img = micro.voxelize(voxel_size=0.01)
    assert img.shape == (2, 100, 100) and np.array_equal(img.data[0], img.data[1])
    # a frame: phase 1 along the sides, the ligaments ending on it, its share in the density
    framed = ScaleRichNetwork(n_ligaments=300, decay=0.5, initial_thickness=0.05, frame=0.05)
    micro = framed.generate(PLANE, seed=1)
    img = micro.voxelize(voxel_size=(0.005, 0.005, 0.004))
    assert img.data[:10].all() and img.data[-10:].all() and img.data[:, :10].all()
    assert micro.growth.cell_vertices.min() >= 0.05 - 1e-12
    assert micro.growth.cell_vertices.max() <= 0.95 + 1e-12
    voids = micro.growth.cell_areas().sum()
    assert micro.volume_fraction == pytest.approx(1.0 - voids)
    assert img.data.mean() == pytest.approx(micro.volume_fraction, abs=0.01)
    theory = framed.theory(PLANE)
    inner = ScaleRichNetwork(n_ligaments=300, decay=0.5, initial_thickness=0.05).theory(
        mg.Domain(size=(0.9, 0.9, 0.004), periodic=False)
    )
    assert theory["volume_fraction"] == pytest.approx(0.19 + 0.81 * inner["volume_fraction"])
    assert micro.metadata["summary"]["boundary_ends"] > 0
    with pytest.raises(ValueError, match="frame"):
        ScaleRichNetwork(n_ligaments=5, initial_thickness=0.1, frame=0.6).generate(PLANE, 1)
    with pytest.raises(ValueError, match="cannot generate periodic"):
        model.generate(mg.Domain.cube(1.0, periodic=True), seed=1)


def test_solved_volume_fraction():
    """The initial thickness of each realization solved for its relative density, from the
    estimate of the theory, with or without a frame."""
    for spec in ({"decay": 0.5}, {"decay": 0.0}, {"decay": 0.9, "frame": 0.01},
                 {"decay": 0.5, "angles": [0, 60, 120]}):  # fmt: skip
        model = ScaleRichNetwork(n_ligaments=500, volume_fraction=0.4, **spec)
        for seed in range(3):
            micro = model.generate(PLANE, seed=seed)
            assert micro.volume_fraction == pytest.approx(0.4, abs=1.5e-3)
            assert micro.metadata["summary"]["volume_fraction"] == micro.volume_fraction
    model = ScaleRichNetwork(n_ligaments=500, decay=0.5, volume_fraction=0.4)
    estimate = model.theory(PLANE)["initial_thickness"]
    solved = model.generate(PLANE, seed=1).metadata["summary"]["initial_thickness"]
    assert solved == pytest.approx(estimate, rel=0.15)
    with pytest.raises(ValueError, match="jams"):
        ScaleRichNetwork(n_ligaments=500, decay=0.0, volume_fraction=0.99).generate(PLANE, 1)


def test_parameters():
    for bad in ({"n_ligaments": 0}, {"decay": -0.1}, {"initial_thickness": None},
                {"volume_fraction": 0.3}, {"min_spacing": -1.0}, {"angles": []},
                {"angle_spread": -2.0}, {"pole": [0.5]}, {"polarized_fraction": 0.5},
                {"pole": [0.5, 0.5], "polarized_fraction": 1.5}, {"axis": 3},
                {"gradient_axis": 2}, {"nucleation_gradient": [0, 0]},
                {"nucleation_gradient": [1, -1]}, {"radial_gradient": True},
                {"max_attempts": 0}, {"frame": -0.1}):  # fmt: skip
        with pytest.raises(ValueError):
            ScaleRichNetwork(**{"initial_thickness": 0.1, **bad})
    model = mg.models.model_from_config(
        {"type": "scale_rich_network", "n_ligaments": 20, "volume_fraction": 0.3, "angles": [0]}
    )
    assert model.angles == (0.0,) and not model.isotropic
    assert set(model.theory(PLANE)) == {"length_exponent", "degree_exponent", "thickness_exponent"}


def test_template_and_interface():
    from morphogen import cli
    from morphogen.config import load_config
    from morphogen.gui import api

    cfg = load_config(__import__("yaml").safe_load(cli._templates()["scale_rich_network"]))
    assert cfg.model.name == "scale_rich_network"
    models = {m["name"]: m for m in api.models()}
    entry = models["scale_rich_network"]
    assert entry["family"] == "Lattices and TPMS"
    names = [p["name"] for p in entry["parameters"]]
    assert {"n_ligaments", "decay", "initial_thickness", "volume_fraction", "frame"} <= set(names)
    request = {"model": "scale_rich_network",
               "params": {**entry["example"], "n_ligaments": 60, "volume_fraction": 0.3},
               "domain": {"size": [1.0, 1.0, 0.02], "periodic": False}, "seed": 1,
               "resolution": 32}  # fmt: skip
    out = api.generate(request)
    assert out["resolution"] == 32
