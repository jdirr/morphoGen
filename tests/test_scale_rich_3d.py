"""Scale-rich networks in three dimensions: the polyhedra of the plates, the growth of
plates and beams, their theory, and the 3D model (Both et al. 2026, Fig. 6a)."""

import json
import math

import numpy as np
import pytest

import morphogen as mg
from morphogen.models import Microstructure, ScaleRichBeams, ScaleRichNetwork, ScaleRichPlates
from morphogen.models.scale_rich import density_estimate
from morphogen.models.scale_rich_3d import (
    BOX,
    _point_segment,
    _ray_segment,
    area_sum,
    beam_density_estimate,
    box_polyhedron,
    clip_polyhedron,
    first_area,
    grow_beams,
    grow_plates,
    mean_inverse_distance,
    polygon_area_3d,
    polyhedron_volume,
    section,
)

CUBE = mg.Domain(size=(1.0, 1.0, 1.0), periodic=False)


def _volume(poly) -> float:
    return polyhedron_volume(poly[0], poly[1], poly[3], poly[4])


def _area_vectors(poly) -> np.ndarray:
    verts, fofs = poly[0], poly[1]
    out = []
    for f in range(len(fofs) - 1):
        v = verts[fofs[f] : fofs[f + 1]]
        out.append(0.5 * np.cross(v[1:-1] - v[0], v[2:] - v[0]).sum(axis=0))
    return np.array(out)


def test_polyhedra_clip_and_section():
    box = box_polyhedron([0, 0, 0], [1, 1, 1])
    assert _volume(box) == pytest.approx(1.0)
    # the faces turn counter-clockwise about their outward normals
    assert np.all(np.einsum("fi,fi->f", _area_vectors(box), box[3]) > 0)
    n = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    c = n @ [0.5, 0.5, 0.5]
    hexagon, labels = section(*box, *n, c, 1e-12)
    assert len(hexagon) == 6 and np.all(labels == BOX)
    assert polygon_area_3d(hexagon) == pytest.approx(3.0 * math.sqrt(3.0) / 4.0)
    # counter-clockwise about n
    assert np.cross(hexagon[1] - hexagon[0], hexagon[2] - hexagon[1]) @ n > 0
    lower = clip_polyhedron(*box, *n, c, 7, 1e-12)
    upper = clip_polyhedron(*box, *(-n), -c, 7, 1e-12)
    assert _volume(lower) == pytest.approx(0.5) and _volume(upper) == pytest.approx(0.5)
    assert lower[2][-1] == 7 and np.allclose(lower[3][-1], n) and lower[4][-1] == c
    assert np.all(np.einsum("fi,fi->f", _area_vectors(lower), lower[3]) > 0)
    # successive cuts of random planes keep the volume and convex, oriented faces
    rng = np.random.default_rng(0)
    polys = [box]
    for label in range(300):
        i = int(rng.integers(len(polys)))
        p = polys[i][0].mean(axis=0)
        m = rng.normal(size=3)
        m /= np.linalg.norm(m)
        polys.append(clip_polyhedron(*polys[i], *(-m), -(m @ p), label, 1e-12))
        polys[i] = clip_polyhedron(*polys[i], *m, m @ p, label, 1e-12)
    assert sum(_volume(p) for p in polys) == pytest.approx(1.0, abs=1e-12)
    for p in polys[::20]:
        assert np.all(np.einsum("fi,fi->f", _area_vectors(p), p[3]) > 0)
        centre = p[0].mean(axis=0)
        assert np.all(p[3] @ centre <= p[4] + 1e-12)  # convex: inside every face plane


def test_first_area():
    # Bailey, Borwein & Crandall (2007): the mean inverse distance in the unit cube
    exact = (2 / 5 - 2 * math.pi / 3 + 2 * math.sqrt(2) / 5 - 4 * math.sqrt(3) / 5
             + 2 * math.log(1 + math.sqrt(2)) + 12 * math.log((1 + math.sqrt(3)) / math.sqrt(2))
             - 4 * math.log(2 + math.sqrt(3)))  # fmt: skip
    assert mean_inverse_distance(1, 1, 1) == pytest.approx(exact, rel=1e-12)
    assert first_area(1, 1, 1) == pytest.approx(exact / 2, rel=1e-12)
    assert first_area(2, 2, 2) == pytest.approx(4 * first_area(1, 1, 1))
    assert first_area(2, 1, 1) == pytest.approx(first_area(1, 1, 2))
    # the section through a uniform point with a uniform normal (Monte Carlo)
    rng = np.random.default_rng(1)
    box = box_polyhedron([0, 0, 0], [2, 1, 1])
    areas = []
    for _ in range(20000):
        p = rng.random(3) * [2, 1, 1]
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        areas.append(polygon_area_3d(section(*box, *n, n @ p, 1e-12)[0]))
    error = np.std(areas) / math.sqrt(len(areas))
    assert abs(np.mean(areas) - first_area(2, 1, 1)) < 4 * error


def test_plates_partition_and_topology():
    g = grow_plates([0, 0, 0], [1, 1, 1], 300, lambda t: 0.03 * t**-0.5, 7)
    assert g.n == 300 and g.n_cells == 301 and not g.jammed
    bodies, cells = g.body_volumes(), g.cell_volumes()
    assert np.all(bodies > 0) and np.all(cells >= 0)
    assert bodies.sum() + cells.sum() == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(g.thickness, 0.03 * np.arange(1, 301) ** -0.5)
    # a thin plate is its rim times its thickness
    thin = grow_plates([0, 0, 0], [1, 1, 1], 300, lambda t: 1e-9, 7)
    np.testing.assert_allclose(thin.body_volumes(), 1e-9 * thin.areas, rtol=1e-5)
    # the rim of a plate lies in its mid-plane and ends on earlier plates or the box
    for t, rim in enumerate(g.rims()):
        assert len(rim) >= 3
        np.testing.assert_allclose(rim @ g.normal[t], g.normal[t] @ g.nucleation[t], atol=1e-12)
        faces = g.rim_faces[g.rim_offsets[t] : g.rim_offsets[t + 1]]
        assert np.all((faces == BOX) | ((faces >= 0) & (faces < t)))
    assert g.areas[0] == pytest.approx(polygon_area_3d(g.rims()[0]))
    links = g.links()
    assert np.all(links[:, 0] > links[:, 1]) and len(np.unique(links, axis=0)) == len(links)
    deg = g.degrees()
    assert deg.sum() == 2 * len(links) and deg.mean() > 5
    # the voids are bounded by the faces of the plates and of the box
    assert set(np.unique(g.cells.labels)) <= {BOX, *range(300)}


def test_plates_density_and_jamming():
    alpha, n = 0.3, 300
    s = area_sum(alpha, n, 1, 1, 1)
    for rho1 in (0.3, 1.5):
        lam0 = rho1 / s
        rho = [1.0 - grow_plates([0, 0, 0], [1, 1, 1], n, lambda t, x=lam0: x * t**-alpha,
                                 50 + k).cell_volumes().sum() for k in range(8)]  # fmt: skip
        assert np.mean(rho) == pytest.approx(density_estimate(rho1, 3), rel=0.06)
    assert density_estimate(3.0, 3) == 1.0 and density_estimate(1.0, 3) == pytest.approx(19 / 27)
    jam = grow_plates([0, 0, 0], [1, 1, 1], 300, lambda t: 1.0 / s * 5, 3, max_attempts=50)
    assert jam.jammed and jam.n < 300


def test_plates_tails():
    g = grow_plates([0, 0, 0], [1, 1, 1], 10000, lambda t: 1e-9 * t**-0.5, 11)
    tails = ScaleRichPlates(CUBE, g).statistics(min_tail=100)
    assert tails["thickness"].exponent == pytest.approx(3.0, abs=0.01)
    assert tails["area"].exponent == pytest.approx(2.5, abs=0.12)
    assert tails["degree"].exponent == pytest.approx(2.5, abs=0.15)


def test_plates_orientations():
    domain = mg.Domain(size=(1.0, 1.0, 0.5), periodic=False)
    # normals in the plane z: the plates are vertical, the network of the plane extruded
    model = ScaleRichNetwork(dimension=3, n_ligaments=100, initial_thickness=0.02,
                             orientation={"type": "planar", "normal": [0, 0, 1]})  # fmt: skip
    micro = model.generate(domain, seed=2)
    g = micro.growth
    assert np.allclose(g.normal[:, 2], 0.0)
    for rim in g.rims():
        assert rim[:, 2].min() == pytest.approx(0.0) and rim[:, 2].max() == pytest.approx(0.5)
    assert micro.geometric_anisotropy == pytest.approx(0.0, abs=1e-12)
    # aligned normals: parallel layers
    layers = ScaleRichNetwork(dimension=3, n_ligaments=30, initial_thickness=0.01,
                              orientation={"type": "aligned", "direction": [0, 0, 1]})  # fmt: skip
    assert np.allclose(np.abs(layers.generate(CUBE, seed=1).growth.normal[:, 2]), 1.0)
    # shells about a pole (plates tangential), then radial plates
    for angle, cosine in ((90, 1.0), (0, 0.0)):
        polar = ScaleRichNetwork(dimension=3, n_ligaments=60, initial_thickness=0.01,
                                 pole=[0.5, 0.5, 0.5], pole_angle=angle)  # fmt: skip
        g = polar.generate(CUBE, seed=4).growth
        r = 0.5 - g.nucleation
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        np.testing.assert_allclose(np.abs(np.einsum("ti,ti->t", r, g.normal)), cosine, atol=1e-9)
    spread = ScaleRichNetwork(dimension=3, n_ligaments=60, initial_thickness=0.01,
                              orientation={"type": "aligned", "direction": [0, 0, 1]},
                              angle_spread=10.0).generate(CUBE, seed=4).growth  # fmt: skip
    tilt = np.degrees(np.arccos(np.minimum(np.abs(spread.normal[:, 2]), 1.0)))
    assert 7 < np.sqrt(np.mean(tilt**2) / 2) < 13  # a normal law of 10 degrees per axis
    # a gradient of nucleation along z
    graded = ScaleRichNetwork(dimension=3, n_ligaments=200, initial_thickness=0.005,
                              nucleation_gradient=[0.1, 1.0], gradient_axis=2)  # fmt: skip
    z = graded.generate(CUBE, seed=5).growth.nucleation[:, 2]
    assert np.mean(z > 0.5) > 0.7


def _segment_distance(g, i, j) -> float:
    a, b = g.ends[i]
    length = np.linalg.norm(b - a)
    d = (b - a) / length if length > 0 else np.array([1.0, 0.0, 0.0])
    return _ray_segment(*a, *d, length, *g.ends[j, 0], *g.ends[j, 1])[1]


def test_beams_contacts():
    g = grow_beams([0, 0, 0], [1, 1, 1], 300, lambda t: 0.04 * t**-0.2, 9)
    assert g.n == 300
    np.testing.assert_allclose(np.linalg.norm(g.direction, axis=1), 1.0)
    contacts = {(int(i), int(j)) for i, j in g.links()}
    r = g.radii
    for i in range(g.n):
        for j in range(i):
            d = _segment_distance(g, i, j)
            # the axis of a later beam never enters an earlier one...
            assert d >= r[j] * (1 - 1e-9)
            # ...and the contacts recorded are those of the beams that touch
            assert (d <= r[i] + r[j] + 1e-9) == ((i, j) in contacts)
    # each end lies on the box or on the beam that stopped it
    for i in range(g.n):
        for side in (0, 1):
            j, end = g.neighbours[i, side], g.ends[i, side]
            if j == BOX:
                assert np.min(np.minimum(end, 1.0 - end)) == pytest.approx(0.0, abs=1e-12)
            else:
                assert j < i and (i, j) in contacts
                assert _point_segment(*end, *g.ends[j, 0], *g.ends[j, 1]) <= r[i] + r[j] + 1e-9
    assert np.all(g.degrees() == np.bincount(g.links().ravel(), minlength=g.n))
    # nucleation points clear the other beams
    for t in range(1, g.n):
        p = g.nucleation[t]
        gaps = [_point_segment(*p, *g.ends[j, 0], *g.ends[j, 1]) - r[j] for j in range(t)]
        assert min(gaps) >= r[t] * (1 - 1e-9)


def test_beams_mean_field():
    for alpha, lam0 in ((0.0, 0.03), (0.5, 0.1)):
        law = (lambda x, a: lambda t: x * t**-a)(lam0, alpha)
        rho = [grow_beams([0, 0, 0], [1, 1, 1], 300, law, 20 + k).volumes().sum()
               for k in range(6)]  # fmt: skip
        assert np.mean(rho) == pytest.approx(beam_density_estimate(lam0, alpha, 300, (1, 1, 1)),
                                             rel=0.15)  # fmt: skip


def test_model_3d():
    plates = ScaleRichNetwork(dimension=3, n_ligaments=200, decay=0.4, volume_fraction=0.3)
    micro = plates.generate(CUBE, seed=1)
    assert isinstance(micro, ScaleRichPlates)
    assert micro.volume_fraction == pytest.approx(0.3, abs=1.5e-3)  # within a jump
    summary = micro.metadata["summary"]
    assert summary["ligaments"] == 200 and summary["voids"] == 201
    assert 0 < summary["geometric_anisotropy"] <= 1
    image = micro.voxelize(shape=48)
    assert image.data.mean() == pytest.approx(0.3, abs=0.02)
    fine = micro.voxelize(shape=24, supersampling=3)
    assert fine.data.dtype == np.float32 and fine.data.mean() == pytest.approx(0.3, abs=0.01)
    theory = plates.theory(CUBE)
    assert theory["volume_fraction"] == pytest.approx(0.3)
    assert theory["first_area"] == pytest.approx(first_area(1, 1, 1))
    assert theory["area_exponent"] == theory["degree_exponent"] == 2.5
    assert theory["jamming_thickness"] > theory["endless_jamming_thickness"] > 0
    assert theory["initial_thickness"] == pytest.approx(summary["initial_thickness"], rel=0.1)
    # beams, in a frame
    frame = 1 / 32  # two voxels of the image below
    beams = ScaleRichNetwork(dimension=3, element="beams", n_ligaments=150, decay=0.3,
                             volume_fraction=0.3, frame=frame)  # fmt: skip
    micro = beams.generate(CUBE, seed=2)
    assert isinstance(micro, ScaleRichBeams)
    assert micro.volume_fraction == pytest.approx(0.3, abs=1.5e-3)
    assert micro.voxelize(shape=64, supersampling=2).data.mean() == pytest.approx(0.3, abs=0.01)
    ends = micro.growth.ends
    assert np.all(ends >= frame - 1e-12) and np.all(ends <= 1 - frame + 1e-12)
    assert micro.metadata["summary"]["boundary_ends"] > 0
    assert beams.theory(CUBE)["volume_fraction"] == pytest.approx(0.3)
    assert len(micro.segments()) == 150
    assert set(micro.statistics(min_tail=20)) == {"thickness", "length", "degree"}
    uniform = ScaleRichNetwork(dimension=3, n_ligaments=60, decay=0.0, initial_thickness=0.01)
    assert set(uniform.generate(CUBE, seed=1).statistics(min_tail=10)) == {"area", "degree"}
    # errors
    for bad, match in (
        ({"element": "beams"}, "3D only"),
        ({"dimension": 3, "angles": [0, 90]}, "2D only"),
        ({"orientation": "isotropic"}, "3D only"),
        ({"dimension": 3, "pole": [0.5, 0.5]}, "3 coordinates"),
        ({"dimension": 4}, "dimension"),
        ({"dimension": 3, "element": "struts"}, "plates or beams"),
        ({"dimension": 3, "gradient_axis": 3, "nucleation_gradient": [1, 0]}, "gradient_axis"),
    ):
        with pytest.raises(ValueError, match=match):
            ScaleRichNetwork(**{"n_ligaments": 10, "initial_thickness": 0.01, **bad})
    with pytest.raises(ValueError, match="beyond the network"):
        ScaleRichNetwork(dimension=3, n_ligaments=20, decay=0.0, volume_fraction=0.95,
                         max_attempts=20).generate(CUBE, seed=1)  # fmt: skip


@pytest.mark.parametrize("element", ["plates", "beams"])
def test_serialization_and_fields_3d(element):
    model = ScaleRichNetwork(dimension=3, element=element, n_ligaments=60, decay=0.3,
                             initial_thickness=0.06, frame=0.03)  # fmt: skip
    micro = model.generate(CUBE, seed=6)
    back = Microstructure.from_dict(json.loads(json.dumps(micro.to_dict())))
    assert type(back) is type(micro) and back.model == model
    a, b = micro.voxelize(shape=20), back.voxelize(shape=20)
    assert np.array_equal(a.data, b.data)
    assert np.array_equal(model.generate(CUBE, seed=6).voxelize(shape=20).data, a.data)
    points = a.grid().center_points()
    assert np.array_equal(micro.contains(points).reshape(a.shape), a.data == 1)
    level = micro.level_set(shape=20)
    assert np.array_equal(level.data < 0, a.data == 1)
    assert np.all(level.data[a.data == 0] > 0)
    assert micro.summary()["volume_fraction"] == pytest.approx(micro.volume_fraction)


def test_template_and_interface_3d():
    from morphogen import cli
    from morphogen.config import load_config
    from morphogen.gui import api

    cfg = load_config(__import__("yaml").safe_load(cli._templates()["scale_rich_plates"]))
    assert cfg.model.dimension == 3 and cfg.model.plates
    params = {k: v for k, v in cfg.model.to_config().items() if k != "type"}
    small = ScaleRichNetwork(**{**params, "n_ligaments": 40})
    micro = small.generate(cfg.domain, seed=cfg.seed)
    assert micro.volume_fraction == pytest.approx(0.3, abs=1.5e-3)
    entry = {m["name"]: m for m in api.models()}["scale_rich_network"]
    names = {p["name"] for p in entry["parameters"]}
    assert {"dimension", "element", "orientation"} <= names
    request = {"model": "scale_rich_network",
               "params": {**entry["example"], "n_ligaments": 40, "volume_fraction": 0.2,
                          "dimension": 3, "element": "beams"},
               "domain": {"size": [1.0, 1.0, 1.0], "periodic": False}, "seed": 1,
               "resolution": 24}  # fmt: skip
    out = api.generate(request)
    assert out["resolution"] == 24
