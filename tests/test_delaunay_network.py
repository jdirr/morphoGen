"""Fibre networks on the edges of Delaunay triangulations (Ban et al. 2016): the periodic
triangulation against the replicated points, the coordination and fibre lengths of the
Poisson-Delaunay graph in space and in the plane, hard-core points, the fibres as struts
(a solved fraction, the framework), the template and the interface."""

import itertools
import math

import numpy as np
import pytest
from scipy.spatial import Delaunay

import morphogen as mg
from morphogen import theory
from morphogen.models import DelaunayNetwork, Microstructure
from morphogen.models.foam import periodic_delaunay


def _replicated_edges(points, size, reach=1):
    """The edges at the points of the box of the triangulation of all their images,
    ``reach`` periods around along each axis (3^d images by default)."""
    n, dim = points.shape
    reach = np.broadcast_to(reach, dim)
    shifts = np.array(list(itertools.product(*(range(-k, k + 1) for k in reach))))
    shifts = shifts[np.argsort(np.abs(shifts).sum(axis=1), kind="stable")]
    allp = (points[None] + shifts[:, None] * size).reshape(-1, dim)
    indptr, nb = Delaunay(allp).vertex_neighbor_vertices
    out = set()
    for i in range(n):
        for j in nb[indptr[i] : indptr[i + 1]]:
            a, b, s = i, j % n, tuple(shifts[j // n])
            if (a, b) > (b, a) or (a == b and s < tuple(-np.array(s))):
                a, b, s = b, a, tuple(-np.array(s))
            out.add((a, b, *s))
    return out


@pytest.mark.parametrize("dim", [2, 3])
def test_periodic_triangulation(dim):
    rng = np.random.default_rng(dim)
    size = np.array([3.0, 4.0, 5.0][:dim])
    points = rng.random((150, dim)) * size
    pairs, shifts = periodic_delaunay(points, size)
    mine = {(int(a), int(b), *map(int, s)) for (a, b), s in zip(pairs, shifts, strict=True)}
    assert mine == _replicated_edges(points, size)
    if dim == 2:  # a triangulation of the torus: E = 3 V (Euler), six neighbours on average
        assert len(pairs) == 3 * len(points)
    with pytest.raises(ValueError, match="at least"):
        periodic_delaunay(points[:3], size)


def test_periodic_triangulation_of_thin_boxes():
    """A plate a mean spacing or two thick and a strip of the plane: their margins span
    several periods across, so do their images."""
    rng = np.random.default_rng(5)
    for size, n, reach in (((5.0, 5.0, 1.0), 200, (1, 1, 4)), ((10.0, 1.0), 30, (1, 6))):
        size = np.array(size)
        points = rng.random((n, len(size))) * size
        pairs, shifts = periodic_delaunay(points, size)
        mine = {(int(a), int(b), *map(int, s)) for (a, b), s in zip(pairs, shifts, strict=True)}
        assert mine == _replicated_edges(points, size, reach)
        assert np.abs(shifts).max() >= 1
    micro = DelaunayNetwork(n_points=200, radius=0.05).generate(mg.Domain(size=(5.0, 5.0, 1.0)), 1)
    assert micro.metadata["summary"]["nodes"] == 200


def _statistics(model, domain, seeds=8):
    coordination, length, cv = [], [], []
    for seed in range(seeds):
        frame = model.framework(domain, np.random.default_rng(seed))
        lengths = np.linalg.norm(frame.vectors(), axis=1)
        coordination.append(frame.coordination())
        length.append(lengths.mean() * model.theta(domain) ** (1.0 / model.dimension))
        cv.append(lengths.std() / lengths.mean())
        assert frame.degrees().min() >= model.dimension  # every node in the network
    return np.mean(coordination), np.mean(length), np.mean(cv)


def test_poisson_delaunay_graph():
    """The coordination of the Poisson-Delaunay graph, 2 + 48 pi^2 / 35 = 15.535 in space
    (measured 15.532 +- 0.007 over 8 x 1000 points) and 6 in the plane, and the mean
    fibre length 32 / (9 pi) lambda^(-1/2) in the plane (1.1332 +- 0.0009 for 1.1318); in
    space 1.286 lambda^(-1/3), more even for hard-core points (CV 0.28 against 0.33)."""
    space = mg.Domain.cube(10.0)
    z, length, cv = _statistics(DelaunayNetwork(n_points=1000, radius=0.05), space)
    assert z == pytest.approx(2.0 + 48.0 * math.pi**2 / 35.0, abs=0.04)
    assert z == pytest.approx(theory.VORONOI_MEAN_FACES, abs=0.04)
    assert length == pytest.approx(1.286, abs=0.006) and cv == pytest.approx(0.332, abs=0.01)
    hard = DelaunayNetwork(n_points=1000, radius=0.05, regularity=0.6)
    z_hard, _, cv_hard = _statistics(hard, space)
    assert z_hard < z - 0.05 and cv_hard < cv - 0.03
    plane = mg.Domain(size=(40.0, 40.0, 1.0))
    model = DelaunayNetwork(n_points=1600, radius=0.05, dimension=2)
    z, length, cv = _statistics(model, plane)
    th = model.theory(plane)
    assert z == th["coordination"] == 6.0
    assert length == pytest.approx(th["mean_fibre_length"], abs=0.005)
    assert th["fibre_length_per_area"] == pytest.approx(32.0 / (3.0 * math.pi))
    _, _, cv_hard = _statistics(DelaunayNetwork(n_points=1600, radius=0.05, dimension=2,
                                                regularity=0.6), plane)  # fmt: skip
    assert cv_hard < cv - 0.1
    assert "coordination" not in hard.theory(space)


def test_hardcore_points_keep_their_distance():
    model = DelaunayNetwork(n_points=300, radius=0.05, regularity=0.8)
    domain = mg.Domain.cube(6.0)
    frame = model.framework(domain, np.random.default_rng(1))
    least = 0.8 * (6.0 * 0.384 * domain.volume / (math.pi * 300)) ** (1.0 / 3.0)
    assert np.linalg.norm(frame.vectors(), axis=1).min() >= least * (1.0 - 1e-9)


def test_fibres_of_the_network():
    model = DelaunayNetwork(n_points=200, volume_fraction=0.1)
    micro = model.generate(mg.Domain.cube(5.0), seed=1)
    summary = micro.metadata["summary"]
    assert summary["nodes"] == 200 and summary["fibres"] == len(micro.segments)
    assert summary["coordination"] == pytest.approx(2.0 * summary["fibres"] / 200)
    assert micro.voxelize(voxel_size=0.025).data.mean() == pytest.approx(0.1, abs=0.005)
    frame = micro.metadata["framework"]
    assert len(frame["bars"]) == summary["fibres"] and len(frame["nodes"]) == 200
    again = Microstructure.from_dict(micro.to_dict())
    assert np.array_equal(again.segments.start, micro.segments.start)
    assert np.array_equal(model.generate(mg.Domain.cube(5.0), seed=1).segments.end,
                          micro.segments.end)  # fmt: skip
    # a planar network: the fibres in the middle plane of the domain, thick nodes
    planar = DelaunayNetwork(n_points=300, radius=0.1, node_ratio=1.5, dimension=2)
    micro = planar.generate(mg.Domain(size=(10.0, 10.0, 0.6)), seed=2)
    assert np.allclose(micro.segments.start[:, 2], 0.3) and np.allclose(micro.spheres.radii, 0.15)
    assert micro.metadata["summary"]["coordination"] == 6.0
    # a window: Poisson points of the intensity, the periodic network of the box cut by it
    window = DelaunayNetwork(intensity=2.0, radius=0.05).generate(
        mg.Domain.cube(4.0, periodic=False), seed=3
    )
    assert abs(window.metadata["summary"]["nodes"] - 128) < 5 * math.sqrt(128)
    # the fibres leaving the window by a face enter it again by the opposite one: the image
    # of the periodic box, and the radius of a fraction alike
    for spec in ({"radius": 0.08, "node_ratio": 1.5}, {"volume_fraction": 0.1}):
        model = DelaunayNetwork(intensity=2.0, **spec)
        cut = model.generate(mg.Domain.cube(4.0, periodic=False), seed=3)
        whole = model.generate(mg.Domain.cube(4.0, periodic=True), seed=3)
        assert np.array_equal(cut.voxelize(shape=48).data, whole.voxelize(shape=48).data)
        assert cut.metadata["summary"]["radius"] == whole.metadata["summary"]["radius"]
    for bad in ({"dimension": 4}, {"dimension": 2, "relaxation": 2}, {"n_points": 3},
                {"node_ratio": 0.5}, {"node_ratio": math.nan}, {"node_ratio": math.inf},
                {"regularity": 0.95}):  # fmt: skip
        with pytest.raises(ValueError):
            DelaunayNetwork(**{"n_points": 50, "radius": 0.1, **bad})


def test_template_and_interface():
    from morphogen import cli
    from morphogen.config import load_config
    from morphogen.gui import api

    cfg = load_config(__import__("yaml").safe_load(cli._templates()["delaunay_network"]))
    assert cfg.model.name == "delaunay_network"
    models = {m["name"]: m for m in api.models()}
    entry = models["delaunay_network"]
    assert entry["family"] == "Fibres and layers" and entry["example"]["n_points"] == 200
    names = [p["name"] for p in entry["parameters"]]
    assert {"n_points", "radius", "regularity", "dimension"} <= set(names)
    request = {"model": "delaunay_network", "params": {**entry["example"], "n_points": 40},
               "domain": {"size": 2.5, "periodic": True}, "seed": 1, "resolution": 24}  # fmt: skip
    out = api.generate(request)
    assert out["resolution"] == 24
