"""Nested models: Cox spheres, coatings and hollow members, hierarchical lattices of
tubes, fractal sponges and nested combinations."""

import dataclasses
import itertools
import math

import numpy as np
import pytest

import morphogen as mg
from morphogen.analysis.fractal import box_counting
from morphogen.models.hierarchical import FCC_NODES, frame

CUBE = mg.Domain.cube(1.0, periodic=True)


def test_cox_spheres():
    """Germs in the aggregates only, as many as a Poisson process of their intensity in
    the aggregates realized; the fraction of a Cox Boolean model of small spheres."""
    aggregates = {"type": "boolean_spheres", "radius": 0.15, "volume_fraction": 0.3}
    model = mg.models.CoxSpheres(aggregates=aggregates, radius=0.01, local_fraction=0.25)
    domain = mg.Domain.cube(1.5, periodic=True)
    micro = model.generate(domain, seed=4)
    centers = micro.spheres.centers
    assert micro.aggregates.contains(centers).all()
    grid = micro.grid((150,) * 3)
    inside = micro.aggregates.voxelize(grid=grid).data == 1
    p = inside.mean()
    theta_in, theta_out = model.intensities
    assert theta_out == 0.0
    expected = theta_in * p * domain.volume
    assert abs(len(centers) - expected) < 4.0 * math.sqrt(expected)
    q = math.exp(-theta_in * model.mean_volume)
    measured = (micro.voxelize(grid=grid).data == 1).mean()
    assert measured == pytest.approx(p * (1.0 - q), rel=0.05)
    t = model.theory()
    assert t["aggregate_fraction"] == pytest.approx(0.3) and t["volume_fraction"] == pytest.approx(
        0.3 * 0.25
    )
    # the aggregates as a phase of their own, around the spheres
    model = mg.models.CoxSpheres(
        aggregates=aggregates, radius=0.02, local_fraction=0.2, aggregate_phase=3
    )
    assert not model.has_level_set and model.active_phases()[3].startswith("aggregates")
    micro = model.generate(CUBE, seed=1)
    img = micro.voxelize((64,) * 3).data
    agg = micro.aggregates.voxelize((64,) * 3).data == 1
    assert set(np.unique(img)) == {0, 1, 3} and np.all(agg[img == 3]) and not agg[img == 0].any()
    back = mg.models.Microstructure.from_dict(micro.to_dict())
    assert np.array_equal(back.voxelize((32,) * 3).data, micro.voxelize((32,) * 3).data)


def test_hardcore_and_nested_cox_spheres():
    """Hard-core spheres among the germs: no overlap, the local fraction in the
    aggregates; Cox spheres in Cox spheres (three scales)."""
    model = mg.models.CoxSpheres(
        aggregates={"type": "hardcore_spheres", "radius": 0.2, "volume_fraction": 0.25},
        radius=0.03,
        local_fraction=0.2,
        hardcore=True,
        min_gap=0.002,
    )
    micro = model.generate(CUBE, seed=2)
    s = micro.spheres
    assert micro.aggregates.contains(s.centers).all()
    d = s.centers[:, None, :] - s.centers[None, :, :]
    d -= np.rint(d)  # periodic
    dist = np.linalg.norm(d, axis=-1) + np.eye(len(s.centers)) * 10
    assert np.all(dist >= s.radii[:, None] + s.radii[None, :] + 0.002 - 1e-12)
    agg = micro.aggregates.voxelize((96,) * 3).data == 1
    volume = 4.0 / 3.0 * math.pi * np.sum(s.radii**3)
    assert volume / (agg.mean() * CUBE.volume) == pytest.approx(0.2, rel=0.05)
    with pytest.raises(ValueError, match="local_fraction"):
        mg.models.CoxSpheres(aggregates=model.aggregates, intensity=10.0, hardcore=True)
    deep = mg.models.CoxSpheres(
        aggregates={
            "type": "cox_spheres",
            "aggregates": {"type": "boolean_spheres", "radius": 0.25, "volume_fraction": 0.3},
            "radius": 0.06,
            "local_fraction": 0.5,
        },
        radius=0.01,
        local_fraction=0.3,
    )
    micro = deep.generate(CUBE, seed=1)
    assert micro.aggregates.aggregates.contains(micro.aggregates.spheres.centers).all()
    assert micro.aggregates.contains(micro.spheres.centers).all()
    with pytest.raises(ValueError, match="exactly one"):
        mg.models.CoxSpheres(aggregates=deep.aggregates)


def test_coatings_and_shells():
    """Coating of Boolean spheres: the dilated Boolean model; hollow spheres: exact
    shells; shells of other structures from the distance transform."""
    base = mg.models.BooleanSpheres(radius=0.05, volume_fraction=0.2)
    model = mg.models.Coating(model=base, thickness=0.015)
    micro = model.generate(CUBE, seed=3)
    img = micro.voxelize((128,) * 3).data
    t = model.theory()
    assert (img == 2).mean() == pytest.approx(t["coating_fraction"], rel=0.06)
    assert np.array_equal(img == 1, micro.inner.voxelize((128,) * 3).data == 1)
    assert not model.has_level_set and model.active_phases()[2] == "coating"
    hollow = mg.models.Coating(
        model={"type": "hardcore_spheres", "radius": 0.1, "volume_fraction": 0.15},
        thickness=0.03,
        side="inside",
    )
    assert hollow.has_level_set
    micro = hollow.generate(CUBE, seed=1)
    img = micro.voxelize((96,) * 3)
    n = len(micro.inner.spheres)
    exact = n * 4.0 / 3.0 * math.pi * (0.1**3 - 0.07**3)
    assert img.data.mean() == pytest.approx(exact, rel=0.03)
    points = img.grid().center_points()
    assert np.array_equal(micro.contains(points).reshape(img.shape), img.data == 1)
    ls = micro.level_set(grid=img.grid()).data
    assert np.mean((ls < 0) == (img.data == 1)) > 0.99
    filled = mg.models.Coating(model=hollow.model, thickness=0.03, side="inside", core_phase=2)
    img = filled.generate(CUBE, seed=1).voxelize((48,) * 3).data
    assert set(np.unique(img)) == {0, 1, 2}
    gyroid = mg.models.TPMS(surface="gyroid", cell_size=1.0, volume_fraction=0.5)
    shell = mg.models.Coating(model=gyroid, thickness=0.04, side="inside")
    fraction = shell.generate(CUBE, seed=0).voxelize((96,) * 3).data.mean()
    surface = mg.analysis.measure(gyroid.generate(CUBE).voxelize((96,) * 3), ["specific_surface"])
    assert fraction == pytest.approx(surface["specific_surface"] * 0.04, rel=0.15)
    for bad, match in (
        ({"thickness": 0.0}, "thickness"),
        ({"side": "around"}, "side"),
        ({"phase": 1}, "phase"),
        ({"core_phase": 1}, "core_phase"),
    ):
        with pytest.raises(ValueError, match=match):
            mg.models.Coating(**{"model": base, "thickness": 0.01, **bad})


def test_hollow_strut_lattice():
    model = mg.models.StrutLattice(cell="octet", cell_size=0.5, strut_radius=0.05,
                                   wall_thickness=0.015)  # fmt: skip
    micro = model.generate(CUBE, seed=0)
    # walls of two voxels alias on a grid aligned with the struts: sample points instead
    points = np.random.default_rng(0).random((200_000, 3))
    fraction = micro.contains(points).mean()
    expected = model.theory()["volume_fraction"]
    assert fraction == pytest.approx(expected, abs=4.0 * math.sqrt(expected / len(points)))
    img = micro.voxelize((64,) * 3).data
    solid = mg.models.StrutLattice(cell="octet", cell_size=0.5, strut_radius=0.05)
    assert np.all(solid.generate(CUBE).voxelize((64,) * 3).data[img == 1] == 1)
    assert img.mean() < 0.6 * solid.theory()["volume_fraction"]
    # hollow struts: the fraction peaks once the bores fill the cell
    cubic = mg.models.StrutLattice(
        cell="cubic", cell_size=0.5, volume_fraction=0.04, wall_thickness=0.01
    )
    r, t = cubic.radius / 0.5, 0.02  # exact for cubic cells
    exact = 3 * math.pi * (r * r - (r - t) ** 2) - 8 * math.sqrt(2) * (r**3 - (r - t) ** 3)
    assert exact == pytest.approx(0.04, rel=1e-4)
    with pytest.raises(ValueError, match=r"at most 0\.05"):
        _ = dataclasses.replace(cubic, volume_fraction=0.1).radius
    octet = mg.models.StrutLattice(cell="octet", cell_size=0.5, volume_fraction=0.08,
                                   wall_thickness=0.02)  # fmt: skip
    assert octet.theory()["volume_fraction"] == pytest.approx(0.08, rel=1e-3)
    fraction = octet.generate(CUBE, seed=0).contains(points).mean()
    assert fraction == pytest.approx(0.08, rel=0.03)
    with pytest.raises(ValueError, match="wall_thickness"):
        mg.models.StrutLattice(strut_radius=0.05, wall_thickness=0.06)


def test_frames_of_hierarchical_lattices():
    """Frames of generation 1: columns of regular octahedra (all edges equal) between the
    nodes; generation 2: the compressed edges replaced by frames."""
    p, q = np.zeros(3), np.array([0.0, 0.0, 1.0])
    for n in (1, 2, 4):
        tubes, owners = frame(p, q, 1, (n,))
        lengths = np.array([np.linalg.norm(b - a) for a, b in tubes])
        edge = 1.0 / (math.sqrt(2.0 / 3.0) * (n + 2))
        assert len(tubes) == 9 * (n + 1) and np.allclose(lengths, edge)
        ends = np.array([e for t in tubes for e in t])
        assert np.allclose(ends.min(axis=0)[2], 0.0) and np.allclose(ends.max(axis=0)[2], 1.0)
        assert owners.count(1) == 3 * (n + 1)  # tension: the triangles
    tubes, owners = frame(p, q, 2, (2, 3))
    assert len(tubes) == 9 + 18 * 36 and owners.count(2) == 9 and owners.count(0) == 18 * 24
    model = mg.models.HierarchicalLattice(generations=2, cells=(2, 2), tube_radius=0.0004)
    unit = model.unit_tubes()
    bonds = 4 * (6 + 3)
    assert len(unit) == bonds * (9 + 18 * 27)
    # the edges of the frames on the nearest (sqrt 1/2) and next-nearest (1) bonds
    shrink = math.sqrt(2.0 / 3.0) * 4
    expected = np.array([b / shrink**k for b in (math.sqrt(0.5), 1.0) for k in (1, 2)])
    assert np.allclose(np.abs(unit.lengths[:, None] - expected).min(axis=1), 0.0, atol=1e-9)
    assert np.allclose(model.edges, [math.sqrt(0.5), *expected[:2]])
    assert model.edges[-1] == pytest.approx(math.sqrt(0.5) / (math.sqrt(2.0 / 3.0) * 4) ** 2)
    radii = mg.models.HierarchicalLattice(generations=1, cells=2, tube_radius=(0.01, 0.004)).radii
    assert radii == (0.01, 0.004)
    with pytest.raises(ValueError, match="blobs"):
        mg.models.HierarchicalLattice(generations=2, cells=4, tube_radius=0.01)
    with pytest.raises(ValueError, match="one per generation"):
        mg.models.HierarchicalLattice(generations=2, cells=(2, 2, 2))


def test_hierarchical_lattices():
    """Generation 0 is the fcc lattice of nearest and next-nearest neighbours; periodic
    tilings repeat the cell; hollow tubes keep less material."""
    plain = mg.models.HierarchicalLattice(generations=0, tube_radius=0.03)
    start, end = plain.bonds()
    assert len(start) == 4 * 9
    lengths = np.linalg.norm(end - start, axis=1)
    assert np.isclose(lengths, math.sqrt(0.5)).sum() == 24 and np.isclose(lengths, 1).sum() == 12
    assert np.allclose(start[::9], FCC_NODES)
    model = mg.models.HierarchicalLattice(generations=1, cells=2, tube_radius=0.008)
    one = model.generate(CUBE, seed=0).voxelize((64,) * 3).data
    two = model.generate(mg.Domain.cube(2.0, periodic=True)).voxelize((128,) * 3).data
    assert np.array_equal(two[:64, 64:, :64], one)
    hollow = mg.models.HierarchicalLattice(
        generations=1, cells=2, tube_radius=0.008, wall_thickness=0.003
    )
    frac = hollow.generate(CUBE).voxelize((128,) * 3).data.mean()
    solid = model.generate(CUBE).voxelize((128,) * 3).data.mean()
    assert 0.3 * solid < frac < 0.8 * solid
    t = model.theory()
    assert t["tubes_per_cell"] == 36 * 27 and 0 < t["volume_fraction"] < 0.05


def test_fractal_sponges():
    menger = mg.models.FractalSponge(level=3)
    img = menger.generate(CUBE).voxelize((81,) * 3)
    t = menger.theory()
    assert img.data.mean() == pytest.approx((20 / 27) ** 3, abs=1e-12) == t["volume_fraction"]
    dim = box_counting(img, sizes=[3, 9, 27])["dimension"]
    assert dim == pytest.approx(math.log(20) / math.log(3), abs=1e-9) == t["fractal_dimension"]
    for pattern, kept in (("vicsek", 7), ("cantor_dust", 8)):
        m = mg.models.FractalSponge(pattern=pattern, level=2)
        assert m.generate(CUBE).voxelize((27,) * 3).data.mean() == pytest.approx((kept / 27) ** 2)
    custom = mg.models.FractalSponge(subdivisions=2, kept=[[0, 0, 0], [1, 1, 1]], level=4)
    data = custom.generate(mg.Domain.cube(2.0, periodic=True)).voxelize((32,) * 3).data
    assert data.mean() == pytest.approx((2 / 8) ** 4)
    assert np.array_equal(data[:16, :16, :16], data[16:, 16:, 16:])
    frac = custom.generate(CUBE).voxelize((8,) * 3, supersampling=2).data
    assert frac.mean() == pytest.approx((2 / 8) ** 4)
    for bad, match in (
        ({"pattern": "koch"}, "pattern"),
        ({"subdivisions": 4}, "3 x 3 x 3"),
        ({"kept": [[3, 0, 0]]}, "kept"),
        ({"level": 0}, "level"),
    ):
        with pytest.raises(ValueError, match=match):
            mg.models.FractalSponge(**bad)
    with pytest.raises(ValueError, match="whole cubes"):
        menger.generate(mg.Domain.cube(1.5, periodic=True))


def test_nested_combination():
    """Each model fills the phase 1 of the previous one, keeping its other phases."""
    model = mg.models.Combination(
        operation="nested",
        models=[
            {"type": "tpms", "surface": "gyroid", "cell_size": 1.0, "volume_fraction": 0.5},
            {
                "type": "cox_spheres",
                "aggregates": {"type": "boolean_spheres", "radius": 0.1, "volume_fraction": 0.3},
                "radius": 0.02,
                "local_fraction": 0.3,
                "aggregate_phase": 2,
            },
        ],
    )
    assert model.active_phases()[2].startswith("aggregates")
    micro = model.generate(CUBE, seed=5)
    img = micro.voxelize((64,) * 3).data
    host = micro.parts[0].voxelize((64,) * 3).data == 1
    assert set(np.unique(img)) == {0, 1, 2} and not (img[~host] != 0).any()
    points = micro.grid((64,) * 3).center_points()
    assert np.array_equal(micro.contains(points).reshape(img.shape), img == 1)
    lattice = mg.models.Combination(
        operation="nested",
        models=[
            {"type": "tpms", "surface": "gyroid", "cell_size": 1.0, "volume_fraction": 0.5},
            {"type": "strut_lattice", "cell": "cubic", "cell_size": 0.125, "strut_radius": 0.02},
        ],
    )
    t = lattice.theory()
    sub = mg.models.StrutLattice(cell="cubic", cell_size=0.125, strut_radius=0.02).theory()
    assert t["volume_fraction"] == pytest.approx(0.5 * sub["volume_fraction"])
    micro = lattice.generate(CUBE)
    ls = micro.level_set((48,) * 3).data
    assert np.mean((ls < 0) == (micro.voxelize((48,) * 3).data == 1)) > 0.97
    for a, b in itertools.pairwise(micro.parts):
        assert a.domain == b.domain
