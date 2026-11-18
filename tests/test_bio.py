"""Bio-inspired models: nacre, Bouligand plywoods, honeycombs and wood, Haversian bone,
open-cell foams and trabecular bone, branching networks grown by space colonization."""

import math

import numpy as np
import pytest

import morphogen as mg
from morphogen.homogenization import VOID, Material, homogenize
from morphogen.models.bouligand import periodic_families
from morphogen.models.branching import murray_radii
from morphogen.models.foam import periodic_edges, stretch_factor
from morphogen.models.honeycomb import PrismaticCells
from morphogen.models.osteons import CANAL, CEMENT, INTERSTITIAL, LAMELLAR, OsteonStructure
from morphogen.models.planar import PlanarCells, hardcore_germs, jamming_distance


def monte_carlo(micro, n=200_000, seed=0):
    """Fraction of phase 1 at random points of the domain, and its standard error."""
    d = micro.domain
    pts = d.lower + np.random.default_rng(seed).random((n, 3)) * d.lengths
    p = float(micro.contains(pts).mean())
    return p, math.sqrt(max(p * (1 - p), 1e-12) / n)


def test_planar_cells_and_hardcore_germs():
    """Cells and distances to their sides against brute force, periodic or not."""
    rng = np.random.default_rng(1)
    size = np.array([3.0, 2.0])
    germs = rng.random((40, 2)) * size
    pts = rng.random((500, 2)) * size
    for periodic in (True, False):
        label, dist = PlanarCells(germs, size, periodic).locate(pts)
        v = germs[None, :, :] - pts[:, None, :]
        if periodic:
            v -= size * np.round(v / size)
        d2 = (v**2).sum(axis=2)
        assert np.array_equal(label, d2.argmin(axis=1))
        own = v[np.arange(len(pts)), label]
        gap = np.linalg.norm(v - own[:, None, :], axis=2)
        side = (d2 - d2.min(axis=1, keepdims=True)) / (2.0 * np.where(gap > 0, gap, np.inf))
        side[np.arange(len(pts)), label] = np.inf
        assert np.allclose(dist, side.min(axis=1))
    # within reach only: beyond, at least the reach
    _, near = PlanarCells(germs, size, True).locate(pts, reach=0.05)
    _, exact = PlanarCells(germs, size, True).locate(pts)
    assert np.allclose(near[exact < 0.05], exact[exact < 0.05])
    assert np.all(near[exact >= 0.05] >= 0.05 - 1e-12)
    d = 0.8 * jamming_distance(200, 6.0)
    g = hardcore_germs(200, size, d, rng)
    v = g[:, None, :] - g[None, :, :]
    v -= size * np.round(v / size)
    dist = np.linalg.norm(v, axis=2) + np.eye(200) * 10
    assert dist.min() >= d * (1 - 1e-9) and np.all((g >= 0) & (g < size))


def test_nacre_bricks():
    """Staggered bricks: the exact volume fraction, the shift of the layers, the level
    set, and the two-dimensional brick-and-mortar model in a box one tablet wide."""
    model = mg.models.Nacre(
        arrangement="brick", tablet_size=1.0, tablet_thickness=0.25, mortar_thickness=0.05
    )
    domain = mg.Domain(size=(4.0, 4.0, 1.2), periodic=True)
    micro = model.generate(domain, seed=2)
    assert micro.n_layers == 4 and micro.metadata["scale"]["thickness"] == pytest.approx(1.0)
    exact = 0.25 / 0.3 * 0.95**2
    assert model.theory()["volume_fraction"] == pytest.approx(exact)
    p, se = monte_carlo(micro)
    assert p == pytest.approx(exact, abs=4 * se)
    shift = (micro.layers[1] - micro.layers[0]) % 1.0
    assert np.allclose(shift, 0.5)  # each tablet over the corners of four
    img = micro.voxelize((64, 64, 48))
    pts = img.grid().center_points()
    assert np.array_equal(micro.contains(pts).reshape(img.shape), img.data == 1)
    ls = micro.level_set(grid=img.grid()).data
    assert np.mean((ls < 0) == (img.data == 1)) > 0.995
    # one tablet across y, shifted along x only: continuous tablets, staggered along x
    flat = mg.models.Nacre(
        arrangement="brick", stagger="u", aspect=4.0, tablet_size=1.0,
        tablet_thickness=0.25, mortar_thickness=0.05,
    )  # fmt: skip
    img = flat.generate(mg.Domain(size=(4.0, 0.5, 1.2), periodic=True), seed=0).voxelize(
        (77, 8, 47)  # voxel centres off the faces of the tablets
    )
    assert np.all(img.data == img.data[:, :1, :])
    # an overlap with no periodic stack, errors
    with pytest.raises(ValueError, match="rational"):
        mg.models.Nacre(arrangement="brick", overlap=1 / math.pi).generate(domain)
    with pytest.raises(ValueError, match="gap"):
        mg.models.Nacre(gap=0.6)
    with pytest.raises(ValueError, match="arrangement"):
        mg.models.Nacre(arrangement="columns")


def test_nacre_polygonal_tablets():
    """Sheet nacre: polygonal tablets of independent layers; to first order in the gap,
    the cells of Poisson germs leave 2 sqrt(lambda) g of each layer to the mortar."""
    model = mg.models.Nacre(
        tablet_size=0.5, tablet_thickness=0.1, mortar_thickness=0.0, gap=0.01, regularity=0.0
    )
    micro = model.generate(mg.Domain(size=(8.0, 8.0, 0.3), periodic=True), seed=4)
    assert micro.n_layers == 3 and len(micro.layers[0]) == round(64 / 0.25)
    p, _ = monte_carlo(micro, 400_000)
    first = model.theory()["volume_fraction_first_order"]
    assert first == pytest.approx(1.0 - 2 * 0.01 / 0.5)
    assert p == pytest.approx(first, abs=0.006)
    regular = mg.models.Nacre(tablet_size=0.5, regularity=0.8)
    germs = regular.generate(mg.Domain(size=(4.0, 4.0, 0.5), periodic=True), seed=1).layers[0]
    v = germs[:, None, :] - germs[None, :, :]
    v -= 4.0 * np.round(v / 4.0)
    gaps = np.linalg.norm(v, axis=2) + 10 * np.eye(len(germs))
    assert gaps.min() >= 0.8 * jamming_distance(len(germs), 16.0) * (1 - 1e-9)
    # non-periodic: layers over the whole box
    micro = model.generate(mg.Domain(size=(2.0, 2.0, 0.5), periodic=False), seed=1)
    img = micro.voxelize((40, 40, 50)).data
    assert img[:, :, 1:-1].any(axis=(0, 1)).sum() >= 40


def test_bouligand_plywood():
    """Non-overlapping fibres: the fraction pi r^2 / (s h); the fibres turn by the
    pitch from a layer to the next."""
    model = mg.models.Bouligand(
        fiber_radius=0.03, fiber_spacing=0.08, layer_thickness=0.07, pitch_angle=20.0,
        start_angle=10.0,
    )  # fmt: skip
    micro = model.generate(mg.Domain.cube(1.0, periodic=False), seed=3)
    assert np.allclose(np.diff(micro.angles) % 180, 20.0) and micro.angles[0] == 10.0
    exact = math.pi * 0.03**2 / (0.08 * 0.07)
    assert model.theory()["volume_fraction"] == pytest.approx(exact)
    p, se = monte_carlo(micro)
    assert p == pytest.approx(exact, abs=4 * se)
    img = micro.voxelize((96,) * 3)
    ls = micro.level_set(grid=img.grid()).data
    assert np.array_equal(ls <= 0, img.data == 1)
    # the fibres of a layer run along its direction: constant along it in its mid-plane
    k = 5
    z = micro.origin + (k + 0.5) * micro.thickness
    t = math.radians(micro.angles[k])
    base = np.random.default_rng(0).random((200, 3)) * [0.6, 0.6, 0] + [0.2, 0.2, z]
    along = base + 0.1 * np.array([math.cos(t), math.sin(t), 0.0])
    assert np.array_equal(micro.contains(base), micro.contains(along))
    with pytest.raises(ValueError, match="fiber_spacing"):
        mg.models.Bouligand(fiber_radius=0.05, fiber_spacing=0.08)


def test_periodic_bouligand():
    """A periodic box: whole half-turns, and fibre families periodic in its plane."""
    lengths = np.array([1.0, 1.5])
    dirs, gaps = periodic_families(lengths, 0.1, 0.05)
    t = np.radians(dirs)
    normals = np.column_stack([-np.sin(t), np.cos(t)])
    for axis in range(2):  # an integer number of spacings across the box
        x = normals[:, axis] * lengths[axis] / gaps
        assert np.allclose(x, np.round(x), atol=1e-9)
    model = mg.models.Bouligand(
        fiber_radius=0.025, fiber_spacing=0.07, layer_thickness=1 / 12, pitch_angle=15.0
    )
    micro = model.generate(mg.Domain(size=(1.0, 1.0, 1.0), periodic=True), seed=0)
    assert micro.n_layers == 12 and micro.pitch_angle == pytest.approx(15.0, abs=1.5)
    turns = (np.diff(np.append(micro.angles, micro.angles[0])) + 90) % 180 - 90
    assert turns.sum() == pytest.approx(180.0, abs=1e-6) or turns.sum() == pytest.approx(0.0)
    assert np.all(np.abs(micro.spacings / 0.07 - 1) < 0.1)
    fraction = np.mean(math.pi * 0.025**2 / (micro.spacings * micro.thickness))
    assert micro.summary()["fiber_fraction"] == pytest.approx(fraction)
    p, se = monte_carlo(micro)
    assert p == pytest.approx(fraction, abs=4 * se)
    with pytest.warns(RuntimeWarning, match="half-turns"):
        mg.models.Bouligand(fiber_radius=0.025, pitch_angle=45.0).generate(
            mg.Domain.cube(0.25, periodic=True), seed=0
        )


def test_honeycombs():
    """Walls of hexagonal and square honeycombs: 1 - (1 - t/d)^2 of the volume, exactly;
    extruded along the axis; the out-of-plane modulus of the walls, phi E_s."""
    t = 0.1
    for arrangement, size in (("hexagonal", (6.0, 4 * math.sqrt(3.0))), ("square", (6.0, 5.0))):
        model = mg.models.Honeycomb(arrangement=arrangement, cell_size=1.0, wall_thickness=t)
        micro = model.generate(mg.Domain(size=(*size, 0.5), periodic=True), seed=0)
        exact = 1 - (1 - t) ** 2
        assert model.theory()["volume_fraction"] == pytest.approx(exact)
        p, se = monte_carlo(micro)
        assert p == pytest.approx(exact, abs=4 * se), arrangement
        assert micro.n_cells == (48 if arrangement == "hexagonal" else 30)
    img = micro.voxelize((60, 50, 3))
    assert np.all(img.data == img.data[:, :, :1])
    ls = micro.level_set(grid=img.grid()).data
    assert np.mean((ls < 0) == (img.data == 1)) > 0.995
    # random cells: to first order, 2 t sqrt(lambda) of the volume
    model = mg.models.Honeycomb(arrangement="voronoi", regularity=0.0, wall_thickness=0.02)
    micro = model.generate(mg.Domain(size=(20.0, 20.0, 0.1), periodic=True), seed=2)
    p, _ = monte_carlo(micro, 400_000)
    assert p == pytest.approx(model.theory()["volume_fraction_first_order"], rel=0.06)
    with pytest.raises(ValueError, match="lumen"):
        mg.models.Honeycomb(wall_thickness=1.2)
    with pytest.raises(ValueError, match="two cells"):
        mg.models.Honeycomb().generate(mg.Domain(size=(1.2, 4.0, 1.0), periodic=True))


def test_out_of_plane_modulus_of_a_honeycomb():
    """Prismatic walls loaded along their axis: E3 = phi E_s exactly (they contract
    freely across it), whatever the cells (Gibson and Ashby)."""
    model = mg.models.Honeycomb(cell_size=1.0, wall_thickness=0.15, arrangement="voronoi")
    img = model.generate(mg.Domain(size=(4.0, 4.0, 0.25), periodic=True), seed=1).voxelize(
        (64, 64, 4)
    )
    phi = float(img.data.mean())
    solid = Material(young=10.0, poisson=0.3)
    res = homogenize(img, {0: VOID, 1: solid}, properties=["stiffness"], tol=1e-8)
    stiff = res["stiffness.periodic"]
    assert stiff["E3"] == pytest.approx(phi * 10.0, rel=2e-3)
    assert stiff["E1"] < 0.2 * stiff["E3"]  # bending of the walls in the plane


def test_wood():
    """Radial files through growth rings: small thick-walled latewood cells at the end
    of each ring, large thin-walled earlywood cells at its start."""
    model = mg.models.Honeycomb(
        arrangement="wood", cell_size=0.5, wall_thickness=0.05, ring_width=6.0,
        latewood_fraction=0.3, latewood_size=0.4, latewood_wall=2.0,
    )  # fmt: skip
    domain = mg.Domain(size=(12.0, 6.0, 0.1), periodic=True)
    micro = model.generate(domain, seed=5)
    assert np.all((micro.germs >= 0) & (micro.germs < [12.0, 6.0]))
    assert micro.walls.min() == pytest.approx(0.05) and micro.walls.max() == pytest.approx(0.1)
    img = micro.voxelize((480, 240, 1)).data[:, :, 0]
    x = (np.arange(480) + 0.5) / 480 * 12.0 / 6.0 % 1.0  # position in the ring
    early, late = img[(x > 0.1) & (x < 0.4)].mean(), img[x > 0.75].mean()
    assert late > 2.0 * early
    files = np.unique(np.round(micro.germs[:, 1], 6))
    assert len(files) == 12  # files 0.5 wide
    ring = model._ring_cells(6.0)
    assert len(micro.germs) == 12 * 2 * (len(ring) - 1)
    sizes = np.diff(ring) * 6.0
    assert sizes[0] == pytest.approx(sizes.max()) and sizes[-1] == pytest.approx(
        0.4 * sizes[0], rel=0.05
    )
    # small cells with thick walls together, rings starting anywhere in the box or not
    unbounded = model.generate(mg.Domain(size=(12.0, 3.0, 0.1), periodic=False), seed=2)
    for cells in (micro, unbounded):
        sizes = np.concatenate([np.diff(b) for b in cells.files["bounds"]])
        assert np.corrcoef(sizes, cells.walls)[0, 1] < -0.9
    with pytest.raises(ValueError, match="ring_width"):
        mg.models.Honeycomb(arrangement="wood", ring_width=1.0)
    # no rings: files of square cells, straight radial walls (constant along x between
    # the files) and staggered tangential ones; the radial modulus several times the
    # tangential one
    plain = mg.models.Honeycomb(arrangement="wood", cell_size=0.5, wall_thickness=0.05)
    micro = plain.generate(domain, seed=1)
    assert len(micro.germs) == 12 * 24 and np.all(micro.walls == 0.05)
    img = micro.voxelize((240, 240, 1))  # rows of voxels at y = 0.0125 and 0.5125
    assert np.all(img.data[:, 0, 0] == 1) and np.all(img.data[:, 20, 0] == 1)
    pts = img.grid().center_points()
    assert np.array_equal(micro.contains(pts).reshape(img.shape), img.data == 1)
    small = plain.generate(mg.Domain(size=(4.0, 2.0, 0.04), periodic=True), seed=1)
    res = homogenize(
        small.voxelize(voxel_size=0.025),
        {0: VOID, 1: Material(young=10.0, poisson=0.3)},
        properties=["stiffness"],
        tol=1e-6,
    )["stiffness.periodic"]
    assert res["E1"] > 3.0 * res["E2"]


def test_wood_ray_cells():
    """Rays between the files of a periodic box: their walls replace the radial wall
    between the files over their height, the files beside them narrowed; inside, the
    walls of their cells (radially, along the grain and across), across both periodic
    faces of the box."""
    domain = mg.Domain(size=(2.0, 2.0, 2.0), periodic=True)
    bounds = [0.1 * j + np.arange(5) * 0.5 for j in range(5)]  # files 0.4 wide, cells 0.5
    germs = [[0.5 * (b[k] + b[k + 1]) % 2.0, 0.4 * j + 0.2] for j, b in enumerate(bounds)
             for k in range(4)]  # fmt: skip
    rays = {
        "boundary": [2, 0], "z": [0.5, 1.8], "u": [0.3, 0.0], "cells": 1, "width": 0.2,
        "height": 0.6, "cell_length": 1.0, "cell_height": 0.2, "thickness": 0.02,
    }  # fmt: skip
    files = {"origin": 0.0, "width": 0.4, "bounds": bounds, "rays": rays}
    micro = PrismaticCells(domain, 2, germs, 0.04, files)
    plain = PrismaticCells(domain, 2, germs, 0.04, {**files, "rays": None})
    cases = [  # (point, in a wall with the rays, without)
        ((0.8, 0.8, 0.6), False, True),  # the radial wall replaced by a ray cell
        ((0.8, 0.8, 0.705), True, True),  # a wall between two ray cells along the grain
        ((0.8, 0.8, 0.715), False, True),
        ((1.305, 0.8, 0.6), True, True),  # radially, at u = 0.3 + 1
        ((1.32, 0.8, 0.6), False, True),
        ((0.8, 0.895, 0.6), True, False),  # the wall between the ray and the next file:
        ((0.8, 0.915, 0.6), True, False),  # half of the ray wall, half of the file wall
        ((0.8, 0.925, 0.6), False, False),
        ((0.8, 0.785, 0.6), False, True),  # the file below, narrowed
        ((0.95, 0.85, 1.115), True, False),  # the lumen of the file closed above the ray
        ((0.95, 0.85, 1.125), False, False),
        ((0.8, 0.0, 0.1), False, True),  # the second ray, across both periodic faces
        ((0.8, 1.995, 0.1), False, True),
        ((0.8, 0.0, 0.205), True, True),
    ]
    pts = np.array([p for p, _, _ in cases])
    assert micro.contains(pts).tolist() == [w for _, w, _ in cases]
    assert plain.contains(pts).tolist() == [w for _, _, w in cases]
    assert micro.n_rays == 2 and micro.ray_fraction == pytest.approx(2 * 0.2 * 0.6 / 4.0)
    img = micro.voxelize((50, 50, 50))  # the voxels, the level set and the points agree
    assert np.array_equal(micro.contains(img.grid().center_points()).reshape(img.shape),
                          img.data == 1)  # fmt: skip
    assert np.array_equal(micro.level_set(grid=img.grid()).data < 0, img.data == 1)


def test_wood_rays():
    """Rays placed at random: a whole number of them in a periodic box, never closer than
    a ray cell along the grain to one on the same or a neighbouring boundary; their share
    of the volume; rays stiffen the radial direction more than the tangential one."""
    model = mg.models.Honeycomb(
        arrangement="wood", cell_size=0.4, wall_thickness=0.04, ray_fraction=0.1
    )
    domain = mg.Domain(size=(3.2, 3.2, 2.4), periodic=True)
    micro = model.generate(domain, seed=4)
    rays = micro.rays
    assert (rays.width, rays.height, rays.cell_height) == pytest.approx((0.2, 1.2, 0.24))
    assert rays.cell_length == pytest.approx(1.6) and rays.thickness == 0.04
    n = round(0.1 * 3.2 * 2.4 / (0.2 * 1.2))
    assert micro.n_rays == n and micro.ray_fraction == pytest.approx(n * 0.24 / (3.2 * 2.4))
    assert micro.summary()["ray_fraction"] == micro.ray_fraction
    assert model.theory()["ray_height_cells"] == 5
    for r in range(n):
        for s in range(r):
            apart = abs(rays.boundary[r] - rays.boundary[s]) % 8
            dz = abs(rays.z[r] - rays.z[s]) % 2.4
            if min(apart, 8 - apart) <= 1:
                assert min(dz, 2.4 - dz) >= 1.2 + 0.24
    # the share of rays at random points, from the rays at their height
    pts = np.random.default_rng(1).random((40_000, 3)) * [3.2, 3.2, 2.4]
    _, j, _, below, above = micro.cells.parts(pts[:, :2])
    lower = micro.rays.locate(j, pts[:, 2])[0] >= 0
    upper = micro.rays.locate(micro.cells.above(j), pts[:, 2])[0] >= 0
    share = np.mean((lower & (below < 0.1)) | (upper & (above < 0.1)))
    assert share == pytest.approx(micro.ray_fraction, abs=4 * math.sqrt(0.1 * 0.9 / 40_000))
    back = mg.models.Microstructure.from_dict(micro.to_dict())
    assert np.array_equal(back.voxelize((32, 32, 24)).data, micro.voxelize((32, 32, 24)).data)
    # a box that holds fewer whole rays than asked; rays beyond the faces of a box
    with pytest.warns(RuntimeWarning, match="whole rays"):
        model.generate(mg.Domain(size=(3.2, 2.0, 1.6), periodic=True), seed=1)
    big = model.generate(mg.Domain(size=(2.0, 12.0, 12.0), periodic=False), seed=3)
    assert big.ray_fraction == pytest.approx(0.1, abs=0.03)
    img = big.voxelize((20, 120, 120))
    assert np.array_equal(big.contains(img.grid().center_points()).reshape(img.shape),
                          img.data == 1)  # fmt: skip
    # mechanics: the rays run radially, stiffening that direction more than the
    # tangential one, the grain hardly
    moduli = []
    for phi in (0.0, 0.125):
        wood = mg.models.Honeycomb(
            arrangement="wood", cell_size=0.4, wall_thickness=0.08, ray_fraction=phi,
            ray_height=0.72,
        )  # fmt: skip
        img = wood.generate(mg.Domain(size=(1.6, 1.6, 1.44), periodic=True), seed=1).voxelize(
            voxel_size=0.04
        )
        res = homogenize(
            img, {0: VOID, 1: Material(young=10.0, poisson=0.3)}, properties=["stiffness"],
            tol=1e-6,
        )["stiffness.periodic"]  # fmt: skip
        moduli.append((res["E1"], res["E2"], res["E3"]))
    (r0, t0, l0), (r1, t1, l1) = moduli
    assert r1 > 1.15 * r0 and r1 / t1 > 1.05 * r0 / t0 and l1 == pytest.approx(l0, rel=0.1)
    errors = [
        ({"arrangement": "hexagonal", "ray_fraction": 0.1}, "ray_fraction"),
        ({"ray_fraction": 0.5}, "ray_fraction"),
        ({"ray_fraction": 0.1, "ray_width": 0}, "ray_width"),
        ({"ray_fraction": 0.1, "ray_wall": 0.25}, "ray cells"),
        ({"ray_fraction": 0.1, "ray_height": 0.2}, "ray_height"),
        ({"ray_fraction": 0.1, "ray_width": 4}, "rays"),
    ]
    for kwargs, match in errors:
        with pytest.raises(ValueError, match=match):
            mg.models.Honeycomb(
                **{"arrangement": "wood", "cell_size": 0.4, "wall_thickness": 0.04, **kwargs}
            )
    with pytest.raises(ValueError, match="periodic box longer"):
        model.generate(mg.Domain(size=(3.2, 3.2, 1.2), periodic=True))
    with pytest.raises(ValueError, match="do not fit"):
        mg.models.Honeycomb(
            arrangement="wood", cell_size=0.4, wall_thickness=0.04, ray_fraction=0.4
        ).generate(domain, seed=1)


def test_osteons_dead_leaves():
    """The later osteon on top; the share of each part, that of its area (exact)."""
    domain = mg.Domain(size=(2.0, 2.0, 0.5), periodic=False)
    micro = OsteonStructure(domain, 2, [[0.5, 0.5], [0.7, 0.5]], [0.3, 0.3], 0.2, 0.05)
    x = np.array([0.5, 0.3, 0.7, 0.42, 0.22, 1.9])
    phases = micro._phases(np.column_stack([x, np.full(6, 0.5)]))
    # the second osteon cuts the canal of the first, its cement line through its bone
    assert phases.tolist() == [LAMELLAR, LAMELLAR, CANAL, CEMENT, CEMENT, INTERSTITIAL]
    model = mg.models.Osteons(radius=0.1, coverage=0.7, canal_ratio=0.25, cement_thickness=0.01)
    t = model.theory()
    assert t["volume_fraction_3"] == pytest.approx(0.7 * 0.0625)
    assert t["volume_fraction_2"] == pytest.approx(0.7 * (0.01 - 0.09**2) / 0.01)
    assert sum(t[f"volume_fraction_{i}"] for i in range(4)) == pytest.approx(1.0)
    counts = np.zeros(4)
    for seed in range(4):
        img = model.generate(mg.Domain(size=(4.0, 4.0, 0.01), periodic=True), seed=seed)
        data = img.voxelize((800, 800, 1)).data
        counts += np.bincount(data.ravel(), minlength=4) / data.size / 4
    for i in range(4):
        assert counts[i] == pytest.approx(t[f"volume_fraction_{i}"], abs=0.012 if i < 2 else 0.004)
    assert not model.has_level_set and model.active_phases()[CANAL] == "Haversian canals"
    with pytest.raises(ValueError, match="wider"):
        model.generate(mg.Domain.cube(0.15, periodic=True), seed=0)


def test_open_foam_edges():
    """Edges of Poisson-Voronoi tessellations: Meijering's length 5.832 lambda^(2/3) per
    unit volume, 6.768 lambda vertices of degree four; each edge once."""
    rng = np.random.default_rng(3)
    germs = rng.random((400, 3))
    ends, vertices = periodic_edges(germs, np.ones(3))
    length = np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1).sum()
    assert length == pytest.approx(5.832 * 400 ** (2 / 3), rel=0.02)
    assert len(vertices) == pytest.approx(6.768 * 400, rel=0.03)
    assert len(ends) == 2 * len(vertices)  # four edges at each vertex
    assert np.all((ends[:, 0] >= 0) & (ends[:, 0] < 1))
    # each vertex (on the torus) ends four edges
    tips = np.mod(ends.reshape(-1, 3), 1.0)
    keys = np.round(tips * 1e6).astype(np.int64) % 1_000_000
    _, counts = np.unique(keys, axis=0, return_counts=True)
    assert np.all(counts == 4)
    a = 1.7  # isotropic directions stretched: the mean growth of their lengths
    x = np.linspace(0, 1, 200_001)
    assert stretch_factor(a) == pytest.approx(np.trapezoid(np.sqrt(1 + (a * a - 1) * x * x), x))
    assert stretch_factor(1 / a) == pytest.approx(
        np.trapezoid(np.sqrt(1 + (1 / a**2 - 1) * x * x), x)
    )
    assert stretch_factor(1.0 + 1e-9) == pytest.approx(1.0)


def test_open_foams_and_trabecular_bone():
    """Relative density reached on the realization; Lloyd's iterations even the cells;
    elongated cells; the first-order theory of thin struts."""
    domain = mg.Domain.cube(1.0, periodic=True)
    model = mg.models.OpenFoam(n_cells=40, volume_fraction=0.12, node_ratio=1.3)
    micro = model.generate(domain, seed=1)
    p, _ = monte_carlo(micro)
    assert p == pytest.approx(0.12, abs=0.004)
    assert micro.summary()["n_cells"] == 40 and len(micro.spheres) > 0

    def volumes_cv(germs):
        pts = np.random.default_rng(0).random((100_000, 3))
        from scipy.spatial import cKDTree

        _, lab = cKDTree(germs, boxsize=1.0).query(pts)
        v = np.bincount(lab, minlength=len(germs))
        return v.std() / v.mean()

    poisson = mg.models.OpenFoam(n_cells=60, strut_radius=0.01)
    relaxed = mg.models.OpenFoam(n_cells=60, strut_radius=0.01, relaxation=15)
    rng = np.random.default_rng(2)
    size = np.ones(3)
    cv0 = volumes_cv(poisson._germs(rng, 60, size))
    cv1 = volumes_cv(relaxed._germs(rng, 60, size))
    assert cv1 < 0.5 * cv0
    bone = mg.models.OpenFoam(n_cells=60, strut_radius=0.01, anisotropy=2.0, axis=2)
    seg = bone.generate(domain, seed=0).segments
    d = np.abs(seg.end - seg.start)
    assert d[:, 2].mean() > 1.4 * d[:, :2].mean()
    t = mg.models.OpenFoam(intensity=300.0, strut_radius=0.005).theory()
    assert t["edge_length_density"] == pytest.approx(5.832 * 300 ** (2 / 3))
    assert t["volume_fraction_first_order"] == pytest.approx(
        math.pi * 0.005**2 * 5.832 * 300 ** (2 / 3)
    )
    with pytest.raises(ValueError, match="exactly one"):
        mg.models.OpenFoam(n_cells=10)


def test_branching_networks():
    """Space colonization: segments of one step joined in trees from the roots; Murray's
    law at every branching, from tips of the given radius."""
    parents = np.array([-1, 0, 1, 1, 2, 2])
    r = murray_radii(parents, 1.0, 3.0)
    assert np.allclose(r, [3 ** (1 / 3), 3 ** (1 / 3), 2 ** (1 / 3), 1, 1, 1])
    assert np.allclose(
        murray_radii(parents, 1.0, 2.0)[:3], [math.sqrt(3), math.sqrt(3), math.sqrt(2)]
    )
    model = mg.models.BranchingNetwork(n_attractors=800, step=0.03, tip_radius=0.004)
    domain = mg.Domain.cube(1.0, periodic=False)
    tree = model.grow(np.random.default_rng(1), np.ones(3), False)
    child = np.flatnonzero(tree.parents >= 0)
    steps = np.linalg.norm(tree.positions[child] - tree.positions[tree.parents[child]], axis=1)
    assert np.allclose(steps, 0.03)
    assert np.all(tree.parents[child] < child)  # the parents grow first
    assert np.flatnonzero(tree.parents < 0).tolist() == [0]
    assert tree.positions[0, 2] == 0.0  # the root on the lower face
    radii = murray_radii(tree.parents, 0.004, 3.0)
    flow = np.zeros(len(radii))
    np.add.at(flow, tree.parents[child], radii[child] ** 3)
    inner = flow > 0
    assert np.allclose(radii[inner] ** 3, flow[inner])
    assert np.allclose(radii[~inner], 0.004)
    micro = model.generate(domain, seed=1)
    s = micro.summary()
    assert s["reached_fraction"] > 0.9 and s["n_tips"] == int((~inner).sum())
    assert s["root_radius"] == pytest.approx(0.004 * s["n_tips"] ** (1 / 3))
    assert 0.0 < micro.voxelize((48,) * 3).data.mean() < 0.2
    # on the torus: segments from their end in the box
    periodic = mg.models.BranchingNetwork(
        n_attractors=400, step=0.04, roots=2, root_position="random"
    )
    seg = periodic.generate(mg.Domain.cube(1.0, periodic=True), seed=0).segments
    assert np.all((seg.start >= 0) & (seg.start < 1))
    assert np.allclose(seg.lengths, 0.04)
    with pytest.raises(ValueError, match="kill"):
        mg.models.BranchingNetwork(step=0.02, kill=0.5)
    with pytest.raises(ValueError, match="center"):
        mg.models.BranchingNetwork(roots=2, root_position="center")
