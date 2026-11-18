"""Crystallographic textures (:mod:`morphogen.texture`) and orientation maps
(:mod:`morphogen.io.ebsd`, the model ``ebsd``): the symmetries of the lattices, the
disorientation against known rotations and Mackenzie's distribution, mean orientations,
pole figures and inverse pole figures of known orientations, the files of the maps read
and written, the grains of a map of known grains, and a measured polycrystal through the
homogenization."""

import dataclasses

import numba
import numpy as np
import pytest
from scipy import spatial

import morphogen as mg
from morphogen import pipeline
from morphogen import texture as tx
from morphogen.config import load_config
from morphogen.homogenization import Material, homogenize
from morphogen.io import ebsd
from morphogen.orientations import euler_to_matrix, matrix_to_euler, random_orientations

COPPER = Material.cubic(168.4, 121.4, 75.4)
NICKEL = ebsd.Phase("Nickel", "cubic", (3.524, 3.524, 3.524, 90.0, 90.0, 90.0))
TITANIUM = ebsd.Phase("Titanium", "hexagonal", (2.95, 2.95, 4.683, 90.0, 90.0, 120.0))


def turn(axis, degrees):
    """The rotation matrix ``(1, 3, 3)`` of an angle about an axis."""
    return tx._turn(axis, np.radians(degrees))[None]


# -- symmetries and disorientations -------------------------------------------------------


ORDERS = {"cubic": 24, "hexagonal": 12, "tetragonal": 8, "trigonal": 6, "orthorhombic": 4,
          "monoclinic": 2, "triclinic": 1, "m-3": 12, "6/m": 6, "4/m": 4, "-3": 3}  # fmt: skip


@pytest.mark.parametrize("lattice", tx.CLASSES)
def test_symmetries_are_groups_of_rotations(lattice):
    """The rotations of each of the eleven Laue classes: as many as its order, proper,
    orthogonal, distinct, and closed under the product; those of a class of lower
    symmetry are among those of the class of highest symmetry of its system."""
    ops = tx.symmetries(lattice)
    assert ops.shape == (ORDERS[lattice], 3, 3)
    assert np.allclose(np.linalg.det(ops), 1.0)
    assert np.allclose(np.einsum("kij,klj->kil", ops, ops), np.eye(3), atol=1e-12)
    assert len(np.unique(np.round(ops.reshape(len(ops), 9), 9), axis=0)) == len(ops)
    for a in ops:
        for b in ops:
            assert any(np.allclose(a @ b, c, atol=1e-12) for c in ops)
    whole = {"m-3": "cubic", "6/m": "hexagonal", "4/m": "tetragonal", "-3": "trigonal"}
    if lattice in whole:
        larger = tx.symmetries(whole[lattice])
        assert all(any(np.allclose(a, b, atol=1e-12) for b in larger) for a in ops)
    with pytest.raises(ValueError, match="lattice"):
        tx.symmetries("icosahedral")


def test_disorientation_of_known_rotations():
    """The disorientation of a crystal from itself turned about an axis: the angle, folded
    by the symmetries of the lattice (a quarter turn of a cube about one of its edges
    brings it back; a turn of 60 degrees about [111] is the twin of cubic crystals; a
    sixth of a turn about the axis c of a hexagonal crystal brings it back), and the same
    for any orientation of the pair."""
    one = np.eye(3)[None]
    cases = [("cubic", [1, 0, 0], 30.0, 30.0), ("cubic", [1, 0, 0], 60.0, 30.0),
             ("cubic", [1, 0, 0], 90.0, 0.0), ("cubic", [1, 0, 0], 45.0, 45.0),
             ("cubic", [1, 1, 1], 60.0, 60.0), ("cubic", [1, 1, 1], 120.0, 0.0),
             ("cubic", [1, 1, 0], 50.0, 50.0), ("cubic", [1, 1, 0], 180.0, 0.0),
             ("hexagonal", [0, 0, 1], 60.0, 0.0), ("hexagonal", [0, 0, 1], 40.0, 20.0),
             ("hexagonal", [1, 0, 0], 180.0, 0.0), ("hexagonal", [1, 0, 0], 86.3, 86.3),
             ("triclinic", [1, 2, 3], 170.0, 170.0)]  # fmt: skip
    rng = np.random.default_rng(0)
    common = euler_to_matrix(random_orientations(rng, 1))
    for lattice, axis, angle, expected in cases:
        turned = turn(axis, angle)
        assert tx.disorientation(one, turned, lattice)[0] == pytest.approx(expected, abs=1e-6)
        # the same pair in other axes of the sample, and with symmetric equivalents
        a, b = one @ common, turned @ common
        assert tx.disorientation(a, b, lattice)[0] == pytest.approx(expected, abs=1e-6)
        ops = tx.symmetries(lattice)
        again = tx.disorientation(ops[-1][None] @ a, ops[len(ops) // 2][None] @ b, lattice)
        assert again[0] == pytest.approx(expected, abs=1e-6)
    # Euler angles as matrices; one crystal against several; the numbers must agree
    e = random_orientations(rng, 5)
    assert np.allclose(tx.disorientation(e, euler_to_matrix(e)), 0.0, atol=1e-5)
    assert tx.disorientation(e[0], e).shape == (5,)
    with pytest.raises(ValueError, match="crystals"):
        tx.disorientation(e[:2], e[:3])
    with pytest.raises(ValueError, match="orientations"):
        tx.disorientation(np.zeros((2, 4)), e[:2])


def test_mackenzie_distribution():
    """The density of the disorientation of random cubic crystals (Mackenzie 1958): its
    integral is one, its mean 40.74 degrees, it is continuous where its expression
    changes and vanishes at the largest disorientation, 62.80 degrees; the disorientations
    of random pairs follow it; random hexagonal crystals stay below 93.84 degrees."""
    assert tx.MACKENZIE_ANGLES == pytest.approx((45.0, 60.0, 60.7224, 62.7994), abs=1e-4)
    theta = np.linspace(0.0, 63.0, 126001)
    density = tx.mackenzie(theta)
    assert np.trapezoid(density, theta) == pytest.approx(1.0, abs=1e-8)
    assert np.trapezoid(density * theta, theta) == pytest.approx(40.7358, abs=1e-3)
    for angle in tx.MACKENZIE_ANGLES[:3]:
        below, above = tx.mackenzie(np.array([angle - 1e-7, angle + 1e-7]))
        assert below == pytest.approx(above, abs=1e-7)
    assert tx.mackenzie(np.array([10.0]))[0] == pytest.approx(2 / 15 * (1 - np.cos(np.radians(10))))
    assert np.all(tx.mackenzie(np.array([-1.0, 62.8, 70.0])) == 0.0)
    rng = np.random.default_rng(1)
    n = 200_000
    a, b = random_orientations(rng, n), random_orientations(rng, n)
    angles = tx.disorientation(a, b, "cubic")
    assert angles.max() < tx.MACKENZIE_ANGLES[3] and angles.mean() == pytest.approx(40.736, abs=0.1)
    counts, edges = np.histogram(angles, bins=21, range=(0.0, 63.0))
    for lo, hi, count in zip(edges[:-1], edges[1:], counts, strict=True):
        fine = np.linspace(lo, hi, 301)
        share = np.trapezoid(tx.mackenzie(fine), fine)  # (the probability of the interval)
        assert count / n == pytest.approx(share, abs=4.5 * np.sqrt(share * (1 - share) / n) + 1e-9)
    assert tx.disorientation(a[:50_000], b[:50_000], "hexagonal").max() < 93.85


def test_mean_orientation_of_a_grain():
    """The mean of the symmetric equivalents of one orientation is that orientation, with
    no spread; the mean of crystals scattered about an orientation is close to it, with
    their mean disorientation as spread; the weights count."""
    rng = np.random.default_rng(2)
    for lattice in ("cubic", "hexagonal"):
        g0 = euler_to_matrix(random_orientations(rng, 1))
        equivalents = tx.symmetries(lattice) @ g0[0]
        mean, spread = tx.mean_orientation(equivalents, lattice)
        assert tx.disorientation(mean, g0, lattice)[0] < 1e-5 and spread < 1e-5
        scattered = tx.sample_texture(rng, g0, 400, spread=2.0)
        from_centre = tx.disorientation(scattered, g0, lattice)
        assert from_centre.mean() == pytest.approx(2.0 * np.sqrt(2 / np.pi), rel=0.15)
        mean, spread = tx.mean_orientation(scattered, lattice)
        assert tx.disorientation(mean, g0, lattice)[0] < 0.3
        assert spread == pytest.approx(from_centre.mean(), rel=0.05)
    # two orientations, all the weight on one
    two = random_orientations(rng, 2)
    mean, _ = tx.mean_orientation(two, weights=[1.0, 0.0])
    assert tx.disorientation(mean, two[0])[0] < 1e-5
    with pytest.raises(ValueError, match="weight"):
        tx.mean_orientation(two, weights=[1.0])


# -- pole figures --------------------------------------------------------------------------


def test_axes_of_the_families_of_planes():
    """The distinct axes of the families of planes and directions: 3 {100}, 6 {110}, 4
    {111}, 12 {112} and 24 {123} axes of the cube; the axis c, 3 prismatic and 6
    pyramidal axes of the hexagonal lattice, whose angle with c follows the ratio c/a."""
    counts = {(1, 0, 0): 3, (1, 1, 0): 6, (1, 1, 1): 4, (1, 1, 2): 12, (1, 2, 3): 24}
    for indices, count in counts.items():
        axes = tx.crystal_axes(indices)
        assert axes.shape == (count, 3) and np.allclose(np.linalg.norm(axes, axis=1), 1.0)
    assert np.allclose(tx.crystal_axes((0, 0, 0, 1), "hexagonal"), [[0.0, 0.0, 1.0]])
    assert np.allclose(tx.crystal_axes((0, 0, 1), "hexagonal"), [[0.0, 0.0, 1.0]])
    prisms = tx.crystal_axes((1, 0, -1, 0), "hexagonal")
    assert prisms.shape == (3, 3) and np.allclose(prisms[:, 2], 0.0)
    ratio = 1.587  # titanium
    pyramids = tx.crystal_axes((1, 0, -1, 1), "hexagonal", c_over_a=ratio)
    assert pyramids.shape == (6, 3)
    # the normal of (10-11) makes with c the angle atan(2 c / (sqrt(3) a))
    assert np.degrees(np.arccos(np.abs(pyramids[:, 2]))) == pytest.approx(
        np.degrees(np.arctan(2.0 * ratio / np.sqrt(3.0)))
    )
    # a direction of the basal plane: [2 -1 -1 0] is the first axis of the crystal
    assert np.allclose(tx.crystal_axes((2, -1, -1, 0), "hexagonal", plane=False)[0], [1, 0, 0])
    with pytest.raises(ValueError, match="lattice"):
        tx.crystal_axes((1, 0, 0), "tetragonal")
    with pytest.raises(ValueError, match="indices"):
        tx.crystal_axes((0, 0, 0))


def test_poles_and_projections():
    """The poles of the cube orientation: {100} at the centre and on the two axes of the
    figure, {111} at 54.74 degrees from the centre (a radius tan(27.37 degrees) of the
    stereographic projection) on its diagonals; a crystal turned by phi1 about the third
    axis of the sample has its poles turned by phi1; the two projections put a direction
    at the radii tan(t/2) and sqrt(2) sin(t/2), and are inverted."""
    cube = np.zeros((1, 3))
    hundred = tx.project(tx.poles(cube, (1, 0, 0))[0])
    assert sorted(np.round(hundred, 12).tolist()) == [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    eleven = tx.project(tx.poles(cube, (1, 1, 1))[0])
    assert np.allclose(np.linalg.norm(eleven, axis=1), np.tan(0.5 * np.arccos(1 / np.sqrt(3))))
    assert np.allclose(np.abs(eleven[:, 0]), np.abs(eleven[:, 1]))
    turned = tx.poles(np.array([[30.0, 0.0, 0.0]]), (1, 0, 0))[0]
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    expected = [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
    assert all(any(np.allclose(p, e, atol=1e-12) for p in turned) for e in expected)
    rng = np.random.default_rng(3)
    directions = rng.normal(size=(200, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    polar = np.arccos(np.abs(directions[:, 2]))
    for projection, radius in (("stereographic", np.tan(polar / 2)),
                               ("equal_area", np.sqrt(2) * np.sin(polar / 2))):  # fmt: skip
        points = tx.project(directions, projection)
        assert np.allclose(np.linalg.norm(points, axis=1), radius)
        upper = np.where(directions[:, 2:3] < 0, -directions, directions)
        assert np.allclose(tx.unproject(points, projection), upper, atol=1e-12)
        assert np.allclose(tx.project(-directions, projection), points)  # (an axis: both ends)
    with pytest.raises(ValueError, match="projection"):
        tx.project(directions, "gnomonic")


def test_pole_figure_densities():
    """The density of a pole figure, in multiples of that of a random texture: its mean
    over the sphere is one for any texture (an equal-area figure: the mean of its
    points); a random texture gives one everywhere, within its noise; one orientation
    gives, at each of its m poles, the value of the kernel on its axis over m; the weights
    are the volumes of the crystals."""
    rng = np.random.default_rng(4)
    random = tx.pole_figure(random_orientations(rng, 20_000), (1, 1, 1), halfwidth=10.0,
                            resolution=61)  # fmt: skip
    assert np.nanmean(random.density) == pytest.approx(1.0, abs=2e-3)
    assert 0.9 < np.nanmin(random.density) and random.maximum < 1.1
    assert np.isnan(random.density[0, 0]) and np.isfinite(random.density[30, 30])
    for halfwidth in (5.0, 15.0):
        kappa = np.log(2.0) / (1.0 - np.cos(np.radians(halfwidth)))
        # (poles away from the rim of the disc, where the grid cuts the kernel)
        one = tx.pole_figure(np.array([[20.0, 30.0, 40.0]]), (1, 0, 0), halfwidth=halfwidth,
                             resolution=201)  # fmt: skip
        assert np.nanmean(one.density) == pytest.approx(1.0, abs=2e-3)
        cube = tx.pole_figure(np.zeros((1, 3)), (1, 0, 0), halfwidth=halfwidth, resolution=201)
        peak = kappa / np.tanh(kappa) / 3.0  # (the kernel at its axis, three axes)
        assert cube.at([0, 0, 1]) == pytest.approx(peak, rel=1e-6) and cube.maximum <= peak * 1.001
        assert cube.at([1, 0, 0]) == pytest.approx(peak, rel=1e-3)
        # the density is halved at the half-width of the kernel
        away = [np.sin(np.radians(halfwidth)), 0.0, np.cos(np.radians(halfwidth))]
        assert tx.axis_density([away], [[0, 0, 1]], halfwidth=halfwidth)[0] == pytest.approx(
            0.5 * kappa / np.tanh(kappa), rel=1e-3
        )
    # two orientations of weights 3 and 1: the poles of the first three times as dense
    second = np.array([[45.0, 0.0, 0.0]])
    both = tx.pole_figure(np.vstack([np.zeros((1, 3)), second]), (1, 0, 0), weights=[3.0, 1.0],
                          halfwidth=5.0, resolution=201)  # fmt: skip
    first_pole, second_pole = [1.0, 0.0, 0.0], [np.sqrt(0.5), np.sqrt(0.5), 0.0]
    assert both.at(first_pole) == pytest.approx(3.0 * both.at(second_pole), rel=3e-2)
    several = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        alone = tx.pole_figure(np.zeros((1, 3)), (1, 1, 0), resolution=41).density
    finally:
        numba.set_num_threads(several)
    assert np.array_equal(alone, tx.pole_figure(np.zeros((1, 3)), (1, 1, 0), resolution=41).density,
                          equal_nan=True)  # fmt: skip
    with pytest.raises(ValueError, match="halfwidth"):
        tx.pole_figure(np.zeros((1, 3)), (1, 0, 0), halfwidth=0.0)


def test_inverse_pole_figures_and_colours():
    """The direction of the crystal along a direction of the sample, in the fundamental
    sector: the same for all the symmetric equivalents of the crystal and for the
    opposite direction; red, green and blue at the corners [001], [011], [111] of the
    cube and [0001], [2-1-10], [10-10] of the hexagonal lattice."""
    rng = np.random.default_rng(5)
    for lattice in ("cubic", "hexagonal"):
        g = euler_to_matrix(random_orientations(rng, 20))
        d = rng.normal(size=3)
        base = tx.inverse_poles(g, d, lattice)
        assert np.allclose(np.linalg.norm(base, axis=1), 1.0)
        for s in tx.symmetries(lattice):
            assert np.allclose(tx.inverse_poles(s @ g, d, lattice), base, atol=1e-10)
        assert np.allclose(tx.inverse_poles(g, -d, lattice), base, atol=1e-10)
        if lattice == "cubic":
            assert np.all(np.diff(base, axis=1) >= -1e-12) and np.all(base >= 0)
        else:
            azimuth = np.degrees(np.arctan2(base[:, 1], base[:, 0]))
            assert np.all(base[:, 2] >= 0) and np.all((azimuth >= -1e-9) & (azimuth <= 30 + 1e-9))
        colours = tx.ipf_colours(g, d, lattice)
        assert colours.shape == (20, 3) and np.allclose(colours.max(axis=1), 1.0)
        assert np.all(colours >= 0)

    def along_z(direction):
        """A crystal whose given direction lies along the third axis of the sample."""
        c = np.array(direction, dtype=float)
        c /= np.linalg.norm(c)
        a = np.cross(c, [1.0, 0.3, 0.2])
        a /= np.linalg.norm(a)
        return np.stack([a, np.cross(c, a), c], axis=1)[None]  # (g z = c: its third column)

    for direction, colour in (([0, 0, 1], [1, 0, 0]), ([0, 1, 1], [0, 1, 0]),
                              ([1, 1, 1], [0, 0, 1]), ([1, 0, 0], [1, 0, 0]),
                              ([1, -1, 0], [0, 1, 0])):  # fmt: skip
        assert np.allclose(tx.ipf_colours(along_z(direction)), [colour], atol=1e-9)
    m = [np.cos(np.pi / 6), np.sin(np.pi / 6), 0.0]  # the normal of a prism plane
    for direction, colour in (([0, 0, 1], [1, 0, 0]), ([1, 0, 0], [0, 1, 0]), (m, [0, 0, 1])):
        found = tx.ipf_colours(along_z(direction), lattice="hexagonal")
        assert np.allclose(found, [colour], atol=1e-9)
    with pytest.raises(ValueError, match="lattice"):
        tx.inverse_poles(np.zeros((1, 3)), lattice="monoclinic")


def test_textures_drawn_from_measured_orientations():
    """Orientations drawn from measured ones follow their weights, are the measured ones
    without spread, and are turned from them by the given spread."""
    rng = np.random.default_rng(6)
    measured = random_orientations(rng, 3)
    drawn = tx.sample_texture(rng, measured, 6000, weights=[1.0, 2.0, 3.0])
    which = np.array([np.argmin(tx.disorientation(measured, d)) for d in drawn[:600]])
    assert np.all(np.min([tx.disorientation(measured[k], drawn) for k in range(3)], axis=0) < 1e-5)
    assert np.bincount(which, minlength=3) / 600 == pytest.approx([1 / 6, 2 / 6, 3 / 6], abs=0.07)
    spread = tx.sample_texture(rng, measured[:1], 2000, spread=3.0)
    angles = tx.disorientation(measured[0], spread)
    assert angles.mean() == pytest.approx(3.0 * np.sqrt(2 / np.pi), rel=0.1) and angles.min() > 0
    with pytest.raises(ValueError, match="weight"):
        tx.sample_texture(rng, measured, 3, weights=[1.0, -1.0, 1.0])


# -- orientation maps ----------------------------------------------------------------------


def voronoi_map(n=24, shape=(48, 64), seed=7, phases=(NICKEL,), step=(0.5, 0.5)):
    """A map of ``n`` grains (the cells of a plane Voronoi tessellation of random seeds)
    whose orientations are far from each other, these orientations, and the cell of each
    point."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    seeds = rng.uniform([0, 0], [nx, ny], (n, 2))
    yy, xx = np.meshgrid(np.arange(ny) + 0.5, np.arange(nx) + 0.5, indexing="ij")
    labels = spatial.cKDTree(seeds).query(np.column_stack([xx.ravel(), yy.ravel()]))[1]
    lattice = phases[0].lattice
    euler = []
    while len(euler) < n:  # orientations at 15 degrees from each other at least
        e = random_orientations(rng, 1)[0]
        if all(tx.disorientation(e, o, lattice)[0] > 15.0 for o in euler):
            euler.append(e)
    euler = np.array(euler)
    labels = labels.reshape(ny, nx)
    return ebsd.EBSDMap.from_grains(labels, euler, step, phases=phases), euler, labels


def test_maps_written_and_read(tmp_path):
    """A map written as an .ang (radians) and as a .ctf (degrees) file is read back: its
    grid, its steps, its phases, the points that were not indexed, and its orientations
    within the digits of the files."""
    made, _, _ = voronoi_map(phases=(NICKEL, TITANIUM))
    rng = np.random.default_rng(8)
    phase = made.phase.copy()
    phase[rng.random(phase.shape) < 0.05] = 0
    phase[:, 40:][phase[:, 40:] > 0] = 2  # the second phase on the right
    made = ebsd.EBSDMap(made.euler, phase, (0.25, 0.4), made.phases)
    for suffix, digits in ((".ang", 1e-3), (".ctf", 2e-4)):
        path = ebsd.write_ebsd(tmp_path / f"map{suffix}", made)
        back = ebsd.read_ebsd(path)
        assert back.shape == made.shape and back.step == pytest.approx(made.step)
        assert np.array_equal(back.phase, made.phase)
        named = [(p.name, p.lattice) for p in back.phases]
        assert named == [("Nickel", "cubic"), ("Titanium", "hexagonal")]
        assert back.phases[1].cell == pytest.approx(TITANIUM.cell, abs=1e-3)
        assert back.phases[1].c_over_a == pytest.approx(4.683 / 2.95, abs=1e-3)
        for k, lattice in ((1, "cubic"), (2, "hexagonal")):
            mask = made.phase == k
            gap = tx.disorientation(back.euler[mask], made.euler[mask], lattice)
            assert gap.max() < digits
        assert back.size == pytest.approx((64 * 0.25, 48 * 0.4))
        assert np.array_equal(back.indexed, made.indexed)
    # the angles of the files are in the axes of the systems: those of the image of the
    # map (.ctf), or a quarter of a turn about its normal away (.ang, the usual setting)
    cubic = made.phase == 1
    quarter = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for suffix, turn_of_file in ((".ang", quarter), (".ctf", np.eye(3))):
        as_written = ebsd.read_ebsd(tmp_path / f"map{suffix}", axes="file")
        expected = euler_to_matrix(made.euler[cubic]) @ turn_of_file
        assert tx.disorientation(as_written.euler[cubic], expected).max() < 1e-3
        again = ebsd.write_ebsd(tmp_path / f"again{suffix}", as_written, axes="file")
        assert (
            again.read_text().splitlines()[-1]
            == ((tmp_path / f"map{suffix}").read_text().splitlines()[-1])
        )
    assert set(ebsd.read_ang(tmp_path / "map.ang").quality) == {"iq", "ci"}
    assert set(ebsd.read_ctf(tmp_path / "map.ctf").quality) == {"bands", "error", "mad", "bc",
                                                                "bs"}  # fmt: skip
    with pytest.raises(ValueError, match=r"\.ang or a \.ctf"):
        ebsd.read_ebsd(tmp_path / "map.txt")
    with pytest.raises(ValueError, match="axes"):
        ebsd.read_ebsd(tmp_path / "map.ctf", axes="sample")
    with pytest.raises(ValueError, match="setting"):
        ebsd.read_ebsd(tmp_path / "map.ang", setting=5)
    with pytest.raises(ValueError, match="phase"):
        ebsd.EBSDMap(made.euler, made.phase + 5, (1.0, 1.0), made.phases)


ANG_HEXAGONAL = """# TEM_PIXperUM          1.000000
# Phase 1
# MaterialName  	Titanium (Alpha)
# Formula     	Ti
# Symmetry              62
# LatticeConstants      2.950 2.950 4.683  90.000  90.000 120.000
#
# GRID: HexGrid
# XSTEP: 1.000000
# YSTEP: 0.866025
# NCOLS_ODD: 3
# NCOLS_EVEN: 2
# NROWS: 3
#
  0.10000   0.20000   0.30000      0.00000      0.00000  120.0  0.900  0      1  1.200
  0.10000   0.20000   0.30000      1.00000      0.00000  110.0  0.800  0      1  1.100
 12.56637  12.56637  12.56637      2.00000      0.00000   10.0 -1.000  0      1  0.000
  1.00000   0.50000   2.00000      0.50000      0.86603  130.0  0.700  0      1  1.000
  1.00000   0.50000   2.00000      1.50000      0.86603  125.0  0.600  0      1  1.300
  0.10000   0.20000   0.30000      0.00000      1.73205  100.0  0.500  0      1  1.400
  0.10000   0.20000   0.30000      1.00000      1.73205  105.0  0.400  0      1  1.500
  1.00000   0.50000   2.00000      2.00000      1.73205  115.0  0.300  0      1  1.600
"""

CTF_COMMAS = (
    "Channel Text File\nPrj\tsample\nAuthor\t[Unknown]\nJobMode\tGrid\nXCells\t3\nYCells\t2\n"
    "XStep\t0,5\nYStep\t0,5\nAcqE1\t0\nAcqE2\t0\nAcqE3\t0\n"
    "Euler angles refer to Sample Coordinate system (CS0)!\tMag\t500\tCoverage\t100\n"
    "Phases\t2\n"
    "3,524;3,524;3,524\t90;90;90\tNi-superalloy\t11\t225\t3803863129_5.0.6.3\t-906185425\n"
    "4,05;4,05;10,2\t90;90;90\tSigma\t5\t136\n"
    "Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS\n"
    "1\t0,0000\t0,0000\t9\t0\t10,5000\t20,2500\t30,1250\t0,4500\t120\t200\n"
    "1\t0,5000\t0,0000\t9\t0\t10,5000\t20,2500\t30,1250\t0,5000\t121\t201\n"
    "0\t1,0000\t0,0000\t0\t3\t0,0000\t0,0000\t0,0000\t0,0000\t30\t90\n"
    "2\t0,0000\t0,5000\t8\t0\t100,0000\t45,0000\t5,0000\t0,6000\t110\t190\n"
    "2\t0,5000\t0,5000\t8\t0\t100,0000\t45,0000\t5,0000\t0,6500\t111\t191\n"
    "1\t1,0000\t0,5000\t9\t0\t10,5000\t20,2500\t30,1250\t0,4000\t125\t205\n"
)


def test_files_of_the_acquisition_systems(tmp_path):
    """The files as the systems write them: an .ang file on a hexagonal grid (every other
    row half a step away and one point shorter, read row by row; a point of negative
    confidence index is not indexed; the symmetry 62 is the hexagonal lattice), and a .ctf
    file with decimal commas and two phases (the Laue groups 11 and 5: cubic and
    tetragonal)."""
    path = tmp_path / "titanium.ang"
    path.write_text(ANG_HEXAGONAL)
    hexagonal = ebsd.read_ang(path, axes="file")
    assert hexagonal.shape == (3, 3) and hexagonal.step == pytest.approx((1.0, 0.866025))
    assert hexagonal.phases[0].name == "Titanium (Alpha)" and hexagonal.lattice() == "hexagonal"
    assert hexagonal.phases[0].c_over_a == pytest.approx(4.683 / 2.95)
    expected = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 1]])  # (not indexed; the short row)
    assert np.array_equal(hexagonal.phase, expected)
    assert hexagonal.euler[0, 0] == pytest.approx(np.degrees([0.1, 0.2, 0.3]))
    assert hexagonal.euler[1, 1] == pytest.approx(np.degrees([1.0, 0.5, 2.0]))
    assert hexagonal.quality["ci"][2, 2] == pytest.approx(0.3)
    assert hexagonal.quality["iq"][1, 0] == pytest.approx(130.0)
    # two crystals in four sets of neighbours (the points of the corner meet by a diagonal
    # only); the two points without an orientation take the grains of their neighbours
    grains = hexagonal.grains(threshold=5.0)
    assert grains.count == 4 and np.all(grains.labels >= 0)
    assert grains.labels[0, 2] == grains.labels[0, 1] and grains.labels[1, 2] == grains.labels[2, 2]
    assert grains.size.tolist() == [3, 2, 2, 2]
    assert tx.disorientation(grains.orientations[[0, 2]], np.degrees([[0.1, 0.2, 0.3]]),
                             "hexagonal").max() < 1e-9  # fmt: skip
    path = tmp_path / "superalloy.ctf"
    path.write_text(CTF_COMMAS)
    commas = ebsd.read_ctf(path, axes="file")
    assert commas.shape == (2, 3) and commas.step == (0.5, 0.5)
    assert [p.lattice for p in commas.phases] == ["cubic", "tetragonal"]
    assert commas.phases[1].cell == pytest.approx((4.05, 4.05, 10.2, 90.0, 90.0, 90.0))
    assert np.array_equal(commas.phase, [[1, 1, 0], [2, 2, 1]])
    assert commas.euler[0, 1] == pytest.approx([10.5, 20.25, 30.125])
    assert commas.euler[1, 0] == pytest.approx([100.0, 45.0, 5.0])
    assert commas.quality["mad"][1, 1] == pytest.approx(0.65)
    assert commas.orientations(2).shape == (2, 3) and commas.orientations().shape == (5, 3)
    with pytest.raises(ValueError, match="table of points"):
        (tmp_path / "empty.ctf").write_text("Channel Text File\nXCells\t3\n")
        ebsd.read_ctf(tmp_path / "empty.ctf")


CTF_HEXAGONAL = (
    "Channel Text File\nPrj\ttitanium\nAuthor\t[Unknown]\nJobMode\tGrid\nXCells\t2\nYCells\t1\n"
    "XStep\t1\nYStep\t1\nAcqE1\t0\nAcqE2\t0\nAcqE3\t0\n"
    "Euler angles refer to Sample Coordinate system (CS0)!\tMag\t100\tCoverage\t100\n"
    "Phases\t2\n"
    "2.950;2.950;4.683\t90;90;120\tTi-Hex\t9\t194\n"
    "5.417;5.417;5.417\t90;90;90\tPyrite\t10\t205\n"
    "Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS\n"
    "1\t0.0000\t0.0000\t9\t0\t0.0000\t0.0000\t0.0000\t0.4500\t120\t200\n"
    "1\t1.0000\t0.0000\t9\t0\t10.0000\t20.0000\t40.0000\t0.5000\t121\t201\n"
)


def test_axes_of_the_files(tmp_path):
    """The orientations of a file are in the axes of its acquisition system, and are
    brought into those of the image of the map when it is read (the first axis to the
    right, the second upwards, the third towards the viewer): they are those of a .ctf
    file; an .ang file has them turned about the normal, by an angle that depends on the
    setting of the software — its first axis down the image and its second one to the
    right with the usual setting, then up, to the right and to the left with the three
    others. A hexagonal crystal of a .ctf file has its first axis along a* = [1 0 -1 0],
    that of morphoGen along a = [2 -1 -1 0], 30 degrees away about c."""
    # a crystal whose axes are those of the system (g = 1 in the file), seen from the map:
    # the rows of its matrix are the axes of the crystal in those of the image
    one = ebsd.EBSDMap(np.zeros((2, 2, 3)), np.ones((2, 2), dtype=int), (1.0, 1.0))
    ctf = ebsd.write_ebsd(tmp_path / "one.ctf", one, axes="file")
    for axes in ebsd.AXES:
        assert np.allclose(euler_to_matrix(ebsd.read_ctf(ctf, axes=axes).euler[0, 0]), np.eye(3))
    ang = ebsd.write_ebsd(tmp_path / "one.ang", one, axes="file")
    settings = {
        1: [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        2: [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        3: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        4: [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    }
    assert set(settings) == set(ebsd.ANG_SETTINGS)
    for setting, matrix in settings.items():
        in_map = euler_to_matrix(ebsd.read_ang(ang, setting=setting).euler[0, 0])[0]
        assert np.allclose(in_map, matrix, atol=1e-12)
        # (written with the same setting, the file is the same)
        back = ebsd.write_ang(tmp_path / "back.ang", ebsd.read_ang(ang, setting=setting),
                              setting=setting)  # fmt: skip
        as_written = ebsd.read_ang(back, axes="file").euler.reshape(-1, 3)
        assert tx.disorientation(as_written, np.zeros(3), "triclinic").max() < 1e-3
    # the points that were not indexed keep no orientation in other axes
    holes = ebsd.EBSDMap(np.zeros((2, 2, 3)), [[1, 0], [1, 1]]).turned(tx._turn([1, 1, 0], np.pi))
    assert np.all(holes.euler[0, 1] == 0.0) and not np.all(holes.euler[0, 0] == 0.0)
    # the axes of a hexagonal crystal: (0, 0, 0) in a .ctf file is a crystal whose plane
    # (1 0 -1 0) is normal to the first axis of the sample, and whose direction
    # [2 -1 -1 0] is 30 degrees away from it
    path = tmp_path / "titanium.ctf"
    path.write_text(CTF_HEXAGONAL)
    titanium = ebsd.read_ctf(path, axes="file")
    assert [p.lattice for p in titanium.phases] == ["hexagonal", "m-3"]
    assert titanium.euler[0] == pytest.approx(np.array([[0.0, 0.0, 330.0], [10.0, 20.0, 10.0]]))
    prisms = tx.poles(titanium.euler[0, 0], (1, 0, -1, 0), "hexagonal")[0]
    assert any(np.allclose(np.abs(p), [1.0, 0.0, 0.0], atol=1e-12) for p in prisms)
    a = tx.poles(titanium.euler[0, 0], (2, -1, -1, 0), "hexagonal", plane=False)[0]
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    assert any(np.allclose(p, [c, -s, 0.0], atol=1e-12) for p in a)
    written = ebsd.write_ctf(tmp_path / "written.ctf", titanium, axes="file")
    assert written.read_text().splitlines()[-1].split("\t")[5:8] == ["10.0000", "20.0000",
                                                                      "40.0000"]  # fmt: skip
    # (an .ang file has the first axis of its hexagonal crystals along a, as morphoGen)
    as_ang = ebsd.write_ang(tmp_path / "titanium.ang", titanium, axes="file")
    assert ebsd.read_ang(as_ang, axes="file").euler[0, 1] == pytest.approx([10.0, 20.0, 10.0],
                                                                           abs=1e-3)  # fmt: skip


def test_grains_of_a_map():
    """The grains of a map of known grains, whose points are scattered by a degree about
    the orientations of their grains and 4 % of which were not indexed: as many grains,
    each point in the grain of its cell, the mean orientations within a tenth of a degree
    of those of the cells, the spread that of the scatter; the areas and the boundaries."""
    made, euler, cells = voronoi_map(n=24, shape=(96, 128), seed=9)
    ny, nx = made.shape
    rng = np.random.default_rng(10)
    axes = rng.normal(size=(ny * nx, 3))
    angles = np.radians(np.abs(rng.normal(0.0, 1.0, ny * nx)))
    g = euler_to_matrix(made.euler.reshape(-1, 3)) @ tx._turns(axes, angles)
    phase = made.phase.copy()
    lost = rng.random(phase.shape) < 0.04
    phase[lost] = 0
    noisy = ebsd.EBSDMap(matrix_to_euler(g).reshape(ny, nx, 3), phase, made.step, made.phases)
    grains = noisy.grains(threshold=5.0, min_size=10)
    assert grains.count == 24 and np.all(grains.labels >= 0)
    assert grains.size.sum() == ny * nx and grains.areas.sum() == pytest.approx(48.0 * 64.0)
    # each grain is one cell: the orientation of the cell nearest to its mean; the points
    # inside the cells (their four neighbours in the same cell) are in the grain of their
    # cell, those of the boundaries almost all
    cell = np.array([np.argmin(tx.disorientation(euler, e)) for e in grains.orientations])
    assert sorted(cell.tolist()) == list(range(24))
    padded = np.pad(cells, 1, mode="edge")
    inside = np.ones(cells.shape, dtype=bool)
    for shifted in (padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]):
        inside &= shifted == cells
    assert np.array_equal(cell[grains.labels][inside], cells[inside]) and inside.mean() > 0.8
    assert np.mean(cell[grains.labels] == cells) > 0.995
    gap = tx.disorientation(grains.orientations, euler[cell])
    assert gap.max() < 0.15
    assert grains.spread == pytest.approx(np.full(24, np.sqrt(2 / np.pi)), rel=0.25)
    assert np.all(grains.phase == 1) and grains.lattice() == "cubic"
    assert grains.diameters == pytest.approx(2 * np.sqrt(grains.size * 0.25 / np.pi))
    # the boundaries: pairs of neighbours, their lengths in faces of points, their angles
    pairs, faces, angles = tx.boundary_disorientations(grains.labels, grains.orientations)
    assert np.all(pairs[:, 0] < pairs[:, 1]) and len(np.unique(pairs, axis=0)) == len(pairs)
    assert angles.min() > 14.0 and angles.max() < 62.8 and faces.min() >= 1
    k = int(np.argmax(faces))
    first, second = euler[cell[pairs[k, 0]]], euler[cell[pairs[k, 1]]]
    assert angles[k] == pytest.approx(tx.disorientation(first, second)[0], abs=0.3)
    # a threshold above the disorientations of the grains: one grain
    assert noisy.grains(threshold=70.0).count == 1
    # without filling, the points that were not indexed have no grain
    holes = noisy.grains(threshold=5.0, min_size=10, fill=False)
    assert np.array_equal(holes.labels < 0, lost | (holes.labels < 0)) and (holes.labels < 0).any()
    with pytest.raises(ValueError, match="no grain"):
        noisy.grains(min_size=10**6)


def test_maps_turned_and_coloured():
    """A map in other axes of the sample: its orientations follow (a crystal seen from
    axes turned by a quarter turn about the third one has its poles turned back by a
    quarter turn), its points do not; a map of its own grains is itself; the colours of a
    map are those of its orientations, black where it was not indexed."""
    made, _, _ = voronoi_map(n=6, shape=(12, 16), seed=11)
    quarter = tx._turn([0, 0, 1], np.pi / 2)
    turned = made.turned(quarter)
    assert np.array_equal(turned.phase, made.phase)
    before = tx.poles(made.euler[3, 4], (1, 0, 0))[0]
    after = tx.poles(turned.euler[3, 4], (1, 0, 0))[0]
    assert all(any(np.allclose(quarter @ p, q, atol=1e-9) for q in after) for p in before)
    assert tx.disorientation(turned.turned(quarter.T).euler[5, 5], made.euler[5, 5])[0] < 1e-5
    grains = made.grains()
    again = ebsd.EBSDMap.from_grains(grains.labels, grains.orientations, made.step)
    gap = tx.disorientation(again.euler.reshape(-1, 3), made.euler.reshape(-1, 3))
    assert gap.max() < 1e-5
    phase = made.phase.copy()
    phase[0, 0] = 0
    colours = ebsd.EBSDMap(made.euler, phase, made.step).colours([1.0, 0.0, 0.0])
    assert colours.shape == (12, 16, 3) and np.all(colours[0, 0] == 0.0)
    assert np.allclose(colours[3, 4], tx.ipf_colours(made.euler[3, 4], [1.0, 0.0, 0.0])[0])
    cropped = made.cropped(slice(4, 12), slice(2, 9))
    assert cropped.shape == (7, 8) and np.array_equal(cropped.euler, made.euler[2:9, 4:12])
    with pytest.raises(ValueError, match="rotation"):
        made.turned(2.0 * np.eye(3))


# -- measured polycrystals ----------------------------------------------------------------


def test_polycrystal_of_a_map(tmp_path):
    """The model ``ebsd``: the window of the size of the domain at the first corner of the
    map, with the grains found in it and their orientations, on the lattice of the points
    of the map; a window drawn at random for each seed; the elastic stiffness of a map of
    one crystal is that of the crystal in the axes of the sample."""
    made, _, cells = voronoi_map(n=20, shape=(48, 64), seed=12)
    path = ebsd.write_ebsd(tmp_path / "nickel.ctf", made)
    domain = mg.Domain((16.0, 12.0, 0.5), periodic=False)
    assert mg.get_model("ebsd") is mg.models.EBSD
    # by default the sets of fewer than five points are not grains (here a point of a cell
    # that meets it by a diagonal): the 20 cells, a quarter of the map in the window
    default = mg.models.EBSD(file=str(path))
    assert default.active_phases() == {1: "Nickel"}
    assert default.grains().count == 20 and default.expected_count(domain) == 5.0
    assert default.generate(domain, 0).attributes["window"] == [0, 0, 32, 24]
    # every set of points a grain: the map itself
    model = mg.models.EBSD(file=str(path), min_size=1)
    micro = model.generate(domain, 0)
    assert micro.shape == (32, 24, 1) and micro.attributes["window"] == [0, 0, 32, 24]
    # (the rows of the map from the top, the second axis of the domain upwards)
    labels = micro.labels().data[:, ::-1, 0].T
    assert np.all(micro.voxelize().data == 1)
    gap = tx.disorientation(micro.orientations[labels].reshape(-1, 3),
                            made.euler[:24, :32].reshape(-1, 3))  # fmt: skip
    assert gap.max() < 2e-4
    assert sorted(np.unique(labels).tolist()) == list(range(len(micro.orientations)))
    assert len(micro.orientations) >= len(np.unique(cells[:24, :32]))
    drawn = mg.models.EBSD(file=str(path), window="random")
    windows = {tuple(drawn.generate(domain, seed).attributes["window"]) for seed in range(6)}
    assert len(windows) > 3 and all(w[0] <= 32 and w[1] <= 24 for w in windows)
    assert drawn.generate(domain, 3).attributes == drawn.generate(domain, 3).attributes
    # the map in other axes: the orientations of the model follow
    turned = mg.models.EBSD(file=str(path), min_size=1, turn=(90.0, 0.0, 0.0))
    quarter = euler_to_matrix(np.array([[90.0, 0.0, 0.0]]))[0]
    expected = euler_to_matrix(micro.orientations) @ quarter.T
    assert tx.disorientation(turned.generate(domain, 0).orientations, expected).max() < 1e-4
    # the same map as an .ang file with another setting of the system, and as it is
    other = ebsd.write_ebsd(tmp_path / "nickel.ang", made, setting=2)
    for options, turn_of_file in (({"setting": 2}, np.eye(3)),
                                  ({"axes": "file"}, tx._turn([0, 0, 1], np.pi / 2))):  # fmt: skip
        again = mg.models.EBSD(file=str(other), min_size=1, **options).generate(domain, 0)
        expected = euler_to_matrix(micro.orientations) @ turn_of_file
        assert tx.disorientation(again.orientations, expected).max() < 1e-3
    wrong = [({"window": "centre"}, "window"), ({"threshold": 0.0}, "threshold"),
             ({"min_size": 0}, "min_size"), ({"turn": (1.0, 2.0)}, "turn"),
             ({"axes": "sample"}, "axes"), ({"setting": 5}, "setting")]  # fmt: skip
    for bad, message in wrong:
        with pytest.raises(ValueError, match=message):
            mg.models.EBSD(file=str(path), **bad)
    with pytest.raises(ValueError, match="larger than the map"):
        model.generate(mg.Domain((40.0, 12.0, 0.5), periodic=False), 0)
    with pytest.raises(ValueError, match="not periodic"):
        model.generate(mg.Domain((16.0, 12.0, 0.5)), 0)
    with pytest.raises(ValueError, match="no orientation map"):
        mg.models.EBSD(file=str(tmp_path / "missing.ctf")).grains()
    # one crystal: its stiffness in the axes of the sample
    e = np.array([[35.0, 50.0, 20.0]])
    single = ebsd.EBSDMap.from_grains(np.zeros((6, 8), dtype=int), e, (1.0, 1.0))
    crystal = single.grains().microstructure(thickness=2.0, periodic=True)
    image = crystal.voxelize((8, 6, 2))
    result = homogenize(image, {1: COPPER}, grains=crystal.labels((8, 6, 2)).data,
                        orientations=crystal.orientations, properties=["stiffness"])  # fmt: skip
    turned_crystal = homogenize(mg.VoxelImage(np.ones((2, 2, 2), dtype=np.uint8)), {1: COPPER},
                                grains=np.zeros((2, 2, 2), dtype=np.int32), orientations=e,
                                properties=["stiffness"])  # fmt: skip
    assert result["stiffness.periodic"]["tensor"] == pytest.approx(
        turned_crystal["stiffness.periodic"]["tensor"], rel=1e-9, abs=1e-9
    )
    # (the stiffness of the crystal turned into the axes of the sample)
    g = euler_to_matrix(e)[0]
    c = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            c[i, i, j, j] += 121.4
            c[i, j, i, j] += 75.4
            c[i, j, j, i] += 75.4
        c[i, i, i, i] += 168.4 - 121.4 - 2 * 75.4
    in_sample = np.einsum("ai,bj,ck,dl,abcd->ijkl", g, g, g, g, c)
    tensor = result["stiffness.periodic"]["tensor"]
    assert tensor[0, 0] == pytest.approx(in_sample[0, 0, 0, 0], rel=1e-9)
    assert tensor[0, 1] == pytest.approx(in_sample[0, 0, 1, 1], rel=1e-9)
    assert tensor[2, 2] == pytest.approx(in_sample[2, 2, 2, 2], rel=1e-9)


def test_map_of_a_polycrystal_and_back():
    """The map that an acquisition would give of a layer of a polycrystal, without noise:
    the polycrystal of its grains is the layer, cell by cell, with its orientations
    (the rows of a map go down its image, the second axis of a domain up); with noise,
    the orientations are scattered by the given angle and points are not indexed, more
    of them along the boundaries."""
    domain = mg.Domain((12.0, 9.0, 0.5), periodic=False)
    micro = mg.models.Voronoi(intensity=0.02, crystal_orientations=True).generate(domain, 5)
    shape = (48, 36, 1)
    clean = ebsd.synthetic_map(micro, shape)
    assert clean.shape == (36, 48) and clean.step == pytest.approx((0.25, 0.25))
    assert clean.indexed.all()
    cells = micro.labels(shape).data[:, :, 0]
    # the first row of the map is the top of the image: the largest y
    top = tx.disorientation(clean.euler[0], micro.orientations[cells[:, -1]])
    assert top.max() < 1e-9
    back = clean.grains(threshold=2.0).microstructure(thickness=0.5)
    assert back.domain.size == pytest.approx(domain.size) and back.shape == shape
    found = back.labels(shape).data[:, :, 0]
    gap = tx.disorientation(back.orientations[found].reshape(-1, 3),
                            micro.orientations[cells].reshape(-1, 3))  # fmt: skip
    assert gap.max() < 1e-6
    # (one grain of the map per cell of the layer, or more where a cell is cut in two)
    pairs = np.unique(np.stack([found.ravel(), cells.ravel()], axis=1), axis=0)
    assert len(np.unique(pairs[:, 0])) == len(pairs) >= len(np.unique(cells))
    noisy = ebsd.synthetic_map(micro, shape, scatter=0.5, missing=0.05, boundaries=0.5, seed=1)
    assert np.array_equal(noisy.euler, ebsd.synthetic_map(
        micro, shape, scatter=0.5, missing=0.05, boundaries=0.5, seed=1).euler)  # fmt: skip
    lost = ~noisy.indexed
    edge = np.zeros(lost.shape, dtype=bool)
    rows = cells[:, ::-1].T
    edge[:, 1:] |= rows[:, 1:] != rows[:, :-1]
    edge[:, :-1] |= rows[:, 1:] != rows[:, :-1]
    edge[1:] |= rows[1:] != rows[:-1]
    edge[:-1] |= rows[1:] != rows[:-1]
    assert lost[~edge].mean() == pytest.approx(0.05, abs=0.02)
    assert lost[edge].mean() == pytest.approx(0.05 + 0.95 * 0.5, abs=0.06)
    scattered = tx.disorientation(noisy.euler[~lost], clean.euler[~lost])
    assert scattered.mean() == pytest.approx(0.5 * np.sqrt(2 / np.pi), rel=0.1)
    with pytest.raises(ValueError, match="layer"):
        ebsd.synthetic_map(micro, shape, layer=1)
    with pytest.raises(ValueError, match="no orientations"):
        plain = mg.models.Voronoi(intensity=0.02).generate(domain, 5)
        ebsd.synthetic_map(plain, shape)


def test_demonstration_map():
    """The map that comes with morphoGen (the template ``ebsd``) is the one that
    examples/ebsd_map.py writes; its grains, its cube texture and the bulk modulus of its
    polycrystal of nickel."""
    import importlib.util
    from pathlib import Path

    path = ebsd.example_map()
    assert path.name == "ebsd_map.ctf" and ebsd.example_map("ebsd_map.ctf") == path
    measured = ebsd.read_ebsd(path)
    assert measured.shape == (48, 64) and measured.step == (0.5, 0.5)
    assert measured.phases[0].name == "Nickel" and 0.85 < measured.indexed.mean() < 0.92
    grains = measured.grains(threshold=5.0, min_size=5)
    assert 45 <= grains.count <= 65 and np.all(grains.labels >= 0)
    assert grains.spread.mean() == pytest.approx(0.4 * np.sqrt(2 / np.pi), rel=0.2)
    # a third of the grains near the cube orientation: {100} poles along the axes
    cube = tx.disorientation(grains.orientations, np.zeros(3)) < 15.0
    assert 0.25 < grains.areas[cube].sum() / grains.areas.sum() < 0.5
    figure = tx.pole_figure(grains.orientations, (1, 0, 0), weights=grains.areas)
    assert figure.at([0, 0, 1]) > 4.0 and figure.at([1, 0, 0]) > 4.0 > 1.0 > figure.at([1, 1, 1])
    crystal = grains.microstructure()
    result = homogenize(crystal.voxelize((64, 48, 1)), {1: Material.cubic(246.5, 147.3, 124.7)},
                        grains=crystal.labels((64, 48, 1)).data,
                        orientations=crystal.orientations, properties=["bulk_modulus"])  # fmt: skip
    exact = (246.5 + 2 * 147.3) / 3
    assert result["bulk_modulus.periodic"].value == pytest.approx(exact, rel=1e-6)
    with pytest.raises(ValueError, match="no map"):
        ebsd.example_map("missing.ctf")
    with pytest.raises(ValueError, match="no map"):
        ebsd.example_map("ebsd.yaml")
    script = Path(__file__).resolve().parents[1] / "examples" / "ebsd_map.py"
    if not script.is_file():  # an installed package without its examples
        return
    spec = importlib.util.spec_from_file_location("ebsd_map_example", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    written = module.demonstration_map()
    assert np.array_equal(written.phase, measured.phase)
    gap = tx.disorientation(written.euler[written.indexed], measured.euler[measured.indexed])
    assert gap.max() < 2e-4


def test_measured_polycrystal_in_a_configuration(tmp_path):
    """A configuration with the model ``ebsd``: windows of a map drawn for each
    realization, homogenized by the FFT solver; the bulk modulus of a polycrystal of
    cubic crystals is (C11 + 2 C12) / 3 for every window, and the shear modulus lies
    between the bounds of Reuss and Voigt."""
    made, _, _ = voronoi_map(n=30, shape=(60, 80), seed=13)
    path = ebsd.write_ebsd(tmp_path / "copper.ang", made)
    config = {
        "domain": {"size": [20.0, 15.0, 0.5], "periodic": False},
        "model": {"type": "ebsd", "file": str(path), "window": "random"},
        "voxels": {"voxel_size": 0.5},
        "realizations": 2,
        "seed": 1,
        "homogenization": {"method": "fft", "materials": {1: {"cubic": [168.4, 121.4, 75.4]}},
                           "properties": ["bulk_modulus", "shear_modulus"]},
        "outputs": {"directory": str(tmp_path / "out"), "name": "map", "labels": ["vti"]},
    }  # fmt: skip
    rows = pipeline.read_rows(pipeline.run(load_config(config)) / "results.csv")
    assert len(rows) == 2 and rows[0]["n_grains"] != rows[1]["n_grains"]
    # the file of a configuration file is taken from its folder; a map that comes with
    # morphoGen is found by its name wherever the configuration is
    (tmp_path / "study").mkdir()
    (tmp_path / "study" / "run.yaml").write_text(
        "model: {type: ebsd, file: ../copper.ang}\ndomain: {size: [20, 15, 0.5], periodic: false}\n"
    )
    read = load_config(tmp_path / "study" / "run.yaml")
    assert read.model.grains().count == mg.models.EBSD(file=str(path)).grains().count
    packaged = load_config({"model": {"type": "ebsd", "file": "ebsd_map.ctf"},
                            "domain": {"size": [16.0, 12.0, 0.5], "periodic": False}},
                           base_dir=tmp_path)  # fmt: skip
    assert packaged.model.path() == str(ebsd.example_map())
    c11, c12, c44 = 168.4, 121.4, 75.4
    voigt = (c11 - c12 + 3 * c44) / 5
    reuss = 5 * c44 * (c11 - c12) / (4 * c44 + 3 * (c11 - c12))
    for row in rows:
        assert row["bulk_modulus.periodic"] == pytest.approx((c11 + 2 * c12) / 3, rel=1e-6)
        assert reuss < row["shear_modulus.periodic"] < 1.35 * voigt


# -- inverse pole figures, tables, figures and the command ---------------------------------


def test_inverse_pole_figure_densities():
    """The density of an inverse pole figure, in multiples of that of a random texture:
    one everywhere for a random texture; for one crystal, the kernel on its axis over the
    number of its equivalents that the symmetries make of the direction — 24 of a
    direction of a cube inside the sector, 4 of [111], one of the axis c of a hexagonal
    crystal; the corners of the sector and their colours."""
    rng = np.random.default_rng(20)
    for lattice in ("cubic", "hexagonal"):
        random = tx.inverse_pole_figure(random_orientations(rng, 20_000), (0, 0, 1), lattice)
        assert np.nanmean(random.density) == pytest.approx(1.0, abs=5e-3)
        assert 0.95 < np.nanmin(random.density) and random.maximum < 1.05
        assert np.isnan(random.density[0, -1])  # (outside the sector)
    kappa = np.log(2.0) / (1.0 - np.cos(np.radians(5.0)))
    peak = kappa / np.tanh(kappa)

    def along_z(direction):
        c = np.array(direction, dtype=float)
        c /= np.linalg.norm(c)
        a = np.cross(c, [1.0, 0.3, 0.2])
        a /= np.linalg.norm(a)
        return np.stack([a, np.cross(c, a), c], axis=1)[None]  # (g z = c)

    inside = [0.15, 0.45, 0.88]  # (more than 10 degrees from the mirrors of the cube)
    cases = [("cubic", inside, 24), ("cubic", [1, 1, 1], 4), ("cubic", [0, 0, 1], 3),
             ("cubic", [0, 1, 1], 6), ("hexagonal", [0, 0, 1], 1),
             ("hexagonal", [1, 0, 0], 3)]  # fmt: skip
    for lattice, direction, axes in cases:
        figure = tx.inverse_pole_figure(along_z(direction), (0, 0, 1), lattice, halfwidth=5.0,
                                        resolution=401)  # fmt: skip
        assert figure.at(direction) == pytest.approx(peak / axes, rel=2e-2)
        assert figure.maximum == pytest.approx(peak / axes, rel=2e-2)
    # the direction of the sample: the first axis of a crystal of the cube orientation
    assert tx.inverse_pole_figure(np.zeros((1, 3)), (1, 0, 0)).at([0, 0, 1]) > 10.0
    # the corners of the sector on the plane of the figure, and their colours
    corners = np.array([c for c, _ in tx.SECTOR_CORNERS["cubic"]])
    assert tx.sector_points(corners) == pytest.approx(
        np.array([[0.0, 0.0], [np.sqrt(2) - 1, 0.0], [0.3660254, 0.3660254]]), abs=1e-7
    )
    assert np.allclose(tx.direction_colours(corners), np.eye(3), atol=1e-12)
    corners = np.array([c for c, _ in tx.SECTOR_CORNERS["hexagonal"]])
    assert tx.sector_points(corners, "hexagonal") == pytest.approx(
        np.array([[0.0, 0.0], [1.0, 0.0], [np.cos(np.pi / 6), 0.5]]), abs=1e-12
    )
    assert np.allclose(tx.direction_colours(corners, "hexagonal"), np.eye(3), atol=1e-12)
    # the equivalents of a direction are one direction of the sector
    d = rng.normal(size=(50, 3))
    for lattice in ("cubic", "hexagonal"):
        base = tx.fundamental_directions(d, lattice)
        for s in tx.symmetries(lattice):
            assert np.allclose(tx.fundamental_directions(-d @ s.T, lattice), base, atol=1e-10)
    with pytest.raises(ValueError, match="weight"):
        tx.inverse_pole_figure(np.zeros((2, 3)), weights=[1.0])
    with pytest.raises(ValueError, match="lattice"):
        tx.inverse_pole_figure(np.zeros((1, 3)), lattice="trigonal")


def test_tables_and_boundaries_of_the_grains():
    """The grains of a map as a table; their boundaries: the pairs of neighbouring grains
    of one phase, the lengths of their boundaries (the sides of the points, of unequal
    steps) and their disorientations; the summary of the map, phase by phase."""
    made, _, _ = voronoi_map(n=12, shape=(30, 40), seed=21, phases=(NICKEL, TITANIUM),
                             step=(0.5, 0.25))  # fmt: skip
    phase = np.where(made.euler[..., 0] > 180.0, 2, 1)  # (two phases, by the orientations)
    grains = ebsd.EBSDMap(made.euler, phase, made.step, made.phases).grains(threshold=3.0)
    table = grains.table()
    assert list(table) == ["grain", "phase", "phi1", "Phi", "phi2", "points", "area", "diameter",
                           "spread"]  # fmt: skip
    assert table["points"].sum() == 30 * 40 and table["area"].sum() == pytest.approx(150.0)
    assert np.array_equal(table["grain"], np.arange(grains.count))
    pairs, lengths, angles = grains.boundaries()
    assert np.all(pairs[:, 0] < pairs[:, 1]) and np.all(lengths > 0)
    assert np.all(grains.phase[pairs[:, 0]] == grains.phase[pairs[:, 1]])
    # against the sides counted one by one
    lab = grains.labels
    total = {}
    for a, b in zip(lab[:, :-1].ravel(), lab[:, 1:].ravel(), strict=True):
        if a != b and grains.phase[a] == grains.phase[b]:
            total[(min(a, b), max(a, b))] = total.get((min(a, b), max(a, b)), 0.0) + 0.25
    for a, b in zip(lab[:-1].ravel(), lab[1:].ravel(), strict=True):
        if a != b and grains.phase[a] == grains.phase[b]:
            total[(min(a, b), max(a, b))] = total.get((min(a, b), max(a, b)), 0.0) + 0.5
    assert {
        tuple(p): pytest.approx(v) for p, v in zip(pairs.tolist(), lengths, strict=True)
    } == total
    for (a, b), angle in zip(pairs.tolist(), angles, strict=True):
        lattice = grains.lattice(int(grains.phase[a]))
        expected = tx.disorientation(grains.orientations[a], grains.orientations[b], lattice)[0]
        assert angle == pytest.approx(expected, abs=1e-9)
    summary = grains.summary()
    assert summary["grains"] == grains.count and len(summary["phases"]) == 2
    nickel, titanium = summary["phases"]
    assert (nickel["lattice"], titanium["lattice"]) == ("cubic", "hexagonal")
    assert nickel["area_fraction"] + titanium["area_fraction"] == pytest.approx(1.0)
    assert set(nickel["pole_figures"]) == {"100", "110", "111"}
    assert set(titanium["pole_figures"]) == {"0001", "10-10", "10-11"}
    assert nickel["mean_diameter"] <= nickel["mean_diameter_by_area"]
    assert grains.summary(poles=[(1, 1, 2)])["phases"][0]["pole_figures"].keys() == {"112"}
    # one grain: no boundary
    alone = ebsd.EBSDMap(np.zeros((4, 4, 3)), np.ones((4, 4), dtype=int)).grains()
    assert [len(v) for v in alone.boundaries()] == [0, 0, 0]
    assert "mean_disorientation" not in alone.summary()["phases"][0]


def test_figures_of_textures(tmp_path):
    """The figures of the textures are drawn (matplotlib): pole figures on one colour
    range, an inverse pole figure with the directions of the crystals, the map in the
    colours of its orientations with its grain boundaries and its key, the
    disorientations against Mackenzie's density."""
    pytest.importorskip("matplotlib")
    from morphogen.viz import figures

    measured = ebsd.read_ebsd(ebsd.example_map())
    grains = measured.grains(5.0, min_size=5)
    shown = [tx.pole_figure(grains.orientations, i, weights=grains.areas, resolution=41)
             for i in ((1, 0, 0), (1, -1, 1))]  # fmt: skip
    fig = figures.pole_figures_figure(shown, axes=("RD", "TD"), title="nickel")
    assert len(fig.axes) == 3  # (two discs and the colour bar)
    assert fig.axes[0].get_title().startswith("{100}") and "1̅" in fig.axes[1].get_title()
    assert len(figures.pole_figures_figure(shown[0]).axes) == 2
    inverse = tx.inverse_pole_figure(grains.orientations, weights=grains.areas, resolution=41)
    fig = figures.inverse_pole_figure_figure(inverse, points=tx.inverse_poles(grains.orientations))
    assert "along z" in fig.axes[0].get_title()
    fig.savefig(tmp_path / "inverse.png")
    fig = figures.ipf_map_figure(measured, grains=grains)
    assert fig.axes[0].get_xlabel() == "x (um)" and len(fig.axes) == 2
    fig.savefig(tmp_path / "map.png")
    # the boundaries: as many sides as pairs of neighbouring points of two grains
    xs, _ = figures._boundary_lines(grains.labels, measured.step, measured.size[1])
    lab = grains.labels
    sides = np.count_nonzero(lab[:, 1:] != lab[:, :-1]) + np.count_nonzero(lab[1:] != lab[:-1])
    assert np.count_nonzero(np.isnan(xs)) == sides
    _, lengths, angles = grains.boundaries()
    fig = figures.disorientation_figure(angles, lengths)
    assert len(fig.axes[0].lines) == 1 and fig.axes[0].get_legend() is not None
    assert len(figures.disorientation_figure(angles, lattice="hexagonal").axes[0].lines) == 0
    hexagonal = tx.inverse_pole_figure(np.array([[10.0, 40.0, 5.0]]), (1, 0, 0), "hexagonal")
    figures.inverse_pole_figure_figure(hexagonal).savefig(tmp_path / "hexagonal.png")
    for name in ("inverse.png", "map.png", "hexagonal.png"):
        assert (tmp_path / name).stat().st_size > 2000
    with pytest.raises(ValueError, match="no pole figure"):
        figures.pole_figures_figure([])


def test_command_of_the_textures(tmp_path, capsys):
    """``morphogen texture``: the summary of a map and of its grains, the tables, the map
    in the colours of its orientations, the map written in the other format (as it is,
    or cleaned: the mean orientations of its grains at every point)."""
    import json

    from morphogen import cli

    source = str(ebsd.example_map())
    assert cli.main(["texture", source, "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["points"] == [64, 48] and summary["grains"] == 56 and summary["files"] == []
    assert summary["phases"][0]["pole_figures"]["100"] == pytest.approx(5.73, abs=0.01)
    assert summary["phases"][0]["mean_disorientation"] == pytest.approx(38.4, abs=0.1)
    folder = tmp_path / "out"
    options = ["--poles", "111", "10-1", "--direction", "x", "--clean"]
    code = cli.main(["texture", source, "-o", str(folder), *options,
                     "--convert", str(tmp_path / "clean.ang")])  # fmt: skip
    assert code == 0
    text = capsys.readouterr().out
    assert "56 grains" in text and "{111}" in text and "{10-1}" in text and "written:" in text
    rows = (folder / "grains.csv").read_text().splitlines()
    assert rows[0] == "grain,phase,phi1,Phi,phi2,points,area,diameter,spread" and len(rows) == 57
    assert len((folder / "boundaries.csv").read_text().splitlines()) == 131
    from morphogen.viz.png import decode_png

    image = decode_png((folder / "map_ipf_x.png").read_bytes())
    assert image.shape == (48 * 8, 64 * 8, 3)
    cleaned = ebsd.read_ebsd(tmp_path / "clean.ang")
    assert cleaned.indexed.all() and cleaned.grains(5.0).count == 56
    as_it_is = tmp_path / "copy.ang"
    assert cli.main(["texture", source, "--convert", str(as_it_is), "--setting", "2"]) == 0
    capsys.readouterr()
    copy = ebsd.read_ebsd(as_it_is, setting=2)
    measured = ebsd.read_ebsd(source)
    assert np.array_equal(copy.indexed, measured.indexed)
    gap = tx.disorientation(copy.euler[copy.indexed], measured.euler[measured.indexed])
    assert gap.max() < 1e-3
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pass
    else:
        for name in ("map_ipf_x_grains.png", "pole_figures.png", "inverse_pole_figure_x.png",
                     "disorientations.png"):  # fmt: skip
            assert (folder / name).stat().st_size > 2000, name
    assert cli.main(["texture", str(tmp_path / "missing.ctf")]) == 2
    assert "error" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["texture", source, "--poles", "1x1"])


def test_polycrystal_with_a_measured_texture(tmp_path):
    """A computed polycrystal with the texture of a map (``texture: {type: measured}``):
    its orientations are those of the grains of the map, each drawn with the area of its
    grain, then scattered by the given angle; the map of a configuration file is taken
    from its folder."""
    from morphogen.models import model_from_config
    from morphogen.orientations import Measured, fibre_orientations, orientation_from_config

    law = orientation_from_config({"type": "measured", "file": "ebsd_map.ctf"})
    assert isinstance(law, Measured) and law.to_config() == {"type": "measured",
                                                             "file": "ebsd_map.ctf"}  # fmt: skip
    euler, areas, lattice = law.grains()
    assert lattice == "cubic" and len(euler) == len(areas) == 56
    rng = np.random.default_rng(22)
    drawn = law.orientations(rng, 20_000)
    which = np.array([np.argmin(tx.disorientation(euler, d)) for d in drawn[:400]])
    assert np.all(tx.disorientation(drawn[:400], euler[which]) < 1e-5)
    # (the share of the grains near the cube orientation: that of their areas)
    near = tx.disorientation(euler, np.zeros(3)) < 15.0
    share = areas[near].sum() / areas.sum()
    assert np.mean(tx.disorientation(drawn, np.zeros(3)) < 15.0) == pytest.approx(share, abs=0.02)
    axes = law.sample(rng, 100)
    assert axes.shape == (100, 3) and np.allclose(np.linalg.norm(axes, axis=1), 1.0)
    assert fibre_orientations(np.random.default_rng(1), 5, law) == pytest.approx(
        law.orientations(np.random.default_rng(1), 5)
    )
    spec = {"type": "voronoi", "n_cells": 300, "crystal_orientations": True,
            "texture": {"type": "measured", "file": "ebsd_map.ctf", "spread": 2.0}}  # fmt: skip
    model = model_from_config(spec)
    assert model_from_config(model.to_config()) == model
    micro = model.generate(mg.Domain.cube(8.0), 4)
    gaps = np.array([tx.disorientation(euler, o).min() for o in micro.orientations])
    assert gaps.mean() == pytest.approx(2.0 * np.sqrt(2 / np.pi), rel=0.15) and gaps.max() < 12.0
    figure = tx.pole_figure(micro.orientations, (1, 0, 0))
    assert figure.at([0, 0, 1]) > 3.0  # (the cube component of the map)
    # the map of a configuration file: from its folder
    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / "nickel.ctf").write_bytes(ebsd.example_map().read_bytes())
    (tmp_path / "run.yaml").write_text(
        "model: {type: voronoi, n_cells: 20, crystal_orientations: true,\n"
        "        texture: {type: measured, file: maps/nickel.ctf}}\n"
        "domain: {size: 4.0}\n"
    )
    read = load_config(tmp_path / "run.yaml")
    assert read.model.texture.file == str(tmp_path / "maps" / "nickel.ctf")
    assert len(read.model.generate(read.domain, 1).orientations) == 20
    wrong = [({"file": ""}, "file"), ({"spread": -1.0}, "spread"), ({"setting": 7}, "setting"),
             ({"colour": 1}, "invalid orientation")]  # fmt: skip
    for bad, message in wrong:
        with pytest.raises(ValueError, match=message):
            orientation_from_config({"type": "measured", "file": "ebsd_map.ctf", **bad})
    with pytest.raises(ValueError, match="no grain of the phase 3"):
        Measured(file="ebsd_map.ctf", phase=3).grains()


# -- the lattices along loading paths ------------------------------------------------------

#: A hexagonal crystal that slips on its basal plane only (magnesium, MPa): with a tensile
#: axis in the plane of ``c`` and ``a_1``, one system slips (the two others of the basal
#: plane carry half of its resolved shear stress, and slip (1/2)^25 times as fast)
BASAL = {"hexagonal": [59400, 25600, 21400, 61600, 16400],
         "crystal_plasticity": {"families": ["basal"], "tau0": 10.0, "K": 5.0, "n": 25,
                                "c_over_a": 1.624}}  # fmt: skip


def test_lattice_of_a_crystal_in_single_slip():
    """The lattice of a crystal that slips on one system turns as Schmid and Boas found.
    In a tension the material line along the axis keeps its direction (the grips), and
    in the lattice it is sheared by the slip: its components across the slip direction
    and across the slip plane keep their lengths, so that, with the stretch l/l0 and the
    angles of the axis to the slip direction (lambda) and to the slip plane (chi),
    l/l0 = sin(lambda0)/sin(lambda) = sin(chi0)/sin(chi): the axis turns towards the slip
    direction. In a compression the planes normal to the axis keep their normal (the
    platens): with the height h/h0 and the angles of the axis to the normal of the slip
    plane (phi) and to the slip direction, sin(phi) = (h/h0) sin(phi0) and cos(lambda) =
    (h/h0) cos(lambda0): the axis turns towards the normal of the slip plane. Within the
    elastic strains (0.1 %)."""
    from morphogen.homogenization import simulate_point

    magnesium = Material.from_config(BASAL)

    def crystal_at(angle, axis):
        """The crystal whose axis c is at ``angle`` degrees from the axis of the sample
        (0: x, 2: z), its direction a_1 in the plane of the two."""
        t = np.radians(angle)
        if axis == 0:
            c, a1 = np.array([np.cos(t), 0.0, np.sin(t)]), np.array([np.sin(t), 0.0, -np.cos(t)])
        else:
            c, a1 = np.array([np.sin(t), 0.0, np.cos(t)]), np.array([np.cos(t), 0.0, -np.sin(t)])
        return matrix_to_euler(np.stack([a1, np.cross(c, a1), c])[None])[0]

    # tension along x to a stretch of 1.3: the axis c at 50 degrees, a_1 at 40
    path = {"times": [0.0, 300.0], "increments": [60], "strain": {"11": [0.0, 0.3]}}
    pulled = simulate_point(magnesium, path, orientation=crystal_at(50.0, 0), kinematics="finite")
    assert pulled.orientation.shape == (61, 3) and pulled.info["cuts"] == 0
    assert tx.disorientation(pulled.orientation[0], crystal_at(50.0, 0), "hexagonal")[0] < 1e-9
    stretch = 1.0 + pulled.strain[:, 0]
    along = euler_to_matrix(pulled.orientation) @ np.array([1.0, 0.0, 0.0])
    to_direction = np.sqrt(1.0 - along[:, 0] ** 2)  # (sine of the angle to a_1)
    to_plane = np.abs(along[:, 2])  # (sine of the angle to the basal plane)
    assert stretch * to_direction == pytest.approx(np.full(61, np.sin(np.radians(40.0))), rel=2e-3)
    assert stretch * to_plane == pytest.approx(np.full(61, np.sin(np.radians(40.0))), rel=2e-3)
    final = np.degrees(np.arcsin(to_direction[-1]))
    assert final == pytest.approx(np.degrees(np.arcsin(np.sin(np.radians(40.0)) / 1.3)), abs=0.05)
    assert final == pytest.approx(29.63, abs=0.05)  # (from 40 degrees)
    # the curves of the point have the angles of its lattice; its texture is its lattice
    row = pulled.rows()[-1]
    assert [row[k] for k in ("phi1", "Phi", "phi2")] == pytest.approx(pulled.orientation[-1])
    euler, weights = pulled.texture()
    assert np.array_equal(euler, pulled.orientation[-1:]) and np.array_equal(weights, [1.0])
    assert np.array_equal(pulled.texture(0.0)[0], pulled.orientation[:1])
    # compression along z to three quarters of the height: the axis c at 40 degrees
    path = {"times": [0.0, 300.0], "increments": [60], "strain": {"33": [0.0, -0.25]}}
    pressed = simulate_point(magnesium, path, orientation=crystal_at(40.0, 2), kinematics="finite")
    height = 1.0 + pressed.strain[:, 2]
    along = euler_to_matrix(pressed.orientation) @ np.array([0.0, 0.0, 1.0])
    to_normal = np.sqrt(1.0 - along[:, 2] ** 2)  # (sine of the angle to c)
    assert to_normal == pytest.approx(height * np.sin(np.radians(40.0)), rel=2e-3)
    assert np.abs(along[:, 0]) == pytest.approx(height * np.cos(np.radians(50.0)), rel=2e-3)
    assert np.degrees(np.arcsin(to_normal[-1])) == pytest.approx(28.82, abs=0.05)
    # in small strains the lattice does not turn: no orientation, no texture
    small = simulate_point(magnesium, {"times": [0.0, 1.0], "strain": {"11": [0.0, 0.01]}},
                           orientation=crystal_at(50.0, 0))  # fmt: skip
    assert small.orientation is None and "phi1" not in small.rows()[-1]
    with pytest.raises(ValueError, match="no fields"):
        small.texture()


def test_fibres_of_a_taylor_polycrystal():
    """Face-centred cubic crystals that all undergo the same isochoric extension (the
    polycrystal of Taylor), to a stretch of 1.6: the tensile axis goes to <111> and, less,
    to <100>, and leaves <110>; in a compression to 0.6 it goes to <110>. The inverse pole
    figures of the axis, in multiples of a random texture."""
    from morphogen.homogenization import simulate_point

    nickel = Material.from_config({
        "cubic": [250000, 160000, 120000],
        "crystal_plasticity": {"families": ["octahedral"], "tau0": 80, "Q": 20, "b": 10,
                               "K": 25, "n": 10},
    })  # fmt: skip
    start = random_orientations(np.random.default_rng(3), 50)
    segments = 16
    zeros = [0.0] * (segments + 1)
    found = {}
    for name, last in (("tension", 1.6), ("compression", 0.6)):
        stretch = np.linspace(1.0, last, segments + 1)
        lateral = (stretch**-0.5 - 1.0).tolist()
        path = {"times": np.linspace(0.0, 400.0, segments + 1).tolist(),
                "increments": [4] * segments,
                "strain": {"11": (stretch - 1.0).tolist(), "22": lateral, "33": lateral,
                           "23": zeros, "13": zeros, "12": zeros}}  # fmt: skip
        final = [simulate_point(nickel, path, orientation=e, kinematics="finite").orientation[-1]
                 for e in start]  # fmt: skip
        found[name] = tx.inverse_pole_figure(np.array(final), (1.0, 0.0, 0.0), halfwidth=12.0)
    before = tx.inverse_pole_figure(start, (1.0, 0.0, 0.0), halfwidth=12.0)
    assert 0.5 < before.at([1, 1, 1]) < 1.6 and 0.5 < before.at([0, 1, 1]) < 1.6
    pulled, pressed = found["tension"], found["compression"]
    assert pulled.at([1, 1, 1]) > 2.5 and pulled.at([0, 1, 1]) < 0.6
    assert pulled.at([0, 0, 1]) > pulled.at([0, 1, 1])
    assert pressed.at([0, 1, 1]) > 1.6 and pressed.at([1, 1, 1]) < 0.5


def test_texture_of_a_cell_along_a_path(tmp_path, capsys):
    """The texture of a volume element at a time of a loading path: the lattices of its
    crystals (the voxels of an image, the elements of a mesh) with their volumes, from
    the fields of the path; in finite strains they have turned from those of the grains
    by a few tenths of a degree at 1 % of strain. The same from the files of the fields,
    and by the command ``morphogen texture``."""
    import json

    from helpers import box_mesh
    from morphogen import cli
    from morphogen.homogenization import CrystalPlasticity, NonlinearFFTProblem
    from morphogen.image import VoxelImage

    n = 8
    rng = np.random.default_rng(1)
    centres = (np.indices((n, n, n)).reshape(3, -1).T + 0.5) / n
    gaps = centres[:, None, :] - rng.random((5, 3))[None, :, :]
    gaps -= np.round(gaps)
    grains = np.argmin((gaps**2).sum(-1), axis=1).reshape(n, n, n)
    orientations = random_orientations(rng, 5)
    image = VoxelImage(np.zeros((n, n, n), dtype=np.uint8), spacing=(1 / n,) * 3)
    slip = CrystalPlasticity.from_config({"tau0": 60.0, "K": 10.0, "n": 15.0})
    crystal = dataclasses.replace(Material.cubic(168e3, 121e3, 75e3), crystal_plasticity=slip)
    path = {"times": [0, 10], "increments": 5, "strain": {"33": 0.01}}
    problem = NonlinearFFTProblem(image, {0: crystal}, grains=grains, orientations=orientations,
                                  kinematics="finite")  # fmt: skip
    result = problem.run(path, fields=[10.0])
    euler, weights = result.texture()
    assert euler.shape == (n**3, 3) and np.array_equal(weights, np.ones(n**3))
    assert np.array_equal(result.texture(9.0)[0], euler)  # (the nearest of the fields)
    turned = tx.disorientation(euler, orientations[grains.ravel()])
    assert 0.0 < turned.mean() < turned.max() < 5.0
    figure = tx.pole_figure(euler, (1, 1, 1), weights=weights, resolution=41)
    assert np.nanmean(figure.density) == pytest.approx(1.0, abs=0.05)
    # in small strains: no orientation among the fields
    small = NonlinearFFTProblem(image, {0: crystal}, grains=grains,
                                orientations=orientations).run(path, fields=[10.0])  # fmt: skip
    small_fields = small.fields[10.0]
    with pytest.raises(ValueError, match="no orientation"):
        small.texture()
    # the fields written for a path, read back: the same lattices (to single precision)
    written = pipeline.write_path_fields(tmp_path / "tension", result.support, result.fields[10.0])
    read, ones = pipeline.read_path_texture(written)
    assert np.array_equal(ones, weights) and tx.disorientation(read, euler).max() < 1e-3
    assert cli.main(["texture", str(written), "--json", "--poles", "111"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["crystals"] == n**3 and summary["lattice"] == "cubic"
    assert summary["pole_figures"]["111"] == pytest.approx(
        tx.pole_figure(euler, (1, 1, 1)).maximum, rel=1e-3
    )
    assert cli.main(["texture", str(written), "--direction", "z"]) == 0
    assert f"the lattices of {n**3} crystals" in capsys.readouterr().out
    # on a mesh: the volumes of its elements; the elements that are not crystals left out
    mesh = box_mesh((2, 2, 2))
    angles = random_orientations(rng, mesh.n_elements)
    angles[::3] = np.nan
    on_mesh = pipeline.write_path_fields(tmp_path / "mesh", mesh, {"orientation": angles})
    read, volumes = pipeline.read_path_texture(on_mesh)
    kept = ~np.isnan(angles[:, 0])
    assert volumes == pytest.approx(mesh.volumes()[kept], rel=1e-6)
    assert tx.disorientation(read, angles[kept]).max() < 1e-3
    for bad, message in (("tension.txt", "a .vti or a .vtu"), ("none.vti", "no orientation")):
        if bad.endswith(".vti"):
            pipeline.write_path_fields(tmp_path / "none", result.support, small_fields)
        with pytest.raises(ValueError, match=message):
            pipeline.read_path_texture(tmp_path / bad)
    with pytest.raises(ValueError, match="no fields"):
        NonlinearFFTProblem(image, {0: crystal}, grains=grains,
                            orientations=orientations).run(path).texture()  # fmt: skip
