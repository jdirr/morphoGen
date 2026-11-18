"""Morphological measurements on synthetic images with known answers."""

import math

import numpy as np
import pytest

import morphogen as mg
from morphogen import analysis
from morphogen.analysis.surface import DIRECTIONS_3, DIRECTIONS_13, crofton_weights


def _image(data, spacing=1.0, periodic=False):
    return mg.VoxelImage(np.asarray(data), spacing=spacing, periodic=periodic)


def test_fractions():
    data = np.zeros((4, 4, 4), np.uint8)
    data[:2] = 1
    data[0, 0, 0] = 2
    img = _image(data)
    assert analysis.volume_fraction(img, 1) == pytest.approx(31 / 64)
    assert analysis.phase_fractions(img) == pytest.approx({0: 0.5, 1: 31 / 64, 2: 1 / 64})
    frac = _image(np.full((3, 3, 3), 0.25, np.float32))
    assert analysis.volume_fraction(frac) == pytest.approx(0.25)
    assert analysis.volume_fraction(frac, 0) == pytest.approx(0.75)


def test_crofton_weights():
    for dirs in (DIRECTIONS_3, DIRECTIONS_13):
        w = crofton_weights(dirs)
        assert w.sum() == pytest.approx(1.0) and np.all(w > 0)
    np.testing.assert_allclose(crofton_weights(DIRECTIONS_3), 1 / 3)
    w = crofton_weights(DIRECTIONS_13)
    # symmetric weights: axes, face diagonals and body diagonals
    assert np.ptp(w[:3]) < 1e-12 and np.ptp(w[3:9]) < 1e-12 and np.ptp(w[9:]) < 1e-12


def test_specific_surface_of_planes_and_spheres():
    # two planes normal to x in a periodic box of edge 32: S_V = 2 / 32
    data = np.zeros((32, 32, 32), np.uint8)
    data[8:24] = 1
    img = _image(data, periodic=True)
    # P_L(d) = |cos(d, x)| * 2 / 32 exactly; the weighted sum over directions approximates
    # the spherical mean of |cos| (1/2): about -7 % for planes, the worst anisotropy
    d = np.array(DIRECTIONS_13, float)
    cos = np.abs(d[:, 0]) / np.linalg.norm(d, axis=1)
    expected13 = 2 * np.sum(crofton_weights(DIRECTIONS_13) * cos) * 2 / 32
    assert analysis.specific_surface(img) == pytest.approx(expected13, rel=1e-12)
    assert expected13 == pytest.approx(2 / 32, rel=0.08)
    assert analysis.specific_surface(img, directions=3) == pytest.approx(2 / 32 * 2 / 3)
    # a large ball: S_V = 4 pi R^2 / V (periodic image: the test lines cover the whole box;
    # non-periodic estimates refer to the slightly smaller box spanned by voxel centres)
    n, r = 80, 25.0
    x = np.arange(n) + 0.5 - n / 2
    ball = (x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2) <= r * r
    s = analysis.specific_surface(_image(ball.astype(np.uint8), spacing=0.1, periodic=True))
    assert s == pytest.approx(4 * math.pi * (r * 0.1) ** 2 / (n * 0.1) ** 3, rel=0.01)
    with pytest.raises(ValueError):
        analysis.specific_surface(_image(np.zeros((4, 4, 4), np.float32)))


def test_covariance_estimators():
    rng = np.random.default_rng(0)
    data = (rng.random((40, 30, 20)) < 0.3).astype(np.uint8)  # white noise: C(h) = p^2, h > 0
    for periodic in (True, False):
        cov = analysis.covariance(_image(data, periodic=periodic), max_lag=6)
        p = data.mean()
        for axis in "xyz":
            assert cov.values[axis][0] == pytest.approx(p)
            np.testing.assert_allclose(cov.values[axis][1:], p * p, atol=0.01)
    # exact values on a periodic stripe pattern along x (period 4: 1 1 0 0)
    stripes = np.zeros((8, 4, 4), np.uint8)
    stripes[0::4] = stripes[1::4] = 1
    cov = analysis.covariance(_image(stripes, periodic=True), max_lag=4)
    np.testing.assert_allclose(cov.values["x"], [0.5, 0.25, 0.0, 0.25, 0.5])
    np.testing.assert_allclose(cov.values["y"], 0.5)
    cmap = analysis.covariance_map(_image(stripes, periodic=True))
    assert cmap[4, 2, 2] == pytest.approx(0.5) and cmap[6, 2, 2] == pytest.approx(0.0)


def test_connected_components_and_percolation():
    data = np.zeros((10, 10, 10), np.uint8)
    data[1:3, 1:3, 1:3] = 1  # isolated cube
    data[:, 5, 5] = 1  # rod along x touching both faces
    img = _image(data)
    comp = analysis.connected_components(img)
    assert comp.n_components == 2
    assert comp.percolating == {"x": True, "y": False, "z": False}
    assert comp.spanning_fraction == pytest.approx(10 / 18)
    # a cluster crossing the periodic boundary counts once
    data = np.zeros((10, 10, 10), np.uint8)
    data[0:2, 3, 3] = data[8:10, 3, 3] = 1
    assert analysis.connected_components(_image(data, periodic=True)).n_components == 1
    assert analysis.connected_components(_image(data, periodic=False)).n_components == 2
    # 26-connectivity joins diagonal neighbours
    diag = np.zeros((4, 4, 4), np.uint8)
    diag[0, 0, 0] = diag[1, 1, 1] = 1
    assert analysis.connected_components(_image(diag), connectivity=26).n_components == 1
    assert analysis.connected_components(_image(diag), connectivity=6).n_components == 2


def test_cell_statistics():
    labels = np.zeros((4, 2, 2), np.int32)
    labels[2:] = 1
    labels[3, 1, 1] = 2
    img = _image(labels, spacing=0.5)
    np.testing.assert_allclose(analysis.cell_volumes(img), np.array([8, 7, 1]) * 0.125)
    pairs = analysis.cell_adjacency(img).tolist()
    assert pairs == [[0, 1], [1, 2]]
    assert analysis.cell_neighbors(img).tolist() == [1, 2, 1]
    periodic = _image(labels, periodic=True)
    assert analysis.cell_adjacency(periodic).tolist() == [[0, 1], [0, 2], [1, 2]]
    # negative labels (no cell, e.g. an untransformed matrix) are ignored
    labels[0] = -1
    img = _image(labels)
    np.testing.assert_allclose(analysis.cell_volumes(img), [4, 7, 1])
    assert analysis.cell_adjacency(img).tolist() == [[0, 1], [1, 2]]
    assert analysis.cell_neighbors(_image(np.full((2, 2, 2), -1))).tolist() == []


def _sierpinski_carpet(level):
    carpet = np.ones((1, 1), dtype=np.uint8)
    for _ in range(level):
        empty = np.zeros_like(carpet)
        carpet = np.block(
            [[carpet, carpet, carpet], [carpet, empty, carpet], [carpet, carpet, carpet]]
        )
    return carpet[:, :, None]


def test_box_counting_dimension():
    cube = np.ones((32, 32, 32), dtype=np.uint8)
    assert analysis.box_dimension(cube) == pytest.approx(3.0)
    plane = np.zeros((32, 32, 32), dtype=np.uint8)
    plane[:, :, 7] = 1
    assert analysis.box_dimension(plane) == pytest.approx(2.0)
    line = np.zeros((32, 32, 32), dtype=np.uint8)
    line[3, 5, :] = 1
    assert analysis.box_dimension(_image(line)) == pytest.approx(1.0)
    carpet = _sierpinski_carpet(4)  # 81 x 81: D = log 8 / log 3 exactly with sizes 3^k
    out = analysis.box_counting(carpet, sizes=[1, 3, 9, 27])
    assert out["counts"] == [8**4, 8**3, 8**2, 8]
    assert out["dimension"] == pytest.approx(math.log(8) / math.log(3))
    out = analysis.measure(_image(cube), ["fractal_dimension"])
    assert out == {"box_dimension": pytest.approx(3.0)}
    with pytest.raises(ValueError, match="exceed"):
        analysis.box_counting(cube, sizes=[1, 64])


def test_measure_registry():
    micro = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3).generate(
        mg.Domain.cube(8.0), seed=1
    )
    img = micro.voxelize(shape=40)
    out = analysis.measure(
        img, ["volume_fraction", "specific_surface", "connectivity", "phase_fractions"]
    )
    assert {
        "volume_fraction",
        "specific_surface",
        "n_components",
        "percolates_x",
        "fraction_0",
        "fraction_1",
    } <= set(out)
    with pytest.raises(ValueError, match="unknown measure"):
        analysis.measure(img, ["porosity"])
    with pytest.raises(ValueError, match="tessellation"):
        analysis.measure(img, ["cells"])
