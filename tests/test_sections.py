"""Sections computed on their planes (morphogen.viz.sections): the slices of the voxel images
at the same resolution, for every model, without the volumes; their images, and the
measures of the level sets."""

import functools
import math
import sys
import time
import tracemalloc

import numpy as np
import pytest

import morphogen as mg
from morphogen import analysis
from morphogen.models import Laminate, PrimitiveUnion, model_from_config
from morphogen.primitives import Spheres
from morphogen.viz import sections as vs
from morphogen.viz.png import PALETTE, label_colors, slice_of
from test_models_base import CONFIGS


@functools.cache
def realization(name: str):
    """The realization of a model of CONFIGS used by these tests (a cube of edge 8)."""
    params, periodic = CONFIGS[name]
    model = model_from_config({"type": name, **params})
    return model.generate(mg.Domain.cube(8.0, periodic=periodic), seed=3)


def references(micro, grid):
    """Phase, level set (None without one) and labels (None) of a structure on a grid."""
    phase = micro.voxelize(grid=grid).data
    level = None
    if vs.has_level_set(micro):
        try:
            level = micro.level_set(grid=grid).data
        except NotImplementedError:
            level = None
    labels = micro.labels(grid=grid).data if micro.has_labels else None
    return phase, level, labels


def recorded_grids(micro, monkeypatch) -> list:
    """The grids that the methods of ``micro`` receive from now on."""
    seen = []
    for method in ("voxelize", "level_set", "labels"):
        if not hasattr(micro, method):
            continue
        original = getattr(micro, method)

        def record(*args, _original=original, **kwargs):
            seen.append(kwargs.get("grid"))
            return _original(*args, **kwargs)

        monkeypatch.setattr(micro, method, record)
    return seen


def assert_section_is_slice(sec, refs, index):
    phase, level, labels = refs
    axis = sec.axis
    assert sec.phase.dtype == np.uint8
    np.testing.assert_array_equal(sec.phase, slice_of(phase, axis, index))
    assert (sec.level_set is None) == (level is None)
    assert (sec.coverage is None) == (level is None)
    if level is not None:
        assert sec.level_set.dtype == np.float32 and sec.coverage.dtype == np.float32
        ref = slice_of(level, axis, index)
        np.testing.assert_allclose(sec.level_set, ref, rtol=1e-6, atol=1e-6)
        assert 0.0 <= sec.coverage.min() and sec.coverage.max() <= 1.0
    assert (sec.labels is None) == (labels is None)
    if labels is not None:
        assert sec.labels.dtype == np.int32
        np.testing.assert_array_equal(sec.labels, slice_of(labels, axis, index))


# -- grids -------------------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_plane_grids_are_slices_of_the_voxel_grid(axis):
    domain = mg.Domain(size=(6.0, 9.0, 4.5), origin=(-1.0, 2.0, 0.5), periodic=False)
    res = 37
    full = vs.volume_grid(domain, axis, res)
    u, v = (a for a in range(3) if a != axis)
    assert max(full.shape[u], full.shape[v]) == res
    h = full.spacing
    assert abs(h[u] - h[v]) <= max(h[u], h[v]) / res  # square up to the rounding of the edges
    n = full.shape[axis]
    points = full.center_points().reshape(*full.shape, 3)
    for fraction, index in ((0.0, 0), (0.5, n // 2), (1.0, n - 1), (0.43, math.floor(0.43 * n))):
        assert vs.plane_index(domain, axis, fraction, res) == index
        grid = vs.plane_grid(domain, axis, fraction, res)
        assert grid.shape[axis] == 1
        assert [grid.shape[a] for a in (u, v)] == [full.shape[a] for a in (u, v)]
        np.testing.assert_array_equal(grid.spacing, full.spacing)
        # the same cell centres, bit for bit (those of slice_of at the middle)
        np.testing.assert_array_equal(
            grid.center_points().reshape(*grid.shape, 3),
            np.take(points, [index], axis=axis),
        )
        position = grid.centers(axis)[0]
        assert domain.lower[axis] < position < domain.upper[axis]
    with pytest.raises(ValueError, match="axis"):
        vs.plane_grid(domain, 3)
    with pytest.raises(ValueError, match="fraction"):
        vs.plane_grid(domain, 2, 1.5)
    with pytest.raises(ValueError, match="resolution"):
        vs.plane_grid(domain, 2, 0.5, 0)


# -- every model: the slices of its voxel image, without the volume -----------------------------


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_sections_are_the_slices_of_the_voxel_images(name, monkeypatch):
    micro = realization(name)
    res = 24
    full = vs.volume_grid(micro.domain, 2, res)  # a cube: the same grid for the three axes
    assert full.shape == (res, res, res)
    refs = references(micro, full)
    seen = recorded_grids(micro, monkeypatch)
    for axis in (0, 1, 2):
        for fraction in (0.5, 0.0, 1.0, 0.37):
            sec = vs.section(micro, axis, fraction, res)
            assert sec.sampling == "plane"
            assert sec.index == vs.plane_index(micro.domain, axis, fraction, res)
            assert_section_is_slice(sec, refs, sec.index)
            assert sec.position == full.centers(axis)[sec.index]
            # sampled on the plane: never a grid of more than one cell along the axis
            assert seen and all(g.shape[axis] == 1 for g in seen)
            seen.clear()


def test_sections_of_lattice_fields_on_their_lattice():
    micro = realization("cellular_automaton")
    res = micro.shape[0]
    assert vs.volume_grid(micro.domain, 2, res).shape == micro.shape
    phases = micro.voxelize().data  # the cells themselves
    for axis in (0, 1, 2):
        sec = vs.section(micro, axis, 0.5, res)
        np.testing.assert_array_equal(sec.phase, slice_of(phases, axis))


@pytest.mark.parametrize("name", ["tpms", "hardcore_spheres", "voronoi", "gaussian_excursion"])
def test_large_sections_do_not_build_the_volume(name, monkeypatch):
    micro = realization(name)
    res = 2048
    vs.section(micro, 2, 0.5, 32)  # compilation of the kernels
    seen = recorded_grids(micro, monkeypatch)
    tracemalloc.start()
    start = time.perf_counter()
    sec = vs.section(micro, 2, 0.5, res)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert sec.shape == (res, res) and sec.sampling == "plane"
    assert seen and all(g.shape == (res, res, 1) for g in seen)
    # far less than the volume (8 GiB of uint8): the coordinates of the points and their
    # temporaries at most (128 bytes per pixel)
    assert peak < 128 * res * res, f"{peak / 2**20:.0f} MiB"
    # building the volume would take minutes; one thread of a loaded machine (the parallel
    # tests) may take half a minute for the section itself
    assert elapsed < 90.0
    print(f"{name}: {res}^2 section in {elapsed:.2f} s, traced peak {peak / 2**20:.0f} MiB")


# -- domains ------------------------------------------------------------------------------------

ANISOTROPIC = {
    "boolean_spheres": {"radius": 0.7, "volume_fraction": 0.2},
    "tpms": {"surface": "gyroid", "cell_size": 1.5, "volume_fraction": 0.3},
    "voronoi": {"n_cells": 25, "phase_fractions": [0.4, 0.6]},
    "gaussian_excursion": {"correlation_length": 0.8, "volume_fraction": 0.35},
    "random_laminate": {"thickness": 0.6, "spacing": 1.0, "hole_fraction": 0.2, "hole_radius": 0.5},
    "honeycomb": {"cell_size": 1.5, "wall_thickness": 0.2, "arrangement": "voronoi"},
    "dead_leaves": {"radius": {"type": "uniform", "low": 0.6, "high": 1.0}},
    "transparent_dead_leaves": {"radius": 0.9, "transparency": 0.6, "volume_fraction": 0.4},
    "cellular_automaton": {"rule": "conserved", "volume_fraction": 0.4, "steps": 3},
}


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("name", sorted(ANISOTROPIC))
def test_non_periodic_and_anisotropic_domains(name, periodic):
    domain = mg.Domain(size=(6.0, 9.0, 4.5), origin=(-1.0, 2.0, 0.5), periodic=periodic)
    micro = model_from_config({"type": name, **ANISOTROPIC[name]}).generate(domain, seed=5)
    res = 30
    lo, hi = domain.lower, domain.upper
    for axis in (0, 1, 2):
        full = vs.volume_grid(domain, axis, res)
        refs = references(micro, full)
        u, v = (a for a in range(3) if a != axis)
        for fraction in (0.0, 0.5, 1.0):
            sec = vs.section(micro, axis, fraction, res)
            assert sec.shape == (full.shape[v], full.shape[u])
            assert_section_is_slice(sec, refs, sec.index)
            assert lo[axis] < sec.position < hi[axis]
            assert sec.extent == (lo[u], hi[u], lo[v], hi[v])
            assert sec.names == ("xyz"[u], "xyz"[v])
            assert sec.pixel == pytest.approx(max(domain.size[u], domain.size[v]) / res, rel=0.05)


def test_planar_structures_have_one_section():
    thin = mg.Domain(size=(8.0, 8.0, 0.1), periodic=False)
    micro = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3).generate(thin, seed=1)
    assert vs.is_planar(micro, 64)
    (sec,) = vs.sections(micro, 64)
    assert sec.title == "x-y" and sec.shape == (64, 64)
    full = vs.volume_grid(thin, 2, 64)
    assert full.shape == (64, 64, 1)
    assert_section_is_slice(sec, references(micro, full), 0)
    # a planar automaton: one cell through the thickness of a cube
    flat = mg.models.CellularAutomatonModel(rule="conserved", volume_fraction=0.4, dimension=2)
    micro = flat.generate(mg.Domain.cube(8.0), seed=2)
    assert micro.shape[2] == 1 and vs.is_planar(micro, 64)
    assert [s.title for s in vs.sections(micro, 64)] == ["x-y"]
    cube = realization("tpms")
    assert [s.title for s in vs.sections(cube, 32)] == ["y-z", "x-z", "x-y"]


def test_structures_that_fall_back_to_the_volume(monkeypatch):
    # a coating of a TPMS: the distance transform of its voxels
    model = mg.models.Coating(
        model={"type": "tpms", "surface": "gyroid", "cell_size": 4.0, "volume_fraction": 0.3},
        thickness=0.3,
    )
    micro = model.generate(mg.Domain.cube(8.0), seed=1)
    res = 32
    full = vs.volume_grid(micro.domain, 2, res)
    refs = references(micro, full)
    for axis in (0, 2):
        sec = vs.section(micro, axis, 0.5, res)
        assert sec.sampling == "volume"
        assert_section_is_slice(sec, refs, sec.index)
    # beyond the largest volume: a coarser image, enlarged to the pixels asked for
    monkeypatch.setattr(vs, "MAX_VOLUME", 16**3)
    sec = vs.section(micro, 2, 0.5, res)
    assert sec.sampling == "coarse" and sec.shape == (res, res)
    coarse = micro.voxelize(shape=16).data
    np.testing.assert_array_equal(sec.phase, np.repeat(np.repeat(slice_of(coarse, 2), 2, 0), 2, 1))
    # a hollow combination with periodic dead leaves (their kernels need the whole box)
    leaves = {"type": "dead_leaves", "radius": 1.0, "phase_fractions": [0.5, 0.5]}
    spheres = {"type": "boolean_spheres", "radius": 0.8, "volume_fraction": 0.1}
    combined = mg.models.Combination(operation="union", models=[leaves, spheres])
    micro = combined.generate(mg.Domain.cube(8.0), seed=1)
    monkeypatch.setattr(vs, "MAX_VOLUME", 256**3)
    sec = vs.section(micro, 1, 0.5, 24)
    assert sec.sampling == "volume"
    assert_section_is_slice(sec, references(micro, vs.volume_grid(micro.domain, 1, 24)), 12)
    # dead leaves alone are sampled on the plane, in both kinds of boxes
    assert vs.section(realization("dead_leaves"), 0, 0.5, 24).sampling == "plane"


# -- images ---------------------------------------------------------------------------------


def disc_section(res: int = 64, radius: float = 1.3, z: float = 2.0):
    domain = mg.Domain.cube(4.0, periodic=False)
    micro = PrimitiveUnion(domain, spheres=Spheres(np.array([[2.0, 2.0, z]]), np.array([radius])))
    return vs.section(micro, 2, 0.5, res)


def test_coverage_is_the_fraction_of_the_pixels_in_phase_1():
    sec = disc_section()
    h = sec.pixel
    rho2 = 1.3**2 - (sec.position - 2.0) ** 2  # the disc cut by the plane
    assert float(sec.coverage.sum()) * h * h == pytest.approx(math.pi * rho2, rel=2e-3)
    # a straight interface: the exact fractions of the pixels
    x = (np.arange(40) + 0.5) / 40
    phi = np.broadcast_to(x - 0.3137, (25, 40))  # phase 1 on the left
    exact = np.clip(0.5 - (x - 0.3137) * 40, 0.0, 1.0)
    np.testing.assert_allclose(vs.pixel_coverage(phi), np.broadcast_to(exact, (25, 40)), atol=1e-5)
    # independent of the scale of the level set; by its sign where it is flat
    np.testing.assert_allclose(vs.pixel_coverage(7.0 * phi), vs.pixel_coverage(phi), atol=1e-6)
    np.testing.assert_array_equal(vs.pixel_coverage(np.full((3, 4), -2.0)), np.ones((3, 4)))
    # a strip thinner than a pixel keeps a partial coverage
    strip = np.abs(x - x[20]) - 0.2 / 40
    cov = vs.pixel_coverage(np.broadcast_to(strip, (5, 40)))
    assert 0.0 < cov[2, 20] < 1.0 and cov[2, 18] == 0.0


def test_render_blends_the_boundaries():
    sec = disc_section()
    rgb = vs.render(sec)
    assert rgb.shape == (*sec.shape, 3) and rgb.dtype == np.uint8
    inside, outside = sec.coverage == 1.0, sec.coverage == 0.0
    boundary = (sec.coverage > 0.05) & (sec.coverage < 0.95)
    assert inside.sum() > 100 and outside.sum() > 100 and boundary.sum() > 50
    assert (rgb[inside] == PALETTE[1]).all() and (rgb[outside] == PALETTE[0]).all()
    lo = np.minimum(PALETTE[0], PALETTE[1]).astype(int)
    hi = np.maximum(PALETTE[0], PALETTE[1]).astype(int)
    mid = rgb[boundary].astype(int)
    assert ((mid >= lo) & (mid <= hi)).all()
    assert not (mid == PALETTE[0]).all(axis=1).any() and not (mid == PALETTE[1]).all(axis=1).any()
    # the contour darkens the boundary only
    lined = vs.render(sec, contour=True)
    np.testing.assert_array_equal(lined[inside | outside], rgb[inside | outside])
    assert (lined[boundary].astype(int).sum(axis=1) < mid.sum(axis=1)).all()


def test_render_of_phases_and_labels():
    sec = vs.section(realization("voronoi"), 2, 0.5, 48)
    assert sec.coverage is None and sec.labels is not None
    np.testing.assert_array_equal(vs.render(sec), PALETTE[sec.phase])
    np.testing.assert_array_equal(vs.render(sec, "labels"), label_colors(sec.labels))
    lined = vs.render(sec, "labels", contour=True)
    changes = np.zeros(sec.shape, dtype=bool)  # pixels unlike the next one along a row or column
    changes[:, :-1] |= sec.labels[:, :-1] != sec.labels[:, 1:]
    changes[:-1, :] |= sec.labels[:-1, :] != sec.labels[1:, :]
    assert changes.any() and not changes.all()
    assert (lined[changes] == vs.CONTOUR).all()
    np.testing.assert_array_equal(lined[~changes], label_colors(sec.labels)[~changes])
    with pytest.raises(ValueError, match="labels"):
        vs.render(vs.section(realization("tpms"), 2, 0.5, 16), "labels")
    with pytest.raises(ValueError, match="kind"):
        vs.render(sec, "colours")


SEVERAL = {
    "coating": {"model": {"type": "hardcore_spheres", "radius": 1.0, "volume_fraction": 0.2},
                "thickness": 0.3},
    "dead_leaves": {"radius": 1.2, "phase_fractions": [0.4, 0.3, 0.3]},
    "osteons": {"radius": 1.6, "coverage": 0.6, "canal_ratio": 0.25, "cement_thickness": 0.2},
    "voronoi": {"n_cells": 30, "phase_fractions": [0.5, 0.3, 0.2]},
    "combination": {
        "operation": "inclusions",
        "models": [
            {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.2},
            {"type": "boolean_cylinders", "radius": 0.3, "length": 4.0, "volume_fraction": 0.1},
        ],
    },
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(SEVERAL))
def test_sections_of_several_phases(name):
    """The sections of the structures of several phases keep the fraction of each phase
    in each pixel, from the implicit function of each phase: the fractions sum to one,
    the phase of the centre of a pixel has the largest, their means are the fractions of
    the pixels, and the image blends the colours of the phases at the interfaces only."""
    model = model_from_config({"type": name, **SEVERAL[name]})
    micro = model.generate(mg.Domain.cube(8.0, periodic=True), seed=3)
    phases = micro.implicit_phases()
    assert len(phases) > 2
    sec = vs.section(micro, 2, 0.5, 160)
    assert sec.sampling == "plane" and sec.level_set is None and sec.coverage is None
    cover = sec.coverages
    assert sorted(cover) == phases
    stack = np.stack([cover[k] for k in phases])
    assert stack.dtype == np.float32 and stack.min() >= 0.0
    np.testing.assert_allclose(stack.sum(axis=0), 1.0, atol=1e-5)
    np.testing.assert_array_equal(np.array(phases)[np.argmax(stack, axis=0)], sec.phase)
    for k in phases:  # the mean coverage of a phase: its fraction of the section
        assert abs(float(cover[k].mean()) - float(np.mean(sec.phase == k))) < 2e-3
    # pure pixels in the colour of their phase, the others between those of their phases
    rgb = vs.render(sec)
    pure = stack.max(axis=0) == 1.0
    assert 0.5 < pure.mean() < 1.0
    np.testing.assert_array_equal(rgb[pure], PALETTE[sec.phase][pure])
    assert (rgb[~pure] != PALETTE[sec.phase][~pure]).any(axis=1).mean() > 0.9
    # the contour darkens the interfaces only
    lined = vs.render(sec, contour=True)
    np.testing.assert_array_equal(lined[pure], rgb[pure])
    assert (lined[~pure].astype(int).sum(axis=1) <= rgb[~pure].astype(int).sum(axis=1)).all()
    assert (lined[~pure].astype(int).sum(axis=1) < rgb[~pure].astype(int).sum(axis=1)).mean() > 0.9


def test_coverages_of_phases():
    """Three phases in vertical bands, their interfaces at given places in the pixels: the
    fraction of each phase in the pixels they cross."""
    x = np.arange(12) + 0.5  # the centres of pixels of unit edge
    a, b = 4.25, 8.6  # phase 0 below a, phase 1 between a and b, phase 2 above b
    levels = {0: x - a, 1: np.maximum(a - x, x - b), 2: b - x}
    levels = {k: np.tile(f.astype(np.float32), (5, 1)) for k, f in levels.items()}
    phase = np.tile(np.where(x < a, 0, np.where(x < b, 1, 2)).astype(np.uint8), (5, 1))
    cover = vs.phase_coverages(levels, phase)
    np.testing.assert_allclose(sum(cover.values()), 1.0, atol=1e-6)
    np.testing.assert_allclose(cover[0][0, :6], [1, 1, 1, 1, 0.25, 0], atol=1e-6)
    np.testing.assert_allclose(cover[1][0, 3:10], [0, 0.75, 1, 1, 1, 0.6, 0], atol=1e-6)
    np.testing.assert_allclose(cover[2][0, 7:], [0, 0.4, 1, 1, 1], atol=1e-6)
    # where no function covers a pixel, the phase of its centre takes it
    none = vs.phase_coverages({0: np.ones((2, 2)), 1: np.ones((2, 2))}, np.array([[0, 1], [1, 0]]))
    np.testing.assert_array_equal(none[0], [[1, 0], [0, 1]])
    np.testing.assert_array_equal(none[1], [[0, 1], [1, 0]])


def test_many_phases_keep_their_plain_colours():
    """Beyond MAX_PHASES phases the sections do not evaluate one function per phase."""
    fractions = [1.0 / 12] * 12
    model = model_from_config({"type": "voronoi", "n_cells": 60, "phase_fractions": fractions})
    micro = model.generate(mg.Domain.cube(8.0, periodic=True), seed=1)
    assert len(micro.implicit_phases()) > vs.MAX_PHASES
    sec = vs.section(micro, 2, 0.5, 48)
    assert sec.coverages is None
    np.testing.assert_array_equal(vs.render(sec), PALETTE[np.minimum(sec.phase, len(PALETTE) - 1)])


def test_sections_are_displayed_like_slice_of():
    # a box of phase 1 in the corner of high x and low y: right and bottom of the image
    micro = Laminate(mg.Domain(size=(4.0, 3.0, 2.0), periodic=False), 0, [3.0], [4.0])
    sec = vs.section(micro, 2, 0.5, 40)
    assert sec.shape == (30, 40) and sec.names == ("x", "y")
    assert (sec.phase[:, 30:] == 1).all() and (sec.phase[:, :30] == 0).all()
    micro = Laminate(mg.Domain(size=(4.0, 3.0, 2.0), periodic=False), 1, [0.0], [1.0])
    sec = vs.section(micro, 2, 0.5, 40)
    assert (sec.phase[20:] == 1).all() and (sec.phase[:20] == 0).all()  # low y at the bottom


# -- measures of the level sets ---------------------------------------------------------------


def sphere(periodic: bool):
    domain = mg.Domain.cube(4.0, periodic=periodic)
    centre = (0.3, 3.8, 0.1) if periodic else (2.0, 2.1, 1.9)  # across the faces, or inside
    return PrimitiveUnion(domain, spheres=Spheres(np.array([centre]), np.array([1.2])))


@pytest.mark.parametrize("periodic", [True, False])
def test_level_set_measures_of_closed_forms(periodic):
    m = vs.level_set_measures(sphere(periodic))
    assert m["volume_fraction"] == pytest.approx(4 / 3 * math.pi * 1.2**3 / 64, rel=1e-3)
    assert m["specific_surface"] == pytest.approx(4 * math.pi * 1.2**2 / 64, rel=1e-3)
    layers = Laminate(mg.Domain.cube(8.0, periodic=periodic), 2, [0.7, 4.1], [2.2, 5.3])
    m = vs.level_set_measures(layers)
    assert m["volume_fraction"] == pytest.approx(2.7 / 8, rel=1e-6)
    assert m["specific_surface"] == pytest.approx(4 * 64 / 512, rel=1e-6)
    # hard-core spheres: the exact values of the realization and the theory of the model
    micro = mg.models.HardcoreSpheres(radius=0.8, volume_fraction=0.2).generate(
        mg.Domain.cube(8.0, periodic=periodic), seed=3
    )
    m = vs.level_set_measures(micro)
    if periodic:  # each sphere once, whole
        r = micro.spheres.radii
        assert m["volume_fraction"] == pytest.approx(np.sum(4 / 3 * np.pi * r**3) / 512, rel=5e-3)
        assert m["specific_surface"] == pytest.approx(np.sum(4 * np.pi * r**2) / 512, rel=5e-3)
    theory = micro.model.theory(micro.domain)
    assert m["volume_fraction"] == pytest.approx(theory["volume_fraction"], rel=0.03)
    assert m["specific_surface"] == pytest.approx(theory["specific_surface"], rel=0.03)


@pytest.mark.parametrize("name", ["tpms", "gaussian_excursion", "boolean_spheres", "strut_lattice"])
def test_level_set_measures_against_the_voxel_measures(name):
    micro = realization(name)
    m = vs.level_set_measures(micro, 128)
    voxels = analysis.measure(micro.voxelize(shape=256), ["phase_fractions", "specific_surface"])
    assert m["volume_fraction"] == pytest.approx(voxels["fraction_1"], rel=0.01)
    assert m["specific_surface"] == pytest.approx(voxels["specific_surface"], rel=0.015)


def test_level_set_measures_without_scikit_image(monkeypatch):
    monkeypatch.setitem(sys.modules, "skimage", None)
    monkeypatch.setitem(sys.modules, "skimage.measure", None)
    for periodic in (True, False):
        m = vs.level_set_measures(sphere(periodic))
        assert m["specific_surface"] == pytest.approx(4 * math.pi * 1.2**2 / 64, rel=2e-3)
        layers = Laminate(mg.Domain.cube(8.0, periodic=periodic), 2, [0.7, 4.1], [2.2, 5.3])
        assert vs.level_set_measures(layers)["specific_surface"] == pytest.approx(0.5, rel=1e-6)
    micro = realization("gaussian_excursion")
    m = vs.level_set_measures(micro, 128)
    voxels = analysis.measure(micro.voxelize(shape=256), ["specific_surface"])
    assert m["specific_surface"] == pytest.approx(voxels["specific_surface"], rel=0.02)


def test_level_set_measures_need_a_level_set():
    for name in ("voronoi", "osteons", "cox_spheres"):  # more phases than one level set
        with pytest.raises(ValueError, match="no level-set function"):
            vs.level_set_measures(realization(name), 16)
