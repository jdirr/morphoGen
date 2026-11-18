"""Foams as level sets of distance fields (Sonon et al. 2012; Kilingar et al. 2019;
Degdeg & Dirrenberger 2026): the struts on the edges of the tessellation, their
thickening towards the nodes, the curvature of their cross-sections, the calibration
of the solid fraction, closed cells, the open copper foam of Degdeg & Dirrenberger, and
the calibration of the seeds on the sections of the pores."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import morphogen as mg
from morphogen import analysis as an
from morphogen.distributions import Uniform
from morphogen.models.levelset_foam import DistanceFieldFoam

BOX = mg.Domain.cube(1.0, periodic=True)


def _bcc(n=2, radius=0.2):
    """Seeds on a body-centred cubic lattice of n x n x n cells in the unit box: the
    tessellation of Kelvin's truncated octahedra."""
    a = 1.0 / n
    corners = np.stack(np.meshgrid(*[np.arange(n) * a] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    centres = np.vstack([corners, corners + 0.5 * a])
    return centres, np.full((len(centres), 3), radius * a)


def _kelvin(**kwargs):
    centres, axes = _bcc()
    foam = DistanceFieldFoam(BOX, centres, axes, BOX.lower, BOX.lengths, 0.2, **kwargs)
    grid = mg.domain.Grid(BOX, (64, 64, 64))
    pts = np.stack(np.meshgrid(*[grid.centers(a) for a in range(3)], indexing="ij"), axis=-1)
    foam.learn_struts(*foam.distances(pts.reshape(-1, 3)))
    return foam


def test_struts_on_the_edges_of_kelvin_cells():
    foam = _kelvin(c1=50.0, c2=1.0)
    a = 0.5  # lattice constant: vertices (0, a/4, a/2) and permutations; seeds at 0, a/2
    edge_middle = np.array([[a / 8, a / 8, a / 2]])  # equidistant to three seeds
    vertex = np.array([[0.0, a / 4, a / 2]])
    seed = np.array([[a / 2, a / 2, a / 2]])
    square_face = np.array([[0.0, 0.0, a / 2]])  # centre of a square face
    assert foam.contains(edge_middle)[0] and foam.contains(vertex)[0]
    assert not foam.contains(seed)[0] and not foam.contains(square_face)[0]
    # the position along the struts: 0 at mid-span, 1/2 at the nodes
    dn, nn = foam.distances(np.vstack([edge_middle, vertex + np.array([1e-4, 0.0, 0.0])]))
    xi = foam._position(dn, nn)
    assert xi[0] == pytest.approx(0.0, abs=0.02) and xi[1] == pytest.approx(0.5, abs=0.02)


def test_thickening_towards_the_nodes():
    # the width of a strut scales as t O_s (O_p is linear near an edge between equal seeds)
    plain, thick = _kelvin(), _kelvin(c1=50.0, c2=1.0)
    a = 0.5
    start, end = np.array([0.0, a / 4, a / 2]), np.array([a / 4, 0.0, a / 2])  # one edge
    normal = np.array([0.0, 0.0, 1.0])  # across the square face
    widths = {}
    for name, foam in (("plain", plain), ("thick", thick)):
        for s in (0.5, 0.9):
            p = start + s * (end - start)
            r = np.linspace(0.0, 0.1, 2001)
            inside = foam.contains(p + r[:, None] * normal)
            widths[name, s] = r[np.argmin(inside)]
    ratio_mid = widths["thick", 0.5] / widths["plain", 0.5]
    ratio_node = widths["thick", 0.9] / widths["plain", 0.9]
    assert ratio_mid == pytest.approx(1.0, abs=0.05)
    assert ratio_node > 1.3  # A(xi) grows to 4.375 at the nodes: sqrt, 2.09 in width


@pytest.mark.slow
def test_curvature_and_calibration():
    fractions = []
    for kc in (0.5, 0.0, -0.7):
        foam = mg.models.LevelSetFoam(seed_radius=0.1, thickness=0.3, curvature=kc).generate(
            BOX, seed=1
        )
        fractions.append(an.volume_fraction(foam.voxelize((48, 48, 48))))
    assert fractions[0] < fractions[1] < fractions[2]  # concave, flat, convex sections
    model = mg.models.LevelSetFoam(
        seed_radius=0.1, volume_fraction=0.12, curvature=-0.7, c1=50.0, c2=1.0
    )
    foam = model.generate(BOX, seed=2)
    assert an.volume_fraction(foam.voxelize((80, 80, 80))) == pytest.approx(0.12, abs=0.006)
    comp = an.connected_components(foam.voxelize((80, 80, 80)), 1)
    assert comp.percolating == {"x": True, "y": True, "z": True}
    # the same realization from its JSON geometry
    again = DistanceFieldFoam._from_geometry(BOX, foam._geometry())
    pts = np.random.default_rng(0).random((500, 3))
    np.testing.assert_array_equal(again.contains(pts), foam.contains(pts))


def test_closed_cells():
    model = mg.models.LevelSetFoam(
        seed_radius=0.12, thickness=0.2, cells="closed", wall_thickness=0.5
    )
    foam = model.generate(BOX, seed=3)
    img = foam.voxelize((96, 96, 96))
    pores = an.connected_components(img, 0)
    # every cell is a closed pore (the cells cut by the faces merged across them)
    assert pores.n_components == pytest.approx(len(foam.centres), rel=0.1)
    assert an.volume_fraction(img) > an.volume_fraction(
        mg.models.LevelSetFoam(seed_radius=0.12, thickness=0.2)
        .generate(BOX, seed=3)
        .voxelize((96, 96, 96))
    )


@pytest.mark.slow
def test_open_copper_foam_of_degdeg_and_dirrenberger():
    """Seeds of short semi-axes 87.1 to 145.2 um, b = 1.22 a, c = 3.22 a; K_c = -0.7,
    c1 = 50, c2 = 1, t set by the solid fraction 0.261 of the tomography: the mean
    ligament thickness (not fitted) is that of their model, 72.8 um (71.1 +- 21.1 um
    measured), and the pores are elongated along z."""
    model = mg.models.LevelSetFoam(
        seed_radius=Uniform(0.0871, 0.1452),
        aspect=(1.22, 3.22),
        packing_fraction=0.6,
        volume_fraction=0.261,
        curvature=-0.7,
        c1=50.0,
        c2=1.0,
    )
    foam = model.generate(mg.Domain.cube(3.0, periodic=True), seed=3)
    img = foam.voxelize(voxel_size=0.025)  # their grid
    assert an.volume_fraction(img) == pytest.approx(0.261, abs=0.004)
    thickness = an.local_thickness(img)[np.asarray(img.data) == 1]
    assert 1e3 * thickness.mean() == pytest.approx(72.8, rel=0.06)
    assert 1e3 * thickness.std() == pytest.approx(21.1, rel=0.3)
    chords = [an.chord_lengths(img, 0, a).mean() for a in range(3)]
    assert 1.05 < chords[1] / chords[0] < 1.35 and 2.3 < chords[2] / chords[0] < 3.3


@pytest.mark.slow
def test_calibrate_seeds_on_the_sections_of_known_seeds():
    domain = mg.Domain.cube(1.6, periodic=True)
    truth = mg.models.LevelSetFoam(
        seed_radius={"type": "uniform", "low": 0.07, "high": 0.12},
        aspect=(1.3, 1.8),
        packing="dense",
        packing_fraction=0.6,
        volume_fraction=0.26,
    )
    img = truth.generate(domain, seed=1).voxelize((64, 64, 64))
    target = [an.section_sizes(img, 0, a, separate=True) for a in range(3)]
    # round seeds 20 % too small, another realization: the scale and the shape factors
    start = dataclasses.replace(
        truth, seed_radius={"type": "uniform", "low": 0.056, "high": 0.096}, aspect=(1.0, 1.0)
    )
    c = mg.models.calibrate_seeds(
        start, domain, target, shape=(64, 64, 64), seed=2, fit_shape=True, tol=0.02
    )
    assert c.residual < 0.025 and len(c.history) <= 6
    assert c.scale == pytest.approx(1.25, rel=0.03)
    np.testing.assert_allclose(c.aspect, (1.3, 1.8), rtol=0.04)
    assert c.model.seed_radius.to_config()["low"] == pytest.approx(0.056 * c.scale)
    assert c.history[0][:2] == (1.0, (1.0, 1.0))
    # the scale alone, from the sections normal to z
    c = mg.models.calibrate_seeds(
        dataclasses.replace(start, aspect=(1.3, 1.8)),
        domain,
        {2: target[2].mean_radius()},
        shape=(64, 64, 64),
        seed=1,
        tol=0.02,
    )
    assert c.scale == pytest.approx(1.25, rel=0.03) and c.aspect == (1.3, 1.8)
    with pytest.raises(ValueError, match="fit_shape"):
        mg.models.calibrate_seeds(start, domain, {2: 0.1}, shape=(64, 64, 64), fit_shape=True)
