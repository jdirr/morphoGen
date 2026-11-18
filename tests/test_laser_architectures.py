"""Architectures treated pixel by pixel (morphogen.laser.architectures): the majority
phase of the pixels, the opening by the spot, the measures of the treatment, the phases
of the Gauss points, the sheet of a laminate against its iso-strain proof stress, the
moduli of phases of different stiffness, and smooth tracks over a periodic cell."""

from __future__ import annotations

import math

import numpy as np
import pytest

from morphogen.laser.architectures import (
    design_membrane,
    periodic_lines,
    pixelate,
    sheet_tests,
    spot_opening,
    track_membrane,
    treatment_measures,
)
from morphogen.laser.cells import Phase, wang_phases
from morphogen.laser.materials import FlowLaw
from morphogen.laser.periodic import PeriodicCell, cell_test


def test_pixelate_takes_the_majority_and_samples_coarser_images():
    d = np.array([[0, 1], [1, 2]])
    fine = np.kron(d, np.ones((4, 4), int))
    assert np.array_equal(pixelate(fine, (2, 2)), d)
    noisy = fine.copy()
    noisy[0, 0] = 2  # one point of sixteen does not change the majority
    assert np.array_equal(pixelate(noisy, (2, 2)), d)
    assert np.array_equal(pixelate(d, (4, 4)), np.kron(d, np.ones((2, 2), int)))
    with pytest.raises(ValueError):
        pixelate(np.ones((2, 2, 2)), (2, 2))


def test_the_spot_opens_the_designs():
    d = np.zeros((8, 8), int)
    d[2:4, 2:4] = 1  # a square of the spot: kept
    d[6, 0:8] = 1  # a line one cell wide: thinner than the spot
    o = spot_opening(d, 2)
    assert o[2:4, 2:4].all() and not o[6].any()
    # periodic: a square split by the edge of the cell is a whole square; not periodic,
    # the spot overhangs the edges (a line one cell wide along an edge is kept)
    w = np.zeros((8, 8), int)
    w[0, 3:5] = w[7, 3:5] = 1
    assert spot_opening(w, 2).sum() == 4
    w[3, 0:8] = 1
    assert spot_opening(w, 2)[3].sum() == 0
    assert spot_opening(w, 2, periodic=False)[0].sum() == 2
    # the treated cells keep their phase
    m = d.copy()
    m[2:4, 2:4] = 2
    assert set(np.unique(spot_opening(m, 2))) == {0, 2}


def test_treatment_measures():
    lam = np.zeros((10, 10), int)
    lam[[2, 3, 7, 8], :] = 1
    p = treatment_measures(lam)
    assert p["hard_spans"] == (True, False) and p["soft_spans"] == (True, False)
    assert p["treated"] == pytest.approx(0.4) and p["switches"] == 0.0
    assert p["length"] == pytest.approx(0.04)  # 40 mm² by a spot of 1 mm
    assert p["time"] == pytest.approx(40.0 / 10.0 / 60.0)
    assert p["heat_affected"] == pytest.approx(4 * 10 / 60)  # the rows beside the tracks
    grid = np.zeros((9, 9), int)
    grid[::3, :] = grid[:, ::3] = 1
    p = treatment_measures(grid)
    assert p["hard_spans"] == (True, True) and p["soft_spans"] == (False, False)
    island = np.zeros((10, 10), int)
    island[4:6, 4:6] = 1
    p = treatment_measures(island)
    assert p["hard_spans"] == (False, False) and p["soft_spans"] == (True, True)
    hole = np.ones((5, 5), int)
    hole[2, 2] = 0
    assert treatment_measures(hole)["enclosed"] == 1.0


def test_gauss_points_take_the_phase_of_their_pixel():
    d = np.zeros((4, 6), int)
    d[1, 2] = 1
    cell = PeriodicCell.build((6.0, 0.0), (0.0, 4.0), element=0.5, centre=(3.0, 2.0))
    mem = design_membrane(cell.mesh, d, phases=wang_phases())
    pts = cell.mesh.gauss()[0]
    inside = (pts[..., 0] > 2) & (pts[..., 0] < 3) & (pts[..., 1] > 1) & (pts[..., 1] < 2)
    hard = wang_phases()[1].law.sigma0
    assert np.allclose(mem.law[..., 1][inside], hard)
    assert np.allclose(mem.law[..., 1][~inside], wang_phases()[0].law.sigma0)
    with pytest.raises(ValueError):
        design_membrane(cell.mesh, d + 2)


def _uniaxial(law, strain, young=210000.0):
    """The stress of a phase in uniaxial tension at a total strain (bisection)."""
    if young * strain <= law.sigma0:
        return young * strain
    lo, hi = 0.0, strain
    for _ in range(80):
        p = 0.5 * (lo + hi)
        if young * (strain - p) > law.stress(p):
            lo = p
        else:
            hi = p
    return young * (strain - 0.5 * (lo + hi))


def test_the_sheet_of_a_laminate():
    soft, hard = wang_phases(math.inf)
    lam = np.zeros((10, 10), int)
    lam[[2, 3, 7, 8], :] = 1  # 40 % of hard tracks along x
    s = sheet_tests(lam, phases=(soft, hard), angles=(0.0, 90.0), max_strain=0.06)
    # along the tracks: the 0.2 % proof stress of the iso-strain mixture
    lo, hi = 0.002, 0.02
    for _ in range(80):
        e = 0.5 * (lo + hi)
        mix = 0.6 * _uniaxial(soft.law, e) + 0.4 * _uniaxial(hard.law, e)
        lo, hi = (e, hi) if mix > 210000.0 * (e - 0.002) else (lo, e)
    assert s["directions"][0.0]["yield_strength"] == pytest.approx(mix, rel=0.02)
    across = s["directions"][90.0]
    assert across["yield_strength"] < 0.7 * mix  # the soft rows yield first
    assert across["r_value"] < 0.2  # the tracks hold the width
    assert s["yield_strength_spread"] > 0.3


def test_phases_of_different_stiffness():
    """A laminate of bands along x, elastic: its modulus along the bands is the mean of
    the moduli (iso-strain), across them below it and above their harmonic mean."""
    hard = FlowLaw.hollomon(5000.0, 6000.0, 0.1)  # elastic over the test
    phases = (Phase("soft", hard), Phase("stiff", hard))
    d = np.zeros((4, 2), int)
    d[:1] = 1  # a quarter of the sheet
    cell = PeriodicCell.build((2.0, 0.0), (0.0, 4.0), element=0.5, centre=(1.0, 2.0))
    moduli = (100000.0, 200000.0)
    mem = design_membrane(cell.mesh, d, phases=phases, origin=cell.origin, young=moduli)
    assert mem.young == pytest.approx(125000.0)
    assert mem.moduli.min() == 100000.0 and mem.moduli.max() == 200000.0
    slope = {}
    for angle in (0.0, 90.0):
        res = cell_test(mem, cell, angle, max_strain=0.001, step=0.25, max_step=0.25)
        slope[angle] = res.stress[1] / res.strain[1]
    assert slope[0.0] == pytest.approx(125000.0, rel=2e-3)
    assert 1.0 / (0.25 / 200000.0 + 0.75 / 100000.0) < slope[90.0] < slope[0.0]
    with pytest.raises(ValueError):
        design_membrane(cell.mesh, d, phases=phases, origin=cell.origin, young=(1.0, 2.0, 3.0))


def test_tracks_over_a_periodic_cell():
    """A corrugated track and its images: the treated share of the cell is the area of
    the track over the cell's, the moduli of the points mix those of the phases."""
    P, s, A, t = 10.0, 4.0, 1.5, 1.0
    cell = PeriodicCell.build((P, 0.0), (0.0, s), element=0.25, centre=(0.5 * P, 0.5 * s))
    x = np.linspace(-P, 2.0 * P, 601)
    line = np.column_stack([x, 0.5 * s + A * np.sin(2.0 * math.pi * x / P)])
    lines = periodic_lines([line], cell.a1, cell.a2, cell.box(), reach=t)
    assert len(lines) >= 3  # the track and its images above and below
    u = np.linspace(0.0, 2.0 * math.pi, 4001)
    arc = np.trapezoid(np.sqrt(1.0 + (2 * math.pi * A / P * np.cos(u)) ** 2), u) / (2 * math.pi)
    mem = track_membrane(cell.mesh, lines, t, young=(200000.0, 100000.0))
    share = mem.law[..., 0, 8]
    treated = float((share * mem.w0).sum() / mem.w0.sum())
    assert treated == pytest.approx(t * arc / s, rel=0.03)
    assert np.allclose(mem.moduli[..., 0], 200000.0 - 100000.0 * share)
    assert np.isfinite(mem.critical[share > 0.5]).all()
    assert np.isinf(mem.critical[share <= 0.5]).all()
    with pytest.raises(ValueError):
        track_membrane(cell.mesh, lines, t, phases=wang_phases()[:1])
    with pytest.raises(ValueError):
        track_membrane(cell.mesh, lines, t, kinds=[2] * len(lines))  # two phases only
    with pytest.raises(ValueError):
        periodic_lines([line], (1.0, 0.0), (2.0, 0.0), cell.box())


def test_a_track_whose_treatment_changes_along_it():
    """A straight track treated two ways, one half each: the points take the phase of
    their half, the treated share is the track's."""
    soft, hard = wang_phases()
    tempered = Phase("tempered", FlowLaw.hollomon(840.0, 1600.0, 0.1), critical=0.24)
    cell = PeriodicCell.build((10.0, 0.0), (0.0, 4.0), element=0.25, centre=(5.0, 2.0))
    left = np.array([[-1.0, 2.0], [5.0, 2.0]])
    right = np.array([[5.0, 2.0], [11.0, 2.0]])
    mem = track_membrane(
        cell.mesh,
        [left, right],
        1.0,
        phases=(soft, hard, tempered),
        kinds=[1, 2],
        young=(200000.0, 210000.0, 220000.0),
    )
    share = mem.law[..., 0, 8]
    assert float((share * mem.w0).sum() / mem.w0.sum()) == pytest.approx(0.25, rel=0.02)
    x = mem.points[..., 0]
    inside = share > 0.99
    second = mem.law[..., 0, 5]  # the yield stress of the treated law of each point
    assert np.allclose(second[inside & (x < 4.5)], hard.law.sigma0)
    assert np.allclose(second[inside & (x > 5.5)], tempered.law.sigma0)
    assert np.allclose(mem.critical[inside & (x > 5.5), 0], 0.24)
    assert np.allclose(mem.moduli[inside & (x > 5.5), 0], 220000.0)
    assert np.allclose(mem.moduli[share == 0, 0], 200000.0)
