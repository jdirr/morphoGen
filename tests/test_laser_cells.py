"""Cellular design of laser-architectured sheets (morphogen.laser.cells): the replays
against the finite elements (work, plastic strains), the nonlocal fracture, the
measures of the curves, the cells (location, orbits, filter, spots, treatability,
tracks), the estimates against the switches they predict, the automata and the fracture
of the hard points of a sheet."""

from __future__ import annotations

import math

import numpy as np
import pytest

from morphogen.laser import FlowLaw, Specimen, sheet_material
from morphogen.laser.cells import (
    MARTENSITE_CRITICAL,
    CellGrid,
    CellProblem,
    Phase,
    band_fractions,
    beso,
    constant_area_width,
    contrast_phases,
    corrugation,
    design_tracks,
    failure_measures,
    hca,
    make_treatable,
    phase_strength,
    round_design,
    symmetrize,
    topology,
    treatable,
    wang_phases,
)
from morphogen.laser.mechanics import Fracture, Membrane, replay, tensile_test


@pytest.fixture(scope="module")
def coarse():
    """A dog-bone of 1 mm elements, a band of martensite along it, its recorded test."""
    spec = Specimen(element=1.0)
    problem = CellProblem(specimen=spec, element=1.0, size=1.0)
    design = problem.grid.band(0.2)
    return problem, design, problem.evaluate(design, record=True)


def test_replay_reproduces_the_test(coarse):
    problem, _design, ev = coarse
    res, mem = ev.result, ev.membrane
    rp = replay(mem, res.history)
    external = np.concatenate(
        [[0.0], np.cumsum(0.5 * (res.force[1:] + res.force[:-1]) * np.diff(res.displacement))]
    )
    assert np.allclose(rp.work.sum(axis=1)[1:], external[1:], rtol=1e-3)
    assert np.allclose(rp.plastic[-1], mem.p_n)
    assert np.allclose(rp.ratio[-1], mem.fracture.ratios(mem.p_n))
    # another law: the martensite everywhere absorbs more work on the same path
    law = mem.law.copy()
    hard = problem.phases[1].law
    law[..., 0], law[..., 1], law[..., 2], law[..., 3], law[..., 8] = (
        hard.code,
        hard.sigma0,
        hard.a,
        hard.b,
        0.0,
    )
    assert replay(mem, res.history, law=law).work[-1].sum() > rp.work[-1].sum()


def test_nonlocal_fracture_of_a_uniform_field():
    rng = np.random.default_rng(1)
    centroids = rng.uniform(0, 5, (40, 2))
    w0 = np.full((40, 4), 0.25)
    crit = np.full((40, 4, 2), 0.1)
    crit[:10] = np.inf  # ductile elements
    plastic = np.full((40, 4, 2), 0.05)
    for radius in (None, 1.5):
        f = Fracture(centroids, w0, crit, radius)
        r = f.ratios(plastic)
        assert np.allclose(r[10:], 0.5) and np.allclose(r[:10], 0.0)
    # a peak at one element is spread over its neighbours
    plastic[20] = 0.2
    local = Fracture(centroids, w0, crit, None).ratios(plastic)
    spread = Fracture(centroids, w0, crit, 2.0).ratios(plastic)
    assert local[20] == pytest.approx(2.0) and 0.5 < spread[20] < 2.0


def test_failure_measures_of_a_parabola():
    e = np.linspace(0.0, 0.2, 201)
    s = 1000.0 * e * (0.2 - e) / 0.01  # maximum 1000 MPa at 0.1
    m = failure_measures(e, s, np.inf)
    assert m["uniform_elongation"] == pytest.approx(0.1, abs=1e-6)
    assert m["tensile_strength"] == pytest.approx(1000.0, rel=1e-6)
    exact = 1e5 * (0.2 * 0.1**2 / 2 - 0.1**3 / 3)
    assert m["energy"] == pytest.approx(exact, rel=1e-4)
    early = failure_measures(e, s, 0.05)
    assert early["failure_strain"] == pytest.approx(0.05)
    assert early["tensile_strength"] == pytest.approx(750.0, rel=1e-6)
    capped = failure_measures(e, s[:, None], np.array([np.inf]), cap=np.array([750.0]))
    assert capped["failure_strain"][0] == pytest.approx(0.05, abs=1e-6)


def test_grid_locates_and_filters():
    g = CellGrid(30.0, 5.0, 1.0)
    assert g.shape == (5, 30) and g.count == 150
    pts = np.array([[0.5, 0.5], [29.9, 4.9], [-3.0, 2.5], [31.0, 2.5], [10.0, 6.0]])
    assert g.locate(pts).tolist() == [0, 149, 60, 89, -1]
    assert g.locate(pts, extend=False)[2] == -1
    H = g.filter_matrix(1.5)
    assert np.allclose(np.asarray(H.sum(axis=1)).ravel(), 1.0)
    assert H[0, 0] > H[0, 1] > H[0, 31] > 0 and H[0, 2] == 0


def test_orbits_periods_and_symmetry():
    g = CellGrid(30.0, 5.0, 0.5)
    assert g.orbits().max() + 1 == g.count
    assert g.orbits(("point",)).max() + 1 == g.count // 2
    o = g.orbits(("point",), period=10.0)
    assert o.max() + 1 == g.count // 6
    d = symmetrize(np.random.default_rng(0).integers(0, 2, g.shape), o)
    assert np.array_equal(d, d[::-1, ::-1]) and np.array_equal(d, np.roll(d, 20, axis=1))
    with pytest.raises(ValueError):
        g.orbits(period=7.0)


def test_spots_tracks_and_treatability():
    spec = Specimen()
    g = CellGrid.of(spec, 0.5)
    for amplitude, period in ((0.0, None), (1.0, 10.0), (1.5, 15.0)):
        d = g.stamp(corrugation(spec, amplitude, period))
        assert treatable(d)
        tracks = design_tracks(g, d)
        assert len(tracks) == 1
        redrawn = g.stamp(tracks)  # the centre line drawn again by the spot
        assert ((redrawn == 0) & (d > 0)).sum() <= 0.15 * d.sum()
        assert ((redrawn > 0) & (d == 0)).sum() <= 0.15 * d.sum()
        redrawn = g.stamp(design_tracks(g, d, cross=True))  # steep runs crossed
        assert ((redrawn == 0) & (d > 0)).sum() == 0
        assert ((redrawn > 0) & (d == 0)).sum() <= 0.2 * d.sum()
    d = g.band(0.2)
    d[4, 10] = 0  # a notch: a neck thinner than the spot
    assert not treatable(d)
    pix = np.zeros(g.shape, dtype=int)
    pix[3, 5] = 1
    assert not treatable(pix) and make_treatable(pix).sum() == 0
    cov = g.coverage(corrugation(spec, 0.0, None), 1.0)
    assert cov.sum() * g.dx * g.dy == pytest.approx(30.0, rel=0.02)
    assert g.rasterize(corrugation(spec, 1.0, 10.0), 1.0, budget=0.2).mean() == pytest.approx(0.2)


def test_bands_of_constant_area():
    assert constant_area_width(1.2, 0.0, None) == 1.2
    t = constant_area_width(1.2, 1.5, 10.0)
    assert 1.0 < t < 1.1  # Wang et al. 2023, sample 6: 1.01
    spec = Specimen(element=0.5)
    mesh = spec.mesh()
    share = band_fractions(mesh, corrugation(spec, 0.0, None), 1.2)
    _, _, w = mesh.gauss()
    inside = mesh.gauss()[0][..., 0]
    gauge = (inside >= 0) & (inside <= 30)
    assert (share * w)[gauge].sum() / w[gauge].sum() == pytest.approx(0.24, abs=0.015)


def test_phases_and_their_strength():
    ferrite, martensite = wang_phases()
    assert phase_strength(ferrite) == pytest.approx(550.0, rel=0.01)
    assert phase_strength(martensite) == pytest.approx(1553.0, rel=0.01)  # Table 4
    p = Phase.from_config(
        {"name": "m", "law": {"yield_strength": 1300, "k": 1966, "n": 0.06}, "critical": 0.07}
    )
    assert p.law.kind == "swift" and p.critical == 0.07
    law = FlowLaw.hollomon(280.0, 986.0, 0.24)
    assert law.stress(0.0) == pytest.approx(280.0)
    assert law.slope(0.1) == pytest.approx(
        0.24 * law.stress(0.1) / ((280 / 986) ** (1 / 0.24) + 0.1)
    )


def test_phases_of_a_contrast():
    ferrite, martensite = wang_phases()
    top = 1300.0 / 280.0
    for rule in ("graded", "brittle"):  # the laser martensite at the top
        matrix, hard = contrast_phases(top, rule)
        assert matrix == ferrite and hard.law == martensite.law
        assert hard.critical == martensite.critical
    assert math.isinf(contrast_phases(top, "ductile")[1].critical)
    soft = contrast_phases(1.0)[1]  # the matrix at the bottom
    assert soft.law == ferrite.law and math.isinf(soft.critical)
    assert contrast_phases(1.0, "brittle")[1].critical == MARTENSITE_CRITICAL
    # the yield stress is the contrast times the matrix's; a harder state is stronger,
    # hardens less and (graded) breaks earlier
    contrasts = (1.5, 2.5, 3.5, top)
    states = [contrast_phases(c)[1] for c in contrasts]
    assert [p.law.sigma0 for p in states] == pytest.approx([280.0 * c for c in contrasts])
    assert np.all(np.diff([phase_strength(p) for p in states]) > 0)
    assert np.all(np.diff([p.law.b for p in states]) < 0)
    assert np.all(np.diff([p.critical for p in states]) < 0)
    for bad in ((0.9, "graded"), (top + 0.01, "graded"), (2.0, "fragile")):
        with pytest.raises(ValueError):
            contrast_phases(*bad)


def test_estimates_predict_the_switches(coarse):
    problem, design, ev = coarse
    gains = problem.estimates(ev)
    assert gains.shape == (problem.grid.count, 2)
    d = design.ravel()
    assert np.allclose(gains[np.arange(problem.grid.count), d], 0.0)
    # a gap in the track: the weakest section necks, the estimate sees the loss
    c = np.flatnonzero(d == 1)[len(np.flatnonzero(d == 1)) // 2]
    gapped = design.ravel().copy()
    gapped[c] = 0
    actual = problem.evaluate(gapped.reshape(design.shape)).value - ev.value
    assert gains[c, 0] < 0 and actual < 0
    assert gains[c, 0] == pytest.approx(actual, rel=0.6)


def test_automata_run(coarse):
    problem, design, _ev = coarse
    r = beso(problem, design, budget=None, moves="exchange", candidates=2, iterations=2)
    assert r.evaluations >= 1 and r.design.shape == design.shape
    assert problem.budget(r.design) == pytest.approx(problem.budget(design))
    r = beso(problem, design, budget=0.2, moves="swap", iterations=2)
    assert r.evaluation.value >= r.history[0]["value"] - 1e-9
    grey = design.astype(float)
    for fun in (hca, topology):
        r = fun(problem, grey, budget=0.2, iterations=2)
        assert r.design.dtype.kind == "i" and r.design.mean() == pytest.approx(0.2)
    assert round_design(np.array([[0.1, 0.9, 0.6, 0.2]]), 0.5).tolist() == [[0, 1, 1, 0]]
    assert round_design(np.array([0.0, 0.4, 1.0]), levels=3).tolist() == [0, 1, 3]


def test_fracture_of_the_hard_points_of_a_sheet():
    m = sheet_material("annealed_ms1500")
    assert m.critical_strain([200.0]) is None
    from dataclasses import replace

    m = replace(m, fracture={"hardness": [300.0, 480.0], "critical": [math.inf, 0.072]})
    c = m.critical_strain(np.array([200.0, 300.0, 390.0, 480.0, 600.0]))
    assert np.isinf(c[:2]).all()
    assert c[2] == pytest.approx(0.144) and c[3] == pytest.approx(0.072) and c[4] == 0.072
    # a membrane breaks where its brittle points reach their strain
    spec = Specimen(kind="strip", element=1.0, length=10.0, width=2.0)
    mesh = spec.mesh()
    ne = len(mesh.quads)
    law = FlowLaw.hollomon(400.0, 1000.0, 0.2)
    flow = [np.full((ne, 4, 1), v) for v in (law.code, law.sigma0, law.a, law.b)]
    mem = Membrane(mesh, flow, critical=np.full((ne, 4, 1), 0.05))
    res = tensile_test(spec, mem, failure_margin=0.0)
    assert res.failure_strain == pytest.approx(math.expm1(0.05 + 500 / 210000), rel=0.05)


def test_cellular_section_of_a_configuration(tmp_path):
    from morphogen.config import ConfigError
    from morphogen.laser import load_laser_config, run_laser

    data = {
        "laser_sheet": {
            "specimen": {"kind": "dogbone", "length": 30, "width": 5, "thickness": 1.0},
            "element": 1.0,
            "pattern": {"kind": "corrugated", "count": 1, "amplitude": 1.5, "period": 10},
        },
        "cellular": {
            "size": 1.0,
            "element": 1.0,
            "start": {"amplitude": 1.5, "period": 10},
            "symmetry": ["point"],
            "period": 10,
            "min_width": 1,
            "candidates": 1,
            "iterations": 1,
        },
        "outputs": {"directory": str(tmp_path / "out"), "figures": False, "fields": []},
    }
    cfg = load_laser_config(data)
    summary, out = run_laser(cfg)
    assert summary["cellular"]["evaluations"] >= 1
    assert (out / "cells.csv").is_file() and (out / "cells_design.csv").is_file()
    assert "cellular" in summary["designs"]
    with pytest.raises(ConfigError):
        load_laser_config({**data, "cellular": {"method": "simulated_annealing"}})
    with pytest.raises(ConfigError):
        load_laser_config({**data, "cellular": {"phases": "steel"}})
