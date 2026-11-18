"""Load cases of laser-architectured sheets beyond the tensile test (morphogen.laser.
structures): graded meshes, the compact tension specimen against the compliance of
Saxena & Hudak, the L-shaped plate, the displacement-controlled test and its initiation,
the cells over a structure; Lankford's coefficients and Hill's 1948 criterion of the
directions of a treated sheet; the softening of an age-hardened aluminium alloy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from morphogen.laser import FlowLaw, Specimen
from morphogen.laser.cells import CellProblem, Phase
from morphogen.laser.design import hill48
from morphogen.laser.mechanics import Membrane, tensile_test
from morphogen.laser.metallurgy import PeakSoftening
from morphogen.laser.structures import CompactTension, LPlate, ct_mesh, graded, l_mesh


def _membrane(mesh, law, **kw):
    ne = len(mesh.quads)
    flow = [np.full((ne, 4, 1), v) for v in (law.code, law.sigma0, law.a, law.b)]
    return Membrane(mesh, flow, **kw)


def test_graded_coordinates():
    x = graded(0.0, 10.0, 1.0, focus=(4.0,), fine=0.1, growth=1.2)
    assert x[0] == 0.0 and x[-1] == 10.0 and 4.0 in x
    h = np.diff(x)
    assert h.min() == pytest.approx(0.1, rel=0.3) and h.max() <= 1.5  # no sliver at the ends
    assert np.all(np.maximum(h[1:] / h[:-1], h[:-1] / h[1:]) < 1.6)  # graded
    assert len(graded(0.0, 10.0, 1.0)) == 11


def test_meshes_of_the_structures():
    d = ct_mesh(20.0, 0.5, element=1.0, fine=0.25)
    area = d.mesh.gauss()[2].sum()
    assert area == pytest.approx(1.25 * 20 * 1.2 * 20 - 2 * math.pi * 2.5**2, rel=0.01)
    assert np.allclose(d.points["tip"], [10.0, 0.0])
    mu, ml = d.sets["mouth_upper"][0], d.sets["mouth_lower"][0]
    assert mu != ml and np.allclose(d.mesh.nodes[mu], d.mesh.nodes[ml])  # open crack
    lp = l_mesh(40.0, 16.0, element=1.0, fine=0.5)
    assert lp.mesh.gauss()[2].sum() == pytest.approx(40 * 40 - 24 * 24)
    assert np.all(lp.mesh.nodes[lp.sets["top"], 1] == 40.0)


def test_ct_compliance_of_saxena_and_hudak():
    ct = CompactTension(width=20.0, crack=0.5, element=1.0, fine=0.25)
    law = FlowLaw.hollomon(1e6, 2e6, 0.1)  # elastic
    mem = _membrane(ct.mesh, law, springs=ct.springs)
    r = ct.run(mem, max_strain=0.0005, failure_margin=None)
    a = 0.5
    poly = 2.1630 + 12.219 * a - 20.065 * a**2 - 0.9925 * a**3 + 20.609 * a**4 - 9.9314 * a**5
    exact = ((1 + a) / (1 - a)) ** 2 * poly / 210000.0  # load-line, plane stress, B = 1
    assert r.displacement[-1] / r.force[-1] == pytest.approx(exact, rel=0.05)
    assert r.fields["monitors"]["cmod"][-1] > r.displacement[-1]


def test_initiation_at_the_crack_tip_and_the_cells_of_a_structure():
    ferrite = Phase("ferrite", FlowLaw.hollomon(280.0, 986.0, 0.24), critical=0.5, cost=0.0)
    hard = Phase("martensite", FlowLaw.hollomon(1300.0, 1966.0, 0.06), critical=0.075)
    ct = CompactTension(element=2.0, fine=0.25, max_disp=4.0)
    prob = CellProblem(phases=(ferrite, hard), case=ct, size=1.0, fracture_radius=0.5)
    assert prob.grid.x0 == pytest.approx(6.0) and prob.active.mean() > 0.8
    base = prob.evaluate(np.zeros(prob.grid.shape, dtype=int))
    assert math.isfinite(base.result.failure_strain)  # the crack initiates
    band = np.zeros(prob.grid.shape, dtype=int)
    C = prob.grid.centres()
    band[(C[..., 0] >= 11.5) & (C[..., 0] < 13.5)] = 1  # 1.5 mm ahead of the tip
    shielded = prob.evaluate(band)
    assert shielded.measures["energy"] > 1.2 * base.measures["energy"]
    lp = LPlate(element=2.0, fine=0.5, max_disp=2.0, region=(0.0, 40.0, 0.0, 40.0))
    prob = CellProblem(phases=(ferrite, hard), case=lp, size=2.0, fracture_radius=0.5)
    assert 0 < prob.active.sum() < prob.grid.count  # the cut quadrant
    ev = prob.evaluate(np.zeros(prob.grid.shape, dtype=int))
    assert ev.result.force.max() > 0 and prob.budget(np.ones(prob.grid.shape, int)) == 1.0


def test_lankford_and_hill48():
    spec = Specimen(element=0.5)
    r = tensile_test(spec, _membrane(spec.mesh(), FlowLaw.hollomon(280.0, 986.0, 0.24)))
    assert r.r_value() == pytest.approx(1.0, abs=0.01)  # isotropic von Mises
    iso = hill48({0: 1.0, 45: 1.0, 90: 1.0})
    assert iso["hill_f"] == pytest.approx(0.5) and iso["hill_n"] == pytest.approx(1.5)
    assert iso["r_bar"] == 1.0 and iso["delta_r"] == 0.0
    h = hill48({0: 2.0, 45: 1.5, 90: 2.5})
    assert h["r_bar"] == pytest.approx(1.875) and h["delta_r"] == pytest.approx(0.75)
    assert h["hill_g"] + h["hill_h"] == pytest.approx(1.0)
    assert math.isnan(hill48({0: 1.0})["r_bar"])


def test_softening_of_an_age_hardened_alloy():
    m = PeakSoftening()
    assert m.gain(150.0) == 0.0 and m.gain(400.0) < m.gain(300.0) < 0.0
    state = m.initial_state(2)
    for peak in (420.0, 300.0):  # a second, milder cycle does not age the point again
        state["peak"][:] = peak
        m._complete(state, np.ones(2, dtype=bool))
    assert np.allclose(m.hardness(state), m.base_hardness + m.gain(420.0))
    assert np.all(m.treated(state) > 0.5)


def test_alternative_materials():
    from morphogen.laser import Beam, SheetCase, sheet_material, track_kernel

    b = sheet_material("22mnb5")
    _, s0, _, _ = b.strength.flow(b.base_hardness)
    assert float(np.ravel(s0)[0]) == pytest.approx(400.0, rel=0.2)  # as delivered
    al = sheet_material("aa6082_t6")
    assert al.young == 70000.0 and al.base_hardness == 105.0
    beam = Beam(260.0, speed=20.0, radius=0.75, absorptivity=0.3)
    k = track_kernel(beam, 1.0, al.thermal, bottom=beam)  # one beam on each face
    assert 300.0 < k.peak.max() < al.metallurgy.solidus  # softened, not melted
    m = SheetCase(
        material="aa6082_t6",
        element=1.0,
        pattern={"kind": "straight", "count": 1},
        beam=Beam(260.0, speed=20.0, radius=0.75, absorptivity=0.3),
        bottom=Beam(260.0, speed=20.0, radius=0.75, absorptivity=0.3),
    ).evaluate()
    assert m.measures["mean_hardness"] < 105.0 and m.measures["volume_fraction"] > 0.2
    assert m.measures["yield_strength"] < 280.0  # a softened track


@pytest.mark.parametrize("angle", [0.0, 45.0, 90.0])
def test_hill48_tension_of_a_textured_sheet(angle):
    from morphogen.laser.mechanics import hill_from_r

    r0, r45, r90 = 1.6, 1.2, 2.0
    spec = Specimen(element=0.5)
    law = FlowLaw.hollomon(280.0, 986.0, 0.24)
    F, G, H, N, _ = hill_from_r(r0, r45, r90)
    res = tensile_test(
        spec, _membrane(spec.mesh(), law, anisotropy=hill_from_r(r0, r45, r90, -angle))
    )
    s2, c2 = math.sin(math.radians(angle)) ** 2, math.cos(math.radians(angle)) ** 2
    r_hill = (H + (2 * N - F - G - 4 * H) * s2 * c2) / (F * s2 + G * c2)
    assert res.r_value() == pytest.approx(r_hill, rel=0.02)
    iso = tensile_test(spec, _membrane(spec.mesh(), law))
    ratio = 1.0 / math.sqrt(F * s2 + G * c2 + H + (2 * N - F - G - 4 * H) * s2 * c2)
    assert res.yield_strength / iso.yield_strength == pytest.approx(ratio, rel=0.015)
    assert res.iterations < 4 * res.steps  # the consistent tangent of the return


def test_texture_of_a_sheet_and_its_treated_points():
    from dataclasses import replace

    from morphogen.laser import SheetCase, sheet_material

    m = replace(sheet_material("annealed_ms1500"), anisotropy={"r0": 1.6, "r45": 1.2, "r90": 2.0})
    h = m.hill(np.array([0.0, 1.0, 0.5]), angle=30.0)
    assert np.allclose(h[1, :4], [0.5, 0.5, 0.5, 1.5]) and np.isclose(h[0, 4], math.radians(30))
    assert np.allclose(h[2, :4], 0.5 * (h[0, :4] + h[1, :4]))
    case = SheetCase(material=m, element=1.0, pattern={"kind": "none"}, directions=(0, 90))
    meas = case.evaluate().measures
    assert meas["r_value_0"] == pytest.approx(1.6, rel=0.03)
    assert meas["r_value_90"] == pytest.approx(2.0, rel=0.03)


# -- crack growth ---------------------------------------------------------------------------


def test_graded_box_is_uniform_inside_and_grows_outside():
    from morphogen.laser.structures import graded

    x = graded(-5.0, 20.0, 1.0, focus=(10.0,), fine=0.25, growth=1.2, box=(8.0, 20.0))
    inner = x[(x >= 8.0 - 1e-9) & (x <= 20.0 + 1e-9)]
    assert np.allclose(np.diff(inner), 0.25)
    assert np.any(np.isclose(x, 10.0))  # the tip on the grid
    d = np.diff(x)
    assert d.max() <= 1.0 + 1e-9 and d.min() >= 0.25 - 1e-9
    assert np.all(d[1:][x[1:-1] < 8.0] <= 1.2 * d[:-1][x[1:-1] < 8.0] + 1e-9)


def test_a_strip_breaks_at_its_weak_band():
    from morphogen.laser.mechanics import Membrane
    from morphogen.laser.structures import _quadmesh, _tensor, load_test

    xs, ys = np.linspace(0.0, 10.0, 21), np.linspace(0.0, 2.0, 5)
    nodes, quads = _tensor(xs, ys)
    mesh = _quadmesh(nodes, quads)
    ne = len(quads)
    cen = nodes[quads].mean(axis=1)
    weak = np.abs(cen[:, 0] - 5.0) < 0.5
    s0 = np.where(weak, 250.0, 300.0)[:, None, None] * np.ones((ne, 4, 1))
    flow = (np.full((ne, 4, 1), 2.0), s0, np.full((ne, 4, 1), 900.0), np.full((ne, 4, 1), 0.2))
    critical = np.where(weak[:, None, None], 0.15, np.inf) * np.ones((ne, 4, 1))
    mem = Membrane(mesh, flow, critical=critical)
    left = np.flatnonzero(np.isclose(nodes[:, 0], 0.0))
    right = np.flatnonzero(np.isclose(nodes[:, 0], 10.0))
    fixed = np.concatenate([2 * left, [2 * left[0] + 1]])
    res = load_test(
        mem,
        fixed,
        2 * right,
        np.ones(len(right)),
        length=10.0,
        section=2.0,
        max_disp=2.0,
        drop=0.95,
        erosion=True,
    )
    broken, at = res.fields["broken"]
    assert len(broken) >= len(ys) - 1  # the band crossed
    assert np.all(weak[broken])
    assert res.force[-1] < 0.05 * res.force.max()  # separated
    assert np.all(np.diff(at) >= 0)


def test_a_spiral_path():
    from morphogen.laser import Beam
    from morphogen.laser.patterns import make_pattern

    path = make_pattern(
        "spiral",
        (0, 20, -10, 10),
        beam=Beam(300.0, 10.0, 0.5),
        center=(10.0, 0.0),
        pitch=2.0,
        turns=3.0,
        radius=1.0,
    )
    pts = path.tracks[0].points
    r = np.hypot(pts[:, 0] - 10.0, pts[:, 1])
    assert r[0] == pytest.approx(1.0) and r[-1] == pytest.approx(7.0)
    assert np.all(np.diff(r) > 0)
    # the length of an Archimedean spiral: about 2π times the mean radius per turn
    assert path.length == pytest.approx(2 * np.pi * 4.0 * 3.0, rel=0.02)
