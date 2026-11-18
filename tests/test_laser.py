"""Laser-architectured sheets (morphogen.laser): the moving beams against Rosenthal's
solutions and the conservation of energy, the transient solver against the
quasi-steady cycles, the metallurgy against its formulas (lever rule, Andrews, Maynier,
Koistinen-Marburger, Hollomon-Jaffe additivity, Sieverts, mass balances), the flow laws
against the tensile properties they are fitted to, the geometry of the paths, the
membrane against the exact uniaxial tension, finite differences of its tangent, rigid
rotations and the iso-strain mixture of bands, the optimizers and the command line."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import integrate

from morphogen.laser import (
    Alloying,
    Beam,
    FlowLaw,
    GaussianProcess,
    LaserPath,
    LaserTreatment,
    Membrane,
    PeakHardening,
    SheetCase,
    Specimen,
    Tempering,
    ThermalProperties,
    TransformationHardening,
    TransientHeat,
    andrews_ac,
    andrews_ms,
    distance_to_polyline,
    flow_parameters,
    make_pattern,
    maynier_hardness,
    moving_source,
    moving_source_grid,
    optimize,
    rosenthal_line,
    rosenthal_point,
    sheet_material,
    sieverts_nitrogen,
    tensile_test,
    track_kernel,
)
from morphogen.laser.design import expected_improvement

STEEL = ThermalProperties(conductivity=25.0, density=7700.0, heat_capacity=600.0, convection=0.0)


# -- heat ----------------------------------------------------------------------------------


def test_moving_source_tends_to_rosenthal():
    beam = Beam(power=500.0, speed=10.0, radius=0.01, absorptivity=1.0)
    xi = np.array([2.0, 0.5, -1.0, -3.0, -6.0, -2.0])
    y = np.array([1.0, 1.5, 1.0, 2.0, 0.5, 0.0])
    z = np.array([0.0, 0.5, 1.0, 0.2, 0.0, 0.8])
    t = moving_source(xi, y, z, beam, 200.0, STEEL)  # a thick plate
    r = rosenthal_point(xi, y, z, beam.absorbed, beam.speed, STEEL)
    np.testing.assert_allclose(t - 20.0, r - 20.0, rtol=2e-4)
    # a thin plate cooled by its faces: the line source of Rykalin
    lossy = ThermalProperties(25.0, 7700.0, 600.0, convection=200.0, emissivity=0.5)
    t = moving_source(xi, y, np.full_like(xi, 0.01), beam, 0.02, lossy)
    r = rosenthal_line(xi, y, beam.absorbed, beam.speed, 0.02, lossy)
    keep = np.hypot(xi, y) > 0.1
    np.testing.assert_allclose(t[keep] - 20.0, r[keep] - 20.0, rtol=2e-4)


def test_energy_of_a_cross_section():
    # without losses, a cross-section far behind the beam holds the energy P/v per mm
    beam = Beam(power=200.0, speed=10.0, radius=0.5, absorptivity=0.5)
    ys = np.linspace(-60.0, 60.0, 2401)
    t = moving_source_grid([-40.0], ys, [0.25], beam, 0.5, STEEL, configuration="symmetric")
    energy = STEEL.rho_c * 0.5 * np.trapezoid(t[0, :, 0] - 20.0, ys)
    assert energy == pytest.approx(2.0 * beam.absorbed / beam.speed, rel=2e-3)


def test_grid_and_beams_of_both_faces():
    top = Beam(150.0, 10.0, 0.5, 0.5)
    low = Beam(100.0, 10.0, 0.6, 0.5, profile="tophat")
    xs, ys, zs = np.linspace(-4, 1, 7), np.linspace(0, 2, 5), np.array([0.1, 0.5, 0.9])
    g = moving_source_grid(xs, ys, zs, top, 1.0, STEEL, bottom=low)
    pts = np.meshgrid(xs, ys, zs, indexing="ij")
    np.testing.assert_allclose(g, moving_source(*pts, top, 1.0, STEEL, bottom=low), rtol=1e-10)
    # the same beam on both faces: a profile symmetric through the thickness
    s = moving_source_grid(xs, ys, zs, top, 1.0, STEEL, configuration="symmetric")
    np.testing.assert_allclose(s[:, :, 0], s[:, :, 2], rtol=1e-9)
    single = moving_source_grid(xs, ys, zs, top, 1.0, STEEL)
    np.testing.assert_allclose(s - 20.0, single - 20.0 + (single[:, :, ::-1] - 20.0), rtol=1e-9)
    with pytest.raises(ValueError, match="same speed"):
        moving_source(0.0, 0.0, 0.0, top, 1.0, STEEL, bottom=Beam(100.0, 5.0, 0.6))


def test_transient_against_the_quasi_steady_track():
    # the same layers through the thickness: the cells hold their mean temperature
    props = ThermalProperties(25.0, 7700.0, 560.0, convection=200.0)
    beam = Beam(100.0, 10.0, 0.5, 0.42)
    kern = track_kernel(beam, 0.5, props, configuration="symmetric", layers=2)
    solver = TransientHeat((30.0, 8.0), 0.5, props, cell=0.1, layers=2)
    res = solver.run(
        [(np.array([[1.0, 4.0], [29.0, 4.0]]), beam, None, 0.0)], configuration="symmetric"
    )
    i = int(np.argmin(np.abs(solver.x - 20.0)))
    d = np.abs(solver.y - 4.0)
    near = d < 1.5
    np.testing.assert_allclose(res.peak[i, near], kern.peak_at(d[near]), rtol=0.04)
    assert kern.width(800.0) > kern.width(1200.0) > 0.0


# -- metallurgy ------------------------------------------------------------------------------

MS1500 = {"C": 0.228, "Si": 0.242, "Mn": 1.73, "Cr": 0.19}


def test_formulas_of_the_steels():
    ac1, ac3 = andrews_ac(MS1500)
    assert ac1 == pytest.approx(714.7, abs=0.1)
    assert ac3 == pytest.approx(823.9, abs=0.1)
    assert float(andrews_ms(0.228, MS1500)) == pytest.approx(387.665, abs=1e-3)
    hm, _, hf = maynier_hardness(0.228, MS1500, 2.0 / 60.0)  # furnace cooling 2 °C/min
    assert float(hf) == pytest.approx(167.0, abs=10.0)  # Wang 2022: 167 ± 6 HV
    hm, _, _ = maynier_hardness(0.228, MS1500, 1000.0)
    assert 480.0 < float(hm) < 540.0
    assert sieverts_nitrogen(1600.0) == pytest.approx(0.0445, rel=0.02)
    assert sieverts_nitrogen(1600.0, chromium=18.0) > 5.0 * sieverts_nitrogen(1600.0)


def test_lever_rule_and_quench():
    m = TransformationHardening(composition=MS1500, superheat=0.0)
    f, c = m.austenite([m.a1 + 1e-6, 0.5 * (m.a1 + m.a3), m.a3], [0.228, 0.228, 0.228])
    assert f[0] == pytest.approx((0.228 - 0.02) / 0.75, rel=1e-4)  # the pearlite at once
    assert c[0] == pytest.approx(0.77, rel=1e-4)
    assert f[2] == 1.0 and c[2] == pytest.approx(0.228)
    # a cycle above A3, cooled fast: martensite of Koistinen and Marburger at ambient
    st = m.initial_state(1)
    t = np.array([20.0, 500.0, 1000.0, 900.0, 700.0, 450.0, 250.0, 20.0])
    for a, b in itertools.pairwise(t):
        m.step(st, np.array([a]), np.array([b]), 0.05)
    m.finish(st)
    ms = float(andrews_ms(0.228, MS1500))
    km = 1.0 - math.exp(-0.011 * (ms - 20.0))
    assert st["martensite"][0] == pytest.approx(km, rel=1e-9)
    assert m.hardness(st)[0] > 450.0


def test_tempering_is_additive():
    tmp = Tempering(constant=20.0)
    m = TransformationHardening(composition=MS1500, martensite=1.0, tempering=tmp)
    st = m.initial_state(1)
    temp, hours, n = 600.0, 0.25, 50
    for _ in range(n):
        m.step(st, np.array([temp]), np.array([temp]), hours * 3600.0 / n)
    p = (temp + 273.15) * (20.0 + math.log10(hours))
    assert st["ptemper"][0] == pytest.approx(p, rel=1e-6)
    h = m.hardness(st)[0]
    h0 = float(m.martensite_hardness)
    assert h == pytest.approx(float(tmp.factor(p, h0)), rel=1e-9) and h < h0


def test_peak_hardening_and_its_tempering():
    m = PeakHardening()
    st = m.initial_state(3)
    for peak in (1425.0, 1000.0, 600.0):
        cyc = np.array([20.0, peak, 0.5 * (peak + 20.0), 250.0, 20.0])
        st2 = m.initial_state(1)
        for a, b in itertools.pairwise(cyc):
            m.step(st2, np.array([a]), np.array([b]), 0.2)
        m.finish(st2)
        assert m.hardness(st2)[0] == pytest.approx(m.base_hardness + float(m.gain(peak)), rel=1e-9)
    del st
    # a second cycle below the onset tempers the gain of the first
    st = m.initial_state(1)
    for cyc in ([20.0, 1425.0, 700.0, 250.0, 20.0], [20.0, 750.0, 600.0, 250.0, 20.0]):
        for a, b in itertools.pairwise(cyc):
            m.step(st, np.array([a]), np.array([b]), 0.5)
        m.finish(st)
    assert m.base_hardness < m.hardness(st)[0] < m.base_hardness + float(m.gain(1425.0))


def test_melt_pool_mass_balance():
    alloy = Alloying(kind="carburizing", coating=20.0, efficiency=0.5)
    st = {"carbon": np.full(10, 0.026), "nitrogen": np.zeros(10)}
    melted = np.zeros(10, dtype=bool)
    melted[2:6] = True
    vol = np.full(10, 0.01)  # mm³
    out = alloy.mix(
        st, melted, vol, {"top": 0.02, "bottom": 0.02}, density=7700.0, melt_temperature=1600.0
    )
    mass = 7700.0 * 1e-6 * 0.04  # g of melt
    added = 0.5 * 20.0 * 1e-6 * 0.04  # g of carbon
    assert out["C"] == pytest.approx(100.0 * (0.026e-2 * mass + added) / (mass + added), rel=1e-9)
    np.testing.assert_allclose(st["carbon"][~melted], 0.026)
    nit = Alloying(kind="nitriding", efficiency=0.5)
    st = {"carbon": np.full(10, 0.026), "nitrogen": np.zeros(10)}
    out = nit.mix(st, melted, vol, {}, density=7700.0, melt_temperature=1600.0, chromium=18.7)
    assert out["N"] == pytest.approx(0.5 * sieverts_nitrogen(1600.0, 18.7), rel=1e-9)


# -- materials -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rp, rm, ag", [(346.0, 502.9, 0.1939), (300.0, 534.0, 0.167), (1300.0, 1600.0, 0.04)]
)
def test_flow_laws_fit_the_tensile_properties(rp, rm, ag):
    law = FlowLaw.from_tensile(rp, rm, ag)
    assert float(law.stress(0.002)) == pytest.approx(rp, rel=1e-6)
    su, young = rm * (1.0 + ag), 210000.0
    pu = math.log1p(ag) - su / young
    assert float(law.stress(pu)) == pytest.approx(su, rel=1e-6)
    assert float(law.slope(pu)) == pytest.approx(su, rel=1e-6)  # Considère
    kind, *_ = flow_parameters(rp, rm, ag)
    assert law.kind == ("ludwik" if kind[0] == 0 else "voce")
    with pytest.raises(ValueError):
        flow_parameters(600.0, 500.0, 0.1)


def test_presets():
    m = sheet_material("aisi430")
    assert m.base_hardness == pytest.approx(160.0)
    rp, rm, ag = m.strength.properties(241.8)
    assert (float(rp), float(rm), float(ag)) == pytest.approx((539.7, 702.5, 0.1365))
    ms = sheet_material("annealed_ms1500")
    assert ms.base_hardness == pytest.approx(167.0)
    with pytest.raises(ValueError, match="one of"):
        sheet_material("steel")


# -- patterns --------------------------------------------------------------------------------


def test_paths_and_their_measures():
    beam = Beam(150.0, 10.0, 0.5)
    amp, per = 1.5, 15.0
    path = make_pattern(
        "corrugated",
        (0.0, 30.0, 0.0, 5.0),
        beam=beam,
        count=1,
        amplitude=amp,
        period=per,
        margin=0.0,
        resolution=400,
    )
    k = 2.0 * math.pi * amp / per
    exact = (
        2.0
        * integrate.quad(lambda u: math.sqrt(1.0 + (k * math.cos(u)) ** 2), 0.0, 2.0 * math.pi)[0]
        * per
        / (2.0 * math.pi)
    )
    assert path.length == pytest.approx(exact, rel=1e-5)
    assert path.min_radius() == pytest.approx(per**2 / (4.0 * math.pi**2 * amp), rel=1e-3)
    assert path.time == pytest.approx(path.length / 10.0)
    two = make_pattern("straight", (0.0, 30.0, 0.0, 5.0), beam=beam, count=2, margin=5.0)
    assert two.min_gap() == pytest.approx(2.5, abs=1e-9)
    assert two.clipped_length((0.0, 0.0), (30.0, 5.0)) == pytest.approx(60.0)
    assert two.time == pytest.approx(80.0 / 10.0 + 30.0)
    code = two.to_gcode()
    assert code.count("G1 ") == 2 and code.count("G4 ") == 1
    pts = np.random.default_rng(0).uniform(-2, 8, (200, 2))
    poly = np.array([[0.0, 0.0], [3.0, 1.0], [5.0, 4.0]])
    brute = np.min(
        [
            np.hypot(
                *(
                    pts
                    - (
                        a
                        + np.clip(((pts - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)[:, None]
                        * (b - a)
                    )
                ).T
            )
            for a, b in itertools.pairwise(poly)
        ],
        axis=0,
    )
    np.testing.assert_allclose(distance_to_polyline(pts, poly), brute, rtol=1e-12)


def test_honeycomb_draws_each_side_once():
    beam = Beam(150.0, 10.0, 0.5)
    cell = 2.0
    path = make_pattern("honeycomb", (0.0, 20.0, 0.0, 10.0), beam=beam, cell=cell, margin=2.0)
    seg = np.concatenate([np.hypot(*np.diff(t.points, axis=0).T) for t in path.tracks])
    np.testing.assert_allclose(seg, cell, rtol=1e-9)  # every move is one side
    mids = np.vstack([0.5 * (t.points[1:] + t.points[:-1]) for t in path.tracks])
    assert len(np.unique(np.round(mids, 6), axis=0)) == len(mids)  # no side twice


# -- process ---------------------------------------------------------------------------------


def test_tracks_harden_and_temper():
    beam = Beam(150.0, 10.0, 0.5, 0.55)
    low = Beam(100.0, 10.0, 0.6, 0.55, profile="tophat")
    tr = LaserTreatment("annealed_ms1500", configuration="symmetric", layers=2)
    y = np.linspace(0.0, 5.0, 101)
    pts = np.column_stack([np.full_like(y, 15.0), y])
    one = make_pattern("straight", (0.0, 30.0, 0.0, 5.0), beam=beam, bottom=low, count=1)
    res = tr.apply(one, pts)
    kern = tr.kernel(beam, low)
    # the hardened band: where the peak passed A3 in every layer, martensite
    full = kern.peak_at(np.abs(y - 2.5)).min(axis=1) > tr.model.a3 + 20.0
    assert full.sum() > 10
    assert np.all(res.hardness[full] > 480.0)
    assert np.allclose(res.hardness[np.abs(y - 2.5) > 1.5], 167.0)
    np.testing.assert_allclose(res.hardness, res.hardness[::-1], atol=1e-6)  # symmetric
    # a second track nearby tempers the edge of the first
    two = LaserPath([one.tracks[0], one.tracks[0].shifted(dy=1.6)])
    res2 = tr.apply(two, pts)
    edge = (y > 2.5) & (y < 3.2) & (res.hardness[:, 0] > 400.0)
    assert np.any(res2.hardness[edge, 0] < res.hardness[edge, 0] - 50.0)
    # the tabulated response of an untouched sheet equals the integrated cycles
    probe = tr.apply(LaserPath([one.tracks[0]]), pts)
    st = tr.model.initial_state(len(pts) * 2)
    cyc = kern.cycles(np.abs(y - 2.5))
    temps = np.transpose(cyc, (1, 0, 2)).reshape(len(kern.times), -1)
    tr._integrate(st, temps, kern.times)
    direct = tr.model.hardness(st).reshape(-1, 2)
    gap = np.abs(probe.hardness - direct)
    assert gap.mean() < 2.5 and np.sum(gap > 15.0) <= 2  # the jump at A1 between two distances


# -- mechanics -------------------------------------------------------------------------------


def _uniform(mesh, law, layers=1):
    ne = len(mesh.quads)
    kind = 0 if law.kind == "ludwik" else 1
    return tuple(np.full((ne, 4, layers), v) for v in (kind, law.sigma0, law.a, law.b))


def test_tension_of_a_homogeneous_strip_is_exact():
    law = FlowLaw.from_tensile(346.0, 502.9, 0.1939)
    sp = Specimen(kind="strip", length=10.0, width=2.0, thickness=0.5, clamped=False, element=1.0)
    mesh = sp.mesh()
    res = tensile_test(sp, Membrane(mesh, _uniform(mesh, law), thickness=0.5), max_strain=0.25)
    e_young, nu = 210000.0, 0.3
    p = np.linspace(0.0, 0.5, 20001)
    sig = law.stress(p)
    e = np.expm1(sig / e_young + p)
    s = sig * np.exp(-2.0 * nu * sig / e_young - p)
    plastic = res.strain > e[0]
    np.testing.assert_allclose(res.stress[plastic], np.interp(res.strain[plastic], e, s), rtol=2e-4)
    assert res.tensile_strength == pytest.approx(s.max(), rel=1e-4)
    assert res.uniform_elongation == pytest.approx(e[np.argmax(s)], rel=2e-3)


def test_tangent_matches_finite_differences():
    sp = Specimen(kind="strip", length=2.0, width=1.0, thickness=1.0, element=1.0)
    mesh = sp.mesh()
    rng = np.random.default_rng(1)
    for law in (FlowLaw("voce", 300.0, 300.0, 20.0), FlowLaw("ludwik", 127.0, 594.0, 0.237)):
        mem = Membrane(mesh, _uniform(mesh, law), thickness=1.0)
        u = 3e-3 * rng.standard_normal(mem.ndof)
        _, k, _, bad = mem.evaluate(u)
        du = rng.standard_normal(mem.ndof)
        fp = mem.evaluate(u + 1e-7 * du)[0]
        fm = mem.evaluate(u - 1e-7 * du)[0]
        fd = (fp - fm) / 2e-7
        assert not bad
        assert np.linalg.norm(k @ du - fd) < 2e-5 * np.linalg.norm(fd)


def test_rigid_rotation_turns_the_stresses():
    sp = Specimen(kind="strip", length=1.0, width=1.0, thickness=1.0, element=1.0)
    mesh = sp.mesh()
    law = FlowLaw("voce", 300.0, 200.0, 10.0)
    mem = Membrane(mesh, _uniform(mesh, law), thickness=1.0)
    x = mesh.nodes
    stretch = np.column_stack([0.01 * x[:, 0], -0.004 * x[:, 1]])
    _, _, st, _ = mem.evaluate(stretch.ravel())
    mem.commit(stretch.ravel(), st)
    sig0, p0 = st[0][:, :, 0].copy(), st[1].copy()
    deformed = x + stretch
    for theta in np.linspace(0.1, 0.5 * math.pi, 8):
        c, s = math.cos(theta), math.sin(theta)
        r = np.array([[c, -s], [s, c]])
        u = deformed @ r.T - x
        _, _, st, bad = mem.evaluate(u.ravel())
        mem.commit(u.ravel(), st)
    assert not bad
    np.testing.assert_allclose(st[1], p0, atol=1e-12)
    # a quarter turn exchanges the normal stresses
    sig = st[0][:, :, 0]
    np.testing.assert_allclose(sig[..., 0], sig0[..., 1], atol=1e-6 * np.abs(sig0).max())
    np.testing.assert_allclose(sig[..., 1], sig0[..., 0], atol=1e-6 * np.abs(sig0).max())


def test_bands_along_the_load_mix_at_equal_strain():
    soft = FlowLaw.from_tensile(300.0, 534.0, 0.167)
    hard = FlowLaw.from_tensile(700.0, 900.0, 0.08)
    sp = Specimen(kind="strip", length=8.0, width=4.0, thickness=1.0, clamped=False, element=0.5)
    mesh = sp.mesh()
    pts = mesh.gauss()[0]
    is_hard = pts[..., 1] < 1.0  # a quarter of the width
    flow = []
    for att in ("kind", "sigma0", "a", "b"):
        hv = 0 if hard.kind == "ludwik" else 1
        sv = 0 if soft.kind == "ludwik" else 1
        hv_, sv_ = (hv, sv) if att == "kind" else (getattr(hard, att), getattr(soft, att))
        flow.append(np.where(is_hard, hv_, sv_)[..., None].astype(float))
    res = tensile_test(sp, Membrane(mesh, tuple(flow), thickness=1.0), max_strain=0.05)
    e_young, nu = 210000.0, 0.3

    def eng(law):
        p = np.linspace(0.0, 0.3, 30001)
        sig = law.stress(p)
        return np.expm1(sig / e_young + p), sig * np.exp(-2.0 * nu * sig / e_young - p)

    (eh, sh), (es, ss) = eng(hard), eng(soft)
    late = res.strain > 0.01
    mix = 0.25 * np.interp(res.strain[late], eh, sh) + 0.75 * np.interp(res.strain[late], es, ss)
    np.testing.assert_allclose(res.stress[late], mix, rtol=5e-3)


# -- design ----------------------------------------------------------------------------------


def test_gaussian_process_and_expected_improvement():
    rng = np.random.default_rng(3)
    x = rng.random((25, 2))
    y = np.sin(3.0 * x[:, 0]) + x[:, 1] ** 2
    gp = GaussianProcess(x, y)
    m, _ = gp.predict(x)
    np.testing.assert_allclose(m, y, atol=1e-3)
    xs = rng.random((50, 2))
    m, _s = gp.predict(xs)
    assert np.max(np.abs(m - (np.sin(3.0 * xs[:, 0]) + xs[:, 1] ** 2))) < 0.05
    ei = expected_improvement(np.array([0.0, 1.0]), np.array([1e-9, 1.0]), 0.5)
    assert ei[0] == pytest.approx(0.5) and 0.0 < ei[1] < 0.5


def test_case_and_its_optimization():
    case = SheetCase(
        specimen=Specimen(kind="dogbone", length=20.0, width=5.0, thickness=1.0),
        pattern={"kind": "corrugated", "count": 1, "amplitude": 1.0, "period": 10.0},
        element=1.0,
        layers=2,
    )
    res = case.evaluate()
    m = res.measures
    assert 0.15 < m["volume_fraction"] < 0.45
    assert m["tensile_strength"] > 600.0 and 0.0 < m["uniform_elongation"] < 0.17
    assert m["min_radius"] == pytest.approx(100.0 / (4.0 * math.pi**2), rel=2e-2)
    out = optimize(
        case,
        {"pattern.amplitude": [0.0, 1.5]},
        method="grid",
        levels=3,
        constraints={"yield_strength": [350.0, None]},
    )
    assert out.evaluations == 3
    feasible = [h for h in out.history if h["feasible"]]
    best = max(feasible, key=lambda h: h["strength_ductility"])
    assert out.values["pattern.amplitude"] == best["pattern.amplitude"]
    with pytest.raises(ValueError, match="objective"):
        optimize(case, {"pattern.amplitude": [0.0, 1.0]}, objective="stiffness")


def test_command_line(tmp_path):
    import yaml

    from morphogen.cli import main

    cfg = {
        "laser_sheet": {
            "material": "aisi430",
            "configuration": "symmetric",
            "layers": 1,
            "laser": {"beam": {"power": 100, "speed": 10, "radius": 0.5}},
            "specimen": {"kind": "strip", "length": 12, "width": 4, "thickness": 0.5},
            "element": 1.0,
            "pattern": {"kind": "straight", "count": 1},
            "compare": {"base": {"kind": "none"}},
        },
        "outputs": {"directory": str(tmp_path / "out"), "figures": False, "fields": ["vti"]},
    }
    path = tmp_path / "laser.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert main(["laser", str(path)]) == 0
    for name in ("laser.json", "curves.csv", "path.gcode", "path.csv", "fields.vti"):
        assert (tmp_path / "out" / name).is_file()
    import json

    summary = json.loads((tmp_path / "out" / "laser.json").read_text(encoding="utf-8"))
    treated = summary["designs"]["pattern"]["measures"]
    base = summary["designs"]["base"]["measures"]
    assert treated["mean_hardness"] > base["mean_hardness"]
    assert treated["yield_strength"] > base["yield_strength"]


# -- bending and directions ---------------------------------------------------------------


def test_three_point_bending_closed_forms():
    from morphogen.laser.bending import three_point_bending

    ns, ny, nl = 31, 6, 8
    span, width, h, young = 40.0, 10.0, 1.0, 210000.0
    shape = (ns, ny, nl)
    res = three_point_bending(
        np.ones(shape, int),
        np.full(shape, 300.0),
        np.zeros(shape),
        np.ones(shape),
        span=span,
        width=width,
        thickness=h,
        young=young,
    )
    z = (np.arange(nl) + 0.5) * h / nl - 0.5 * h
    inertia = width * h / nl * np.sum(z**2)  # the layers' second moment
    assert res.stiffness == pytest.approx(48.0 * young * inertia / span**3, rel=2e-3)
    assert res.max_load == pytest.approx(300.0 * width * h**2 / span, rel=1e-3)  # 4 M_p / S
    # a strip harder on its upper face: the neutral axis moves towards it
    s0 = np.full(shape, 300.0)
    s0[:, :, :2] = 900.0
    tab = three_point_bending(
        np.ones(shape, int),
        s0,
        np.zeros(shape),
        np.ones(shape),
        span=span,
        width=width,
        thickness=h,
    ).sections
    assert tab["axes"][ns // 2, -1] > 0.05 * h


def test_rotated_patterns_and_directions():
    beam = Beam(150.0, 10.0, 0.5)
    lengths = []
    for rot in (0.0, 45.0, 90.0):
        p = make_pattern(
            "corrugated",
            (0.0, 30.0, 0.0, 5.0),
            beam=beam,
            count=1,
            amplitude=1.0,
            period=10.0,
            rotation=rot,
        )
        lengths.append(p.clipped_length((0.0, 0.0), (30.0, 5.0)))
    # the same treated sheet cut in three directions: about the same length of tracks
    assert max(lengths) < 1.15 * min(lengths)
    turned = make_pattern("straight", (0.0, 30.0, 0.0, 5.0), beam=beam, count=2, rotation=90.0)
    for t in turned:
        assert np.allclose(t.points[:, 0], t.points[0, 0])  # tracks across the specimen
    case = SheetCase(
        specimen=Specimen(kind="dogbone", length=20.0, width=5.0, thickness=1.0),
        pattern={"kind": "straight", "count": 1},
        element=1.0,
        layers=2,
        directions=(0.0, 90.0),
    )
    m = case.evaluate().measures
    assert m["tensile_strength_0"] > m["tensile_strength_90"]  # along the track: in parallel
    assert m["strength_anisotropy"] == pytest.approx(
        abs(m["tensile_strength_0"] - m["tensile_strength_90"]) / m["tensile_strength"]
    )


def test_bending_case_and_its_configurations():
    strip = Specimen(kind="strip", length=24.0, width=6.0, thickness=1.0)
    load = {"kind": "bending", "span": 20.0, "deflection": 0.5}
    base = SheetCase(specimen=strip, pattern={"kind": "none"}, loading=load, layers=4).evaluate()
    sym = SheetCase(
        specimen=strip, pattern={"kind": "straight", "count": 2}, loading=load, layers=4
    ).evaluate()
    m0, m1 = base.measures, sym.measures
    assert math.isnan(m0["tensile_strength"])
    assert m1["bending_stiffness"] == pytest.approx(m0["bending_stiffness"], rel=1e-9)  # E kept
    assert m1["bending_load_at_deflection"] > 1.2 * m0["bending_load_at_deflection"]
    assert m1["bending_load"] > m0["bending_load"]


def test_aisi430_fully_treated_as_measured():
    # Wang et al. 2022: ten tracks at 1 mm pitch, double laser under argon (the preset's
    # hardening is fitted to these mean hardnesses)
    tr = LaserTreatment("aisi430", configuration="symmetric", layers=4)
    xs, ys = np.linspace(10.0, 20.0, 11), np.linspace(-4.5, 4.5, 46)
    pts = np.column_stack([np.repeat(xs, len(ys)), np.tile(ys, len(xs))])
    for (pa, pb), measured in (
        ((75.0, 75.0), 211.5),
        ((100.0, 100.0), 241.8),
        ((150.0, 100.0), 233.6),
    ):
        path = make_pattern(
            "straight",
            (0.0, 30.0, -5.0, 5.0),
            beam=Beam(pa, 10.0, 0.5, 0.42),
            bottom=Beam(pb, 10.0, 0.6, 0.42, profile="tophat"),
            count=10,
            spacing=1.0,
        )
        assert tr.apply(path, pts).hardness.mean() == pytest.approx(measured, abs=6.0)
