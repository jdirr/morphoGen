"""The periodic cell of a laser-architectured sheet: lattices, the periodic map, the
homogenized tensile tests against closed forms."""

import math
from dataclasses import replace

import numpy as np
import pytest

from morphogen.laser import SheetCase, sheet_material
from morphogen.laser.materials import FlowLaw
from morphogen.laser.mechanics import Membrane, hill_from_r
from morphogen.laser.periodic import (
    PeriodicCell,
    cell_test,
    lattice,
    mean_nominal,
    reduce_basis,
)
from morphogen.laser.specimen import QuadMesh

SOFT = FlowLaw.hollomon(280.0, 986.0, 0.24)
HARD = FlowLaw.hollomon(1300.0, 1966.0, 0.06)


def _flow(law, shape):
    return tuple(np.full(shape, v, dtype=float) for v in (law.code, law.sigma0, law.a, law.b))


def test_gradients_of_a_skewed_and_turned_element():
    # a linear field has the exact gradient on any parallelogram (the transpose of the
    # inverse Jacobian is wrong as soon as the element is turned)
    th = math.radians(35.0)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    nodes = np.array([[0.0, 0.0], [2.0, 0.0], [2.6, 1.0], [0.6, 1.0]]) @ rot.T
    mesh = QuadMesh(nodes, np.array([[0, 1, 2, 3]]), *(np.zeros(0, np.int64),) * 4, (1, 1))
    grad = np.array([[0.3, -0.2], [0.1, 0.4]])
    u = nodes @ grad.T
    _, dndx, w = mesh.gauss()
    g = np.einsum("ai,gaj->gij", u, dndx[0])
    assert np.allclose(g, grad)
    assert w.sum() == pytest.approx(2.0)


def test_lattices_of_the_patterns():
    a1, a2 = lattice({"kind": "straight", "angle": 30.0, "spacing": 4.0})
    assert abs(a1[0] * a2[1] - a1[1] * a2[0]) == pytest.approx(8.0)  # 2 mm along the tracks
    a1, a2 = lattice({"kind": "straight", "angle": 30.0, "spacing": 4.0}, along=4.0)
    assert abs(a1[0] * a2[1] - a1[1] * a2[0]) == pytest.approx(16.0)
    assert a1 @ a2 == pytest.approx(0.0, abs=1e-12)
    # crossed: the area of the cell is s1 s2 / sin 2θ, the basis reduced
    a1, a2 = lattice({"kind": "crossed", "angle": 20.0, "spacing": 3.0, "spacing2": 5.0})
    assert abs(a1[0] * a2[1] - a1[1] * a2[0]) == pytest.approx(15.0 / math.sin(math.radians(40)))
    assert abs(a1 @ a2) <= 0.5 * (a1 @ a1) + 1e-9
    assert a1 @ a1 <= a2 @ a2 + 1e-9
    a1, a2 = lattice({"kind": "honeycomb", "cell": 2.0, "angle": 10.0, "rotation": 20.0})
    assert np.linalg.norm(a1) == pytest.approx(2.0 * math.sqrt(3.0))
    assert np.linalg.norm(a2) == pytest.approx(2.0 * math.sqrt(3.0))
    b1, b2 = reduce_basis([1.0, 0.0], [7.3, 1.0])
    assert np.allclose(b1, [1.0, 0.0]) and np.allclose(b2, [0.3, 1.0])
    with pytest.raises(ValueError):
        lattice({"kind": "straight", "count": 1})  # a family needs its spacing
    with pytest.raises(ValueError):
        reduce_basis([1.0, 2.0], [2.0, 4.0])


@pytest.mark.parametrize("angle", [0.0, 30.0, 70.0])
def test_the_map_carries_the_mean_gradient(angle):
    a1, a2 = lattice({"kind": "honeycomb", "cell": 1.0, "angle": 15.0})
    cell = PeriodicCell.build(a1, a2, element=0.4, centre=(3.0, 1.0))
    T = cell.map(angle)
    rng = np.random.default_rng(1)
    q = rng.normal(0.0, 0.01, T.shape[1])
    a, b, c = 0.05, -0.02, 0.01
    q[-3:] = (a, b, c)
    u = (T @ q).reshape(-1, 2)
    _, dndx, w = cell.mesh.gauss()
    g = np.einsum("eai,egaj->egij", u[cell.mesh.quads], dndx)
    mean = (g * w[..., None, None]).sum(axis=(0, 1)) / w.sum()
    th = math.radians(angle)
    n, t = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
    H = a * np.outer(n, n) + b * np.outer(t, t) + c * (np.outer(n, t) + np.outer(t, n))
    assert np.allclose(mean, H, atol=1e-12)
    assert w.sum() == pytest.approx(cell.area)


def test_an_untreated_sheet_is_isotropic_in_the_cell():
    cell = PeriodicCell.build([2.0, 0.0], [0.0, 2.0], element=0.5)
    ne = len(cell.mesh.quads)
    mem = Membrane(cell.mesh, _flow(SOFT, (ne, 4, 1)))
    a = cell_test(mem, cell, 0.0, max_strain=0.15, step=0.002)
    b = cell_test(mem, cell, 35.0, max_strain=0.15, step=0.002)
    assert a.r_value() == pytest.approx(1.0, abs=0.01)
    assert b.r_value() == pytest.approx(1.0, abs=0.01)
    assert b.yield_strength == pytest.approx(a.yield_strength, rel=1e-6)
    # the nominal curve of the law: true stress K (ε0 + p)^n over the stretch
    e = 0.1
    p = math.log(1 + e) - SOFT.stress(0.1) / 210000.0
    nominal = float(SOFT.stress(p)) / (1 + e)
    assert np.interp(e, a.strain, a.stress) == pytest.approx(nominal, rel=0.01)


def test_biaxial_stretching_of_an_untreated_sheet():
    """Linear paths of the true strains: the minor strain keeps the ratio, the mean
    nominal stresses are those of von Mises, the major force peaks at the exponent."""
    cell = PeriodicCell.build([2.0, 0.0], [0.0, 2.0], element=1.0)
    ne = len(cell.mesh.quads)
    mem = Membrane(cell.mesh, _flow(SOFT, (ne, 4, 1)))
    # uniaxial: the mean nominal stress of the membrane is the stress of the test
    res = cell_test(
        mem, cell, 0.0, max_strain=0.1, monitors={"P": lambda q: float(mean_nominal(mem)[0, 0])}
    )
    assert np.allclose(res.fields["monitors"]["P"], res.stress, atol=1e-8)
    for rho in (0.0, 0.5, 1.0):
        res = cell_test(mem, cell, 30.0, ratio=rho, max_strain=0.4, drop=1.0, step=0.005)
        mac = res.fields["macroscopic"]
        e1, e2 = np.log1p(mac["a"]), np.log1p(mac["b"])
        assert np.allclose(e2, rho * e1, atol=1e-9)
        assert np.abs(mac["c"]).max() < 1e-9
        # P_tt / P_nn = (σ_t / σ_n) λ_n / λ_t, σ_t / σ_n = (2ρ + 1) / (ρ + 2)
        nom = res.fields["nominal"]
        k = len(e1) // 2
        alpha = (2 * rho + 1) / (rho + 2)
        assert nom["tt"][k] / nom["nn"][k] == pytest.approx(
            alpha * math.exp(e1[k] - e2[k]), rel=0.02
        )
        # the diffuse necking along n: P_nn ∝ ε^n e^(-ε) peaks at ε = n (rigid plastic)
        top = int(np.argmax(res.stress))
        assert math.log1p(res.strain[top]) == pytest.approx(0.24, abs=0.015)


def test_a_cell_of_given_shape():
    """A skewed cell meshed by the numbers of elements asked: its nodes match on
    opposite edges through the lattice vectors."""
    cell = PeriodicCell.build([3.0, 1.0], [-1.0, 2.0], shape=(6, 4))
    assert cell.mesh.shape == (6, 4)
    assert len(cell.mesh.quads) == 24
    nodes = cell.mesh.nodes
    d = cell.shift[:, :1] * cell.a1 + cell.shift[:, 1:] * cell.a2
    assert np.allclose(nodes, nodes[cell.master] + d)
    with pytest.raises(ValueError):
        PeriodicCell.build([3.0, 0.0], [0.0, 2.0], shape=(1, 4))


def test_the_shear_free_or_held():
    """A textured sheet stretched at 30° from its axes: with the shear free it shears;
    held, its principal axes stay along the test and it carries a mean shear stress."""
    cell = PeriodicCell.build([2.0, 0.0], [0.0, 2.0], element=1.0)
    ne = len(cell.mesh.quads)
    aniso = np.broadcast_to(np.array(hill_from_r(1.6, 1.2, 2.0)), (ne, 4, 1, 5))
    mem = Membrane(cell.mesh, _flow(SOFT, (ne, 4, 1)), anisotropy=aniso)
    kw = {"ratio": -0.25, "max_strain": 0.1, "drop": 1.0, "step": 0.02}
    free = cell_test(mem, cell, 30.0, **kw)
    held = cell_test(mem, cell, 30.0, free_shear=False, **kw)
    assert abs(free.fields["macroscopic"]["c"][-1]) > 1e-3
    mac = held.fields["macroscopic"]
    assert np.abs(mac["c"]).max() == 0.0
    assert np.allclose(np.log1p(mac["b"]), -0.25 * np.log1p(mac["a"]), atol=1e-9)
    nom = held.fields["nominal"]
    assert abs(nom["nt"][-1]) > 0.02 * nom["nn"][-1]
    assert abs(free.fields["nominal"]["nt"][-1]) < 0.2 * abs(nom["nt"][-1])
    # isotropic: the same either way
    iso = Membrane(cell.mesh, _flow(SOFT, (ne, 4, 1)))
    a = cell_test(iso, cell, 30.0, **kw)
    b = cell_test(iso, cell, 30.0, free_shear=False, **kw)
    assert np.allclose(a.stress, b.stress, rtol=1e-6)


def test_a_test_stops_when_asked():
    cell = PeriodicCell.build([2.0, 0.0], [0.0, 2.0], element=1.0)
    ne = len(cell.mesh.quads)
    mem = Membrane(cell.mesh, _flow(SOFT, (ne, 4, 1)))
    res = cell_test(mem, cell, 0.0, max_strain=0.3, step=0.01, until=lambda q: q[-3] >= 0.05)
    a = res.fields["macroscopic"]["a"]
    assert 0.05 <= a[-1] < 0.05 + 0.02 * 0.3 + 1e-9
    assert a[-2] < 0.05


@pytest.mark.parametrize("angle", [0.0, 30.0, 45.0, 75.0, 90.0])
def test_a_textured_sheet_follows_hill(angle):
    F, G, H, N, _ = hill_from_r(1.6, 1.2, 2.0)
    cell = PeriodicCell.build([2.0, 0.0], [0.0, 2.0], element=1.0)
    ne = len(cell.mesh.quads)
    aniso = np.broadcast_to(np.array(hill_from_r(1.6, 1.2, 2.0)), (ne, 4, 1, 5))
    mem = Membrane(cell.mesh, _flow(SOFT, (ne, 4, 1)), anisotropy=aniso)
    res = cell_test(mem, cell, angle, max_strain=0.12)
    s2, c2 = math.sin(math.radians(angle)) ** 2, math.cos(math.radians(angle)) ** 2
    r = (H + (2 * N - F - G - 4 * H) * s2 * c2) / (F * s2 + G * c2)
    assert res.r_value() == pytest.approx(r, rel=0.02)
    ratio = 1.0 / math.sqrt(F * s2 * s2 + G * c2 * c2 + H * (c2 - s2) ** 2 + 2 * N * s2 * c2)
    ref = cell_test(mem, cell, 0.0, max_strain=0.12)
    assert res.yield_strength / ref.yield_strength == pytest.approx(ratio, rel=0.02)


def test_a_laminate_in_parallel_and_in_series():
    cell = PeriodicCell.build([2.0, 0.0], [0.0, 2.0], element=0.25)
    ne = len(cell.mesh.quads)
    hard = cell.mesh.nodes[cell.mesh.quads].mean(axis=1)[:, 1] < cell.origin[1] + 1.0
    flow = tuple(
        np.where(hard[:, None, None], np.full((ne, 4, 1), vh), np.full((ne, 4, 1), vs))
        for vh, vs in zip(
            (HARD.code, HARD.sigma0, HARD.a, HARD.b),
            (SOFT.code, SOFT.sigma0, SOFT.a, SOFT.b),
            strict=True,
        )
    )
    mem = Membrane(cell.mesh, flow)
    along = cell_test(mem, cell, 0.0, max_strain=0.06)
    across = cell_test(mem, cell, 90.0, max_strain=0.06)
    e = 0.05
    p = math.log(1 + e)
    mix = 0.5 * (HARD.stress(p - 1300.0 / 210000.0) + SOFT.stress(p - 280.0 / 210000.0)) / (1 + e)
    assert np.interp(e, along.strain, along.stress) == pytest.approx(float(mix), rel=0.01)
    # across, the hard layers hold the width of the soft ones: nearly plane strain
    assert across.r_value(0.05) < 0.15
    assert np.interp(e, across.strain, across.stress) < 0.7 * float(mix)


def test_the_cell_of_a_treated_sheet():
    mat = replace(sheet_material("annealed_ms1500"), anisotropy={"r0": 1.6, "r45": 1.2, "r90": 2.0})
    case = SheetCase(
        material=mat,
        element=0.5,
        pattern={"kind": "straight", "angle": 0.0, "count": None, "spacing": 6.0},
        loading={"kind": "cell", "max_strain": 0.12},
        directions=(0.0, 90.0),
    )
    cell = case.cell()
    assert cell.area == pytest.approx(12.0)  # 6 mm across the tracks, 2 mm along them
    res = case.evaluate()
    m = res.measures
    assert 0.1 < m["volume_fraction"] < 0.4
    assert m["path_length"] > 0  # the path's measures: on the specimen
    # the hard tracks along the rolling direction: strong along it, nearly no width
    # strain across it
    assert m["yield_strength_0"] > 1.1 * m["yield_strength_90"]
    assert m["r_value_90"] < 0.7 < m["r_value_0"]
    plain = SheetCase(material=mat, element=0.5, pattern={"kind": "none"}, loading={"kind": "cell"})
    assert plain.evaluate().measures["r_value"] == pytest.approx(1.6, rel=0.02)


# -- Yld2000-2d ---------------------------------------------------------------------------

DC04 = (151.0, 166.0, 163.0, 192.0, 1.83, 1.39, 2.11, 0.87)  # Banabic & Sester 2012


def test_yld2000_is_von_mises_when_isotropic_and_quadratic():
    from morphogen.laser.mechanics import _yld_phi

    g = np.empty(3)
    rng = np.random.default_rng(3)
    for sx, sy, sxy in rng.normal(size=(5, 3)):
        phi = _yld_phi(sx, sy, sxy, np.ones(8), 2.0, g)
        assert phi / 2 == pytest.approx(sx * sx - sx * sy + sy * sy + 3 * sxy * sxy)
        num = []
        for j in range(3):
            x = np.array([sx, sy, sxy])
            xp, xm = x.copy(), x.copy()
            xp[j] += 1e-6
            xm[j] -= 1e-6
            num.append(
                (
                    _yld_phi(*xp, np.ones(8), 2.0, np.empty(3))
                    - _yld_phi(*xm, np.ones(8), 2.0, np.empty(3))
                )
                / 2e-6
            )
        assert np.allclose(g, num, rtol=1e-6, atol=1e-6)


def test_yld2000_fitted_to_a_deep_drawing_steel():
    from morphogen.laser import yld2000_from_data, yld2000_profile

    c = yld2000_from_data(*DC04, a=6.0, angle=30.0)
    assert len(c) == 10 and c[8] == 6.0 and c[9] == pytest.approx(math.radians(30.0))
    prof = yld2000_profile(c[:8], c[8], angles=(0.0, 45.0, 90.0))
    assert np.allclose(prof["stress"], [1.0, 166 / 151, 163 / 151], atol=1e-8)
    assert prof["stress_b"] == pytest.approx(192 / 151, abs=1e-8)
    assert np.allclose(prof["r"], [1.83, 1.39, 2.11], atol=1e-8)
    assert prof["r_b"] == pytest.approx(0.87, abs=1e-8)
    with pytest.raises(ValueError):
        yld2000_from_data(151, 166, 163, 192, 1.83, -1.0, 2.11)


@pytest.mark.parametrize("angle", [0.0, 30.0, 60.0, 90.0])
def test_a_yld2000_sheet_in_the_cell(angle):
    from morphogen.laser import yld2000_from_data, yld2000_profile

    c = yld2000_from_data(*DC04)
    cell = PeriodicCell.build([2.0, 0.0], [0.0, 2.0], element=1.0)
    ne = len(cell.mesh.quads)
    mem = Membrane(cell.mesh, _flow(SOFT, (ne, 4, 1)), anisotropy=np.array(c))
    res = cell_test(mem, cell, angle, max_strain=0.12)
    ref = cell_test(mem, cell, 0.0, max_strain=0.12)
    prof = yld2000_profile(c[:8], c[8], angles=(angle,))
    assert res.r_value() == pytest.approx(prof["r"][0], rel=0.025)
    assert res.yield_strength / ref.yield_strength == pytest.approx(prof["stress"][0], rel=0.01)
    assert res.iterations <= 3 * res.steps  # the consistent tangent


def test_the_yld2000_texture_of_a_material():
    mat = replace(
        sheet_material("annealed_ms1500"),
        anisotropy={
            "criterion": "yld2000",
            **dict(zip(("s0", "s45", "s90", "sb", "r0", "r45", "r90", "rb"), DC04, strict=True)),
            "memory": 0.5,
        },
    )
    crit = mat.yield_criterion(np.array([0.0, 1.0]), angle=-30.0)
    assert crit.shape == (2, 10)
    assert crit[0, 8] == 6.0 and crit[0, 9] == pytest.approx(math.radians(-30.0))
    # the treated point halfway to isotropy
    assert np.allclose(crit[1, :8], 1.0 + 0.5 * (crit[0, :8] - 1.0))
    with pytest.raises(ValueError):
        replace(mat, anisotropy={"criterion": "barlat91"}).yield_criterion(0.0)
