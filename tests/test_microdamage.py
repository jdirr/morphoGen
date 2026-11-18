"""Cleavage planes regularized by their microdamage (Aslan & Forest 2009): the band of a
plane that breaks and its closed form, the law of the planes at a prescribed microdamage,
the equation of the microdamage on voxels and on meshes, a bar that breaks across a band
whose width does not depend on the grid, the planes broken through, a polycrystal."""

import itertools
import math

import numpy as np
import pytest

from helpers import box_mesh
from morphogen.homogenization import Material
from morphogen.homogenization import crystal as cr
from morphogen.homogenization.behaviours import update_points
from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.homogenization.fft_nonlinear import NonlinearFFTProblem
from morphogen.homogenization.finite import update_points_finite
from morphogen.homogenization.materials import material_table
from morphogen.homogenization.nonlinear import (
    NonlinearFEProblem,
    behaviour_table,
    simulate_point,
)
from morphogen.image import VoxelImage

#: Cubic elasticity [c11, c12, c44] of iron (MPa) and its Young modulus along <100>
IRON = (230000.0, 135000.0, 117000.0)
E100 = (IRON[0] - IRON[1]) * (IRON[0] + 2 * IRON[1]) / (IRON[0] + IRON[1])
#: The planes of the bar: their strength (MPa), their fracture energy (N/mm), the width of
#: their damage band (mm) and the length of the bar
STRENGTH, ENERGY, WIDTH, LENGTH = 500.0, 0.2, 0.025, 0.1
#: The zone where the band starts: slightly weaker, wider than the band
WEAK, ZONE = 0.99, 0.03125


def iron(strength=STRENGTH, softening=40.0, modulus=60000.0, length=0.0035, k=0.5, **more):
    planes = {"planes": "cube", "strength": strength, "softening": softening, "K": k, "n": 1.0,
              "regularization": {"modulus": modulus, "length": length}}  # fmt: skip
    return Material.from_config(
        {"cubic": list(IRON), "crystal_plasticity": {"cleavage": planes, **more}}
    )


# -- the band and its parameters -------------------------------------------------------------------


def test_band_of_a_plane_that_breaks():
    """The closed form of the band: its width, the opening that breaks it and the
    fracture energy; the parameters found from the energy and the width; the damage and
    the microdamage of the closed form solve the equation of the microdamage and keep
    the strength of the planes at the stress of the bar."""
    y0, w, chi, ell = 500.0, 40.0, 60000.0, 0.0035
    h = w * y0
    r = h / chi
    theta = math.pi - math.atan(math.sqrt(r / (1 - r)))
    omega = math.sqrt(r / (1 - r)) / ell
    width, energy = cr.microdamage_band(y0, w, chi, ell)
    assert width == pytest.approx(2 * theta / omega, rel=1e-14)
    assert energy == pytest.approx(y0**2 * (width + 2 * ell) / (2 * h), rel=1e-14)
    found = cr.microdamage_parameters(y0, energy, width, r)
    assert found["softening"] == pytest.approx(w, rel=1e-12)
    assert found["modulus"] == pytest.approx(chi, rel=1e-12)
    assert found["length"] == pytest.approx(ell, rel=1e-12)
    # by default a third: the band is some seven lengths wide
    third = cr.microdamage_parameters(y0, 0.2, 0.025)
    assert third["modulus"] == pytest.approx(3 * third["softening"] * y0, rel=1e-12)
    assert 0.025 / third["length"] == pytest.approx(
        2 * (math.pi - math.atan(math.sqrt(0.5))) * math.sqrt(2), rel=1e-12
    )
    # the fields of the closed form at a stress of the bar
    sigma = 200.0
    s = y0 - sigma
    beta = omega * ell * s / (h * math.sin(theta))
    alpha = beta * (1 - r)
    x = np.linspace(-0.03, 0.03, 60001)
    inside = np.abs(x) < width / 2
    damage = np.where(inside, s / h + beta * np.cos(omega * x), 0.0)
    micro = np.where(inside, s / h + alpha * np.cos(omega * x),
                     s / chi * np.exp(-(np.abs(x) - width / 2) / ell))  # fmt: skip
    # the damage comes to nothing at the edge of the band, where the microdamage is smooth
    edge = np.argmin(np.abs(x - width / 2))
    assert abs(s / h + beta * math.cos(theta)) < 1e-12 * s / h
    assert abs(micro[edge + 1] - micro[edge - 1]) < 2e-3 * micro[edge]
    # chi_d - ell^2 chi_d'' = d
    dx = x[1] - x[0]
    second = (micro[2:] - 2 * micro[1:-1] + micro[:-2]) / dx**2
    residual = micro[1:-1] - ell**2 * second - damage[1:-1]
    away = np.abs(np.abs(x[1:-1]) - width / 2) > 5 * dx
    assert np.abs(residual[away]).max() < 1e-6 * damage.max()
    # the strength of the planes of the band is the stress; beyond it, more
    strength = y0 - h * damage + chi * (damage - micro)
    assert np.allclose(strength[inside], sigma, rtol=0, atol=1e-9 * y0)
    assert np.all(strength[~inside] >= sigma - 1e-9 * y0)
    # the opening of the band and the work that breaks the bar
    opening = np.trapezoid(damage, x)
    assert opening == pytest.approx(s * (width + 2 * ell) / h, rel=1e-6)
    assert energy == pytest.approx(0.5 * y0 * y0 * (width + 2 * ell) / h, rel=1e-14)
    # the largest damage of the band as it breaks, and how far ahead of its microdamage
    assert (beta * (1 + 1 / (omega * ell / math.sin(theta)))) * y0 / s == pytest.approx(
        (y0 / h) * (1 + 1 / math.sqrt(1 - r)), rel=1e-12
    )
    assert (beta - alpha) * y0 / s == pytest.approx((y0 / chi) / math.sqrt(1 - r), rel=1e-12)
    for bad in ((y0, w, h, ell), (y0, w, 0.5 * h, ell)):
        with pytest.raises(ValueError, match="micromorphic modulus above"):
            cr.microdamage_band(*bad)
    for bad in ((y0, 0.0, 0.02, 1 / 3), (y0, 0.2, 0.02, 1.0), (y0, 0.2, -1.0, 0.5)):
        with pytest.raises(ValueError, match="microdamage"):
            cr.microdamage_parameters(*bad)


def test_regularization_in_a_configuration():
    """The modulus and the length with the softening of the planes, or the width of the
    band with their fracture energy; the internal variables and the fields."""
    m = iron()
    cp = m.crystal_plasticity
    assert cp.regularized and cp.cleavage_regularization == (60000.0, 0.0035)
    assert cp.n_planes == 3 and cp.micro_width == 4
    n = cp.n_systems
    assert cr.state_width(n, micro=cp.micro_width) == cr.state_width(n) + 4
    assert cp.field_names == ("cleavage_opening", "cleavage_damage", "cleavage_microdamage")
    assert np.all(cp.kinds_array()[:, 10] == 60000.0)
    again = Material.from_config(
        {"cubic": list(IRON), "crystal_plasticity": cp.to_config()}
    ).crystal_plasticity
    assert again == cp
    assert again.to_config()["cleavage"]["regularization"] == {"modulus": 60000.0, "length": 0.0035}
    # from the energy of the planes and the width of their band
    planes = {"planes": "cube", "strength": 500.0, "energy": 0.2, "K": 0.5, "n": 1.0,
              "regularization": {"width": 0.025}}  # fmt: skip
    band = Material.from_config(
        {"cubic": list(IRON), "crystal_plasticity": {"cleavage": planes}}
    ).crystal_plasticity
    found = cr.microdamage_parameters(500.0, 0.2, 0.025)
    assert band.cleavage_regularization == (found["modulus"], found["length"])
    assert band.cleavage[0][1][1] == pytest.approx(found["softening"], rel=1e-14)
    width, energy = cr.microdamage_band(
        500.0, band.cleavage[0][1][1], *band.cleavage_regularization
    )
    assert (width, energy) == (pytest.approx(0.025, rel=1e-12), pytest.approx(0.2, rel=1e-12))
    half = {**planes, "regularization": {"width": 0.025, "ratio": 0.5}}
    other = Material.from_config(
        {"cubic": list(IRON), "crystal_plasticity": {"cleavage": half}}
    ).crystal_plasticity
    assert other.cleavage_regularization[0] == pytest.approx(
        2 * other.cleavage[0][1][1] * 500.0, rel=1e-12
    )
    # planes that are not regularized keep their variables and their fields
    plain = Material.from_config(
        {"cubic": list(IRON), "crystal_plasticity": {"cleavage": {"planes": "cube",
         "strength": 500.0, "softening": 40.0, "K": 0.5, "n": 1.0}}}
    ).crystal_plasticity  # fmt: skip
    assert not plain.regularized and plain.micro_width == 0
    assert plain.field_names == ("cleavage_opening", "cleavage_damage")
    assert "regularization" not in plain.to_config()["cleavage"]

    def refused(planes, match):
        with pytest.raises(ValueError, match=match):
            Material.from_config({"cubic": list(IRON), "crystal_plasticity": {"cleavage": planes}})

    base = {"planes": "cube", "strength": 500.0, "softening": 40.0, "K": 0.5, "n": 1.0}
    refused({**base, "regularization": {"modulus": 20000.0, "length": 0.0035}}, "modulus")
    refused({**base, "regularization": {"modulus": 60000.0, "length": 0.0}}, "length")
    refused({**base, "regularization": {"modulus": 60000.0}}, "modulus, length")
    refused({**base, "regularization": 0.01}, "modulus, length")
    refused({**base, "regularization": {"width": 0.025}}, "fracture energy")
    refused({**planes, "length": 0.01}, "fracture energy")
    refused({**planes, "regularization": {"width": 0.025, "ratio": 1.5}}, "ratio")
    refused({**planes, "regularization": {"width": "wide"}}, "numbers")


# -- the law of the planes at a prescribed microdamage ---------------------------------------------


def _kernel(material, finite=False):
    mats = {0: material}
    table = material_table([0], mats, "elastic", grains=[0], orientations=np.zeros((1, 3)))
    return table, behaviour_table([0], mats, table, finite=finite)


def _point(table, beh, grad, old, dt, finite=False):
    nc = 9 if finite else 6
    new = np.zeros_like(old)
    s, a = np.zeros((1, nc)), np.zeros((1, nc, nc))
    update = update_points_finite if finite else update_points
    failed = update(beh.kind, table.mat, beh.pmat, beh.data, grad[None], 0.0, dt, old, new, s, a,
                    0, 1.0, False)  # fmt: skip
    assert failed == 0
    return s[0], a[0], new


def _opening_system(cp, axis):
    """The opening system of the plane normal to an axis of the crystal, and its plane."""
    kinds = cp.kinds_array()
    normals, _ = cp.systems()
    system = next(
        s for s in range(cp.n_systems) if kinds[s, 0] == cr.OPENING and abs(normals[s, axis]) > 0.9
    )
    return system, int(kinds[system, 3])


def test_planes_at_a_prescribed_microdamage():
    """A crystal strained along [001] at a prescribed microdamage: the opening of its
    (001) plane and its stress, in closed form; its tangent; a point alone, whose
    microdamage is its damage, softens with it; a plane broken through keeps no strength,
    whatever it opens; two planes that open soften together, with the damage of both."""
    y0, w, chi, kd, dt = 500.0, 40.0, 60000.0, 0.5, 0.5
    m = iron(k=kd)
    cp = m.crystal_plasticity
    table, beh = _kernel(m)
    n = cp.n_systems
    slot = cr.state_width(n)
    assert beh.width == slot + 4
    c11, c12 = IRON[0], IRON[1]
    h = w * y0
    system, plane = _opening_system(cp, 2)
    eps = 0.0025
    grad = np.array([0.0, 0.0, eps, 0.0, 0.0, 0.0])
    for micro in (0.0, 0.004, -1.0):
        old = np.zeros((1, beh.width))
        old[0, slot] = micro
        stress, tangent, new = _point(table, beh, grad, old, dt)
        if micro < 0:  # a point alone: Y_0 (1 - w d)
            exact = (c11 * eps - y0) / (c11 - h + kd / dt)
            strength = y0 - h * exact
        else:  # Y_0 (1 - w d) + chi_H (d - chi_d)
            exact = (c11 * eps - y0 + chi * micro) / (c11 - h + chi + kd / dt)
            strength = y0 - h * exact + chi * (exact - micro)
        assert new[0, cr.CP_ALPHA + system] == pytest.approx(exact, rel=1e-12)
        assert new[0, cr.CP_ALPHA + n + system] == pytest.approx(exact, rel=1e-12)
        assert stress[2] == pytest.approx(c11 * (eps - exact), rel=1e-12)
        assert stress[2] == pytest.approx(strength + kd * exact / dt, rel=1e-10)
        assert new[0, slot] == micro and not new[0, slot + 1 :].any()  # kept as they are
        fd = np.zeros((6, 3))
        for j in range(3):
            step = np.zeros(6)
            step[j] = 1e-8
            fd[:, j] = (
                _point(table, beh, grad + step, old, dt)[0]
                - _point(table, beh, grad - step, old, dt)[0]
            ) / 2e-8
        assert np.abs(tangent[:, :3] - fd).max() < 1e-7 * np.abs(fd).max()
        fields = cp.state_fields(new)
        assert fields["cleavage_opening"][0] == pytest.approx(exact, rel=1e-12)
        assert fields["cleavage_microdamage"][0] == pytest.approx(exact if micro < 0 else micro)
        lost = max(1 - strength / y0, 0.0)  # (stronger than at first: none)
        assert fields["cleavage_damage"][0] == pytest.approx(lost, rel=1e-10, abs=1e-12)
    # a microdamage that has taken the strength of the planes: the floor of their law
    old = np.zeros((1, beh.width))
    old[0, slot] = 0.05
    stress, _, new = _point(table, beh, grad, old, dt)
    assert 0.0 < stress[2] < 2 * kd * eps / dt + 1e-5 * y0
    assert cp.state_fields(new)["cleavage_damage"][0] == 1.0
    # a plane broken through: marked with the damage that it had, it opens with no stress
    # to speak of, and the damage of the point stays that of the mark
    old = np.zeros((1, beh.width))
    old[0, slot] = 0.004
    old[0, slot + 1 + plane] = 1.0 + 0.01
    for column in (cr.CP_ALPHA + system, cr.CP_ALPHA + n + system, 2):
        old[0, column] = 0.012  # it has opened since
    wide = np.array([0.0, 0.0, 0.02, 0.0, 0.0, 0.0])
    stress, _, new = _point(table, beh, wide, old, dt)
    exact = (c11 * (0.02 - 0.012) - cr._BROKEN * y0) / (c11 + kd / dt)
    assert new[0, cr.CP_ALPHA + system] - 0.012 == pytest.approx(exact, rel=1e-9)
    assert stress[2] == pytest.approx(cr._BROKEN * y0 + kd * exact / dt, rel=1e-6)
    assert new[0, slot + 1 + plane] == 1.01
    assert cp.state_fields(new)["cleavage_damage"][0] == 1.0
    # the other planes of the point: their strength for the damage of the mark, 0.01
    lateral = np.array([0.006, 0.0, 0.012, 0.0, 0.0, 0.0])
    stress, _, new = _point(table, beh, lateral, old, dt)
    other, _ = _opening_system(cp, 0)
    delta = new[0, cr.CP_ALPHA + other]
    assert delta > 1e-5
    strength = y0 - h * (0.01 + delta) + chi * (0.01 + delta - 0.004)
    assert stress[0] == pytest.approx(strength + kd * delta / dt, rel=1e-9)
    # two planes that open alike: one threshold, for the damage of both
    micro = 0.002
    old = np.zeros((1, beh.width))
    old[0, slot] = micro
    both = np.array([eps, eps, 0.0, 0.0, 0.0, 0.0])
    stress, tangent, new = _point(table, beh, both, old, dt)
    exact = ((c11 + c12) * eps - y0 + chi * micro) / (c11 + c12 + 2 * (chi - h) + kd / dt)
    first, second = _opening_system(cp, 0)[0], _opening_system(cp, 1)[0]
    assert new[0, cr.CP_ALPHA + first] == pytest.approx(exact, rel=1e-10)
    assert new[0, cr.CP_ALPHA + second] == pytest.approx(exact, rel=1e-10)
    strength = y0 - h * 2 * exact + chi * (2 * exact - micro)
    assert stress[0] == pytest.approx(strength + kd * exact / dt, rel=1e-10)
    assert stress[0] == pytest.approx(stress[1], rel=1e-12)


@pytest.mark.parametrize("finite", [False, True])
def test_tangent_of_regularized_planes(finite):
    """The consistent tangent of a crystal that slips and whose regularized planes open
    under a general strain, in small and in finite strains."""
    band = {"modulus": 60000.0, "length": 0.0035}
    planes = {"planes": "cube", "strength": 500.0, "softening": 40.0, "K": 20.0, "n": 1.0,
              "coupling": 3.0, "regularization": band}  # fmt: skip
    slip = {"bcc110": {"tau0": 150.0, "Q": 60.0, "b": 15.0}}
    m = Material.from_config(
        {"cubic": list(IRON),
         "crystal_plasticity": {"K": 10.0, "n": 10.0, "families": slip, "cleavage": planes}}
    )  # fmt: skip
    cp = m.crystal_plasticity
    mats = {0: m}
    table = material_table([0], mats, "elastic", grains=[0],
                           orientations=np.array([[12.0, 31.0, 47.0]]))  # fmt: skip
    beh = behaviour_table([0], mats, table, finite=finite)
    n = cp.n_systems
    slot = cr.state_width(n, finite)
    dt = 0.3
    small = np.array([0.0030, -0.0006, 0.0022, 0.0012, -0.0008, 0.0010])
    if finite:
        half = 0.5 * small[3:]
        grad0 = np.concatenate([small[:3], half, half])
    else:
        grad0 = small
    old = np.zeros((1, beh.width))
    old[0, slot] = 0.0005
    # a first step to a state where the crystal slips and its planes have opened
    _, _, state = _point(table, beh, grad0, old, dt, finite)
    assert cp.state_fields(state, finite)["cleavage_opening"][0] > 1e-4
    assert state[0, 6] > 1e-5
    state[0, slot] = 0.0012  # its microdamage has followed
    grad = 1.15 * grad0
    _, tangent, new = _point(table, beh, grad, state, dt, finite)
    opened = (
        new[0, cr.CP_ALPHA + n : cr.CP_ALPHA + 2 * n]
        - state[0, cr.CP_ALPHA + n : cr.CP_ALPHA + 2 * n]
    )
    kinds = cp.kinds_array()[:, 0]
    assert (opened[kinds == cr.OPENING] > 1e-6).any() and (opened[kinds == cr.SLIP] > 1e-7).any()
    nc = grad.size
    fd = np.zeros((nc, nc))
    for j in range(nc):
        step = np.zeros(nc)
        step[j] = 1e-7
        fd[:, j] = (
            _point(table, beh, grad + step, state, dt, finite)[0]
            - _point(table, beh, grad - step, state, dt, finite)[0]
        ) / 2e-7
    assert np.abs(tangent - fd).max() < 2e-6 * np.abs(fd).max()


def test_point_alone_softens_with_its_damage():
    """A material point has no gradient: its microdamage is its damage, and its planes
    soften linearly with it, down to nothing; planes that soften faster than the crystal
    unloads are refused."""
    y0, w = 500.0, 40.0
    m = iron(k=1e-3)
    path = {"times": [0, 1], "increments": 100,
            "strain": {"11": [0.0], "22": [0.0], "33": [0.03], "23": [0.0], "31": [0.0],
                       "12": [0.0]}}  # fmt: skip
    res = simulate_point(m, path)
    c11 = IRON[0]
    eps, stress = res.strain[:, 2], res.stress[:, 2]
    opening = res.series["cleavage_opening"]
    exact = np.clip((c11 * eps - y0) / (c11 - w * y0), 0.0, None)
    broken = exact >= 1 / w  # the opening that takes all the strength: 1 / w
    assert np.allclose(opening[~broken], exact[~broken], rtol=0, atol=2e-6)
    assert np.allclose(stress[~broken], c11 * (eps[~broken] - exact[~broken]), rtol=0, atol=0.5)
    assert broken.any() and np.all(np.abs(stress[broken][1:]) < 0.1)
    assert np.allclose(res.series["cleavage_damage"][~broken], w * exact[~broken], atol=1e-4)
    assert np.all(res.series["cleavage_damage"][broken][1:] == 1.0)
    # its microdamage: its damage
    assert np.allclose(res.series["cleavage_microdamage"], opening, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="softens faster"):
        simulate_point(iron(softening=500.0, modulus=600000.0), path)


# -- the equation of the microdamage ---------------------------------------------------------------


def _slab(x, half, ell, period):
    """The microdamage of a unit damage over a slab of half-width ``half`` about the
    middle of a periodic cell: the sum of its images."""
    out = np.zeros_like(x)
    for k in range(-8, 9):
        r = np.abs(x - period / 2 + k * period)
        out += np.where(r < half, 1.0 - np.exp(-half / ell) * np.cosh(r / ell),
                        np.sinh(half / ell) * np.exp(-r / ell))  # fmt: skip
    return out


def test_equation_of_the_microdamage():
    """``chi - ell^2 lap chi = d`` for the damage of a slab, on voxels and on meshes of
    linear and of quadratic tetrahedra, against its closed form: the microdamage keeps the
    integral of the damage, and is positive on the grid."""
    n, ell, half = 64, 0.0035, 0.0125
    m = iron()
    image = VoxelImage(np.zeros((n, 1, 1), dtype=np.uint8), spacing=(LENGTH / n,) * 3)
    x = (np.arange(n) + 0.5) * LENGTH / n
    source = (np.abs(x - LENGTH / 2) < half).astype(float)
    exact = _slab(x, half, ell, LENGTH)
    fft = NonlinearFFTProblem(image, {0: m})
    found = fft._smooth(np.arange(n), source, ell)
    assert found.sum() == pytest.approx(source.sum(), rel=1e-12)
    assert found.min() > 0.0
    assert np.abs(found - exact).max() < 0.02
    # the damage of some of the voxels only: nothing in the others
    part = np.arange(n // 2 - 4, n // 2 + 4)
    some = fft._smooth(part, np.ones(part.size), ell)
    assert some.shape == (8,) and np.all((some > 0.3) & (some < 1.0))
    for order, bound in ((1, 0.012), (2, 0.005)):
        mesh = voxel_tet_mesh(image)
        if order == 2:
            mesh = mesh.quadratic()
        fe = NonlinearFEProblem(mesh, {0: m}, "periodic")
        centres = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)[:, 0]
        values = (np.abs(centres - LENGTH / 2) < half).astype(float)
        found = fe._smooth(np.arange(mesh.n_elements), values, ell)
        volumes = np.abs(mesh.volumes())
        assert np.sum(found * volumes) == pytest.approx(np.sum(values * volumes), rel=1e-10)
        slab = np.minimum((centres / (LENGTH / n)).astype(int), n - 1)
        means = np.bincount(slab, weights=found * volumes) / np.bincount(slab, weights=volumes)
        assert np.abs(means - exact).max() < bound
    # kinematic conditions: no flux through the faces of the box
    mesh = box_mesh((6, 2, 2), order=1)
    fe = NonlinearFEProblem(mesh, {0: m}, "kubc")
    found = fe._smooth(np.arange(mesh.n_elements), np.ones(mesh.n_elements), 0.3)
    assert np.allclose(found, 1.0, rtol=0, atol=1e-10)


# -- a bar that breaks -----------------------------------------------------------------------------


def _bar_materials(found, k=0.05):
    def material(strength):
        band = {"modulus": found["modulus"], "length": found["length"]}
        softening = found["softening"] * STRENGTH / strength
        planes = {"planes": "cube", "strength": strength, "K": k, "n": 1.0,
                  "softening": softening, "regularization": band}  # fmt: skip
        return Material.from_config(
            {"cubic": list(IRON), "crystal_plasticity": {"cleavage": planes}}
        )

    return {0: material(STRENGTH), 1: material(WEAK * STRENGTH)}


def _bar_image(n):
    """A bar of ``n`` voxels along [100], with a zone slightly weaker in its middle."""
    zone = round(ZONE * n / LENGTH)
    data = np.zeros((n, 1, 1), dtype=np.uint8)
    data[n // 2 - zone // 2 : n // 2 + (zone + 1) // 2] = 1
    return VoxelImage(data, spacing=(LENGTH / n,) * 3)


def _bar(n=64, steps=48, tol=None, end=0.0095):
    found = cr.microdamage_parameters(STRENGTH, ENERGY, WIDTH)
    problem = NonlinearFFTProblem(_bar_image(n), _bar_materials(found))
    if tol is not None:
        problem.microdamage_tol = tol
    path = {"times": [0, 1], "increments": steps, "strain": {"11": [end]}}
    return problem.run(path, fields="steps"), found


def test_bar_breaks_across_its_band():
    """A bar pulled along [100] breaks across a band of the weaker zone: its softening
    branch, the damage and the microdamage of the band follow the closed form; the work
    that breaks it is the fracture energy of the planes; broken, the planes of the band
    keep no stress, their damage and the microdamage stay as they were, and the band does
    not widen as the crack opens."""
    n = 64
    res, found = _bar(n)
    # (a step may be divided once, as the rounding of the processor has it: a runner of
    # Windows divided one where this machine divides none)
    assert res.info["cuts"] <= 1 and res.info["microdamage_unsettled"] == 0
    assert res.info["microdamage_passes"] > len(res.times)
    y0 = WEAK * STRENGTH
    h = found["softening"] * STRENGTH
    chi, ell = found["modulus"], found["length"]
    r = h / chi
    theta = math.pi - math.atan(math.sqrt(r / (1 - r)))
    omega = math.sqrt(r / (1 - r)) / ell
    energy = y0**2 * (WIDTH + 2 * ell) / (2 * h)
    assert energy == pytest.approx(WEAK**2 * ENERGY, rel=1e-12)
    stress, strain = res.stress[:, 0], res.strain[:, 0]
    peak = int(np.argmax(stress))
    assert 0.985 * y0 < stress[peak] < 1.001 * y0  # (the steps do not stop at the peak)
    # the softening branch: strain = sigma / E + (Y_0 - sigma) (b + 2 ell) / (|H| L)
    branch = np.flatnonzero((np.arange(stress.size) > peak) & (stress > 0.04 * y0))
    assert branch.size > 15
    exact = stress / E100 + (y0 - stress) * (WIDTH + 2 * ell) / (h * LENGTH)
    assert np.allclose(strain[branch], exact[branch], rtol=5e-3)
    # the fields of the band, half-way down
    k = branch[branch.size // 2]
    fields = res.fields[res.times[k]]
    opening = np.asarray(fields["cleavage_opening"]).ravel()
    micro = np.asarray(fields["cleavage_microdamage"]).ravel()
    x = (np.arange(n) + 0.5) * LENGTH / n
    x = x - np.sum(x * opening) / np.sum(opening)  # about the middle of the band
    s = y0 - stress[k]
    beta = omega * ell * s / (h * math.sin(theta))
    inside = np.abs(x) < WIDTH / 2
    exact_d = np.where(inside, s / h + beta * np.cos(omega * x), 0.0)
    exact_m = np.where(inside, s / h + beta * (1 - r) * np.cos(omega * x),
                       s / chi * np.exp(-(np.abs(x) - WIDTH / 2) / ell))  # fmt: skip
    assert np.count_nonzero(opening > 1e-9) == round(WIDTH * n / LENGTH)  # 16 voxels
    assert np.abs(opening - exact_d).max() < 5e-3 * exact_d.max()
    assert np.abs(micro - exact_m).max() < 5e-3 * exact_m.max()
    # the share of their strength that the planes have lost: that of the stress, in the band
    damage = np.asarray(fields["cleavage_damage"]).ravel()
    assert np.allclose(damage[opening > 1e-9], 1 - stress[k] / y0, rtol=0, atol=5e-3)
    assert np.all(np.diff(damage[x > WIDTH / 2]) < 0)  # and less and less beyond it
    # the work that breaks the bar: the fracture energy of the planes
    work = np.trapezoid(stress, strain) * LENGTH
    assert work == pytest.approx(energy, rel=5e-3)
    # broken through
    after = np.flatnonzero(stress < 1e-3 * y0)
    after = after[after > peak]
    assert after.size >= 5 and np.all(stress[after] < 0.05)
    first, last = (res.fields[res.times[j]] for j in (after[1], after[-1]))
    broken = np.asarray(last["cleavage_damage"]).ravel() == 1.0
    assert 2 <= broken.sum() <= round(WIDTH * n / LENGTH)
    assert np.array_equal(broken, np.asarray(first["cleavage_damage"]).ravel() == 1.0)
    assert np.allclose(first["cleavage_microdamage"], last["cleavage_microdamage"], atol=1e-9)
    gap = (
        np.asarray(last["cleavage_opening"]).ravel() - np.asarray(first["cleavage_opening"]).ravel()
    )
    assert np.all(gap[broken] > 1e-3) and np.all(np.abs(gap[~broken]) < 1e-9)
    # the bar is cut: its elongation is the opening of its crack
    stretch = np.sum(np.asarray(last["cleavage_opening"]).ravel()) / n
    assert stretch == pytest.approx(strain[-1], rel=1e-3)


@pytest.mark.slow
def test_band_does_not_depend_on_the_grid():
    """The work that breaks the bar on grids of 32, 64 and 128 voxels (8, 16 and 32
    across the band), with a microdamage settled ten times closer: the fracture energy of
    the planes, to the second order of the size of the voxels."""
    energy = WEAK**2 * ENERGY
    errors = []
    for n in (32, 64, 128):
        res, _ = _bar(n, steps=95, tol=1e-4)
        assert res.info["cuts"] == 0
        work = np.trapezoid(res.stress[:, 0], res.strain[:, 0]) * LENGTH
        errors.append(work / energy - 1.0)
        assert res.stress[-1, 0] < 0.05
    assert abs(errors[0]) < 0.02 and abs(errors[1]) < 0.005 and abs(errors[2]) < 0.002
    assert abs(errors[1]) < 0.4 * abs(errors[0])


def test_bar_on_a_mesh():
    """The bar on a mesh of tetrahedra: it breaks across its band, for a work that comes
    to the fracture energy of the planes as the mesh is refined."""
    found = cr.microdamage_parameters(STRENGTH, ENERGY, WIDTH)
    energy = WEAK**2 * ENERGY
    path = {"times": [0, 1], "increments": 95, "strain": {"11": [0.0095]}}
    errors = []
    for n in (32, 64):
        mesh = voxel_tet_mesh(_bar_image(n))
        problem = NonlinearFEProblem(mesh, _bar_materials(found), "periodic")
        res = problem.run(path)
        stress, strain = res.stress[:, 0], res.strain[:, 0]
        assert stress.max() == pytest.approx(WEAK * STRENGTH, rel=2e-3)
        assert stress[-1] < 0.1
        errors.append(np.trapezoid(stress, strain) * LENGTH / energy - 1.0)
    assert 0.0 < errors[1] < 0.04 and errors[1] < 0.5 * errors[0] < 0.06


@pytest.mark.parametrize("kinematics", ["small", "finite"])
def test_cell_without_gradient_is_its_point(kinematics):
    """A homogeneous crystal on voxels and on a mesh: its damage is uniform, its
    microdamage is its damage, and it softens as its material point."""
    m = iron(k=5.0)
    strain = {key: [0.012 if key == "33" else 0.0] for key in ("11", "22", "33", "23", "31", "12")}
    path = {"times": [0, 1], "increments": 30, "strain": strain}
    ref = simulate_point(m, path, kinematics=kinematics)
    assert ref.series["cleavage_damage"][-1] > 0.3
    image = VoxelImage(np.zeros((2, 2, 2), dtype=np.uint8))
    fft = NonlinearFFTProblem(image, {0: m}, kinematics=kinematics)
    fe = NonlinearFEProblem(box_mesh((1, 1, 1), order=1, perturb=0.2), {0: m}, "periodic",
                            kinematics=kinematics)  # fmt: skip
    for problem in (fft, fe):
        problem.microdamage_tol = 1e-6
        res = problem.run(path)
        assert res.info["cuts"] == 0
        assert np.allclose(res.stress, ref.stress, rtol=0, atol=2e-4 * np.abs(ref.stress).max())
        for name in ("cleavage_opening", "cleavage_microdamage"):
            assert np.allclose(res.series[name], ref.series[name], rtol=0, atol=2e-6)


def test_solvers_without_the_equation_refuse():
    """The loading paths of the extended finite elements have no solver of the
    microdamage: they refuse the planes that it regularizes, before the first step."""
    import morphogen as mg
    from morphogen.homogenization.xfem import LevelSetSample, NonlinearXFEMProblem
    from morphogen.mesh.conforming import ImplicitBody

    def level_set(points):
        return np.abs(points[:, 0] % 1.0 - 0.5) - 0.25

    body = ImplicitBody(mg.Domain.cube(1.0, periodic=True), level_set)
    sample = LevelSetSample.of(body, (4, 2, 2))
    elastic = Material(young=200000.0, poisson=0.3)
    problem = NonlinearXFEMProblem(sample, {0: iron(), 1: elastic})
    path = {"times": [0, 1], "increments": 2, "strain": {"33": [0.001]}}
    with pytest.raises(ValueError, match="microdamage of cleavage planes"):
        problem.run(path)


@pytest.mark.slow
def test_polycrystal_cleaves_across_bands():
    """A columnar polycrystal of iron whose {100} planes are regularized, pulled across
    its grains: a crack crosses it from plane to plane, in a band some voxels wide; the
    work that breaks the cell is of the order of the fracture energy of the planes times
    its section (more: the planes of the grains are inclined), and the planes broken
    through are a small share of the cell."""
    from morphogen import pipeline
    from morphogen.config import load_config

    n, size = 32, 0.1
    planes = {"planes": "cube", "strength": 500.0, "energy": 0.05, "K": 50.0, "n": 1.0,
              "regularization": {"width": 5 * size / n}}  # fmt: skip
    crystal = {"K": 10.0, "n": 10.0, "families": {"bcc110": {"tau0": 300.0, "Q": 100.0, "b": 20.0}},
               "cleavage": planes}  # fmt: skip
    config = {
        "model": {"type": "voronoi", "n_cells": 5, "crystal_orientations": True},
        "domain": {"size": [size, size, size / n], "periodic": True},
        "seed": 2,
        "voxels": {"shape": [n, n, 1]},
        "homogenization": {
            "method": "fft",
            "materials": {1: {"cubic": list(IRON), "crystal_plasticity": crystal}},
            "properties": [],
            "paths": {"tension": {"times": [0, 1], "increments": 40, "strain": {"11": [0, 0.006]}}},
        },
    }
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        config["outputs"] = {"directory": folder, "name": "iron"}
        out = pipeline.run(load_config(config))
        table = next(Path(out).glob("*tension*.csv"))
        names = table.read_text().splitlines()[0].split(",")
        data = np.loadtxt(table, delimiter=",", skiprows=1)
    column = {name: data[:, i] for i, name in enumerate(names)}
    stress, strain = column["S11"], column["E11"]
    assert 450.0 < stress.max() < 650.0
    assert stress[-1] < 0.1 * stress.max()  # broken across
    work = np.trapezoid(stress, strain) * size / 0.05
    assert 1.0 < work < 3.0
    assert 0.0 < column["cleavage_damage"][-1] < 0.5
    assert column["cleavage_microdamage"][-1] > 0.0


# -- the alternate scheme and its acceleration ---------------------------------------------------


def _updates(start, target, count):
    """The updates of the microdamage that the alternate scheme gives to equilibria whose
    damage would have the microdamage ``target(x)``, from ``start``: the microdamage of
    each equilibrium."""
    from morphogen.homogenization.nonlinear import _Microdamage

    scheme = _Microdamage.__new__(_Microdamage)
    scheme.parts = []
    scheme._restart()
    weight = np.ones(start.size)
    out = [start]
    for _ in range(count):
        out.append(scheme._accelerate(out[-1], target(out[-1]), weight))
    return out


def test_acceleration_of_the_alternate_scheme():
    """The updates of the microdamage for maps with a known fixed point. A contraction
    of one rate: two plain updates, then the secant extrapolation, which lands on the
    fixed point; a slow contraction: the extrapolation stays within ten residuals of the
    plain update; a residual that grows: plain updates, then relaxed ones."""
    from morphogen.homogenization import nonlinear as nl

    fixed = np.array([0.3, 1.0, 2.5, 0.0])
    start = np.zeros(4)

    def contraction(rate):
        return lambda x: fixed + rate * (x - fixed)

    seq = _updates(start, contraction(0.8), 3)
    assert np.allclose(seq[1], contraction(0.8)(seq[0]), rtol=0, atol=1e-15)
    assert np.allclose(seq[2], contraction(0.8)(seq[1]), rtol=0, atol=1e-15)
    assert np.allclose(seq[3], fixed, rtol=0, atol=1e-13)
    # a rate of 0.99: a hundred residuals to the fixed point, ten at a time
    seq = _updates(start, contraction(0.99), 3)
    residual = contraction(0.99)(seq[2]) - seq[2]
    assert np.allclose(seq[3] - contraction(0.99)(seq[2]), nl._REACH * residual, rtol=1e-12)
    # rates that differ from an element to the next: faster than the plain updates
    rates = np.array([0.5, 0.8, 0.9, 0.7])

    def mixed(x):
        return fixed + rates * (x - fixed)

    plain = start
    for _ in range(12):
        plain = mixed(plain)
    assert np.abs(_updates(start, mixed, 12)[-1] - fixed).max() < 0.2 * np.abs(plain - fixed).max()
    # a residual that grows (no fixed point close by): relaxed after three passes
    seq = _updates(start, lambda x: x + 0.01 * (1.0 + 2.0 * x.sum()) * np.ones(4), 6)
    steps = [b - a for a, b in itertools.pairwise(seq)]
    grown = [0.01 * (1.0 + 2.0 * x.sum()) for x in seq[:-1]]
    assert all(np.allclose(s, g) for s, g in zip(steps[:3], grown[:3], strict=True))
    assert all(
        np.allclose(s, nl._RELAXATION * g) for s, g in zip(steps[3:], grown[3:], strict=True)
    )


def test_accelerated_scheme_follows_the_plain_one():
    """The bar with the plain alternate scheme (every equilibrium solved to the
    tolerance, the microdamage updated as it is) and with the accelerated one: the same
    curve, in half the equilibria, and a work closer to that of a microdamage settled
    ten times closer."""
    found = cr.microdamage_parameters(STRENGTH, ENERGY, WIDTH)
    path = {"times": [0, 1], "increments": 48, "strain": {"11": [0.0095]}}
    runs = {}
    for name, options in (
        ("plain", {"acceleration": False}),
        ("accelerated", {}),
        ("settled", {"acceleration": False, "tolerance": 1e-4}),
    ):
        problem = NonlinearFFTProblem(_bar_image(64), _bar_materials(found))
        problem.microdamage_options(**options)
        runs[name] = problem.run(path)
        assert runs[name].info["cuts"] <= 1  # (as for the bar above)
    plain, fast = runs["plain"], runs["accelerated"]
    assert fast.info["microdamage_passes"] < 0.6 * plain.info["microdamage_passes"]
    # the curves at their common times (a divided step adds times to one of them)
    both = np.intersect1d(fast.times.round(12), plain.times.round(12))
    stresses = [r.stress[np.isin(r.times.round(12), both), 0] for r in (fast, plain)]
    assert np.abs(stresses[0] - stresses[1]).max() < 0.01 * STRENGTH
    work = {
        name: np.trapezoid(res.stress[:, 0], res.strain[:, 0]) * LENGTH
        for name, res in runs.items()
    }
    energy = WEAK**2 * ENERGY
    assert all(abs(value / energy - 1.0) < 5e-3 for value in work.values())
    assert abs(work["accelerated"] - work["settled"]) < abs(work["plain"] - work["settled"])


def test_options_of_the_microdamage():
    """The tolerance, the passes and the acceleration of the alternate scheme: options of
    the solvers of the paths, and of the configurations (``nonlinear: {microdamage:
    ...}``); a step whose passes are spent is kept, and counted."""
    from morphogen.config import ConfigError, load_config

    found = cr.microdamage_parameters(STRENGTH, ENERGY, WIDTH)
    problem = NonlinearFFTProblem(_bar_image(32), _bar_materials(found))
    assert (problem.microdamage_tol, problem.microdamage_passes) == (1e-3, 200)
    assert problem.microdamage_acceleration is True
    problem.microdamage_options(tolerance=1e-4)
    assert (problem.microdamage_tol, problem.microdamage_passes) == (1e-4, 200)
    problem.microdamage_options(passes=2, acceleration=False)
    assert (problem.microdamage_passes, problem.microdamage_acceleration) == (2, False)
    for bad in ({"tolerance": 0.0}, {"tolerance": 1.5}, {"passes": 0}, {"passes": 2.5},
                {"acceleration": 1}):  # fmt: skip
        with pytest.raises(ValueError, match="microdamage"):
            problem.microdamage_options(**bad)
    # two passes a step: the steps of the softening branch are kept before they settle
    res = problem.run({"times": [0, 1], "increments": 24, "strain": {"11": [0.006]}})
    assert res.info["microdamage_unsettled"] > 5
    assert res.info["microdamage_passes"] <= 2 * (len(res.times) - 1)
    assert res.stress[:, 0].max() == pytest.approx(WEAK * STRENGTH, rel=0.02)
    # in a configuration
    planes = {"planes": "cube", "strength": 500.0, "energy": 0.05, "K": 50.0, "n": 1.0,
              "regularization": {"width": 0.02}}  # fmt: skip
    config = {
        "model": {"type": "voronoi", "n_cells": 3, "crystal_orientations": True},
        "domain": {"size": 0.1, "periodic": True},
        "voxels": {"shape": 6},
        "homogenization": {
            "method": "fft",
            "materials": {1: {"cubic": list(IRON), "crystal_plasticity": {"cleavage": planes}}},
            "properties": [],
            "paths": {"tension": {"times": [0, 1], "increments": 2, "strain": {"33": [0, 1e-3]}}},
            "nonlinear": {"microdamage": {"tolerance": 1e-4, "passes": 50, "acceleration": False}},
        },
    }
    loaded = load_config(config)
    assert loaded.homogenization.nonlinear["microdamage"] == {
        "tolerance": 1e-4,
        "passes": 50,
        "acceleration": False,
    }
    for bad, match in (
        ({"tolerance": 2}, "tolerance"),
        ({"passes": 0}, "passes"),
        ({"acceleration": "yes"}, "acceleration"),
        ({"speed": 1}, "unknown keys"),
        (5, "a mapping"),
    ):
        config["homogenization"]["nonlinear"] = {"microdamage": bad}
        with pytest.raises(ConfigError, match=match):
            load_config(config)
