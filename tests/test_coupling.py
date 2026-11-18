"""Couplings: transient diffusion (Crank's slab), thermal oxidation (Ernault et al.
2019), ageing laws and graded phases, hygromorph multilayers (Timoshenko)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from morphogen.coupling import (
    ClosedLoopOxidation,
    Hydrolysis,
    Layer,
    Multilayer,
    PropertyLaw,
    graded_materials,
    slab_profile,
    slab_uptake,
    timoshenko_curvature,
    transient_diffusion,
)
from morphogen.homogenization.apparent import make_problem, thermal_expansion
from morphogen.homogenization.materials import Material
from morphogen.image import VoxelImage


def test_slab_against_crank():
    n, length, d = 100, 1e-3, 1e-12
    image = VoxelImage(np.zeros((n, 1, 1), np.uint8), spacing=(length / n,) * 3)
    t_ref = length**2 / d
    times = np.array([0.002, 0.01, 0.05, 0.2]) * t_ref
    res = transient_diffusion(image, d, times, solubility=0.02, steps=200)
    np.testing.assert_allclose(res.relative_uptake, slab_uptake(times, length, d), rtol=0.02)
    z = (np.arange(n) + 0.5) * length / n - 0.5 * length
    np.testing.assert_allclose(
        res.fields[2][:, 0, 0] / 0.02, slab_profile(z, times[2], length, d), atol=0.005
    )
    # early times: M / M_inf = 4 sqrt(D t / pi) / L
    assert res.relative_uptake[0] == pytest.approx(
        4.0 * math.sqrt(d * times[0] / math.pi) / length, rel=0.03
    )


def test_periodic_plate_and_solubilities():
    # a plate exposed on z, periodic in x and y (conjugate gradients beyond 60000 voxels)
    n = 40
    image = VoxelImage(np.zeros((n, n, n), np.uint8), spacing=(1.0,) * 3, periodic=True)
    times = [20.0, 200.0]
    res = transient_diffusion(image, 1.0, times, exposed=("z-", "z+"), steps=60)
    line = transient_diffusion(
        VoxelImage(np.zeros((1, 1, n), np.uint8)), 1.0, times, exposed=("z-", "z+"), steps=60
    )
    for a, b in zip(res.fields, line.fields, strict=True):
        np.testing.assert_allclose(a, np.broadcast_to(b, a.shape), atol=1e-6)
    # two phases of different solubilities: c = S u, u continuous, at equilibrium u = 1
    labels = np.zeros((30, 1, 1), np.uint8)
    labels[10:20] = 1
    res = transient_diffusion(
        VoxelImage(labels), {0: 1.0, 1: 0.1}, [1e4], solubility={0: 1.0, 1: 5.0}
    )
    np.testing.assert_allclose(
        res.fields[-1][:, 0, 0], np.where(labels[:, 0, 0] == 1, 5.0, 1.0), rtol=1e-6
    )
    assert res.relative_uptake[-1] == pytest.approx(1.0, rel=1e-6)


def test_thermal_oxidation_of_an_epoxy():
    model = ClosedLoopOxidation()
    hour = 3600.0
    times = np.array([50.0, 111.0, 150.0, 200.0, 240.0, 300.0, 400.0]) * hour
    film = model.thin_film(times)
    amide = film["PO"]
    # Ernault et al. (2019) measured about 0.2 and 0.7 mol/l after 111 h and 240 h
    assert amide[1] == pytest.approx(0.2, rel=0.5) and amide[4] == pytest.approx(0.7, rel=0.3)
    rate = np.gradient(amide, times)
    assert 150 * hour <= times[np.argmax(rate)] <= 300 * hour  # then the sites run out
    assert np.all(np.diff(film["PH"]) < 0)
    plate = model.profile(3e-3, np.array([240.0, 800.0]) * hour, nodes=81)
    for k in range(2):
        a = plate["PO"][k]
        assert a[-1] < 0.05 * a[0]  # the core is hardly oxidized
        layer = model.oxidized_layer(plate["depth"], a)
        assert 50e-6 < layer < 300e-6  # Ernault et al.: about 200 micrometres
    assert plate["PO"][0][0] <= model.thin_film([240.0 * hour])["PO"][0]


def test_property_laws_and_graded_phases():
    law = PropertyLaw(
        tracer=(0.0, 1.0), values={"young": (2.4e3, 3.2e3), "expansion": (0.0, -0.01)}
    )
    np.testing.assert_allclose(law([0.5, 2.0])["young"], [2.8e3, 3.2e3])
    # a periodic laminate: phase 0 inert, phase 1 aged along x
    n = 16
    labels = np.zeros((n, 4, 4), np.uint8)
    labels[n // 2 :] = 1
    tracer = np.broadcast_to(np.linspace(0.0, 1.0, n)[:, None, None], labels.shape)
    image = VoxelImage(labels, periodic=True)
    graded, materials = graded_materials(
        image,
        tracer,
        {1: law},
        {0: Material(young=2.4e3, poisson=0.37, expansion=0.0)},
        levels=4,
        poisson=0.37,
    )
    assert set(np.unique(graded.data)) == {0, 2, 3, 4, 5} and len(materials) == 5
    alphas = sorted(materials[k].expansion for k in (2, 3, 4, 5))
    np.testing.assert_allclose(
        np.diff(alphas), 0.01 * (1.0 - 8.0 / 15.0) / 4, rtol=1e-9
    )  # four classes
    # equal stiffnesses: the apparent expansion is the mean of the eigenstrains (exact)
    same = {
        k: Material(young=3e3, poisson=0.37, expansion=m.expansion) for k, m in materials.items()
    }
    a, _ = thermal_expansion(make_problem(graded, same, "elastic", "periodic"))
    mean = np.mean([same[int(k)].expansion for k in graded.data.ravel()])
    np.testing.assert_allclose(np.diag(a), mean, rtol=1e-6, atol=1e-12)


def test_hydrolysis_of_cellulose():
    h = Hydrolysis(dp0=1500.0, rate=2e-10)
    t = h.time_to()  # to the critical degree of polymerization of Vibert et al.
    assert h.dp(t) == pytest.approx(750.0) and h.scissions(t) == pytest.approx(1.0)
    hot = h.rate_at(h.temperature + 10.0) / h.rate
    assert hot == pytest.approx(math.exp(110e3 / 8.314462618 * (1 / 353.15 - 1 / 363.15)))


def test_bilayer_against_timoshenko():
    h1, h2, e1, e2, de = 0.4e-3, 0.6e-3, 3.0e3, 1.0e3, 0.01
    plate = Multilayer(
        [Layer(h1, e1, 0.3), Layer(h2, e2, 0.3, beta_x=de, beta_y=de)], nodes_per_layer=10
    )
    c = (plate.index == 1).astype(float)
    _, kappa = plate.deformation(c)
    np.testing.assert_allclose(kappa, timoshenko_curvature(de, h1, h2, e1, e2), rtol=1e-9)


def test_hygromorph_bends_in_sequence():
    # a passive layer on an active one exposed from below, swelling more along y
    passive = Layer(0.5e-3, 3.5e3, 0.35, diffusivity=1e-12, solubility=0.01)
    active = Layer(0.5e-3, 2.0e3, 0.35, beta_x=0.2, beta_y=0.3, diffusivity=4e-12, solubility=0.1)
    plate = Multilayer([active, passive])
    times = np.array([0.01, 0.05, 0.2, 1.0, 5.0]) * (1e-3) ** 2 / 1e-12
    hist = plate.curvature_history(times, exposed="bottom")
    kx, ky = hist["curvatures"].T
    assert np.all(np.diff(-kx) > 0) and np.all(ky / kx > 1.0)  # bends away from the active face
    final = plate.deformation(plate.moisture([1e3 * times[-1]], exposed="bottom")[0])[1]
    np.testing.assert_allclose(hist["curvatures"][-1], final, rtol=0.02)
    # one layer exposed on both faces: Crank's uptake
    single = Multilayer([Layer(1e-3, 1e3, diffusivity=1e-12)], nodes_per_layer=100)
    c = single.moisture(times[:3], exposed="both")
    np.testing.assert_allclose(c.mean(axis=1), slab_uptake(times[:3], 1e-3, 1e-12), rtol=0.01)


def test_diffusion_in_configurations(tmp_path):
    from morphogen import config, pipeline

    data = {
        "model": {"type": "boolean_spheres", "radius": 0.15, "volume_fraction": 0.3},
        "domain": {"size": 1.0, "periodic": True},
        "seed": 1,
        "voxels": {"shape": 12},
        "diffusion": {
            "phases": {
                0: {"diffusivity": 1.0, "solubility": 1.0},
                1: {"diffusivity": 0.1, "solubility": 3.0},
            },
            "exposed": ["z-", "z+"],
            "activity": {"times": [0, 0.01], "values": [0, 1]},
            "times": [0.01, 0.05, 1.0],
            "fields": True,
        },
        "outputs": {"directory": str(tmp_path / "out"), "name": "s"},
    }
    cfg = config.load_config(data)
    assert cfg.diffusion.exposed == ("z-", "z+") and cfg.diffusion.boundary()(0.005) == 0.5
    pipeline.run(cfg)
    rows = (tmp_path / "out" / "s_0000_diffusion.csv").read_text().splitlines()
    uptake = [float(r.split(",")[2]) for r in rows[1:]]
    assert len(uptake) == 3 and uptake[0] < uptake[1] < uptake[2] <= 1.0 + 1e-9
    assert uptake[2] == pytest.approx(1.0, abs=0.02)
    assert (tmp_path / "out" / "s_0000_diffusion_3.vti").exists()
    for wrong, message in (
        ({"exposed": ["w+"]}, "exposed"),
        (
            {"phases": {0: {"diffusivity": 1.0, "solubility": 0.0}, 1: {"diffusivity": 1.0}}},
            "solubility",
        ),
        ({"phases": {0: {"diffusivity": 1.0}}}, "phase"),
        ({"times": [1.0, 0.5]}, "times"),
    ):
        with pytest.raises(config.ConfigError, match=message):
            config.load_config({**data, "diffusion": {**data["diffusion"], **wrong}})


def _rotate_q(q, angle):
    """Plane-stress stiffness turned by ``angle`` about the normal (engineering shear)."""
    c, s = np.cos(angle), np.sin(angle)
    t = np.array(
        [[c * c, s * s, 2 * c * s], [s * s, c * c, -2 * c * s], [-c * s, c * s, c * c - s * s]]
    )
    r = np.diag([1.0, 1.0, 2.0])
    return np.linalg.inv(t) @ q @ r @ t @ np.linalg.inv(r)


def test_orthotropic_layers():
    iso = Layer(0.5, 2.0, 0.3, beta_x=0.01, beta_y=0.01)
    same = Layer(
        0.5, 2.0, 0.3, beta_x=0.01, beta_y=0.01, stiffness=tuple(map(tuple, iso.plane_stiffness()))
    )
    passive = Layer(0.5, 4.0, 0.3)
    c = np.ones(80)
    np.testing.assert_allclose(
        Multilayer([passive, iso]).deformation(c)[1],
        Multilayer([passive, same]).deformation(c)[1],
        rtol=1e-12,
    )
    # a fibre-reinforced active layer: stiff and hardly swelling along its fibres (x)
    q = np.array([[30.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 1.5]])
    along_x = Layer(0.5, 30.0, stiffness=tuple(map(tuple, q)), beta_x=0.001, beta_y=0.02)
    q_y = q[[1, 0, 2]][:, [1, 0, 2]]
    along_y = Layer(0.5, 3.0, stiffness=tuple(map(tuple, q_y)), beta_x=0.02, beta_y=0.001)
    kx = Multilayer([passive, along_x]).deformation(c)[1]
    ky = Multilayer([passive, along_y]).deformation(c)[1]
    np.testing.assert_allclose(kx, ky[::-1], rtol=1e-12)  # turning the fibres swaps the curvatures
    assert kx[1] > 5 * kx[0]  # it bends across its fibres
    _, k = Multilayer([passive, along_x]).deformation(c, full=True)
    assert abs(k[2]) < 1e-12 * abs(k).max()
    # fibres at 45 degrees and a swelling across them: the strip twists
    q45 = _rotate_q(q, np.pi / 4)
    b = np.array([0.001, 0.02])
    beta = np.array(
        [0.5 * (b[0] + b[1]), 0.5 * (b[0] + b[1]), b[0] - b[1]]
    )  # strain turned by 45 degrees
    off = Layer(
        0.5, 1.0, stiffness=tuple(map(tuple, q45)), beta_x=beta[0], beta_y=beta[1], beta_xy=beta[2]
    )
    _, k = Multilayer([passive, off]).deformation(c, full=True)
    assert abs(k[2]) > 0.5 * abs(k[0])


def test_layers_from_microstructures():
    mats = {
        0: Material(young=2.0, poisson=0.3, conductivity=1.0e-12 * 0.02, expansion=0.002),
        1: Material(young=30.0, poisson=0.25, conductivity=5.0e-12 * 0.15, expansion=0.01),
    }
    solubility = {0: 0.02, 1: 0.15}
    homogeneous = VoxelImage(np.zeros((4, 4, 4), np.uint8), periodic=True)
    layer = Layer.from_microstructure(1.0e-3, homogeneous, mats, solubility=solubility)
    np.testing.assert_allclose(
        layer.plane_stiffness(), Layer(1.0, 2.0, 0.3).plane_stiffness(), rtol=1e-6, atol=1e-9
    )
    assert layer.beta_x == pytest.approx(0.002, rel=1e-6) and layer.beta_y == pytest.approx(
        0.002, rel=1e-6
    )
    assert layer.diffusivity == pytest.approx(
        1.0e-12, rel=1e-6
    ) and layer.solubility == pytest.approx(0.02)
    # a laminate of layers normal to the thickness: Q the mean of the Q of the phases,
    # the permeability across it their harmonic mean
    labels = np.zeros((4, 4, 10), np.uint8)
    labels[:, :, 6:] = 1
    lam = Layer.from_microstructure(
        1.0e-3, VoxelImage(labels, periodic=True), mats, solubility=solubility
    )
    q0, q1 = Layer(1.0, 2.0, 0.3).plane_stiffness(), Layer(1.0, 30.0, 0.25).plane_stiffness()
    np.testing.assert_allclose(lam.plane_stiffness(), 0.6 * q0 + 0.4 * q1, rtol=1e-6, atol=1e-9)
    perm = 1.0 / (0.6 / (1.0e-12 * 0.02) + 0.4 / (5.0e-12 * 0.15))
    assert lam.solubility == pytest.approx(0.6 * 0.02 + 0.4 * 0.15)
    assert lam.diffusivity == pytest.approx(perm / lam.solubility, rel=1e-6)
    # the same laminate stacked along x, the layer normal to x: its in-plane shear is 23
    stacked_x = VoxelImage(np.moveaxis(labels, 2, 0), periodic=True)
    lam_x = Layer.from_microstructure(1.0e-3, stacked_x, mats, solubility=solubility, normal=0)
    np.testing.assert_allclose(lam_x.plane_stiffness(), lam.plane_stiffness(), rtol=1e-6, atol=1e-9)
    assert lam_x.diffusivity == pytest.approx(lam.diffusivity, rel=1e-6)
