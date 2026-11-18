"""Acoustics of porous media: the parameters of the equivalent fluid of a rigid frame and
the absorption of a layer.

References: the thermal permeability of slits and tubes (the Poisson problem in closed
form) and of the flow along cylinders; the tortuosity and the viscous length of arrays of
cylinders and of spheres (Rayleigh's and Maxwell's dipole fields: ``1 + c`` and ``(1 + c)
/ 2`` of the thermal length; ``1 + c / 2`` and ``2 (1 + c / 2) / 3``); the model of
Johnson, Champoux, Allard and Lafarge against the exact responses of slits and of tubes
(Zwikker & Kosten, Stinson), its limits at low and high frequencies, and the absorption
of a thick layer.
"""

import numpy as np
import pytest

import morphogen as mg
from morphogen.homogenization import (
    VOID,
    Material,
    absorption,
    acoustic_parameters,
    equivalent_fluid,
    homogenize,
    permeability_tensor,
)
from morphogen.homogenization.acoustics import (
    AIR,
    PARAMETERS,
    Fluid,
    along_axis,
    pore_response,
    thermal_permeability,
)

SOLID = Material(young=1.0, poisson=0.3)


def cylinders(n, fraction):
    x = (np.arange(n) + 0.5) / n - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    data = (X**2 + Y**2 < fraction / np.pi).astype(np.uint8)[:, :, None]
    return mg.VoxelImage(data, 1.0 / n, periodic=True)


def spheres(n, fraction):
    radius = (3 * fraction / (4 * np.pi)) ** (1 / 3)
    x = (np.arange(n) + 0.5) / n - 0.5
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return mg.VoxelImage((X**2 + Y**2 + Z**2 < radius**2).astype(np.uint8), 1.0 / n, periodic=True)


def test_thermal_permeability_of_slits_and_tubes():
    """``lap(theta) = -1`` with ``theta = 0`` on the walls: the parabola of a slit (its
    mean at the nodes, as the flow of the lattice Boltzmann method) and ``R^2 / 8`` in a
    tube drawn in voxels."""
    n, width = 18, 16
    data = np.ones((1, n, 1), np.uint8)
    data[0, :width, 0] = 0
    slit = mg.VoxelImage(data, 0.5, periodic=True)
    exact = 0.25 * (width / n) * (width**2 / 12 + 1 / 24)
    assert thermal_permeability(slit, pores=0) == pytest.approx(exact, rel=1e-9)
    assert thermal_permeability(slit, {0: VOID, 1: SOLID}) == pytest.approx(exact, rel=1e-9)
    m = 48
    x = (np.arange(m) + 0.5) / m - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    tube = mg.VoxelImage((X**2 + Y**2 >= 0.35**2).astype(np.uint8)[:, :, None], 1.0 / m, True)
    porosity = 1.0 - tube.data.mean()
    assert thermal_permeability(tube, pores=0) == pytest.approx(
        porosity * (porosity / np.pi) / 8, rel=0.03
    )
    solid = mg.VoxelImage(np.ones((2, 2, 2), np.uint8), 1.0, periodic=True)
    assert thermal_permeability(solid, pores=0) == 0.0
    with pytest.raises(ValueError, match="without a wall"):
        thermal_permeability(solid, pores=1)


def test_parameters_of_a_square_array_of_cylinders():
    """Across the cylinders, Rayleigh's field: a tortuosity ``1 + c`` and a viscous length
    ``(1 + c) / 2`` of the thermal one (twice the volume of the pores over the area of
    their walls); along them a uniform field, and the flow of the Poisson problem."""
    image = cylinders(48, 0.1)
    c = float(image.data.mean())
    radius = np.sqrt(c / np.pi)
    out = acoustic_parameters(image, {0: VOID, 1: SOLID}, tol=1e-5)
    assert set(PARAMETERS) <= set(out)
    assert out["porosity"] == pytest.approx(1 - c)
    assert out["thermal_length"] == pytest.approx((1 - c) / (np.pi * radius), rel=0.03)
    assert out["specific_surface"] == pytest.approx(2 * np.pi * radius, rel=0.03)
    for axis in (1, 2):
        assert out[f"tortuosity{axis}"] == pytest.approx(1 + c, rel=0.01)
        ratio = out[f"viscous_length{axis}"] / out["thermal_length"]
        assert ratio == pytest.approx((1 + c) / 2, rel=0.03)
    assert out["tortuosity3"] == 1.0
    assert out["viscous_length3"] == pytest.approx(out["thermal_length"], rel=1e-12)
    assert out["tortuosity"] == pytest.approx(out["tortuosity1"])  # the means: in the plane
    # the flow along the cylinders is the Poisson problem of the thermal permeability
    assert out["thermal_permeability"] == pytest.approx(out["permeability3"], rel=5e-3)
    assert out["thermal_permeability"] > out["permeability1"]
    flow = permeability_tensor(image, pores=0, tol=1e-5)
    assert np.allclose(out["permeability_tensor"], flow["tensor"], rtol=1e-6, atol=1e-12)
    # the six parameters of the model, along an axis or their means
    assert along_axis(out, 2)["permeability"] == out["permeability3"]
    assert along_axis(out)["tortuosity"] == out["tortuosity"]
    assert set(along_axis(out)) == {"porosity", "permeability", "tortuosity", "viscous_length",
                                    "thermal_length", "thermal_permeability"}  # fmt: skip


def test_parameters_of_a_simple_cubic_array_of_spheres():
    """Maxwell's field around spheres: a tortuosity ``1 + c / 2`` and a viscous length
    ``2 (1 + c / 2) / 3`` of the thermal one, the same along the three axes."""
    image = spheres(32, 0.1)
    c = float(image.data.mean())
    out = acoustic_parameters(image, pores=0, tol=1e-4, thermal=False)
    assert "thermal_permeability" not in out
    assert out["tortuosity"] == pytest.approx(1 + c / 2, rel=0.01)
    assert out["viscous_length"] / out["thermal_length"] == pytest.approx(
        2 * (1 + c / 2) / 3, rel=0.04
    )
    radius = (3 * c / (4 * np.pi)) ** (1 / 3)
    assert out["thermal_length"] == pytest.approx(2 * (1 - c) / (4 * np.pi * radius**2), rel=0.04)
    for name in ("tortuosity", "viscous_length", "permeability"):
        values = [out[f"{name}{i}"] for i in (1, 2, 3)]
        assert np.ptp(values) < 1e-3 * np.mean(values)
    assert out["permeability"] > 0.0


def test_closed_and_narrow_pores():
    """Pores closed along an axis: no path, an infinite tortuosity; pores that an erosion
    of one voxel would close keep a viscous length (their dilation alone)."""
    data = np.ones((12, 12, 1), np.uint8)
    data[:, 4:7, 0] = 0  # a slit of three voxels along x
    out = acoustic_parameters(mg.VoxelImage(data, 1.0, periodic=True), pores=0)
    assert out["tortuosity1"] == 1.0 and np.isinf(out["tortuosity2"])
    assert out["viscous_length2"] == 0.0 and out["permeability2"] == 0.0
    assert out["viscous_length1"] == pytest.approx(3.0, rel=0.35)  # the width of the slit
    # (walls along the lattice: the worst case of the 13 directions of the area estimate)
    assert out["thermal_length"] == pytest.approx(3.0, rel=0.1)
    with pytest.raises(ValueError, match="no pore"):
        acoustic_parameters(mg.VoxelImage(data, 1.0, periodic=True), pores=5)
    with pytest.raises(ValueError, match="cubic voxels"):
        acoustic_parameters(mg.VoxelImage(data, (1.0, 2.0, 1.0), periodic=True), pores=0)


@pytest.mark.parametrize("shape,size,factor,tolerance", [("slit", 2e-4, 12.0, 0.03),
                                                         ("tube", 1e-4, 8.0, 0.06)])  # fmt: skip
def test_the_equivalent_fluid_against_exact_pores(shape, size, factor, tolerance):
    """Slits ``size`` wide and tubes of radius ``size``: ``k0 = k0' = size^2 / 12`` (or
    ``/ 8``), ``Lambda = Lambda' = size``, no tortuosity; the model follows their exact
    density and bulk modulus over three decades of frequency."""
    f = np.logspace(1, 4.3, 60)
    exact = pore_response(f, shape, size)
    parameters = {"porosity": 1.0, "permeability": size**2 / factor, "tortuosity": 1.0,
                  "viscous_length": size, "thermal_length": size}  # fmt: skip
    model = equivalent_fluid(f, thermal_permeability=size**2 / factor, **parameters)
    assert np.abs(model["density"] / exact["density"] - 1).max() < tolerance
    assert np.abs(model["bulk_modulus"] / exact["bulk_modulus"] - 1).max() < 0.02
    # without the thermal permeability: that of circular pores (exact for the tubes)
    plain = equivalent_fluid(f, **parameters)
    error = np.abs(plain["bulk_modulus"] / exact["bulk_modulus"] - 1).max()
    assert error < (0.02 if shape == "tube" else 0.06)
    assert model["resistivity"] == pytest.approx(AIR.viscosity * factor / size**2)
    # Darcy's law and an isothermal gas at low frequencies, the fluid itself at high ones
    low = equivalent_fluid([0.01], thermal_permeability=size**2 / factor, **parameters)
    omega = 2 * np.pi * 0.01
    assert low["density"][0] * 1j * omega / low["resistivity"] == pytest.approx(1.0, abs=1e-3)
    assert low["bulk_modulus"][0] == pytest.approx(AIR.pressure, rel=1e-4)
    high = equivalent_fluid([1e10], thermal_permeability=size**2 / factor, **parameters)
    assert high["density"][0] == pytest.approx(AIR.density, rel=1e-3)
    assert high["bulk_modulus"][0] == pytest.approx(AIR.heat_ratio * AIR.pressure, rel=1e-3)
    assert high["wavenumber"][0].real == pytest.approx(2 * np.pi * 1e10 / AIR.sound_speed, rel=1e-3)


def test_absorption_at_oblique_incidence_and_in_a_diffuse_field():
    """A wave at an angle is refracted in the layer: a layer of air itself on a wall has
    the impedance ``-i Z0 cot(k0 d cos θ) / cos θ`` and absorbs nothing; a frame so
    resistive that the sound in it travels across the layer reacts locally: the
    absorption of its normal impedance at that angle; a diffuse field: Paris' average,
    in closed form for a real impedance; an anisotropic frame."""
    from morphogen.homogenization.acoustics import surface_absorption

    f = np.array([250.0, 1000.0, 4000.0])
    theta, d = np.radians(50.0), 0.04
    # air in wide straight pores of porosity 1: the equivalent fluid is air
    air = {"porosity": 1.0, "permeability": 1.0, "tortuosity": 1.0, "viscous_length": 1.0,
           "thermal_length": 1.0}  # fmt: skip
    gap = absorption(f, d, incidence=theta, **air)
    k0 = 2 * np.pi * f / AIR.sound_speed
    exact = -1j * AIR.impedance / (np.cos(theta) * np.tan(k0 * d * np.cos(theta)))
    assert gap["surface_impedance"] == pytest.approx(exact, rel=2e-3)
    assert np.all(np.abs(gap["absorption"]) < 2e-3)
    # normal incidence: as before
    foam = {"porosity": 0.95, "permeability": AIR.viscosity / 10000.0, "tortuosity": 1.05,
            "viscous_length": 100e-6, "thermal_length": 200e-6}  # fmt: skip
    normal = absorption(f, 0.05, **foam)
    assert absorption(f, 0.05, incidence=0.0, **foam)["absorption"] == pytest.approx(
        normal["absorption"], rel=1e-14
    )
    # a dense frame: the wave in it across the layer, the surface reacts locally
    dense = {**foam, "permeability": AIR.viscosity / 2e6}
    local = absorption(f, 0.05, **dense)["surface_impedance"]
    oblique = absorption(f, 0.05, incidence=theta, **dense)
    assert oblique["surface_impedance"] == pytest.approx(local, rel=0.01)
    assert oblique["absorption"] == pytest.approx(surface_absorption(local, theta), rel=0.01)
    assert not np.allclose(oblique["absorption"], absorption(f, 0.05, **dense)["absorption"])
    # the light foam refracts less: its impedance changes with the angle
    light = absorption(f, 0.05, incidence=theta, **foam)
    assert np.abs(light["surface_impedance"] / normal["surface_impedance"] - 1).max() > 0.03
    # a diffuse field: Paris' closed form for real impedances
    for zeta in (0.5, 1.0, 1.567, 4.0):
        paris = 8 / zeta**2 * (1 + zeta - 1 / (1 + zeta) - 2 * np.log(1 + zeta))
        assert surface_absorption(zeta * AIR.impedance, "diffuse") == pytest.approx(paris, rel=1e-9)
    assert surface_absorption(1.567 * AIR.impedance, "diffuse") == pytest.approx(0.951, abs=1e-3)
    diffuse = absorption(f, 0.05, incidence="diffuse", **foam)
    assert diffuse["absorption_by_angle"].shape == (64, 3) and len(diffuse["angles"]) == 64
    angles = np.radians([15.0, 45.0, 75.0])
    by_angle = np.array([absorption(f, 0.05, incidence=t, **foam)["absorption"] for t in angles])
    assert np.all(diffuse["absorption"] > by_angle.min(axis=0))
    assert np.all(diffuse["absorption"] < by_angle.max(axis=0))
    limited = absorption(f, 0.05, incidence="diffuse", limit=np.radians(78.0), **foam)
    assert not np.allclose(limited["absorption"], diffuse["absorption"])
    assert absorption(f, 0.05, incidence="diffuse", **dense)["absorption"] == pytest.approx(
        surface_absorption(local, "diffuse"), rel=0.01
    )
    # an anisotropic frame: easier along the layer, the wave refracts more
    easy = {**foam, "permeability": 10 * foam["permeability"], "tortuosity": 1.0}
    anisotropic = absorption(f, 0.05, incidence=theta, transverse=easy, **foam)
    assert not np.allclose(anisotropic["absorption"], light["absorption"], rtol=1e-3)
    assert absorption(f, 0.05, transverse=easy, **foam)["absorption"] == pytest.approx(
        normal["absorption"], rel=1e-12
    )  # at normal incidence the frame along the layer does not matter
    for bad, match in (({"incidence": 2.0}, "incidence"), ({"incidence": "random"}, "incidence"),
                       ({"incidence": "diffuse", "limit": 3.0}, "limit")):  # fmt: skip
        with pytest.raises(ValueError, match=match):
            absorption(f, 0.05, **foam, **bad)


def test_absorption_of_a_layer():
    foam = {"porosity": 0.95, "permeability": AIR.viscosity / 10000.0, "tortuosity": 1.05,
            "viscous_length": 100e-6, "thermal_length": 200e-6}  # fmt: skip
    f = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    layer = absorption(f, 0.05, **foam)
    alpha = layer["absorption"]
    assert np.all((alpha > 0.0) & (alpha < 1.0)) and alpha[0] < 0.15 < 0.8 < alpha[-1]
    assert np.all(np.diff(alpha[:4]) > 0.0)  # it rises up to the first quarter wavelength
    assert np.allclose(1 - np.abs(layer["reflection"]) ** 2, alpha)
    # a thick layer absorbs as a half space of the equivalent fluid
    medium = equivalent_fluid(f, **foam)
    half = (
        1
        - np.abs((medium["impedance"] - AIR.impedance) / (medium["impedance"] + AIR.impedance)) ** 2
    )
    assert np.allclose(absorption(f, 50.0, **foam)["absorption"], half, rtol=1e-6)
    # a thin layer absorbs nothing, a denser gas changes the answer
    assert absorption([100.0], 1e-5, **foam)["absorption"][0] < 1e-3
    heavy = Fluid(density=5.0, viscosity=1.5e-5, heat_ratio=1.1, prandtl=0.8)
    assert heavy.sound_speed == pytest.approx(np.sqrt(1.1 * 101325.0 / 5.0))
    assert absorption(f, 0.05, fluid=heavy, **foam)["absorption"][2] != pytest.approx(alpha[2])
    assert AIR.sound_speed == pytest.approx(343.2, rel=1e-3)
    for bad, match in (
        ({"frequencies": [-1.0]}, "frequencies"),
        ({"porosity": 1.5}, "porosity"),
        ({"tortuosity": 0.5}, "tortuosity"),
        ({"viscous_length": 0.0}, "viscous_length"),
        ({"thermal_permeability": -1.0}, "thermal_permeability"),
    ):
        arguments = {"frequencies": f, **foam, **bad}
        with pytest.raises(ValueError, match=match):
            equivalent_fluid(arguments.pop("frequencies"), **arguments)
    with pytest.raises(ValueError, match="thickness"):
        absorption(f, 0.0, **foam)
    with pytest.raises(ValueError, match="shape"):
        pore_response(f, "square", 1e-4)
    with pytest.raises(ValueError, match="viscosity"):
        Fluid(viscosity=0.0)


def test_acoustics_in_homogenize_and_configurations(tmp_path, capsys):
    import json

    import yaml

    from morphogen import cli, pipeline
    from morphogen.config import ConfigError, load_config

    image = cylinders(24, 0.2)
    out = homogenize(image, {0: VOID, 1: SOLID}, properties=["acoustic", "hydraulic_permeability"])
    res, flow = out["acoustic.periodic"], out["hydraulic_permeability.periodic"]
    assert np.allclose(flow["tensor"], res["permeability_tensor"]) and flow["mean"] > 0.0
    assert res["tortuosity1"] == pytest.approx(1.2, rel=0.03)
    # the absorption of 3 cm of these fibres of 20 micrometres at 2 kHz, sound across them
    radius = np.sqrt(image.data.mean() / np.pi)
    scale = 10e-6 / radius  # metres per unit of the image
    si = {**along_axis(res, 0)}
    for name, power in (("permeability", 2), ("viscous_length", 1), ("thermal_length", 1),
                        ("thermal_permeability", 2)):  # fmt: skip
        si[name] *= scale**power
    # (a dense mat: its resistance, 18 kPa s/m, far above the impedance of air, reflects)
    alpha = absorption([2000.0], 0.03, **si)["absorption"][0]
    assert equivalent_fluid([2000.0], **si)["resistivity"] * 0.03 > 10 * AIR.impedance
    assert 0.2 < alpha < 0.6
    data = {
        "model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.3},
        "domain": {"periodic": True},
        "seed": 4,
        "voxels": {"voxel_size": 0.25},
        "homogenization": {"method": "fft",
                           "materials": {0: "void", 1: {"young": 1, "poisson": 0.3}},
                           "properties": ["acoustic"], "tolerance": 1e-4},
        "rve": {"sizes": [4.0, 5.0], "realizations": [2, 2], "bootstrap": 0,
                "properties": ["acoustic.tortuosity", "acoustic.permeability",
                               "acoustic.viscous_length.periodic", "acoustic.thermal_length"]},
        "outputs": {"directory": str(tmp_path / "out"), "name": "bed"},
    }  # fmt: skip
    outdir = pipeline.run(load_config(data))
    report = json.loads((outdir / "rve_report.json").read_text(encoding="utf-8"))
    rows = pipeline.read_rows(outdir / "results.csv")
    for row in rows:  # the viscous length below the thermal one, the tortuosity above 1
        assert (
            0.0 < row["acoustic.periodic.viscous_length"] < row["acoustic.periodic.thermal_length"]
        )
        assert 1.0 < row["acoustic.periodic.tortuosity"] < 2.0
        assert row["acoustic.periodic.thermal_permeability"] > row["acoustic.periodic.permeability"]
    for name in ("tortuosity", "permeability", "viscous_length", "thermal_length"):
        stats = report["properties"][f"acoustic.{name}.periodic"]["statistics"]
        assert [s["n"] for s in stats] == [2, 2] and all(s["mean"] > 0 for s in stats)
    with pytest.raises(ConfigError, match="unknown result 'resistivity'"):
        load_config({**data, "rve": {**data["rve"], "properties": ["acoustic.resistivity"]}})
    # the command line
    path = tmp_path / "rods.vti"
    mg.io.save_image(image, path)
    (tmp_path / "materials.yaml").write_text(
        yaml.safe_dump({0: "void", 1: {"young": 1.0, "poisson": 0.3}}), encoding="utf-8"
    )
    args = ["homogenize", str(path), "--materials", str(tmp_path / "materials.yaml"),
            "--property", "acoustic"]  # fmt: skip
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert "tortuosity" in text and "viscous_length" in text and "thermal_permeability" in text
    assert cli.main([*args, "--json"]) == 0
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["acoustic.periodic"]["tortuosity3"] == 1.0
