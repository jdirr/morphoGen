"""Bloch waves in periodic cells: consistent mass matrices, the path of the Brillouin zone,
a homogeneous cell against the folded branches of the acoustic waves, a bilayer against
Rytov's dispersion relation and its band gap; the scalar waves: the sound of layered
fluids against their exact dispersion and Wood's law, the gap of a photonic crystal of
rods (Joannopoulos et al.) and the sound slowed by rigid cylinders."""

import dataclasses

import numpy as np
import pytest
from scipy.optimize import brentq

from morphogen.homogenization.bloch import (
    MASS,
    SCALAR_WAVES,
    band_structure,
    brillouin_path,
    long_wavelength_speeds,
    mass_matrix,
    scalar_band_structure,
    voxel_tet_mesh,
    wave_coefficients,
)
from morphogen.homogenization.materials import Material
from morphogen.image import VoxelImage


def test_mass_matrices_and_path():
    for order in (1, 2):
        assert MASS[order].sum() == pytest.approx(1.0)  # the shape functions sum to 1
    np.testing.assert_allclose(  # corner, corner pair, corner on and off an edge, edges
        420 * MASS[2][[0, 0, 0, 0, 4, 4, 4], [0, 1, 4, 5, 4, 5, 9]], [6, 1, -4, -6, 32, 16, 8]
    )
    img = VoxelImage(np.array([[[0, 1]]], np.uint8), spacing=(0.5, 1.0, 2.0), periodic=True)
    mesh = voxel_tet_mesh(img).quadratic()
    rho = np.where(mesh.phase == 1, 3.0, 1.0)
    assert mass_matrix(mesh, rho).sum() == pytest.approx(
        (1.0 + 3.0) * 0.5 * 1.0 * 2.0
    )  # voxels of 1
    ks, dist, labels = brillouin_path((1.0, 2.0, 4.0), points=4)
    np.testing.assert_allclose(
        ks[[0, 4, 8, 12, 16]],
        np.pi * np.array([[0, 0, 0], [1, 0, 0], [1, 0.5, 0], [0, 0, 0], [1, 0.5, 0.25]]),
    )
    assert [name for _, name in labels] == ["Γ", "X", "M", "Γ", "R"]
    assert labels[-1][0] == pytest.approx(dist[-1]) and np.all(np.diff(dist) > 0)


def test_homogeneous_cell_against_folded_acoustic_branches():
    materials = {0: Material(young=1.0, poisson=0.25, density=1.0)}
    ct, cl = np.sqrt(0.4), np.sqrt(1.2)  # mu = lambda = 0.4
    img = VoxelImage(np.zeros((4, 4, 4), np.uint8), spacing=(0.25, 0.25, 0.25), periodic=True)
    ks = np.pi * np.array([[1.0, 0.0, 0.0], [0.3, 0.2, 0.1], [0.0, 0.0, 0.0]])
    bands = band_structure(img, materials, ks, bands=10)
    lattice = 2 * np.pi * (np.array(list(np.ndindex(5, 5, 5))) - 2)
    for k, found in zip(ks, bands.frequencies, strict=True):
        q = np.linalg.norm(k + lattice, axis=1)
        exact = np.sort(np.concatenate([ct * q, ct * q, cl * q]))[:10]
        np.testing.assert_allclose(found, exact, rtol=0.01, atol=1e-6)
    assert np.all(bands.frequencies[2, :3] < 1e-6)  # the translations at Γ
    with pytest.raises(ValueError, match="density"):
        band_structure(img, {0: Material(young=1.0, poisson=0.25)}, ks[:1])
    with pytest.raises(ValueError, match="order"):
        band_structure(img, materials, ks[:1], order=3)


def _rytov(q, c, z, d=(0.5, 0.5), top=10.0):
    """Frequencies of the plane waves normal to a bilayer at the wavenumber ``q``."""

    def f(w):
        a, b = w * d[0] / c[0], w * d[1] / c[1]
        mix = 0.5 * (z[0] / z[1] + z[1] / z[0])
        return np.cos(a) * np.cos(b) - mix * np.sin(a) * np.sin(b) - np.cos(q)

    w = np.linspace(1e-6, top, 20000)
    v = f(w)
    return [brentq(f, w[i], w[i + 1]) for i in np.flatnonzero(np.sign(v[:-1]) != np.sign(v[1:]))]


def test_bilayer_against_rytov():
    young, density, nu = (1.0, 10.0), (1.0, 2.0), 0.25
    materials = {p: Material(young=young[p], poisson=nu, density=density[p]) for p in (0, 1)}
    cl = [np.sqrt(1.2 * e / r) for e, r in zip(young, density, strict=True)]
    ct = [np.sqrt(0.4 * e / r) for e, r in zip(young, density, strict=True)]
    zl = [r * c for r, c in zip(density, cl, strict=True)]
    zt = [r * c for r, c in zip(density, ct, strict=True)]
    data = np.zeros((8, 2, 2), np.uint8)
    data[4:] = 1
    img = VoxelImage(data, spacing=(0.125, 0.05, 0.05), periodic=True)  # thin: lateral modes high
    for q in (np.pi, 0.5 * np.pi):
        found = band_structure(img, materials, np.array([[q, 0.0, 0.0]]), bands=8).frequencies[0]
        exact = np.sort(_rytov(q, ct, zt) * 2 + _rytov(q, cl, zl))[:8]
        np.testing.assert_allclose(found, exact, rtol=0.005)
    # along Γ-X: no wave between the first longitudinal and the second shear branches
    bands = band_structure(img, materials, ("G", "X"), bands=5, points=6)
    n, low, high = bands.gaps()[0]
    assert n == 3
    assert low == pytest.approx(min(_rytov(np.pi, cl, zl)), rel=0.005)
    assert high == pytest.approx(_rytov(np.pi, ct, zt)[1], rel=0.005)
    np.testing.assert_allclose(
        bands.normalized(1.0, 2 * np.pi), bands.frequencies / (2 * np.pi) ** 2
    )
    pytest.importorskip("matplotlib")
    ax = bands.plot()
    assert len(ax.lines) == 5 + 2 and [t.get_text() for t in ax.get_xticklabels()] == ["Γ", "X"]


def rods(n=16, radius=0.2):
    """A square lattice of rods along z (phase 1), one voxel thick."""
    x = (np.arange(n) + 0.5) / n - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    data = (X**2 + Y**2 < radius**2).astype(np.uint8)[:, :, None]
    return VoxelImage(data, 1.0 / n, periodic=True)


def test_sound_in_layered_fluids_against_the_exact_dispersion():
    """Two fluids in layers: ``cos(q d) = cos(w d1 / c1) cos(w d2 / c2) - (Z1 / Z2 + Z2 /
    Z1) / 2 sin(w d1 / c1) sin(w d2 / c2)`` with the impedances ``Z = rho c``, the gaps
    of the Bragg mirror, and at long wavelengths Wood's law."""
    n, d = 16, 1.0
    data = np.zeros((n, 1, 1), np.uint8)
    data[: n // 2] = 1
    image = VoxelImage(data, d / n, periodic=True)
    rho, bulk = {0: 1.0, 1: 4.0}, {0: 1.0, 1: 9.0}
    a = {p: 1.0 / rho[p] for p in rho}
    b = {p: 1.0 / bulk[p] for p in bulk}
    bands = scalar_band_structure(image, a, b, ("G", "X"), bands=3, points=8)
    c = {p: np.sqrt(bulk[p] / rho[p]) for p in rho}
    z = {p: rho[p] * c[p] for p in rho}

    def exact(w):
        s = np.sin(0.5 * d * w / c[0]) * np.sin(0.5 * d * w / c[1])
        return (np.cos(0.5 * d * w / c[0]) * np.cos(0.5 * d * w / c[1])
                - 0.5 * (z[0] / z[1] + z[1] / z[0]) * s)  # fmt: skip

    q = bands.wavevectors[:, 0]
    for band, tol in enumerate((1e-6, 5e-4, 2e-3)):
        assert np.abs(np.cos(q * d) - exact(bands.frequencies[:, band])).max() < tol
    # the first gap, at the edge of the zone: between the two roots of exact(w) = -1
    (first, low, high), *_ = bands.gaps()
    assert first == 1
    edges = [brentq(lambda w: exact(w) + 1.0, lo, hi) for lo, hi in ((1.0, 3.0), (4.5, 6.0))]
    assert (low, high) == pytest.approx(edges, rel=1e-3)
    # long waves: the mean density across the layers, the mean compliance (Wood)
    speeds = long_wavelength_speeds(image, a, b)
    wood = 1.0 / np.sqrt(np.mean(list(rho.values())) * np.mean(list(b.values())))
    assert speeds["speeds"][0] == pytest.approx(wood, rel=1e-9)
    assert speeds["speeds"][1] == pytest.approx(np.sqrt(0.625 / speeds["mass"]), rel=1e-9)
    slow = scalar_band_structure(image, a, b, np.array([[0.0, 0, 0], [0.02, 0, 0]]), bands=1)
    assert slow.frequencies[1, 0] / 0.02 == pytest.approx(wood, rel=1e-4)
    # the same waves from the materials: fluids as solids of small shear modulus
    mats = {p: Material.from_config({"bulk": bulk[p], "shear": 1e-9, "density": rho[p]})
            for p in rho}  # fmt: skip
    from_materials = band_structure(image, mats, ("G", "X"), bands=3, points=8, waves="acoustic")
    assert np.allclose(from_materials.frequencies, bands.frequencies, rtol=1e-6, atol=1e-5)


def test_antiplane_shear_waves_are_those_of_the_elastic_solid():
    """A homogeneous cell: the antiplane shear waves travel at ``sqrt(mu / rho)``, as the
    transverse branch of the elastic waves."""
    image = VoxelImage(np.zeros((6, 1, 1), np.uint8), 1.0 / 6, periodic=True)
    steel = Material(young=200.0, poisson=0.3, density=8.0)
    ks = np.array([[0.0, 0, 0], [0.4, 0, 0], [0.8, 0, 0]])
    shear = band_structure(image, {0: steel}, ks, bands=1, waves="shear")
    speed = np.sqrt(steel.lame[1] / 8.0)
    assert shear.frequencies[1:, 0] == pytest.approx(speed * ks[1:, 0], rel=1e-5)
    elastic = band_structure(image, {0: steel}, ks, bands=3)
    assert elastic.frequencies[1:, 0] == pytest.approx(shear.frequencies[1:, 0], rel=1e-4)


def test_the_band_gap_of_a_photonic_crystal_of_rods():
    """Alumina rods (permittivity 8.9, radius 0.2 a) in air on a square lattice
    (Joannopoulos, Johnson, Winn and Meade, Photonic Crystals, 2008): a gap of the waves
    polarized along the rods (TM) between 0.32 and 0.44 (in units of 2 pi c / a), none of
    the other polarization (TE)."""
    image = rods()
    mats = {0: Material(permittivity=1.0), 1: Material(permittivity=8.9)}
    path = ("G", "X", "M", "G")
    tm = band_structure(image, mats, path, bands=3, points=4, waves="tm")
    (first, low, high), *_ = tm.gaps()
    assert first == 1
    assert low / (2 * np.pi) == pytest.approx(0.32, abs=0.01)
    assert high / (2 * np.pi) == pytest.approx(0.44, abs=0.01)
    te = band_structure(image, mats, path, bands=3, points=4, waves="te")
    assert not [g for g in te.gaps() if g[0] == 1]
    # long waves: the indices of the two polarizations, between the bounds of Wiener
    f = image.data.mean()
    tm_speed = long_wavelength_speeds(image, *wave_coefficients("tm", mats))["speeds"][0]
    te_speed = long_wavelength_speeds(image, *wave_coefficients("te", mats))["speeds"][0]
    arithmetic, harmonic = 1 - f + 8.9 * f, 1.0 / (1 - f + f / 8.9)
    assert tm_speed == pytest.approx(1.0 / np.sqrt(arithmetic), rel=1e-9)  # E along the rods
    assert 1.0 / np.sqrt(arithmetic) < te_speed < 1.0 / np.sqrt(harmonic)


def test_sound_slowed_by_rigid_cylinders():
    """A sonic crystal of rigid cylinders in air: left out of the cell, they make the path
    of the sound tortuous, ``c = c0 / sqrt(1 + f)`` at long wavelengths (the effective
    density of Rayleigh's and Maxwell's formula in the plane)."""
    image = rods(n=24, radius=0.25)
    f = image.data.mean()
    a, b = {0: 1.0, 1: 0.0}, {0: 1.0, 1: 1.0}
    speeds = long_wavelength_speeds(image, a, b)
    assert speeds["mass"] == pytest.approx(1 - f)
    assert speeds["speeds"][0] == pytest.approx(1.0 / np.sqrt(1 + f), rel=0.01)
    assert speeds["speeds"][2] == pytest.approx(1.0, rel=1e-9)  # along the cylinders
    ks = np.array([[0.0, 0, 0], [0.1, 0, 0]])
    bands = scalar_band_structure(image, a, b, ks, bands=2)
    assert bands.frequencies[1, 0] / 0.1 == pytest.approx(speeds["speeds"][0], rel=5e-3)


def test_scalar_waves_check_their_coefficients():
    assert set(SCALAR_WAVES) == {"acoustic", "shear", "tm", "te"}
    image = rods(n=4)
    air = Material(permittivity=1.0)
    with pytest.raises(ValueError, match="waves: one of"):
        band_structure(image, {0: air, 1: air}, waves="optical")
    with pytest.raises(ValueError, match="permittivity of phase 1"):
        wave_coefficients("tm", {0: air, 1: Material(young=1.0, poisson=0.3)})
    with pytest.raises(ValueError, match="density of phase 0"):
        wave_coefficients("acoustic", {0: Material(young=1.0, poisson=0.3)})
    with pytest.raises(ValueError, match="isotropic elastic moduli"):
        wave_coefficients(
            "shear", {0: dataclasses.replace(Material.cubic(168.0, 121.0, 75.0), density=8.9)}
        )
    a, b = wave_coefficients("te", {0: air, 1: Material(permittivity=4.0, permeability=2.0)})
    assert a == {0: 1.0, 1: 0.25} and b == {0: 1.0, 1: 2.0}
    assert wave_coefficients("tm", {0: air, 1: Material(void=True)})[0][1] == 0.0
    with pytest.raises(ValueError, match="no coefficients for phase"):
        scalar_band_structure(image, {0: 1.0}, {0: 1.0})
    with pytest.raises(ValueError, match="positive coefficients"):
        scalar_band_structure(image, {0: 1.0, 1: -1.0}, {0: 1.0, 1: 1.0})
    with pytest.raises(ValueError, match="no phase carries the waves"):
        scalar_band_structure(image, {0: 0.0, 1: 0.0}, {0: 1.0, 1: 1.0})
    with pytest.raises(ValueError, match="bands"):
        scalar_band_structure(image, {0: 1.0, 1: 1.0}, {0: 1.0, 1: 1.0}, bands=10**6)


def test_bands_from_the_command_line(tmp_path, capsys):
    """``morphogen bands``: the gap of the photonic crystal of rods, its table and figure;
    the elastic bands of a homogeneous cell."""
    import csv
    import json

    import morphogen as mg
    from morphogen import cli

    path = tmp_path / "rods.vti"
    mg.io.save_image(rods(), path)
    args = ["bands", str(path), "--material", "0=permittivity:1", "1=permittivity:8.9",
            "--waves", "tm", "--path", "G", "X", "M", "G", "--bands", "3"]  # fmt: skip
    args += ["--points", "4"]
    table, figure = tmp_path / "bands.csv", tmp_path / "bands.png"
    assert cli.main([*args, "-o", str(table), "--figure", str(figure)]) == 0
    text = capsys.readouterr().out
    assert "3 bands of the tm waves at 13 wave vectors along Γ - X - M - Γ" in text
    assert "band gap between the bands 1 and 2" in text
    with table.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 13 and set(rows[0]) == {"distance", "kx", "ky", "kz", "omega1", "omega2",
                                                "omega3"}  # fmt: skip
    assert float(rows[0]["omega1"]) == pytest.approx(0.0, abs=1e-3)
    assert figure.stat().st_size > 5000
    assert cli.main([*args, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    (gap,) = [g for g in out["gaps"] if g["below"] == 1]
    assert gap["low"] / (2 * np.pi) == pytest.approx(0.32, abs=0.01)
    assert np.shape(out["frequencies"]) == (13, 3) and out["path"] == ["Γ", "X", "M", "Γ"]
    # elastic waves of a homogeneous cell: no gap; the errors of the options
    cell = tmp_path / "cell.npy"
    np.save(cell, np.zeros((3, 3, 3), np.uint8))
    steel = ["--material", "0=young:200,poisson:0.3,density:8"]
    assert cli.main(["bands", str(cell), *steel, "--path", "G", "X", "--bands", "4",
                     "--points", "2", "--order", "1"]) == 0  # fmt: skip
    assert "no band gap along this path" in capsys.readouterr().out
    assert cli.main(["bands", str(cell), *steel, "--path", "G", "Q"]) != 0
    assert "corners of the Brillouin zone" in capsys.readouterr().err
