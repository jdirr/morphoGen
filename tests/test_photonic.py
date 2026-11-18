"""Electromagnetic Bloch waves of three-dimensional photonic crystals by plane waves: the
light cones of a homogeneous medium and of a uniaxial crystal (exact), the Bragg mirror
against its exact dispersion at normal and oblique incidence (both polarizations,
dielectric and magnetic layers), interfaces anywhere in the cells of the plane waves, a
crystal of rods against the two scalar polarizations and out of its plane, the long waves
against the homogenized permittivity, the band gap of the diamond lattice of spheres (Ho,
Chan & Soukoulis 1990; the values of MPB), the iterative solver against the matrices."""

import numpy as np
import pytest
from scipy.optimize import brentq

import morphogen.homogenization.photonic as photonic
from morphogen.homogenization import (
    Material,
    band_structure,
    electromagnetic_band_structure,
    homogenize,
)
from morphogen.homogenization.photonic import PlaneWaves, smoothed_inverse
from morphogen.image import VoxelImage


def reciprocal(shape, size=1.0):
    """The reciprocal vectors of a grid of a cube (n x 3)."""
    axes = [2 * np.pi * np.fft.fftfreq(n, d=size / n) for n in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def layers(n, cut):
    """A cell of ``n`` voxels along x: phase 0 up to the voxel ``cut``, then phase 1."""
    data = np.zeros((n, 1, 1), np.uint8)
    data[cut:] = 1
    return VoxelImage(data, 1.0 / n, periodic=True)


def spheres(n, centres, radius):
    """Spheres (phase 1) of a cubic cell of edge 1, ``n`` voxels per edge."""
    x = (np.arange(n) + 0.5) / n
    grid = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1)
    inside = np.zeros((n, n, n), bool)
    for centre in centres:
        d = grid - np.asarray(centre, dtype=float)
        d -= np.rint(d)
        inside |= (d**2).sum(axis=-1) < radius**2
    return VoxelImage(inside.astype(np.uint8), 1.0 / n, periodic=True)


def diamond(n):
    """The diamond lattice of spheres of radius a/4 (they overlap) in its cubic cell."""
    fcc = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    return spheres(n, [np.add(f, s) for f in fcc for s in (0.125, -0.125)], 0.25)


def bragg(k, ky, eps, mu, d, top, polarization):
    """The frequencies below ``top`` of a stack of two layers (thicknesses ``d``) at the
    Bloch wave number ``k`` across them and ``ky`` along them (Yeh, Yariv & Hong 1977):
    ``cos(k L) = cos(k1 d1) cos(k2 d2) - (p + 1 / p) / 2 sin(k1 d1) sin(k2 d2)`` with ``ki
    = sqrt(eps_i mu_i w² - ky²)`` and ``p`` the ratio of ``ki / mu_i`` (the electric field
    along the layers, ``"s"``) or of ``ki / eps_i`` (``"p"``)."""

    def f(w):
        kx = [np.sqrt(complex(e * m * w * w - ky * ky)) for e, m in zip(eps, mu, strict=True)]
        z = [q / (m if polarization == "s" else e) for q, e, m in zip(kx, eps, mu, strict=True)]
        p = z[1] / z[0]
        value = np.cos(kx[0] * d[0]) * np.cos(kx[1] * d[1])
        value -= 0.5 * (p + 1.0 / p) * np.sin(kx[0] * d[0]) * np.sin(kx[1] * d[1])
        return value.real - np.cos(k * (d[0] + d[1]))

    ws = np.linspace(1e-6, top, 4001)
    values = np.array([f(w) for w in ws])
    change = np.flatnonzero(values[:-1] * values[1:] < 0)
    return np.array([brentq(f, ws[i], ws[i + 1], xtol=1e-13) for i in change])


def stack_error(image, eps, mu, d, k, ky, *, coarsen=1, bands=8):
    """The largest relative error of the first bands of a stack against :func:`bragg`."""
    waves = electromagnetic_band_structure(
        image, dict(enumerate(eps)), np.array([[k, ky, 0.0]]), permeability=dict(enumerate(mu)),
        bands=bands, coarsen=coarsen,
    ).frequencies[0]  # fmt: skip
    top = 1.3 * waves[-1]
    exact = np.sort(np.concatenate([bragg(k, ky, eps, mu, d, top, p) for p in "sp"]))[:bands]
    return float(np.abs(waves / exact - 1.0).max())


def test_light_cones_of_a_homogeneous_medium():
    """A homogeneous cell: ``w = |k + G| / sqrt(eps mu)``, each wave twice (its two
    polarizations), to round-off; and by the iterative solver."""
    image = VoxelImage(np.zeros((6, 6, 6), np.uint8), 1.0 / 6, periodic=True)
    ks = np.pi * np.array([[0.3, 0.2, 0.1], [0.5, 0.0, 0.9], [0.0, 0.0, 0.0]])
    waves = electromagnetic_band_structure(image, {0: 4.0}, ks, permeability=2.25, bands=8)
    g = reciprocal((6, 6, 6))
    for k, row in zip(ks, waves.frequencies, strict=True):
        exact = np.sort(np.repeat(np.linalg.norm(g + k, axis=1) / 3.0, 2))[:8]
        # at the centre of the zone the uniform fields: the root of a round-off
        assert row == pytest.approx(exact, abs=1e-11 if k.any() else 1e-6)
    image = VoxelImage(np.zeros((10, 10, 10), np.uint8), 0.1, periodic=True)
    waves = electromagnetic_band_structure(image, {0: 4.0}, ks[:2], bands=8)  # 2000 unknowns
    g = reciprocal((10, 10, 10))
    for k, row in zip(ks, waves.frequencies, strict=False):
        exact = np.sort(np.repeat(np.linalg.norm(g + k, axis=1) / 2.0, 2))[:8]
        assert row == pytest.approx(exact, rel=1e-6)


def test_uniaxial_crystal():
    """A homogeneous uniaxial crystal (axis z): the ordinary wave ``w = q / sqrt(eps_o)``
    and the extraordinary one, ``w² = qz² / eps_o + (qx² + qy²) / eps_e``."""
    image = VoxelImage(np.zeros((4, 4, 4), np.uint8), 0.25, periodic=True)
    k = np.array([0.7, -0.4, 1.1])
    ordinary, extraordinary = 2.0, 5.0
    waves = electromagnetic_band_structure(
        image, {0: (ordinary, ordinary, extraordinary)}, k[None], bands=6
    )
    q = reciprocal((4, 4, 4)) + k
    exact = np.concatenate(
        [
            np.linalg.norm(q, axis=1) / np.sqrt(ordinary),
            np.sqrt(q[:, 2] ** 2 / ordinary + (q[:, 0] ** 2 + q[:, 1] ** 2) / extraordinary),
        ]
    )
    assert waves.frequencies[0] == pytest.approx(np.sort(exact)[:6], abs=1e-11)
    # the same crystal turned: its axis along x
    turned = electromagnetic_band_structure(
        image, {0: np.diag([extraordinary, ordinary, ordinary])}, k[[2, 1, 0]][None], bands=6
    )
    assert turned.frequencies[0] == pytest.approx(waves.frequencies[0], abs=1e-11)


def test_bragg_mirror_at_normal_and_oblique_incidence():
    """Layers of permittivities 1 and 12: the exact dispersion of both polarizations, at
    normal and oblique incidence, to the second order in the plane waves."""
    eps, mu, d = (1.0, 12.0), (1.0, 1.0), (0.6, 0.4)
    errors = {}
    for n in (40, 80):
        image = layers(n, int(0.6 * n))
        errors[n] = [
            stack_error(image, eps, mu, d, k, ky)
            for k, ky in ((0.4 * np.pi, 0.0), (0.4 * np.pi, 2.0), (0.7 * np.pi, 5.0))
        ]
    assert max(errors[40]) < 1.2e-3 and max(errors[80]) < 3e-4
    assert max(errors[80]) < 0.3 * max(errors[40])
    # the gap of the mirror at normal incidence: no wave between the first two bands
    waves = electromagnetic_band_structure(
        layers(80, 48), {0: 1.0, 1: 12.0}, ("G", "X"), bands=4, points=8
    )
    (first, low, high), *_ = waves.gaps()
    assert first == 2  # the two polarizations of the first band, then the gap
    edge = bragg(np.pi, 0.0, eps, mu, d, 1.2 * high, "s")
    assert (low, high) == pytest.approx(tuple(edge[:2]), rel=1e-3)


def test_magnetic_layers():
    """Layers of different permeabilities (the pencil of the induction): the exact
    dispersion, and the duality of the two fields: the stack of permittivities and
    permeabilities exchanged has the same waves."""
    eps, mu, d = (2.0, 3.0), (1.0, 6.0), (0.5, 0.5)
    image = layers(80, 40)
    for k, ky in ((0.4 * np.pi, 0.0), (0.6 * np.pi, 3.0)):
        assert stack_error(image, eps, mu, d, k, ky) < 5e-4
        assert stack_error(image, mu, eps, d, k, ky) < 5e-4
    # magnetic layers alone, by their materials
    mats = {0: Material(permittivity=1.0), 1: Material(permittivity=1.0, permeability=9.0)}
    ks = np.array([[0.5 * np.pi, 1.0, 0.0]])
    magnetic = band_structure(image, mats, ks, bands=6, waves="electromagnetic").frequencies
    exact = np.sort(
        np.concatenate(
            [
                bragg(ks[0, 0], 1.0, (1.0, 1.0), (1.0, 9.0), d, 1.3 * magnetic[0, -1], p)
                for p in "sp"
            ]
        )
    )[:6]
    assert magnetic[0] == pytest.approx(exact, rel=5e-4)


def test_interfaces_inside_the_cells_of_the_plane_waves():
    """An interface that cuts the cells of a coarser grid of plane waves: the cell takes
    the tensor of its layers (harmonic mean across, arithmetic mean along), and the waves
    keep their accuracy; without it the interface is moved to a face of the cells."""
    eps, mu = (1.0, 12.0), (1.0, 1.0)
    fine = layers(320, 198)  # 80 plane waves: the interface in the middle of a cell
    d = (198 / 320, 122 / 320)
    assert stack_error(fine, eps, mu, d, 0.4 * np.pi, 2.0, coarsen=4) < 3e-4
    for face in (49, 50):  # the faces of that cell
        assert stack_error(layers(80, face), eps, mu, d, 0.4 * np.pi, 2.0) > 3e-3
    # the tensor of the cell that is cut (its voxels 196 to 199), and the cells of one phase
    field = np.where(np.asarray(fine.data) == 1, 12.0, 1.0)
    inverse = smoothed_inverse(field, 4)
    assert inverse.shape == (3, 3, 80, 1, 1)
    assert inverse[:, :, 49, 0, 0] == pytest.approx(np.diag([0.5 + 0.5 / 12.0, 1 / 6.5, 1 / 6.5]))
    assert inverse[:, :, 48, 0, 0] == pytest.approx(np.eye(3))
    assert inverse[:, :, 50, 0, 0] == pytest.approx(np.eye(3) / 12.0)
    assert smoothed_inverse(field) == pytest.approx(1.0 / field)
    with pytest.raises(ValueError, match="does not divide"):
        smoothed_inverse(field, 3)


def rods(n=12, radius=0.2):
    x = (np.arange(n) + 0.5) / n - 0.5
    data = (x[:, None] ** 2 + x[None, :] ** 2 < radius**2).astype(np.uint8)[:, :, None]
    return VoxelImage(data, 1.0 / n, periodic=True)


def test_crystal_of_rods_in_and_out_of_its_plane():
    """Alumina rods on a square lattice: in the plane the waves are the two scalar
    polarizations together (finite elements on the same voxels); out of the plane, which
    the scalar waves refuse, they lie above the light line of the densest medium."""
    image = rods()
    mats = {0: Material(permittivity=1.0), 1: Material(permittivity=8.9)}
    ks = np.pi * np.array([[1.0, 0, 0], [1, 1, 0], [0.5, 0.5, 0], [0.4, 0.1, 0]])
    tm = band_structure(image, mats, ks, bands=4, waves="tm").frequencies
    te = band_structure(image, mats, ks, bands=4, waves="te").frequencies
    both = np.sort(np.hstack([tm, te]), axis=1)[:, :6]
    light = band_structure(image, mats, ks, bands=6, waves="electromagnetic").frequencies
    assert light == pytest.approx(both, rel=0.015)
    # the cells of a finer image averaged: nearer the waves of the finer image
    fine = np.sort(
        np.hstack(
            [
                band_structure(rods(48), mats, ks[:1], bands=4, waves=w).frequencies
                for w in ("tm", "te")
            ]
        )
    )[0, :6]
    smoothed = band_structure(
        rods(48), mats, ks[:1], bands=6, waves="electromagnetic", coarsen=4
    ).frequencies[0]
    assert smoothed == pytest.approx(fine, rel=0.01)
    assert np.abs(smoothed / fine - 1).max() < 0.7 * np.abs(light[0] / fine - 1).max()
    # out of the plane
    k = np.array([[0.3, 0.0, 2.0]])
    for waves in ("tm", "te"):
        with pytest.raises(ValueError, match="wave vectors in the plane"):
            band_structure(image, mats, k, bands=2, waves=waves)
    steel = Material(young=200.0, poisson=0.3, density=8.0)
    with pytest.raises(ValueError, match="the waves that leave it: 'elastic'"):
        band_structure(image, {0: steel, 1: steel}, k, bands=2, waves="shear")
    out = band_structure(image, mats, k, bands=4, waves="electromagnetic").frequencies[0]
    assert np.all(out > np.linalg.norm(k) / np.sqrt(8.9)) and out[0] < np.linalg.norm(k)
    # the default path of a cell one voxel thick stays in its plane
    default = band_structure(image, mats, bands=2, points=2, waves="electromagnetic")
    assert [name for _, name in default.labels] == ["Γ", "X", "M", "Γ"]
    assert default.frequencies.shape == (7, 2)


def test_long_waves_and_the_homogenized_permittivity():
    """At long wavelengths light sees the homogenized medium: in layers, the arithmetic
    mean of the permittivity for a field along them and its harmonic mean across (exact);
    in a cubic lattice of spheres, the permittivity of the FFT solver."""
    data = np.zeros((1, 1, 16), np.uint8)
    data[:, :, :4] = 1
    image = VoxelImage(data, 1.0 / 16, periodic=True)
    eps = {0: 2.0, 1: 9.0}
    arithmetic, harmonic = 0.75 * 2.0 + 0.25 * 9.0, 1.0 / (0.75 / 2.0 + 0.25 / 9.0)
    k = 1e-3
    along = electromagnetic_band_structure(image, eps, np.array([[k, 0.0, 0.0]]), bands=2)
    assert along.frequencies[0] / k == pytest.approx(
        [1 / np.sqrt(arithmetic), 1 / np.sqrt(harmonic)], rel=1e-6
    )
    across = electromagnetic_band_structure(image, eps, np.array([[0.0, 0.0, k]]), bands=2)
    assert across.frequencies[0] / k == pytest.approx(1 / np.sqrt(arithmetic), rel=1e-6)
    image = spheres(24, [(0.5, 0.5, 0.5)], 0.3)
    mats = {0: Material(permittivity=1.0), 1: Material(permittivity=12.0)}
    waves = band_structure(
        image, mats, np.array([[0.05, 0.0, 0.0]]), bands=2, waves="electromagnetic", coarsen=2
    )
    effective = homogenize(image, mats, properties=["permittivity"])["permittivity.periodic"]
    assert waves.frequencies[0] / 0.05 == pytest.approx(1.0 / np.sqrt(effective.value), rel=0.01)


def test_band_gap_of_the_diamond_lattice_of_spheres():
    """Dielectric spheres (permittivity 11.56, radius a/4) on the diamond lattice have a
    complete band gap between their second and third bands (Ho, Chan & Soukoulis 1990):
    from 0.396 to 0.441 c/a in the tutorial of MPB (Johnson & Joannopoulos 2001, 16 plane
    waves per primitive vector), here between the bands 8 and 9 of the cubic cell of four
    primitive cells; its edges, at the points K and L of the face-centred lattice, hold
    to 1 % as the plane waves are refined."""
    waves = electromagnetic_band_structure(
        diamond(64), {0: 1.0, 1: 11.56}, bands=10, points=2, coarsen=4
    )
    assert [g[0] for g in waves.gaps()] == [8]
    low, high = (edge / (2 * np.pi) for edge in waves.gaps()[0][1:])
    assert low == pytest.approx(0.3963, rel=0.02) and high == pytest.approx(0.4408, rel=0.02)
    assert (high - low) / (0.5 * (high + low)) == pytest.approx(0.106, abs=0.01)
    assert waves.normalized(1.0, 1.0) == pytest.approx(waves.frequencies / (2 * np.pi))
    ks = np.pi * np.array([[0.5, 0.5, 0.0], [1.0, 1.0, 1.0]])  # K and L
    finer = electromagnetic_band_structure(
        diamond(80), {0: 1.0, 1: 11.56}, ks, bands=10, coarsen=4
    ).frequencies / (2 * np.pi)
    assert (finer[0, 7], finer[1, 8]) == pytest.approx((low, high), rel=0.01)


def test_iterative_solver_against_the_matrices(monkeypatch):
    """The block Davidson method finds the frequencies of the diagonalized matrices, for
    dielectric and for magnetic crystals, from scratch and from the waves of a neighbour;
    its operators are Hermitian."""
    image = spheres(8, [(0.5, 0.5, 0.5)], 0.33)
    ks = np.pi * np.array([[1.0, 0.3, 0.0], [1.0, 0.5, 0.2], [0.0, 0.0, 0.0]])
    cases = (
        {"permittivity": {0: 1.0, 1: 13.0}},
        {"permittivity": {0: 1.0, 1: 5.0}, "permeability": {0: 1.0, 1: 3.0}},
        {"permittivity": {0: 1.0, 1: np.diag([4.0, 9.0, 2.0])}},
    )
    for case in cases:
        exact = electromagnetic_band_structure(image, path=ks, bands=7, **case).frequencies
        with monkeypatch.context() as patch:
            patch.setattr(photonic, "DENSE_LIMIT", 0)
            found = electromagnetic_band_structure(image, path=ks, bands=7, **case).frequencies
        # the squares of the frequencies: the uniform fields of the centre have none
        assert found**2 == pytest.approx(exact**2, rel=1e-6, abs=1e-5)
    inverse = 1.0 / np.where(np.asarray(image.data) == 1, 13.0, 1.0)
    waves = PlaneWaves(inverse, image.size, 1.0 / np.where(image.data == 1, 2.0, 1.0)).at(ks[0])
    rng = np.random.default_rng(1)
    x = rng.standard_normal((waves.unknowns, 3)) + 1j * rng.standard_normal((waves.unknowns, 3))
    for operator in (waves.stiffness, waves.mass, waves.precondition):
        matrix = x.conj().T @ operator(x)
        assert matrix == pytest.approx(matrix.conj().T, rel=1e-10, abs=1e-10)
        assert np.linalg.eigvalsh(matrix).min() > 0.0
    assert operator(x[:, 0]) == pytest.approx(operator(x)[:, 0])


def test_refusals():
    image = spheres(4, [(0.5, 0.5, 0.5)], 0.3)
    air, glass = Material(permittivity=1.0), Material(permittivity=2.25)
    with pytest.raises(ValueError, match="waves: one of"):
        band_structure(image, {0: air, 1: glass}, waves="optical")
    with pytest.raises(ValueError, match="permittivity of phase 1"):
        band_structure(image, {0: air, 1: Material(void=True)}, waves="electromagnetic")
    with pytest.raises(ValueError, match="no permittivity for phase"):
        electromagnetic_band_structure(image, {0: 1.0})
    with pytest.raises(ValueError, match="a positive permittivity"):
        electromagnetic_band_structure(image, {0: 1.0, 1: -2.0})
    with pytest.raises(ValueError, match="a positive permeability"):
        electromagnetic_band_structure(image, {0: 1.0, 1: 2.0}, permeability=0.0)
    with pytest.raises(ValueError, match="isotropic"):
        electromagnetic_band_structure(image, {0: 1.0, 1: (1.0, 2.0, 3.0)}, coarsen=2)
    with pytest.raises(ValueError, match="does not divide"):
        electromagnetic_band_structure(image, {0: 1.0, 1: 2.0}, coarsen=3)
    with pytest.raises(ValueError, match="bands"):
        electromagnetic_band_structure(image, {0: 1.0, 1: 2.0}, bands=10**6)
    with pytest.raises(ValueError, match="image of phases"):
        electromagnetic_band_structure(
            VoxelImage(np.zeros((2, 2, 2)), 0.5, periodic=True), {0: 1.0}
        )
    from morphogen.homogenization.bloch import voxel_tet_mesh

    with pytest.raises(TypeError, match="a voxel image"):
        band_structure(voxel_tet_mesh(image), {0: air, 1: glass}, waves="electromagnetic")
    # the scalar polarizations are those of a cell invariant along z
    with pytest.raises(ValueError, match="invariant along z"):
        band_structure(image, {0: air, 1: glass}, ("G", "X"), waves="tm")
    # a uniform permeability given phase by phase is that of the cell
    uniform = electromagnetic_band_structure(
        image, {0: 1.0, 1: 2.0}, ("G", "X"), permeability={0: 4.0, 1: 4.0}, bands=3, points=1
    )
    scalar = electromagnetic_band_structure(image, {0: 1.0, 1: 2.0}, ("G", "X"), bands=3, points=1)
    assert uniform.frequencies == pytest.approx(scalar.frequencies / 2.0, abs=1e-12)


def test_light_from_the_command_line(tmp_path, capsys):
    """``morphogen bands --waves electromagnetic``: the gap of the Bragg mirror."""
    import json

    import morphogen as mg
    from morphogen import cli

    path = tmp_path / "mirror.vti"
    mg.io.save_image(layers(160, 96), path)
    args = ["bands", str(path), "--material", "0=permittivity:1", "1=permittivity:12",
            "--waves", "electromagnetic", "--path", "G", "X", "--bands", "4", "--points", "4",
            "--coarsen", "2", "--json"]  # fmt: skip
    assert cli.main(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["waves"] == "electromagnetic" and out["gaps"][0]["below"] == 2
    edge = bragg(np.pi, 0.0, (1.0, 12.0), (1.0, 1.0), (0.6, 0.4), 4.0, "s")
    assert out["gaps"][0]["low"] == pytest.approx(edge[0], rel=2e-3)
