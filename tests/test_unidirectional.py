"""Unidirectional fibre composites in closed form (morphogen.homogenization.unidirectional):
the composite cylinder assemblage against its limits, Hill's universal relations and the
Hashin–Shtrikman bounds it attains, the generalized self-consistent transverse shear
modulus between Hashin's bounds with their dilute slope, the conductivities and the
expansions (Levin's relation, exact on the FFT solution of a fibre array), the tows as
materials and in configurations, and the FFT stiffness of hexagonal and random arrays of
long fibres (Richardson-extrapolated in the voxel size) against the closed forms and
Rayleigh's multipole solution; misaligned plies, their stiffness turned along the segments
of their fibres and averaged, against the laminates and the voxels."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import morphogen as mg
from morphogen.homogenization import Material, homogenize
from morphogen.homogenization.postprocess import engineering_constants
from morphogen.homogenization.unidirectional import (
    Hill,
    composite_cylinders,
    fibre_directions,
    hill_moduli,
    hill_relations,
    misaligned_conductivity,
    misaligned_stiffness,
    transverse_shear_bounds,
    turned_conductivity,
    turned_stiffness,
)

GLASS = Material(young=72.0, poisson=0.22, conductivity=1.0, expansion=5e-6)
EPOXY = Material(young=3.5, poisson=0.35, conductivity=0.2, expansion=60e-6)
#: A carbon fibre, transversely isotropic about its axis 3 (E_L 230, E_T 15.7, G_LT 15 GPa,
#: nu_LT 0.20, nu_TT 0.36; conductivities 5 across, 50 along; expansions 7e-6 and -0.5e-6)
CARBON = Material.hexagonal(18.08, 6.54, 4.97, 232.05, 15.0, conductivity=(5.0, 50.0),
                            expansion=(7e-6, -0.5e-6))  # fmt: skip


def hill_of(c: np.ndarray) -> np.ndarray:
    """Hill's moduli k, l, n, m, p of a stiffness, averaged over the transverse axes."""
    k = 0.25 * (c[0, 0] + c[1, 1]) + 0.5 * c[0, 1]
    m = 0.5 * (0.25 * (c[0, 0] + c[1, 1]) - 0.5 * c[0, 1] + c[5, 5])
    return np.array([k, 0.5 * (c[0, 2] + c[1, 2]), c[2, 2], m, 0.5 * (c[3, 3] + c[4, 4])])


def test_composite_cylinders_closed_forms():
    """The limits (the matrix, the fibres), Hill's universal relations, the
    Hashin–Shtrikman lower bounds of ``k`` and ``p``, the generalized self-consistent
    ``m`` between Hashin's bounds with the slope of the dilute limit, and the engineering
    constants of the stiffness."""
    for fraction, phase in ((0.0, EPOXY), (1.0, GLASS)):
        r = composite_cylinders(GLASS, EPOXY, fraction)
        assert np.allclose(r.stiffness, hill_moduli(phase).stiffness, rtol=1e-12)
    m, f = hill_moduli(EPOXY), hill_moduli(GLASS)
    for vf in (0.2, 0.5, 0.65):
        r = composite_cylinders(GLASS, EPOXY, vf)
        c = r.constants()
        young, poisson = hill_relations(GLASS, EPOXY, vf, r.moduli.k)
        assert c["young_axial"] == pytest.approx(young, rel=1e-12)
        assert c["poisson_axial"] == pytest.approx(poisson, rel=1e-12)
        assert c["young_axial"] > vf * 72.0 + (1 - vf) * 3.5  # above the rule of mixtures
        vm = 1 - vf
        k_lower = m.k + vf / (1 / (f.k - m.k) + vm / (m.k + m.m))
        p_lower = m.p + vf / (1 / (f.p - m.p) + vm / (2 * m.p))
        assert r.moduli.k == pytest.approx(k_lower, rel=1e-12)
        assert r.moduli.p == pytest.approx(p_lower, rel=1e-12)
        lo, hi = transverse_shear_bounds(GLASS, EPOXY, vf)
        assert lo < r.moduli.m < hi
        for scheme, value in (("lower", lo), ("upper", hi)):
            other = composite_cylinders(GLASS, EPOXY, vf, transverse_shear=scheme)
            assert other.moduli.m == value and other.moduli.k == r.moduli.k
        # Hill: (l - l_m) / (k - k_m) is that of the phases
        slope = (r.moduli.l - m.l) / (r.moduli.k - m.k)
        assert slope == pytest.approx((f.l - m.l) / (f.k - m.k), rel=1e-12)
        s = np.linalg.inv(r.stiffness)
        assert c["young_transverse"] == pytest.approx(1 / s[0, 0])
        assert c["shear_transverse"] == pytest.approx(1 / s[5, 5])
    # the dilute limit of the self-consistent scheme: that of one fibre in the matrix
    for vf in (1e-4, 1e-5):
        lo, _ = transverse_shear_bounds(GLASS, EPOXY, vf)
        r = composite_cylinders(GLASS, EPOXY, vf)
        assert (r.moduli.m - m.m) / vf == pytest.approx((lo - m.m) / vf, rel=2 * vf)
    with pytest.raises(ValueError, match="fibre_fraction"):
        composite_cylinders(GLASS, EPOXY, 1.2)
    with pytest.raises(ValueError, match="transverse_shear"):
        composite_cylinders(GLASS, EPOXY, 0.5, transverse_shear="mori")


def test_conductivities_and_expansions():
    """The rule of mixtures along the fibres, Hashin's two-dimensional formula across them
    (Maxwell's form), and the expansions: those of phases expanding alike, the mean of
    phases of the same moduli, the axial expansion between Schapery's estimate and the
    fibres'."""
    vf = 0.55
    r = composite_cylinders(GLASS, EPOXY, vf)
    across, along = r.conductivity
    assert along == pytest.approx(vf * 1.0 + (1 - vf) * 0.2)
    maxwell = 0.2 * (1.2 + vf * 0.8) / (1.2 - vf * 0.8)
    assert across == pytest.approx(maxwell, rel=1e-12)
    same = composite_cylinders(Material(young=72.0, poisson=0.22, expansion=3e-6),
                               Material(young=3.5, poisson=0.35, expansion=3e-6), vf)  # fmt: skip
    assert same.expansion == pytest.approx((3e-6, 3e-6), rel=1e-9)
    twins = composite_cylinders(Material(young=3.5, poisson=0.35, expansion=5e-6), EPOXY, vf)
    assert twins.expansion == pytest.approx((vf * 5e-6 + (1 - vf) * 60e-6,) * 2, rel=1e-9)
    schapery = (72.0 * 5e-6 * vf + 3.5 * 60e-6 * (1 - vf)) / (72.0 * vf + 3.5 * (1 - vf))
    assert schapery < r.expansion[1] < 1.1 * schapery
    assert r.expansion[0] > vf * 5e-6 + (1 - vf) * 60e-6  # the transverse Poisson effect
    tow = composite_cylinders(CARBON, EPOXY, 0.6)
    assert tow.conductivity[1] == pytest.approx(0.6 * 50.0 + 0.4 * 0.2)
    assert tow.expansion[1] < 1e-6 < tow.expansion[0]
    # Hill's relation with a transversely isotropic fibre
    f, m = hill_moduli(CARBON), hill_moduli(EPOXY)
    slope = (tow.moduli.l - m.l) / (tow.moduli.k - m.k)
    assert slope == pytest.approx((f.l - m.l) / (f.k - m.k), rel=1e-12)


def test_tows_as_materials():
    """The tow as a hexagonal material (axis 3 along the fibres) and back, in
    configurations (``unidirectional``), and the errors."""
    r = composite_cylinders(CARBON, EPOXY, 0.6)
    tow = r.material()
    assert tow.symmetry == "hexagonal"
    assert np.allclose(tow.stiffness, r.stiffness)
    assert dataclasses.astuple(hill_moduli(tow)) == pytest.approx(dataclasses.astuple(r.moduli))
    spec = {"unidirectional": {"fibre": {"hexagonal": [18.08, 6.54, 4.97, 232.05, 15.0],
                                         "conductivity": [5.0, 50.0]},
                               "matrix": {"young": 3.5, "poisson": 0.35, "conductivity": 0.2},
                               "fibre_fraction": 0.6}}  # fmt: skip
    from_config = Material.from_config(spec)
    assert np.allclose(from_config.stiffness, r.stiffness)
    assert np.allclose(np.diag(from_config.conductivities), [r.conductivity[0]] * 2
                       + [r.conductivity[1]])  # fmt: skip
    lower = Material.from_config({"unidirectional": dict(spec["unidirectional"],
                                                         transverse_shear="lower")})  # fmt: skip
    assert np.asarray(lower.stiffness)[5, 5] < np.asarray(from_config.stiffness)[5, 5]
    with pytest.raises(ValueError, match="fibre, matrix, fibre_fraction"):
        Material.from_config({"unidirectional": {"fibre": {"young": 1.0, "poisson": 0.2}}})
    with pytest.raises(ValueError, match="transversely isotropic"):
        hill_moduli(Material.cubic(10.0, 4.0, 5.0))
    with pytest.raises(ValueError, match="isotropic phases"):
        hill_relations(CARBON, EPOXY, 0.5, 10.0)
    assert Hill(1.0, 0.5, 3.0, 0.4, 0.6).stiffness[2, 2] == 3.0


def test_hill_and_levin_exact_on_fibre_arrays():
    """Hill's relation ``(l - l_m) / (k - k_m) = (l_f - l_m) / (k_f - k_m)`` and Levin's
    expansion hold for the FFT solution of an array of fibres, whatever its resolution:
    uniform strains are admissible in the discrete problem."""
    model = mg.models.LongFibers(radius=0.5, volume_fraction=0.5, min_gap=0.1)
    micro = model.generate(mg.Domain(size=(4.0, 4.0, 1.0), periodic=True), seed=2)
    img = micro.voxelize((48, 48, 1))
    out = homogenize(img, {0: EPOXY, 1: GLASS}, properties=["stiffness", "thermal_expansion"])
    c = out["stiffness.periodic"]["tensor"]
    k, l = hill_of(c)[:2]
    m, f = hill_moduli(EPOXY), hill_moduli(GLASS)
    assert (l - m.l) / (k - m.k) == pytest.approx((f.l - m.l) / (f.k - m.k), rel=1e-6)
    alpha = np.asarray(out["thermal_expansion.periodic"]["tensor"], dtype=float)
    alpha = np.diag(alpha) if alpha.ndim == 2 else alpha[:3]
    vf = float(img.data.mean())
    sf, sm = (np.linalg.inv(hill_moduli(x).stiffness) for x in (GLASS, EPOXY))
    levin = (vf * 5e-6 + (1 - vf) * 60e-6) * np.r_[1.0, 1.0, 1.0, 0, 0, 0] + (
        np.linalg.inv(c) - vf * sf - (1 - vf) * sm
    ) @ np.linalg.solve(sf - sm, (5e-6 - 60e-6) * np.r_[1.0, 1.0, 1.0, 0, 0, 0])
    assert alpha[:3] == pytest.approx(levin[:3], rel=1e-5)


def rayleigh_hexagonal(ratio: float, f: float) -> float:
    """The two-dimensional conductivity (over that of the matrix) of a hexagonal array of
    cylinders at the fraction ``f`` and of the conductivity ``ratio`` times that of the
    matrix, by Rayleigh's multipole method (Perrins, McKenzie & McPhedran 1979): ``1 - 2f
    / (T + f - 0.075422 f^6 / (T - 1.060283 f^12))``, ``T = (1 + ratio) / (1 - ratio)``. The
    antiplane shear of fibres (the axial shear modulus) is the same problem."""
    t = (1.0 + ratio) / (1.0 - ratio)
    return 1.0 - 2.0 * f / (t + f - 0.075422 * f**6 / (t - 1.060283 * f**12))


@pytest.mark.slow
def test_fft_stiffness_of_fibre_arrays():
    """The FFT stiffness of long fibres on two resolutions, extrapolated linearly in the
    voxel size (the error of the staircase of the voxels): a hexagonal array (fraction
    0.6545) within 2.5 % of the assemblage for ``k`` and ``p`` (``p`` within 1.5 % of
    Rayleigh's solution, 1.0 % above the assemblage), ``m`` within 2 % of the
    self-consistent scheme and ``n`` within 0.5 %; a random array at 0.5 whose fibres keep
    0.4 radius apart, within 3 % (``k``, ``m``) and 5 % (``p``)."""
    hexagonal = mg.models.LongFibers(radius=0.5, volume_fraction=0.6, arrangement="hexagonal")
    random = mg.models.LongFibers(radius=0.5, volume_fraction=0.5, min_gap=0.2)
    cases = (
        (hexagonal, 12.0, 1, (192, 384), (0.025, 0.02, 0.005, 0.025)),
        (random, 8.0, 3, (128, 256), (0.03, 0.03, 0.005, 0.05)),
    )
    for model, size, seed, (coarse, fine), (tk, tm, tn, tp) in cases:
        micro = model.generate(mg.Domain(size=(size, size, 1.0), periodic=True), seed=seed)
        vf = micro.summary()["volume_fraction"]
        values = []
        for n in (coarse, fine):
            img = micro.voxelize((n, n, 1))
            c = homogenize(img, {0: EPOXY, 1: GLASS}, properties=["stiffness"])
            values.append(hill_of(c["stiffness.periodic"]["tensor"]))
        k, _, n_, m, p = 2.0 * values[1] - values[0]
        r = composite_cylinders(GLASS, EPOXY, vf).moduli
        assert k == pytest.approx(r.k, rel=tk) and k > r.k * (1 - 0.005)
        assert m == pytest.approx(r.m, rel=tm)
        assert n_ == pytest.approx(r.n, rel=tn)
        assert p == pytest.approx(r.p, rel=tp) and p > r.p * (1 - 0.005)
        if model is hexagonal:
            ratio = hill_moduli(GLASS).p / hill_moduli(EPOXY).p
            exact = rayleigh_hexagonal(ratio, vf) * hill_moduli(EPOXY).p
            assert exact / r.p - 1 == pytest.approx(0.010, abs=0.002)
            assert p == pytest.approx(exact, rel=0.015)
            assert math.isclose(vf, 120 * math.pi * 0.25 / 144.0)


def test_misaligned_stiffness_in_closed_form():
    """A stiffness transversely isotropic about the axis 3 turned by theta toward y: its
    modulus along z that of the laminates, ``1/E = cos^4/E_L + sin^4/E_T + (1/G_LT - 2
    nu_LT/E_L) sin^2 cos^2``; an isotropic stiffness unchanged by any turn; one direction
    the turned stiffness under the three averages; over directions scattered about z,
    Voigt >= layers >= Reuss (their differences positive semidefinite); the directions of
    the segments of long fibres, whose RMS angle is their misalignment."""
    c = Hill(k=10.0, l=5.0, n=130.0, m=4.0, p=5.0).stiffness
    e = engineering_constants(c)
    for degrees in (0.0, 5.0, 30.0, 60.0, 90.0):
        t = math.radians(degrees)
        turned = turned_stiffness(c, [[0.0, math.sin(t), math.cos(t)]])[0]
        s2, c2 = math.sin(t) ** 2, math.cos(t) ** 2
        compliance = (c2 * c2 / e["E3"] + s2 * s2 / e["E1"]
                      + (1.0 / e["G31"] - 2.0 * e["nu31"] / e["E3"]) * s2 * c2)  # fmt: skip
        assert engineering_constants(turned)["E3"] == pytest.approx(1.0 / compliance, rel=1e-12)
    rng = np.random.default_rng(0)
    isotropic = Hill(k=14.0, l=10.0, n=18.0, m=4.0, p=4.0).stiffness  # (lambda 10, mu 4)
    assert np.abs(turned_stiffness(isotropic, rng.normal(size=(20, 3))) - isotropic).max() < 1e-12
    directions = np.column_stack([0.1 * rng.normal(size=(200, 2)), np.ones(200)])
    one = misaligned_stiffness(c, directions[:1])
    assert all(np.allclose(one[k], turned_stiffness(c, directions[:1])[0]) for k in one)
    out = misaligned_stiffness(c, directions, np.full(200, 2.0))
    assert np.linalg.eigvalsh(out["voigt"] - out["layers"]).min() > -1e-9
    assert np.linalg.eigvalsh(out["layers"] - out["reuss"]).min() > -1e-9
    e3 = [engineering_constants(out[k])["E3"] for k in ("voigt", "layers", "reuss")]
    assert e3[0] > e3[1] > e3[2] and e3[0] < e["E3"]
    with pytest.raises(ValueError, match="weight"):
        misaligned_stiffness(c, directions, np.ones(3))
    model = mg.models.LongFibers(radius=0.5, volume_fraction=0.5, misalignment=3.0)
    ply = model.generate(mg.Domain(size=(10.0, 10.0, 16.0), periodic=True), seed=1)
    d, lengths = fibre_directions(ply)
    assert np.all(d[:, 2] > 0.0) and np.allclose(np.linalg.norm(d, axis=1), 1.0)
    rms = math.degrees(math.sqrt(float(np.mean(np.arccos(d[:, 2]) ** 2))))
    assert rms == pytest.approx(ply.summary()["misalignment_rms"], rel=1e-9)
    assert len(lengths) == len(d) and np.all(lengths > 0.0)


def test_misaligned_conductivity_in_closed_form():
    """A conductivity of fibres along z (10 along them, 1 across) turned by theta in the
    plane xz: k_L cos^2 + k_T sin^2 along z, (k_L - k_T) sin cos its entry xz; one direction,
    the turned tensor by every average; directions in the plane xz, the axial conductivity
    of Voigt their mean and that of the layers their harmonic mean; Voigt >= layers >=
    Reuss; an isotropic conductivity unchanged; a tensor not 3 x 3 refused."""
    k = np.diag([1.0, 1.0, 10.0])
    theta = np.radians(np.linspace(0.0, 30.0, 7))
    d = np.column_stack([np.sin(theta), np.zeros(7), np.cos(theta)])
    turned = turned_conductivity(k, d)
    axial = 10.0 * np.cos(theta) ** 2 + np.sin(theta) ** 2
    np.testing.assert_allclose(turned[:, 2, 2], axial, rtol=1e-12)
    np.testing.assert_allclose(turned[:, 0, 2], 9.0 * np.sin(theta) * np.cos(theta),
                               rtol=1e-12, atol=1e-15)  # fmt: skip
    one = misaligned_conductivity(k, d[3:4])
    assert all(np.allclose(one[a], turned[3], rtol=1e-12) for a in one)
    out = misaligned_conductivity(k, d)
    assert out["voigt"][2, 2] == pytest.approx(axial.mean(), rel=1e-12)
    assert out["layers"][2, 2] == pytest.approx(1.0 / np.mean(1.0 / axial), rel=1e-12)
    assert np.linalg.eigvalsh(out["voigt"] - out["layers"]).min() > -1e-12
    assert np.linalg.eigvalsh(out["layers"] - out["reuss"]).min() > -1e-12
    iso = misaligned_conductivity(2.0 * np.eye(3), d, np.arange(1.0, 8.0))
    assert all(np.allclose(iso[a], 2.0 * np.eye(3), rtol=1e-12) for a in iso)
    with pytest.raises(ValueError, match="3 x 3"):
        turned_conductivity(np.eye(2), d)


def test_fibre_directions_of_coated_and_hollow_fibres():
    """The segments of the fibres weighted with their volumes: those that a coating covers,
    those of hollow fibres about their lumens; a structure without segments refused."""
    from morphogen.models.nested import Coating

    domain = mg.Domain(size=(10.0, 10.0, 16.0), periodic=True)
    fibres = mg.models.LongFibers(radius=0.5, volume_fraction=0.4, misalignment=3.0)
    coated = Coating(model=fibres, thickness=0.1, side="outside", phase=2).generate(domain, seed=1)
    d, volumes = fibre_directions(coated)
    segments = coated.inner.segments
    lengths = np.linalg.norm(segments.end - segments.start, axis=1)
    assert len(d) == len(lengths) and np.all(d[:, 2] > 0.0)
    np.testing.assert_allclose(volumes, np.pi * segments.radii**2 * lengths, rtol=1e-12)
    model = mg.models.HardcoreFibers(radius=0.5, length=3.0, number=6, lumen=0.5)
    d, volumes = fibre_directions(model.generate(domain, seed=1))  # of the fibres, not lumens
    assert len(d) == 6 and volumes.sum() == pytest.approx(6 * np.pi * 0.25 * 3.0, rel=1e-12)
    with pytest.raises(ValueError, match="segments of the fibres"):
        fibre_directions(mg.models.Voronoi(n_cells=4).generate(domain, seed=1))


def _has_mmg2d() -> bool:
    from morphogen.mesh.conforming import mmg_executable

    try:
        mmg_executable(None, "mmg2d")
    except (FileNotFoundError, RuntimeError, ImportError):
        return False
    return True


@pytest.mark.slow
@pytest.mark.skipif(not _has_mmg2d(), reason="mmg2d (pymmg) is not installed")
def test_misaligned_ply_between_its_bounds():
    """A ply of fibres misaligned by 5 degrees (8 fibres of radius 2 filling 0.4 of the box,
    1.2 apart at least): the axial modulus of its section in its plane, that of straight
    fibres (92.5), above the Voigt bound of its stiffness turned along the segments of its
    fibres (90.8); the FFT solution of its voxels (0.4: the gaps three voxels at least, which
    the voxels keep open) between the bounds of Voigt and Reuss (76.2), at 81.8."""
    epoxy, carbon = Material(young=3.5, poisson=0.35), Material(young=230.0, poisson=0.2)
    model = mg.models.LongFibers(radius=2.0, volume_fraction=0.4, min_gap=1.2, misalignment=5.0)
    ply = model.generate(mg.Domain(size=(16.0, 16.0, 48.0), periodic=True), seed=2)
    section = mg.mesh.plane_mesh(ply, size=0.6, order=2)
    out = homogenize(section, {0: epoxy, 1: carbon}, properties=["stiffness"])
    aligned = out["stiffness.periodic"]["tensor"]
    bounds = misaligned_stiffness(aligned, *fibre_directions(ply))
    image = ply.voxelize(voxel_size=0.4)
    fft = homogenize(image, {0: epoxy, 1: carbon}, properties=["stiffness"])
    e3 = {k: engineering_constants(c)["E3"] for k, c in bounds.items()}
    straight = engineering_constants(aligned)["E3"]
    voxels = engineering_constants(fft["stiffness.periodic"]["tensor"])["E3"]
    assert straight > e3["voigt"] > voxels > e3["layers"] >= e3["reuss"] * (1 - 1e-9)
    assert straight == pytest.approx(92.5, rel=0.01) and voxels == pytest.approx(81.8, rel=0.01)


def _ply_config(tmp_path, **homogenization) -> dict:
    """A ply of fibres misaligned by 5 degrees meshed in its plane, its stiffness and that of
    the misaligned ply (homogenization.misaligned)."""
    return {
        "model": {"type": "long_fibers", "radius": 2.0, "volume_fraction": 0.3, "min_gap": 0.5,
                  "misalignment": 5.0},
        "domain": {"size": [16.0, 16.0, 48.0], "periodic": True},
        "seed": 2,
        "mesh": {"plane": True, "size": 1.0, "order": 1},
        "homogenization": {
            "method": "fe",
            "misaligned": True,
            "materials": {0: {"young": 3.5, "poisson": 0.35, "conductivity": 0.2},
                          1: {"young": 230.0, "poisson": 0.2, "conductivity": 10.0}},
            "boundary_conditions": ["periodic", "kubc"],
            "properties": ["stiffness", "conductivity_tensor"],
            **homogenization,
        },
        "outputs": {"directory": str(tmp_path / "out"), "name": "ply"},
    }  # fmt: skip


@pytest.mark.skipif(not _has_mmg2d(), reason="mmg2d (pymmg) is not installed")
def test_runs_record_the_misaligned_ply(tmp_path):
    """A run of a ply misaligned by 5 degrees, meshed in its plane, records beside the
    stiffness of its section (that of straight fibres: an axial modulus of 69.9) those of
    the misaligned ply under each condition: Voigt (68.6), the layers (59.3) and Reuss
    (59.2), in this order, their entries and engineering constants, and its axial
    conductivity in the same order; the option refused without the stiffness or a
    transport tensor, for tetrahedra, and for a structure without fibres."""
    import csv

    from morphogen.config import ConfigError, load_config
    from morphogen.pipeline import run

    run(load_config(_ply_config(tmp_path), base_dir=tmp_path))
    with open(tmp_path / "out" / "results.csv", newline="") as stream:
        values = {k: float(v) for k, v in next(csv.DictReader(stream)).items() if v}
    for bc in ("periodic", "kubc"):
        e3 = [values[f"stiffness{k}.{bc}.E3"] for k in ("", "_voigt", "_layers", "_reuss")]
        assert e3[0] > e3[1] > e3[2] > e3[3]
        assert {f"stiffness_voigt.{bc}.{k}" for k in ("C11", "C66", "E1", "G12")} <= set(values)
        k33 = [values[f"conductivity_tensor{k}.{bc}.C33"] for k in ("", "_voigt", "_layers",
                                                                     "_reuss")]  # fmt: skip
        assert k33[0] > k33[1] > k33[2] >= k33[3] * (1 - 1e-12)
        assert f"conductivity_tensor_reuss.{bc}.mean" in values
    e3 = [values[f"stiffness{k}.periodic.E3"] for k in ("", "_voigt", "_layers", "_reuss")]
    assert e3 == pytest.approx([69.9, 68.6, 59.3, 59.2], rel=0.02)
    with pytest.raises(ConfigError, match="true or false"):
        load_config(_ply_config(tmp_path, misaligned="yes"), base_dir=tmp_path)
    with pytest.raises(ConfigError, match=r"properties: \[stiffness"):
        load_config(_ply_config(tmp_path, properties=["shear_modulus"]), base_dir=tmp_path)
    tetrahedra = _ply_config(tmp_path)
    tetrahedra["mesh"] = {"size": 1.0}
    with pytest.raises(ConfigError, match="sections of the plane"):
        load_config(tetrahedra, base_dir=tmp_path)
    grains = _ply_config(tmp_path)
    grains["model"] = {"type": "voronoi", "n_cells": 4}
    grains["homogenization"]["materials"] = {1: {"young": 1.0, "poisson": 0.3}}
    with pytest.raises(ValueError, match=r"homogenization\.misaligned: .*segments"):
        run(load_config(grains, base_dir=tmp_path))
