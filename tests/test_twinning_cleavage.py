"""Twinning and cleavage of crystals: the systems of the hexagonal lattice, the twins and
their shears, the cleavage planes; twins that shear in one sense until the crystal has
twinned; cleavage planes that open under a normal stress and soften; the same in
polycrystals, on voxels and on meshes."""

import dataclasses
import math

import numpy as np
import pytest

from helpers import box_mesh
from morphogen.homogenization import Material
from morphogen.homogenization import crystal as cr
from morphogen.homogenization.behaviours import update_points
from morphogen.homogenization.crystal import (
    CLEAVAGE,
    HEXAGONAL,
    TWINS,
    CrystalPlasticity,
    cleavage_systems,
    slip_systems,
    twin_systems,
)
from morphogen.homogenization.fft_nonlinear import NonlinearFFTProblem
from morphogen.homogenization.finite import update_points_finite
from morphogen.homogenization.materials import material_table
from morphogen.homogenization.nonlinear import (
    NonlinearFEProblem,
    behaviour_table,
    simulate_point,
)
from morphogen.image import VoxelImage
from morphogen.orientations import euler_to_matrix

#: Hexagonal elasticity [c11, c12, c13, c33, c44] of magnesium and of zinc (MPa), and
#: the ratios c / a of their lattices
MAGNESIUM, RATIO_MG = (59400.0, 25600.0, 21400.0, 61600.0, 16400.0), 1.624
ZINC, RATIO_ZN = (165000.0, 31100.0, 50000.0, 61800.0, 39600.0), 1.856
#: Cubic elasticity [c11, c12, c44] of iron (MPa)
IRON = (230000.0, 135000.0, 117000.0)


def hexagonal(constants, **plasticity) -> Material:
    return Material.from_config({"hexagonal": list(constants), "crystal_plasticity": plasticity})


def cubic(**plasticity) -> Material:
    return Material.from_config({"cubic": list(IRON), "crystal_plasticity": plasticity})


def modulus(material, axis=2) -> float:
    """The Young modulus of a crystal along one of its axes."""
    return 1.0 / np.linalg.inv(material.matrix("elastic"))[axis, axis]


# -- the lattices ---------------------------------------------------------------------------------


def test_hexagonal_slip_systems():
    counts = {"basal": 3, "prismatic": 3, "pyramidal_a": 6, "pyramidal_ca": 12, "pyramidal2_ca": 6}
    assert set(counts) == set(HEXAGONAL)
    ratio = RATIO_MG
    for family, count in counts.items():
        n, d = slip_systems(family, ratio)
        assert n.shape == d.shape == (count, 3)
        assert np.allclose(np.linalg.norm(n, axis=1), 1) and np.allclose(
            np.linalg.norm(d, axis=1), 1
        )
        assert np.allclose((n * d).sum(axis=1), 0, atol=1e-12)
        # the family is that of the lattice: its rotations map the systems on themselves
        mine = {cr._key(a, b) for a, b in zip(n, d, strict=True)}
        for r in cr._rotations("hexagonal"):
            for a, b in zip(n @ r.T, d @ r.T, strict=True):
                assert any(cr._key(s * a, t * b) in mine for s in (1, -1) for t in (1, -1))
    n, d = slip_systems("basal", ratio)
    assert np.allclose(np.abs(n), [0, 0, 1]) and np.allclose(d[:, 2], 0)
    n, d = slip_systems("prismatic", ratio)
    assert np.allclose(n[:, 2], 0) and np.allclose(d[:, 2], 0)
    # <c + a>: of length sqrt(1 + (c / a)^2) a, at the angle of c / a to the basal plane
    n, d = slip_systems("pyramidal2_ca", ratio)
    assert np.allclose(np.abs(d[:, 2]), ratio / math.hypot(1.0, ratio))
    # the normal of {10-11} is at atan(2 c / (sqrt 3 a)) to the axis c
    n, _ = slip_systems("pyramidal_a", ratio)
    assert np.allclose(np.abs(n[:, 2]), math.cos(math.atan(2 * ratio / math.sqrt(3))))
    with pytest.raises(ValueError, match="c_over_a"):
        slip_systems("basal")
    with pytest.raises(ValueError, match="positive"):
        slip_systems("basal", -1.0)
    assert len(cr._rotations("cubic")) == 24 and len(cr._rotations("hexagonal")) == 12
    assert cr.lattice_of("octahedral") == "cubic" and cr.lattice_of("twin1012") == "hexagonal"


@pytest.mark.parametrize("ratio", [1.587, RATIO_MG, RATIO_ZN])
def test_twin_systems(ratio):
    """The twins of the cubic and hexagonal lattices: their numbers, their shears (the
    formulas of the modes) and the sense of their shear."""
    g = ratio
    shears = {
        "twin111": 1 / math.sqrt(2),
        "twin112": 1 / math.sqrt(2),
        "twin1012": abs(g * g - 3) / (math.sqrt(3) * g),
        "twin1011": (4 * g * g - 9) / (4 * math.sqrt(3) * g),
        "twin1122": 2 * (g * g - 2) / (3 * g),
        "twin1121": 1 / g,
    }
    assert set(shears) == set(TWINS)
    for family, expected in shears.items():
        lattice = TWINS[family][0]
        n, d, shear = twin_systems(family, ratio if lattice == "hexagonal" else None)
        assert len(n) == (12 if lattice == "cubic" else 6)
        assert shear == pytest.approx(expected, rel=1e-12)
        assert np.allclose((n * d).sum(axis=1), 0, atol=1e-12)
        # a twin and its reverse are not the same system: no pair of opposite shears
        keys = {cr._key(a, b) for a, b in zip(n, d, strict=True)}
        assert len(keys) == len(n) and not any(
            cr._key(a, -b) in keys for a, b in zip(n, d, strict=True)
        )
        # the shear leaves the second undistorted plane as it is: eta_2 keeps its length
        r = ratio if lattice == "hexagonal" else None
        normal = cr._vector(TWINS[family][1], lattice, r, True)
        normal /= np.linalg.norm(normal)
        eta1 = cr._vector(TWINS[family][2], lattice, r, False)
        eta1 /= np.linalg.norm(eta1)
        eta2 = cr._vector(TWINS[family][3], lattice, r, False)
        signed = -2.0 * (eta2 @ eta1) / (eta2 @ normal)
        sheared = eta2 + signed * (eta2 @ normal) * eta1
        assert np.linalg.norm(sheared) == pytest.approx(np.linalg.norm(eta2), rel=1e-12)
        assert abs(signed) == pytest.approx(shear, rel=1e-12)
    # {10-12}: the twin lengthens the axis c below c / a = sqrt 3, shortens it above
    n, d, _ = twin_systems("twin1012", ratio)
    assert np.all(np.sign(n[:, 2] * d[:, 2]) == (1 if ratio < math.sqrt(3) else -1))
    # {11-21} lengthens it and {11-22} shortens it, for these ratios
    n, d, _ = twin_systems("twin1121", ratio)
    assert np.all(n[:, 2] * d[:, 2] > 0)
    n, d, _ = twin_systems("twin1122", ratio)
    assert np.all(n[:, 2] * d[:, 2] < 0)
    with pytest.raises(ValueError, match="twin family"):
        twin_systems("twin")
    with pytest.raises(ValueError, match="no shear"):
        twin_systems("twin1012", math.sqrt(3.0))


def test_cleavage_systems():
    counts = {"cube": (3, 2), "dodecahedral": (6, 2), "octahedral": (4, 3), "basal": (1, 3),
              "prismatic": (3, 2)}  # fmt: skip
    assert set(counts) == set(CLEAVAGE)
    for planes, (n_planes, n_shear) in counts.items():
        ratio = RATIO_ZN if CLEAVAGE[planes][0] == "hexagonal" else None
        n, d, kinds, groups = cleavage_systems(planes, ratio)
        assert len(n) == n_planes * (1 + n_shear)
        assert groups.tolist() == np.repeat(np.arange(n_planes), 1 + n_shear).tolist()
        opening = kinds == cr.OPENING
        assert opening.sum() == n_planes and np.all(kinds[~opening] == cr.ACCOMMODATION)
        assert np.allclose(d[opening], n[opening])  # the opening is along the normal
        assert np.allclose((n[~opening] * d[~opening]).sum(axis=1), 0, atol=1e-12)
        # the planes are distinct, and the directions of each one span it
        normals = n[opening]
        assert np.all(np.abs(normals @ normals.T)[~np.eye(n_planes, dtype=bool)] < 1 - 1e-9)
        for plane in range(n_planes):
            assert np.linalg.matrix_rank(d[(groups == plane) & ~opening]) == 2
    with pytest.raises(ValueError, match="cleavage planes"):
        cleavage_systems("pyramidal")


def test_configuration():
    spec = {
        "c_over_a": RATIO_ZN,
        "K": 1.0,
        "n": 5.0,
        "families": {"basal": {"tau0": 1.5}, "pyramidal2_ca": {"tau0": 15.0}},
        "twins": {"twin1012": {"tau0": 10.0, "Q": 5.0, "b": 2.0}},
        "twin_saturation": 0.8,
        "cleavage": {"planes": "basal", "strength": 80.0, "energy": 0.01, "length": 0.05},
    }
    cp = CrystalPlasticity.from_config(spec)
    assert (cp.n_slip, cp.n_twins, cp.n_cleavage, cp.n_systems) == (9, 6, 4, 19)
    assert cp.extended and not cp.reorients
    assert cp.field_names == ("twin_fraction", "cleavage_opening", "cleavage_damage")
    assert set(cp.field_names) < set(cr.FIELDS)
    assert cp.cleavage == (("basal", (80.0, 400.0, 1.0, 1.0, 5.0)),)  # 80 x 0.05 / 0.01
    assert CrystalPlasticity.from_config(cp.to_config()) == cp
    par, kinds = cp.params_array(), cp.kinds_array()
    assert par.shape == (19, 7) and kinds.shape == (19, cr.CP_WIDTH - cr.CP_KIND)
    assert kinds[:, 0].tolist() == [0] * 9 + [1] * 6 + [2, 3, 3, 3]
    shear = twin_systems("twin1012", RATIO_ZN)[2]
    assert np.allclose(kinds[9:15, 2], 1.0 / (0.8 * shear)) and np.all(kinds[:9, 2] == 0)
    assert np.all(kinds[15:, 1] == 400.0) and np.all(kinds[15:, 3] == 0) and kinds[0, 3] == -1
    assert np.allclose(par[15:], [80.0, 0, 0, 0, 0, 1.0, 5.0])
    # the slips and the twins harden each other, not the cleavage planes
    h = cp.interaction_matrix()
    assert (
        h.shape == (19, 19)
        and np.all(h[:15, :15] == 1)
        and not h[15:].any()
        and not h[:, 15:].any()
    )
    # the fields of points from their cumulated slips
    slips = np.zeros((2, 19))
    slips[0, 9:15] = 0.5 * 0.8 * shear / 6  # half of the saturation
    slips[1, 15:] = [0.001, 0.0005, 0.0, 0.0]
    fields = cp.point_fields(slips)
    assert np.allclose(fields["twin_fraction"], [0.4, 0.0])
    assert np.allclose(fields["cleavage_opening"], [0.0, 0.001])
    assert np.allclose(fields["cleavage_damage"], [0.0, 1 - math.exp(-400 * 0.0015)])
    # a material keeps them, and writes them again
    zinc = hexagonal(ZINC, **spec)
    assert Material.from_config(zinc.to_config()) == zinc
    # a classical crystal is as it was: no twins, no cleavage planes
    plain = CrystalPlasticity.from_config({"tau0": 10, "K": 1, "n": 5})
    assert not plain.extended and plain.n_systems == 12 and plain.field_names == ()
    assert set(plain.to_config()) == {"families"}
    # twins or cleavage planes alone: no slip family by default; several families of planes
    brittle = CrystalPlasticity.from_config(
        {
            "K": 1,
            "n": 1,
            "cleavage": {
                "planes": {"cube": {"strength": 5}, "dodecahedral": {}},
                "strength": 9,
                "softening": 2,
                "shear": 3,
            },
        }
    )
    assert brittle.families == () and brittle.n_systems == 9 + 18
    assert brittle.cleavage[0][1][:3] == (5.0, 2.0, 3.0) and brittle.cleavage[1][1][0] == 9.0
    assert brittle.params_array()[1, 0] == 15.0  # an accommodation system: shear x strength
    assert brittle.kinds_array()[:, 3].max() == 8  # nine planes
    assert (
        CrystalPlasticity.from_config({"tau0": 1, "K": 1, "n": 5, "twins": "twin111"}).n_twins == 12
    )
    for bad, match in [
        ({**spec, "families": {"octahedral": {"tau0": 1.0}}}, "one lattice"),
        ({**spec, "c_over_a": None}, "c_over_a"),
        ({"tau0": 1, "K": 1, "n": 5, "c_over_a": 1.6}, "hexagonal lattice"),
        ({**spec, "twins": ["twin"]}, "twin family"),
        ({**spec, "twin_saturation": 1.5}, "twin_saturation"),
        ({**spec, "twin_saturation": "all"}, "numbers"),
        ({**spec, "cleavage": {"strength": 1.0, "softening": 1.0}}, "its planes"),
        (
            {**spec, "cleavage": {"planes": "cube", "strength": 1.0, "softening": 1.0}},
            "one lattice",
        ),
        ({**spec, "cleavage": {"planes": "basal", "softening": 1.0}}, "strength"),
        ({**spec, "cleavage": {"planes": "basal", "strength": 1.0}}, "softening, or"),
        (
            {**spec, "cleavage": {"planes": "basal", "strength": 1.0, "energy": 1.0}},
            "softening, or",
        ),
        (
            {
                **spec,
                "cleavage": {
                    "planes": "basal",
                    "strength": 1.0,
                    "softening": 1.0,
                    "energy": 1.0,
                    "length": 1.0,
                },
            },
            "softening, or",
        ),
        (
            {
                **spec,
                "cleavage": {
                    "planes": "basal",
                    "strength": 1.0,
                    "softening": 1.0,
                    "toughness": 1.0,
                },
            },
            "unknown parameter",
        ),
        (
            {**spec, "cleavage": {"planes": "basal", "strength": -1.0, "softening": 1.0}},
            "strength > 0",
        ),
        ({**spec, "interaction": np.ones((9, 9)).tolist()}, "15 x 15"),
        ({**spec, "interaction": [1.0, 1.0, 0.6, 12.3, 1.6, 1.3]}, "octahedral systems only"),
        ({"families": [], "tau0": 1, "K": 1, "n": 5}, "at least one"),
    ]:
        with pytest.raises(ValueError, match=match):
            CrystalPlasticity.from_config(bad)
    with pytest.raises(ValueError, match="give \\['K', 'n'\\]"):
        CrystalPlasticity.from_config(
            {"cleavage": {"planes": "cube", "strength": 1, "softening": 1}}
        )
    # planes that soften faster than the crystal unloads are refused: a snap-back of the point
    with pytest.raises(ValueError, match="softens faster"):
        cubic(K=1, n=1, cleavage={"planes": "cube", "strength": 500.0, "softening": 300.0})
    assert cubic(K=1, n=1, cleavage={"planes": "cube", "strength": 500.0, "softening": 200.0})


# -- twins at a material point --------------------------------------------------------------------


@pytest.mark.parametrize(
    "constants, ratio, sign",
    [(MAGNESIUM, RATIO_MG, 1), (ZINC, RATIO_ZN, -1)],
)
def test_twinning_along_the_axis_c(constants, ratio, sign):
    """A crystal strained along its axis c with {10-12} twins only: they shear when the
    axis lengthens in magnesium (c / a < sqrt 3) and when it shortens in zinc, at the
    stress of Schmid's law; the other way the crystal is elastic. The crystal twins until
    its saturation, and is then elastic again."""
    tau0, k, n, saturation = 30.0, 2.0, 4.0, 0.9
    twins = {"twins": {"twin1012": {"tau0": tau0, "K": k, "n": n}}, "twin_saturation": saturation}
    m = hexagonal(constants, c_over_a=ratio, **twins)
    normals, directions, shear = twin_systems("twin1012", ratio)
    schmid = sign * normals[0, 2] * directions[0, 2]
    assert np.allclose(sign * normals[:, 2] * directions[:, 2], schmid) and 0.49 < schmid < 0.5
    young = modulus(m)
    steps, total = 80, 0.1
    # the other way: no twin, the elastic crystal
    res = simulate_point(m, {"times": [0, 1], "increments": 8, "strain": {"33": -sign * total}})
    assert np.allclose(res.stress[:, 2], young * res.strain[:, 2], rtol=1e-9)
    assert res.series["twin_fraction"].max() == 0.0 and res.plastic.max() == 0.0
    # the way of the twins
    res = simulate_point(m, {"times": [0, 1], "increments": steps, "strain": {"33": sign * total}})
    stress, strain = sign * res.stress[:, 2], sign * res.strain[:, 2]
    fraction = res.series["twin_fraction"]
    assert np.all(np.diff(fraction) >= 0) and fraction[-1] > fraction[1] >= 0.0
    # the strain of the twins: six systems of one Schmid factor, sheared by the fraction
    assert np.allclose(strain - stress / young, schmid * shear * fraction, rtol=0, atol=1e-9)
    # the flow rule at the end of each step, with the barrier of the saturation
    dt = 1.0 / steps
    rate = shear * np.diff(fraction) / 6 / dt  # of each system
    left = 1.0 - fraction[1:] / saturation
    flowing = rate > 1e-9
    resistance = tau0 * (1.0 + 1e-3 * (1.0 / left - 1.0)) + k * rate ** (1.0 / n)
    assert flowing.sum() > 40
    assert np.allclose(schmid * stress[1:][flowing], resistance[flowing], rtol=1e-6)
    # first twins: at the critical resolved shear stress over the Schmid factor
    first = np.flatnonzero(flowing)[0] + 1
    assert tau0 / schmid < stress[first] < 1.2 * tau0 / schmid
    # saturated: nearly all the crystal that may twin has, and the response is elastic again
    assert 0.999 * saturation < fraction[-1] < saturation
    assert strain[-1] > schmid * shear * saturation  # strained beyond what the twins give
    slope = (stress[-1] - stress[-2]) / (strain[-1] - strain[-2])
    assert slope == pytest.approx(young, rel=2e-3)
    # the cumulated plastic strain is that of the twins: uniaxial, sqrt(2/3 e:e) of a
    # shear of the six systems
    assert res.plastic[-1] > 0.9 * schmid * shear * saturation


def test_twinning_of_a_cubic_crystal():
    """{111}<11-2> twins of a face-centred cubic crystal along [001]: four systems of
    Schmid factor sqrt 2 / 3 in compression, eight of half of it in tension, so that it
    twins at half the stress in compression."""
    tau0 = 50.0
    m = cubic(twins={"twin111": {"tau0": tau0, "K": 0.5, "n": 10.0}})
    normals, directions, _ = twin_systems("twin111")
    schmid = normals[:, 2] * directions[:, 2]
    assert np.sum(np.isclose(schmid, -math.sqrt(2) / 3)) == 4
    assert np.sum(np.isclose(schmid, math.sqrt(2) / 6)) == 8
    flow = {}
    for sign in (1, -1):
        res = simulate_point(m, {"times": [0, 1], "increments": 20, "strain": {"33": sign * 0.01}})
        flow[sign] = abs(res.stress[-1, 2])
        # the plastic strain over the Schmid factor and the shear of the twin
        schmid_factor = math.sqrt(2) / (6 if sign > 0 else 3)
        plastic = 0.01 - flow[sign] / modulus(m)
        assert res.series["twin_fraction"][-1] == pytest.approx(
            plastic / (schmid_factor * math.sqrt(0.5)), rel=1e-6
        )
    assert flow[1] == pytest.approx(tau0 / (math.sqrt(2) / 6), rel=0.02)
    assert flow[-1] == pytest.approx(tau0 / (math.sqrt(2) / 3), rel=0.02)
    assert flow[1] / flow[-1] == pytest.approx(2.0, rel=0.01)


def test_basal_slip_and_twins():
    """A hexagonal crystal pulled at 45 degrees of its axis c, in the plane normal to a_1:
    two basal systems of Schmid factor sqrt 3 / 4 slip (the third is not loaded), at the
    stress of Schmid's law; the twins, harder, do not form."""
    tau0, k, n = 5.0, 1.0, 8.0
    m = hexagonal(
        MAGNESIUM,
        c_over_a=RATIO_MG,
        K=k,
        n=n,
        families={"basal": {"tau0": tau0}},
        twins={"twin1012": {"tau0": 200.0}},
    )
    euler = (0.0, 45.0, 0.0)
    axis = euler_to_matrix(euler)[0][:, 2]  # the tensile axis in the crystal
    normals, directions = slip_systems("basal", RATIO_MG)
    schmid = np.abs((normals @ axis) * (directions @ axis))
    assert np.allclose(np.sort(schmid), [0.0, math.sqrt(3) / 4, math.sqrt(3) / 4])
    rate = 1e-3
    path = {"times": [0, 20], "increments": 40, "strain": {"33": 20 * rate}}
    res = simulate_point(m, path, orientation=euler)
    slip_rate = rate / (2 * math.sqrt(3) / 4)
    expected = (tau0 + k * slip_rate ** (1 / n)) / (math.sqrt(3) / 4)
    assert res.stress[-1, 2] == pytest.approx(expected, rel=1e-6)
    assert res.series["twin_fraction"][-1] == 0.0 and res.plastic[-1] > 0.015


# -- cleavage at a material point -----------------------------------------------------------------


def test_cleavage_normal_to_the_tension():
    """A cubic crystal pulled along [001] with {100} cleavage planes: elastic until the
    normal stress of (001) is its strength, then the opening of the plane softens it,
    ``sigma = Y_0 exp(-w delta) + K d delta / dt``; the work that breaks it is ``Y_0 /
    w``; unloaded, it keeps its opening and comes back to its softened strength."""
    strength, softening, k = 500.0, 100.0, 0.5
    m = cubic(cleavage={"planes": "cube", "strength": strength, "softening": softening,
                        "K": k, "n": 1.0})  # fmt: skip
    young = modulus(m)
    steps, total = 200, 0.06
    res = simulate_point(m, {"times": [0, 1], "increments": steps, "strain": {"33": total}})
    stress, strain = res.stress[:, 2], res.strain[:, 2]
    opening, damage = res.series["cleavage_opening"], res.series["cleavage_damage"]
    assert np.allclose(strain - stress / young, opening, rtol=0, atol=1e-10)
    assert np.allclose(damage, -np.expm1(-softening * opening), rtol=0, atol=1e-12)
    onset = np.flatnonzero(opening > 0)[0]
    assert np.allclose(stress[:onset], young * strain[:onset], rtol=1e-9)
    assert strength * (1 - 2e-2) < stress.max() < strength + k * total * steps
    # the flow rule of the opening at the end of each step
    rate = np.diff(opening) * steps
    expected = strength * np.exp(-softening * opening[1:]) + k * rate
    assert np.allclose(stress[onset:], expected[onset - 1 :], rtol=1e-7, atol=1e-7 * strength)
    # no stress across the other planes, no plastic strain; the plane ends broken
    assert np.allclose(res.stress[:, [0, 1, 3, 4, 5]], 0, atol=1e-7) and res.plastic.max() == 0
    assert damage[-1] > 0.99 and stress[-1] < 0.01 * strength
    work = np.trapezoid(stress, strain)
    assert work == pytest.approx(strength / softening, rel=0.02)
    # unloading and loading again: elastic, up to the softened strength
    path = {"times": [0, 1, 2, 3], "increments": [60, 20, 60],
            "strain": {"33": [0.008, 0.006, 0.012]}}  # fmt: skip
    res = simulate_point(m, path)
    stress, strain = res.stress[:, 2], res.strain[:, 2]
    opening = res.series["cleavage_opening"]
    assert opening[60] > 0.003 and np.allclose(opening[60:81], opening[60])
    assert np.allclose(np.diff(stress[60:81]), young * np.diff(strain[60:81]), rtol=1e-8)
    again = np.flatnonzero(np.diff(opening[80:]) > 1e-12)[0] + 80
    softened = strength * math.exp(-softening * opening[60])
    assert stress[again] < softened < stress[again + 1] + k * 0.01 * 60
    # compressed, the plane does not open
    res = simulate_point(m, {"times": [0, 1], "increments": 5, "strain": {"33": -0.01}})
    assert res.series["cleavage_opening"].max() == 0.0
    assert res.stress[-1, 2] == pytest.approx(-0.01 * young, rel=1e-9)


@pytest.mark.parametrize("shear", [1.0, 4.0])
def test_cleavage_of_an_inclined_plane(shear):
    """A cleavage plane at 30 degrees of the tensile axis: it opens when its normal
    stress, ``sigma cos^2``, is its strength, and its accommodation systems then release
    its shear stress: the crystal ends without stress."""
    strength, angle = 400.0, 30.0
    m = cubic(cleavage={"planes": "cube", "strength": strength, "softening": 60.0, "K": 0.2,
                        "n": 1.0, "shear": shear})  # fmt: skip
    cos2 = math.cos(math.radians(angle)) ** 2
    path = {"times": [0, 1], "increments": 300, "strain": {"33": 0.15}}
    res = simulate_point(m, path, orientation=(0.0, angle, 0.0))
    stress = res.stress[:, 2]
    opening = res.series["cleavage_opening"]
    onset = np.flatnonzero(opening > 0)[0]
    # (the stress at the end of the step in which it opens is already below its peak)
    assert stress[onset - 1] < strength / cos2
    assert 0.95 * strength / cos2 < stress.max() < 1.01 * strength / cos2
    # the shear of the plane, 0.58 of its normal stress, is below its accommodation
    # strength when it opens, and is released as the plane softens
    assert stress[-1] < 0.02 * stress.max() and res.series["cleavage_damage"][-1] > 0.98
    assert np.allclose(res.stress[:, [0, 1, 3, 4, 5]], 0, atol=1e-6)


def test_cleavage_and_slip():
    """Slip shields cleavage: a crystal that flows below the strength of its cleavage
    planes does not cleave; hardened up to it, it does."""
    slip = {"families": ["octahedral"], "tau0": 100.0, "K": 2.0, "n": 10.0}
    planes = {"planes": "cube", "strength": 600.0, "softening": 60.0, "K": 0.5, "n": 1.0}
    schmid = 1.0 / math.sqrt(6.0)
    path = {"times": [0, 1], "increments": 60, "strain": {"33": 0.03}}
    res = simulate_point(cubic(**slip, cleavage=planes), path)
    assert res.stress[-1, 2] < 1.1 * 100.0 / schmid < 600.0
    assert res.series["cleavage_opening"].max() == 0.0 and res.plastic[-1] > 0.02
    # with an isotropic hardening of 400 MPa on the systems, the flow stress passes 600 MPa
    hardened = cubic(**slip, Q=400.0, b=30.0, cleavage=planes)
    res = simulate_point(hardened, path)
    stress, opening = res.stress[:, 2], res.series["cleavage_opening"]
    onset = np.flatnonzero(opening > 0)[0]
    assert stress[onset - 1] < 600.0 and 0.95 * 600.0 < stress.max() < 600.0 * 1.02
    assert stress[-1] < 0.5 * stress.max()
    # p counts the slips, not the opening of the planes
    slipped = res.plastic[-1]
    assert 0 < slipped < 0.03 - opening[-1]
    assert np.allclose(np.diff(res.plastic)[onset + 5 :], 0, atol=1e-6)  # unloaded by the plane


# -- the kernels ----------------------------------------------------------------------------------


def _kernel(material, euler, finite):
    mats = {0: material}
    table = material_table([0], mats, "elastic", grains=[0], orientations=np.array([euler]))
    return table, behaviour_table([0], mats, table, finite=finite)


def _point(table, beh, grad, old, dt, finite, guess=0, new=None):
    nc = 9 if finite else 6
    new = np.zeros_like(old) if new is None else new.copy()
    s, a = np.zeros((1, nc)), np.zeros((1, nc, nc))
    update = update_points_finite if finite else update_points
    failed = update(beh.kind, table.mat, beh.pmat, beh.data, grad[None], 0.0, dt, old, new, s, a,
                    guess, 1.0, False)  # fmt: skip
    assert failed == 0
    return s[0], a[0], new


@pytest.mark.parametrize("finite", [False, True])
def test_consistent_tangent(finite):
    """The tangent of the implicit update against finite differences, for a crystal that
    slips, twins near its saturation and cleaves; the same state from a warm start."""
    m = hexagonal(
        ZINC,
        c_over_a=RATIO_ZN,
        K=20.0,
        n=4.0,
        families={"basal": {"tau0": 10.0, "Q": 8.0, "b": 20.0}, "pyramidal2_ca": {"tau0": 60.0}},
        twins={"twin1012": {"tau0": 30.0}},
        cleavage={
            "planes": "basal",
            "strength": 90.0,
            "softening": 150.0,
            "shear": 0.15,
            "K": 30.0,
            "n": 2.0,
        },
    )
    cp = m.crystal_plasticity
    assert not behaviour_table([0], {0: m}, material_table([0], {0: m}, "elastic")).symmetric
    euler = (20.0, 35.0, 50.0)
    table, beh = _kernel(m, euler, finite)
    assert beh.data[5].shape == (1, cp.n_systems, cr.CP_WIDTH)
    ns = cp.n_systems
    old = np.zeros((1, beh.width))
    v = old[0, cr.CP_ALPHA + ns : cr.CP_ALPHA + 2 * ns]
    shear = twin_systems("twin1012", RATIO_ZN)[2]
    v[9:15] = 0.985 * shear / 6  # the crystal has nearly twinned as far as it may
    v[15] = 0.002  # its basal plane has opened
    # in the axes of the crystal: a tension in the basal plane (the twins of zinc), a
    # smaller one along c (the basal plane opens), shears (basal slip, accommodation)
    g = euler_to_matrix(euler)[0]
    crystal = np.array([[4e-3, 1e-3, 1.5e-3], [1e-3, -1e-3, 1e-3], [1.5e-3, 1e-3, 1.5e-3]])
    sample = g.T @ crystal @ g
    if finite:
        grad = sample.ravel().copy()
        grad[[1, 3]] += [3e-4, -3e-4]  # with a small rotation
    else:
        grad = np.array([sample[0, 0], sample[1, 1], sample[2, 2], 2 * sample[1, 2],
                         2 * sample[2, 0], 2 * sample[0, 1]])  # fmt: skip
    s, a, new = _point(table, beh, grad, old, 0.5, finite)
    moved = new[0, cr.CP_ALPHA + ns : cr.CP_ALPHA + 2 * ns] - v
    # slips, twins, the opening and the accommodation of the plane are all active
    assert moved[:9].max() > 1e-5 and moved[9:15].max() > 1e-7
    assert moved[15] > 1e-5 and moved[16:].max() > 1e-6
    h = 1e-8
    fd = np.zeros_like(a)
    for j in range(grad.size):
        d = np.zeros(grad.size)
        d[j] = h
        up = _point(table, beh, grad + d, old, 0.5, finite)[0]
        down = _point(table, beh, grad - d, old, 0.5, finite)[0]
        fd[:, j] = (up - down) / (2 * h)
    assert np.allclose(a, fd, rtol=0, atol=2e-6 * np.abs(fd).max())
    assert np.abs(a - a.T).max() > 1e-3 * np.abs(a).max()  # not symmetric
    s1, _, new1 = _point(table, beh, grad, old, 0.5, finite, guess=1, new=new)
    assert np.allclose(new1, new, rtol=0, atol=1e-11) and np.allclose(s1, s, atol=1e-7)


# -- cells ----------------------------------------------------------------------------------------


def test_homogeneous_cells():
    """A homogeneous crystal that twins and cleaves follows its material point, on voxels
    and on a mesh, in small and finite strains; its fields are those of the point."""
    m = hexagonal(
        ZINC,
        c_over_a=RATIO_ZN,
        K=10.0,
        n=3.0,
        families={"basal": {"tau0": 5.0, "Q": 10.0, "b": 50.0}},
        twins={"twin1012": {"tau0": 25.0, "Q": 30.0, "b": 30.0}},
        cleavage={"planes": "basal", "strength": 80.0, "softening": 80.0, "K": 5.0, "n": 1.0},
    )
    # the axis c near the axis 3: compressed, the crystal slips on its basal plane, hardens
    # and twins; pulled, its basal plane opens before it slips again
    euler = np.array([10.0, 8.0, 40.0])
    path = {"times": [0, 1, 2], "increments": [20, 20], "strain": {"33": [-0.03, 0.02]}}
    image = VoxelImage(np.zeros((2, 2, 2), dtype=np.uint8))
    grains = np.zeros(image.shape, int)
    for kinematics in ("small", "finite"):
        ref = simulate_point(m, path, orientation=euler, kinematics=kinematics)
        assert ref.info["cuts"] == 0 and ref.plastic[-1] > 0.02
        assert ref.series["twin_fraction"][-1] > 0.3 and ref.series["cleavage_damage"][-1] > 0.9
        scale = np.abs(ref.stress).max()
        fft = NonlinearFFTProblem(image, {0: m}, grains=grains, orientations=euler[None],
                                  kinematics=kinematics)  # fmt: skip
        mesh = box_mesh((1, 1, 1), order=1, perturb=0.2)
        mesh = dataclasses.replace(
            mesh, cell=np.zeros(mesh.n_elements, int), orientations=euler[None]
        )
        fe = NonlinearFEProblem(mesh, {0: m}, "periodic", kinematics=kinematics)
        for problem in (fft, fe):
            res = problem.run(path, fields=True)
            assert res.info["cuts"] == 0  # the steps of the point: the same viscous stresses
            assert np.allclose(res.stress, ref.stress, rtol=0, atol=5e-6 * scale)
            assert res.plastic[-1] == pytest.approx(ref.plastic[-1], rel=1e-5)
            names = m.crystal_plasticity.field_names
            assert set(res.series) == set(names)
            fields = res.fields[2.0]
            for name in names:
                assert np.allclose(res.series[name], ref.series[name], rtol=0, atol=1e-7)
                assert np.allclose(fields[name], ref.series[name][-1], rtol=0, atol=1e-7)
            assert set(res.components()) >= set(names)


def test_crack_band():
    """A bar of one crystal with a weaker layer of voxels, pulled along [001]: the layer
    cleaves alone, and the work that breaks the bar is the fracture energy of its planes
    (the crack band: ``energy`` and ``length``, the size of the voxels), whatever their
    number."""
    energy, strength = 2.0, 300.0
    young = modulus(cubic(K=1, n=1, cleavage={"planes": "cube", "strength": 1.0, "softening": 1.0}))
    for layers in (8, 12):
        h = 1.0 / layers
        planes = {"planes": "cube", "energy": energy, "length": h, "K": 0.2, "n": 1.0}
        materials = {
            0: cubic(cleavage={**planes, "strength": 2 * strength}),
            1: cubic(cleavage={**planes, "strength": strength}),
        }
        assert materials[1].crystal_plasticity.cleavage[0][1][1] == pytest.approx(
            strength * h / energy
        )
        data = np.zeros((1, 1, layers), dtype=np.uint8)
        data[:, :, layers // 2] = 1
        image = VoxelImage(data, spacing=(h, h, h))
        problem = NonlinearFFTProblem(image, materials, grains=np.zeros(data.shape, int),
                                      orientations=np.zeros((1, 3)))  # fmt: skip
        # no snap-back of the bar: its length is below E G_c / Y^2
        assert strength**2 / (young * energy) < 1.0
        res = problem.run({"times": [0, 1], "increments": 80, "strain": {"33": 0.04}}, fields=True)
        stress, strain = res.stress[:, 2], res.strain[:, 2]
        # (the peak falls between two steps)
        assert 0.94 * strength < stress.max() < 1.01 * strength and stress[-1] < 0.02 * strength
        damage = res.fields[1.0]["cleavage_damage"][0, 0]
        assert damage[layers // 2] > 0.98 and np.delete(damage, layers // 2).max() == 0.0
        # the opening of the layer is the elongation of the bar that its stress does not explain
        opening = res.fields[1.0]["cleavage_opening"][0, 0, layers // 2]
        assert h * opening == pytest.approx(strain[-1] - stress[-1] / young, rel=1e-6)
        work = np.trapezoid(stress, strain)  # per unit area: the bar is of length 1
        assert work == pytest.approx(energy, rel=0.03)


def textured_magnesium(shape=8, cells=8, seed=5):
    """A polycrystal of magnesium on voxels, the axes c of its grains near the axis 3."""
    from morphogen.domain import Domain
    from morphogen.models.base import model_from_config

    texture = {"type": "von_mises_fisher", "kappa": 30.0, "axis": [0, 0, 1]}
    model = model_from_config(
        {"type": "voronoi", "n_cells": cells, "crystal_orientations": True, "texture": texture}
    )
    micro = model.generate(Domain(size=1.0, periodic=True), seed)
    image = micro.voxelize(shape=shape)
    magnesium = hexagonal(
        MAGNESIUM,
        c_over_a=RATIO_MG,
        K=5.0,
        n=10.0,
        families={
            "basal": {"tau0": 10.0},
            "prismatic": {"tau0": 60.0},
            "pyramidal2_ca": {"tau0": 100.0},
        },
        twins={"twin1012": {"tau0": 25.0, "Q": 15.0, "b": 20.0}},
    )
    return image, micro.labels(grid=image.grid()), micro.orientations, magnesium


def test_fibre_texture():
    """The axes c of the crystals of a fibre texture follow its law (von Mises-Fisher: the
    mean of 1 - cos of their angle to the axis is 1 / kappa), the crystals turned at random
    about them; without it, the orientations are as they were."""
    from morphogen.domain import Domain
    from morphogen.models.base import model_from_config
    from morphogen.orientations import VonMisesFisher, fibre_orientations

    rng = np.random.default_rng(1)
    axis = np.array([1.0, 2.0, 2.0]) / 3.0
    euler = fibre_orientations(rng, 4000, VonMisesFisher(40.0, tuple(axis)))
    g = euler_to_matrix(euler)
    assert np.allclose(np.linalg.det(g), 1.0) and np.allclose(g @ np.swapaxes(g, 1, 2), np.eye(3))
    c = g[:, 2, :]  # the axis c of each crystal, in the axes of the sample
    assert (1.0 - c @ axis).mean() == pytest.approx(1.0 / 40.0, rel=0.05)
    # the first axis of the crystals is turned uniformly about it: no mean direction
    assert np.linalg.norm(g[:, 0, :].mean(axis=0)) < 0.05
    spec = {"type": "voronoi", "n_cells": 30, "crystal_orientations": True}
    texture = {"type": "von_mises_fisher", "kappa": 50.0, "axis": [0.0, 0.0, 1.0]}
    model = model_from_config({**spec, "texture": texture})
    assert model.to_config()["texture"] == texture
    micro = model.generate(Domain(size=1.0, periodic=True), 3)
    assert np.all(euler_to_matrix(micro.orientations)[:, 2, 2] > 0.85)
    plain = model_from_config(spec)
    assert "texture" not in plain.to_config()
    uniform = plain.generate(Domain(size=1.0, periodic=True), 3).orientations
    assert np.abs(euler_to_matrix(uniform)[:, 2, 2]).min() < 0.3


def test_textured_magnesium():
    """A polycrystal of magnesium whose axes c are near the axis 3: pulled along it, it
    twins and flows at a low stress; compressed, it does not twin and is twice as hard
    (the asymmetry of textured magnesium)."""
    image, grains, orientations, magnesium = textured_magnesium()
    flow, twinned = {}, {}
    for sign in (1, -1):
        problem = NonlinearFFTProblem(image, {1: magnesium}, grains=grains,
                                      orientations=orientations)  # fmt: skip
        path = {"times": [0, 1], "increments": 10, "strain": {"33": sign * 0.02}}
        res = problem.run(path, fields=True)
        flow[sign] = abs(res.stress[-1, 2])
        twinned[sign] = res.series["twin_fraction"][-1]
        field = res.fields[1.0]["twin_fraction"]
        assert field.shape == image.shape
        assert np.nanmean(field) == pytest.approx(twinned[sign], rel=1e-6, abs=1e-12)
    # the twins give 6.4 % of strain at most along c, over a Schmid factor of 1/2
    shear = twin_systems("twin1012", RATIO_MG)[2]
    assert 0.5 * 0.02 / (0.5 * shear) < twinned[1] < 0.02 / (0.5 * shear)
    assert twinned[-1] < 0.01
    assert 25.0 / 0.5 < flow[1] < 0.6 * flow[-1]


@pytest.mark.slow
def test_transgranular_cleavage():
    """A polycrystal of iron at a low temperature, its slip harder than its {100} planes:
    pulled, it breaks by cleavage through its grains, on a surface near one section of
    the cell, and the work that breaks it is a few times the fracture energy of the
    planes times this section (the planes of the grains are inclined: steps, bridges)."""
    from morphogen.domain import Domain
    from morphogen.models.base import model_from_config

    size, shape, energy, strength = 0.1, 10, 0.05, 500.0
    model = model_from_config({"type": "voronoi", "n_cells": 8, "crystal_orientations": True})
    micro = model.generate(Domain(size=size, periodic=True), 2)
    image = micro.voxelize(shape=shape)
    iron = cubic(
        K=10.0,
        n=10.0,
        families={"bcc110": {"tau0": 300.0, "Q": 100.0, "b": 20.0}},
        cleavage={
            "planes": "cube",
            "strength": strength,
            "energy": energy,
            "length": size / shape,
            "K": 50.0,
            "n": 1.0,
        },
    )
    problem = NonlinearFFTProblem(image, {1: iron}, grains=micro.labels(grid=image.grid()),
                                  orientations=micro.orientations)  # fmt: skip
    res = problem.run({"times": [0, 1], "increments": 40, "strain": {"33": 0.01}}, fields=True)
    stress = res.stress[:, 2]
    # the planes the best oriented open first: a peak near their strength (those of a
    # grain are at most at 55 degrees of the axis: cos^2 > 1/3)
    assert strength < stress.max() < 1.5 * strength and stress[-1] < 0.15 * stress.max()
    damage = res.fields[1.0]["cleavage_damage"]
    broken = int((damage > 0.9).sum())
    assert shape**2 <= broken < 2 * shape**2  # one section of the cell, with its steps
    assert res.plastic[-1] < 1e-3  # brittle
    work = np.trapezoid(stress, res.strain[:, 2]) * size
    assert energy < work < 5 * energy


@pytest.mark.slow
def test_cracks_between_the_grains_or_through_them():
    """The template of a polycrystal with cohesive boundaries and cleavage planes: with
    boundaries weaker than the planes the crack follows them and the grains stay whole,
    the work that breaks the cell being the fracture energy of the boundaries times the
    debonded area; with strong boundaries no boundary debonds, and the grains cleave."""
    import yaml

    from morphogen import cli
    from morphogen.config import load_config
    from morphogen.homogenization import homogenize
    from morphogen.mesh.conforming import has_mmg
    from morphogen.pipeline import _conforming_mesh
    from morphogen.seeding import derive_seed

    if not has_mmg():
        pytest.skip("MMG")
    data = yaml.safe_load(cli._templates()["homogenization_grain_fracture"])
    results = {}
    for name, strength in (("between", 300.0), ("through", 3000.0)):
        data["homogenization"]["nonlinear"]["interfaces"]["normal_strength"] = strength
        cfg = load_config(data)
        micro = cfg.model.generate(cfg.domain, derive_seed(cfg.seed, 0))
        mesh, _ = _conforming_mesh(micro, cfg)
        spec = cfg.homogenization
        out = homogenize(mesh, spec.materials, **spec.options(), fields=True)
        results[name] = out["tension.periodic"]
    between, through = results["between"], results["through"]
    # between the grains: the boundaries debond over more than a section of the cell
    area = between.series["debonded_area"][-1]
    assert 1.0 < area < 2.0 and between.series["cleavage_damage"].max() < 1e-3
    assert between.stress[-1, 2] < 1e-3 * between.stress[:, 2].max()
    work = np.trapezoid(between.stress[:, 2], between.strain[:, 2])
    assert work == pytest.approx(0.5 * area, rel=0.08)  # the fracture energy of the boundaries
    # through the grains: no boundary debonds, the cleavage planes of the grains open
    assert through.series["debonded_area"].max() == 0.0
    assert through.series["cleavage_damage"][-1] > 0.1
    assert through.stress[:, 2].max() > 1.5 * between.stress[:, 2].max()
    assert through.stress[-1, 2] < 0.2 * through.stress[:, 2].max()
    damage = through.fields[max(through.fields)]["cleavage_damage"]
    assert 0.1 < np.mean(damage > 0.9) < 0.35
