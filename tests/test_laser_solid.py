"""A treated sheet in its volume: the flow laws of the sheets as von Mises plasticity
(Voce exactly, the power laws fitted), the volume of a hardness map (classes, layers,
free faces), its tensile test by FFT: a homogeneous sheet against its flow law, layers
and stripes along the tension against the mixture of their laws (the membrane's
assumption), stripes across it between its bounds; the cell of a laser treatment."""

import numpy as np
import pytest
from scipy.optimize import brentq

from morphogen.laser import SheetCase, case_volume, sheet_material, sheet_volume, volume_tension
from morphogen.laser.materials import LAWS, FlowLaw
from morphogen.laser.solid import flow_plasticity, hardening_stress

STEEL = sheet_material("aisi430")


def uniaxial(plasticity, strains, young=STEEL.young):
    """The stress of a bar of a plasticity at total strains (monotonic tension)."""
    out = []
    for e in strains:
        if young * e <= plasticity.yield_stress:
            out.append(young * e)
        else:
            f = lambda s, e=e: s - hardening_stress(plasticity, e - s / young)  # noqa: E731
            out.append(brentq(f, 0.0, young * e))
    return np.array(out)


def test_flow_laws_as_von_mises_plasticity():
    """Voce's law is one term of the isotropic hardening; Ludwik's and Swift's power laws
    are fitted within 1.5 % up to a plastic strain of 0.3."""
    voce = flow_plasticity("voce", 600.0, 250.0, 18.0)
    assert voce.yield_stress == 600.0 and voce.isotropic == ((250.0, 18.0),)
    p = np.linspace(0.0, 0.3, 600)
    assert hardening_stress(voce, p) == pytest.approx(FlowLaw("voce", 600.0, 250.0, 18.0).stress(p))
    for kind, s0, a, b in (("ludwik", 290.0, 520.0, 0.45), ("swift", 400.0, 900.0, 0.15),
                           ("ludwik", 700.0, 300.0, 0.6)):  # fmt: skip
        law = FlowLaw(kind, s0, a, b)
        fitted = flow_plasticity(kind, s0, a, b)
        assert fitted.yield_stress == pytest.approx(float(law.stress(0.0)))
        assert hardening_stress(fitted, p) == pytest.approx(law.stress(p), rel=0.015)
        assert len(fitted.isotropic) <= 3 and fitted.hardening >= 0.0
        assert flow_plasticity(LAWS.index(kind), s0, a, b) == fitted  # by its index
    for hv in (160.0, 208.0, 320.0):  # the laws of the hardnesses of a steel
        kind, s0, a, b = STEEL.strength.flow(np.array([hv]), young=STEEL.young)
        law = FlowLaw(LAWS[int(kind[0])], float(s0[0]), float(a[0]), float(b[0]))
        fitted = flow_plasticity(int(kind[0]), s0[0], a[0], b[0])
        assert hardening_stress(fitted, p) == pytest.approx(law.stress(p), rel=0.015)
    with pytest.raises(ValueError, match="strain"):
        flow_plasticity("ludwik", 290.0, 520.0, 0.45, strain=0.0)


def test_volume_of_a_hardness_map():
    """Classes of hardness as phases, layers repeated to cubic voxels, a gap of void
    above the sheet."""
    hardness = np.full((8, 6, 3), 160.0)
    hardness[:, 3:, 0] = 400.0  # a hard stripe in the first layer
    hardness[:, 3:, 1] = 270.0
    image, materials = sheet_volume(hardness, STEEL, (0.2, 0.2), 1.2, classes=4, gap=3)
    assert image.shape == (8, 6, 3 * 2 + 3) and image.periodic
    assert image.spacing == pytest.approx((0.2, 0.2, 0.2))  # two voxels through 0.4 mm
    assert sorted(materials) == [0, 1, 2, 3] and materials[0].void
    assert image.metadata["hardness"] == pytest.approx({1: 160.0, 2: 270.0, 3: 400.0})
    assert image.metadata["solid_voxels"] == 6
    data = np.asarray(image.data)
    assert np.all(data[:, :, 6:] == 0) and np.all(data[:, :3, :6] == 1)
    assert np.all(data[:, 3:, 0:2] == 3) and np.all(data[:, 3:, 2:4] == 2)
    strengths = [materials[k].plasticity.yield_stress for k in (1, 2, 3)]
    assert strengths[0] < strengths[1] < strengths[2]
    assert all(m.young == STEEL.young and m.poisson == STEEL.poisson for m in materials.values()
               if not m.void)  # fmt: skip
    uniform, one = sheet_volume(np.full((4, 4), 200.0), STEEL, (0.5, 0.5), 1.0, depth=1)
    assert uniform.shape == (4, 4, 3) and sorted(one) == [0, 1]
    for bad in ({"classes": 0}, {"classes": 255}, {"gap": 0}, {"depth": 0}):
        with pytest.raises(ValueError):
            sheet_volume(hardness, STEEL, (0.2, 0.2), 1.2, **bad)
    with pytest.raises(ValueError, match="hardness"):
        sheet_volume(-hardness, STEEL, (0.2, 0.2), 1.2)


def test_tension_of_a_homogeneous_sheet():
    """A sheet of one hardness in its volume: the flow law of that hardness in uniaxial
    tension, its width shrinking by half the strain, no stress across its free faces."""
    image, materials = sheet_volume(np.full((6, 6, 3), 300.0), STEEL, (0.25, 0.25), 0.75)
    test = volume_tension(image, materials, strain=0.04, increments=8)
    law = materials[1].plasticity
    assert test["stress"] == pytest.approx(uniaxial(law, test["strain"]), rel=1e-5, abs=1e-6)
    assert test["width"][-1] / test["strain"][-1] == pytest.approx(-0.5, abs=0.03)
    assert np.abs(test["result"].stress[:, 1:]).max() < 1e-3 * test["stress"][-1]
    other = volume_tension(image, materials, axis=1, strain=0.04, increments=8)
    assert other["stress"] == pytest.approx(test["stress"], rel=1e-8)
    # in finite strains, up to 20 %: the flow law between the Kirchhoff stress (the
    # nominal stress times the stretch) and the logarithmic strain
    large = volume_tension(image, materials, strain=0.2, increments=10, kinematics="finite")
    stretch = 1.0 + large["strain"]
    assert large["stress"] * stretch == pytest.approx(uniaxial(law, np.log(stretch)), rel=1e-5,
                                                      abs=1e-6)  # fmt: skip
    assert large["stress"][-1] < 0.85 * (large["stress"] * stretch)[-1]  # the section shrinks
    assert large["result"].log_strain[-1, 0] == pytest.approx(np.log(1.2), rel=1e-9)
    with pytest.raises(ValueError, match="axis"):
        volume_tension(image, materials, axis=2)


def test_layers_and_stripes_against_the_mixture_of_their_laws():
    """Hard and soft parts that share the strain of the tension, side by side or one
    above the other: the mean of their laws (what the membrane assumes of its layers; the
    stresses that their unequal contractions raise count at the second order: 2e-4);
    stripes across the tension: softer, between the soft part alone and that mixture,
    the strain gathered in the soft stripes."""
    stripes = np.full((8, 8, 3), 200.0)
    stripes[:, 4:, :] = 400.0  # along x, half the width
    layers = np.full((6, 6, 4), 200.0)
    layers[:, :, :2] = 400.0  # a hard skin over half the thickness
    curves = {}
    for name, hardness, axis in (("stripes", stripes, 0), ("layers", layers, 0),
                                 ("across", stripes, 1)):  # fmt: skip
        image, materials = sheet_volume(hardness, STEEL, (0.25, 0.25), 0.75, classes=2)
        assert image.metadata["hardness"] == pytest.approx({1: 200.0, 2: 400.0})
        curves[name] = volume_tension(image, materials, axis=axis, strain=0.04, increments=8)
        assert len(curves[name]["strain"]) == 9  # no step divided
    soft, hard = materials[1].plasticity, materials[2].plasticity
    strains = curves["stripes"]["strain"]
    mixture = 0.5 * (uniaxial(soft, strains) + uniaxial(hard, strains))
    for name in ("stripes", "layers"):
        assert curves[name]["stress"] == pytest.approx(mixture, rel=5e-4, abs=1e-6)
    across = curves["across"]["stress"]
    assert np.all(across[1:] > uniaxial(soft, strains)[1:]) and np.all(across[1:] < mixture[1:])
    assert across[-1] == pytest.approx(0.91 * mixture[-1], rel=0.02)
    result = curves["across"]["result"]
    assert abs(result.stress[-1, 0]) < 1e-3 * result.stress[-1, 1]  # no stress along x
    # the strains across the sheet are prescribed, its faces free: no stress across it
    assert np.abs(result.strain[:, [2, 3, 4]]).max() == 0.0
    assert np.abs(result.stress[:, [2, 3, 4]]).max() < 1e-3 * result.stress[-1, 1]


def test_the_cell_of_a_laser_treatment():
    """The periodic cell of a sheet treated by straight tracks, in its volume: hard
    tracks in a softer sheet, stronger along the tracks than across them, as its
    membrane."""
    material = sheet_material("annealed_ms1500")
    case = SheetCase(
        material=material,
        element=0.5,
        pattern={"kind": "straight", "angle": 0.0, "count": None, "spacing": 6.0},
        loading={"kind": "cell"},
    )
    image, materials = case_volume(case, resolution=0.5, classes=4)
    assert image.shape[:2] == (4, 12)  # 2 mm along the tracks, 6 mm across them
    hardness = image.metadata["hardness"]
    assert max(hardness.values()) > 1.5 * min(hardness.values())  # martensite in the tracks
    solid = np.asarray(image.data)[:, :, : image.metadata["solid_voxels"]]
    assert np.all(solid > 0) and np.all(solid == solid[:1])  # the same all along the tracks
    hard = np.mean(solid == max(hardness))
    assert 0.03 < hard < 0.5
    assert min(hardness.values()) == pytest.approx(167.0, abs=1.0)  # the annealed sheet
    along = volume_tension(image, materials, axis=0, strain=0.01, increments=4)
    across = volume_tension(image, materials, axis=1, strain=0.01, increments=4)
    assert along["stress"][-1] > 1.2 * across["stress"][-1]
    assert across["width"][-1] / across["strain"][-1] > -0.35  # the tracks hold the width
    # the membrane of the same cell (plane stress, four layers that share its strains)
    case.directions = (0.0, 90.0)
    case.loading["max_strain"] = 0.012
    membrane = case.evaluate().directions
    for direction, test, rel in ((0.0, along, 0.03), (90.0, across, 0.06)):
        tensile = membrane[direction].tensile
        stress = np.interp(0.01, tensile.strain, tensile.stress)
        assert test["stress"][-1] == pytest.approx(stress, rel=rel)
    # in finite strains the measures are those of the membrane (forces over the initial
    # section, elongations over the initial length): closer
    case.loading["max_strain"] = 0.065
    membrane = case.evaluate().directions
    for direction, axis, rel in ((0.0, 0, 0.015), (90.0, 1, 0.03)):
        tensile = membrane[direction].tensile
        test = volume_tension(image, materials, axis=axis, strain=0.06, increments=6,
                              kinematics="finite")  # fmt: skip
        stress = np.interp(0.06, tensile.strain, tensile.stress)
        assert test["stress"][-1] == pytest.approx(stress, rel=rel)
        small = volume_tension(image, materials, axis=axis, strain=0.06, increments=6)
        assert small["stress"][-1] > 1.03 * test["stress"][-1]
    oblique = SheetCase(material=material, element=0.5, loading={"kind": "cell"},
                        pattern={"kind": "honeycomb", "cell": 3.0})  # fmt: skip
    with pytest.raises(ValueError, match="not rectangular"):
        case_volume(oblique)
