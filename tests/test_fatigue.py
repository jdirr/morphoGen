"""Multiaxial high-cycle fatigue (:mod:`morphogen.homogenization.fatigue`).

The amplitude of a cycle against closed forms: half the range of a proportional cycle,
the larger of the two amplitudes of a tension and a torsion a quarter of a cycle apart,
the circle through the corners of an acute triangle and the circle on the longest side
of an obtuse one. The criteria of Crossland, Sines, Dang Van, Findley and Matake at the
two fatigue limits that calibrate them, under a mean stress and under a hydrostatic
cycle, in any frame. The fields of a cyclic loading path: a homogeneous cell at the
closed form of its criterion (finite elements and FFT), a stiff particle that
concentrates the stress, a criterion per phase, the factor of safety.
"""

import csv
import math

import numpy as np
import pytest

from morphogen import io, pipeline
from morphogen.config import ConfigError, load_config
from morphogen.homogenization import (
    Crossland,
    DangVan,
    Findley,
    Matake,
    Material,
    Sines,
    fatigue,
    homogenize,
    path_fatigue,
    stress_amplitude,
)
from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.homogenization.fatigue import (
    CRITERIA,
    FatigueResult,
    criterion_from_config,
    hydrostatic,
)
from morphogen.image import VoxelImage
from morphogen.mesh.conforming import has_mmg

BENDING, TORSION, REPEATED = 400.0, 260.0, 560.0
ANGLES = np.linspace(0.0, 2.0 * np.pi, 25)  # a cycle: its quarters are steps
VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))


def _cycle(component: int, amplitude: float, mean: float = 0.0, phase: float = 0.0):
    history = np.zeros((ANGLES.size, 6))
    history[:, component] = mean + amplitude * np.sin(ANGLES + phase)
    return history


def _tension(amplitude, mean=0.0, phase=0.0):
    return _cycle(0, amplitude, mean, phase)


def _torsion(amplitude, mean=0.0, phase=0.0):
    return _cycle(5, amplitude, mean, phase)


def _laws() -> dict:
    return {
        "crossland": Crossland.from_limits(BENDING, TORSION),
        "sines": Sines.from_limits(TORSION, REPEATED),
        "dang_van": DangVan.from_limits(BENDING, TORSION),
        "findley": Findley.from_limits(BENDING, TORSION),
        "matake": Matake.from_limits(BENDING, TORSION),
    }


def test_amplitude_of_a_cycle():
    # proportional cycles: half their range, around their mean
    amplitude, mean = stress_amplitude(_tension(300.0, 100.0))
    assert amplitude == pytest.approx(300.0 / math.sqrt(3.0), rel=1e-12)
    np.testing.assert_allclose(mean, [200.0 / 3, -100.0 / 3, -100.0 / 3, 0, 0, 0], atol=1e-9)
    amplitude, mean = stress_amplitude(_torsion(200.0, 50.0))
    assert amplitude == pytest.approx(200.0, rel=1e-12) and mean[5] == pytest.approx(50.0)
    both = _tension(300.0) + _torsion(200.0)
    assert stress_amplitude(both)[0] == pytest.approx(math.hypot(300.0 / math.sqrt(3.0), 200.0))
    # a quarter of a cycle apart: an ellipse, whose amplitude is its larger half-axis
    for shear in (200.0, 100.0):
        apart = _tension(300.0) + _torsion(shear, phase=np.pi / 2)
        assert stress_amplitude(apart)[0] == pytest.approx(max(300.0 / math.sqrt(3.0), shear))
    # three states of shear: the circle through an acute triangle, on the longest side
    # of an obtuse one (sqrt(J2) is the Euclidean norm of the two shears)

    def triangle(*corners):
        history = np.zeros((3, 6))
        history[:, 5], history[:, 3] = np.array(corners).T
        return history

    amplitude, mean = stress_amplitude(triangle((0, 0), (4, 0), (2, 3)))
    assert amplitude == pytest.approx(13.0 / 6.0, rel=1e-9)
    assert mean[5] == pytest.approx(2.0, abs=1e-8) and mean[3] == pytest.approx(5.0 / 6, abs=1e-8)
    amplitude, mean = stress_amplitude(triangle((0, 0), (4, 0), (2, 0.5)))
    assert amplitude == pytest.approx(2.0, rel=1e-12) and abs(mean[3]) < 1e-9
    # a cycle of pressure alone has no amplitude; a state held has none either
    pressure = np.zeros((ANGLES.size, 6))
    pressure[:, :3] = (100.0 * np.sin(ANGLES))[:, None]
    assert stress_amplitude(pressure)[0] < 1e-10
    np.testing.assert_allclose(hydrostatic(pressure), 100.0 * np.sin(ANGLES), atol=1e-12)
    assert stress_amplitude(np.ones((4, 6)))[0] == 0.0
    # arrays of points of any shape
    stack = np.stack([_tension(300.0), _torsion(200.0), both, pressure], axis=1)
    amplitude, mean = stress_amplitude(stack.reshape(ANGLES.size, 2, 2, 6))
    assert amplitude.shape == (2, 2) and mean.shape == (2, 2, 6)
    np.testing.assert_allclose(
        amplitude.ravel(),
        [300.0 / math.sqrt(3.0), 200.0, math.hypot(300.0 / math.sqrt(3.0), 200.0), 0.0],
        rtol=1e-12,
        atol=1e-10,
    )
    for bad in (np.zeros((1, 6)), np.zeros((5, 3)), np.zeros(6), np.full((3, 6), np.nan)):
        with pytest.raises(ValueError, match="stresses"):
            stress_amplitude(bad)


def test_criteria_at_the_limits_that_calibrate_them():
    laws = _laws()
    assert set(laws) == set(CRITERIA) and all(law.name == name for name, law in laws.items())
    bending, torsion = _tension(BENDING), _torsion(TORSION)
    for name in ("crossland", "dang_van"):
        law = laws[name]
        assert law.limit == TORSION
        assert law.equivalent(bending) == pytest.approx(law.limit, rel=1e-9)
        assert law.equivalent(torsion) == pytest.approx(law.limit, rel=1e-9)
    assert laws["crossland"].alpha == pytest.approx(3 * TORSION / BENDING - math.sqrt(3.0))
    assert laws["dang_van"].alpha == pytest.approx(3 * TORSION / BENDING - 1.5)
    sines = laws["sines"]
    assert sines.equivalent(torsion) == pytest.approx(TORSION, rel=1e-9)
    assert sines.equivalent(_tension(REPEATED / 2, REPEATED / 2)) == pytest.approx(TORSION)
    # a mean stress does nothing to a torsion, for the three of them
    for name in ("crossland", "sines", "dang_van"):
        assert laws[name].equivalent(_torsion(TORSION, 100.0)) == pytest.approx(TORSION)
    # the critical planes: to the accuracy of their search
    findley, matake = laws["findley"], laws["matake"]
    ratio = BENDING / TORSION
    assert findley.k == pytest.approx((2 - ratio) / (2 * math.sqrt(ratio - 1)))
    assert findley.limit == pytest.approx(TORSION * math.sqrt(1 + findley.k**2))
    assert findley.equivalent(bending) == pytest.approx(findley.limit, rel=1e-5)
    assert findley.equivalent(torsion) == pytest.approx(findley.limit, rel=1e-5)
    assert matake.k == pytest.approx(2 * TORSION / BENDING - 1) and matake.limit == TORSION
    assert matake.equivalent(bending) == pytest.approx(matake.limit, rel=2e-3)
    assert matake.equivalent(torsion) == pytest.approx(matake.limit, rel=2e-3)
    # the critical plane of a tension: at 45 degrees of its axis for Matake, the shear
    # amplitude half the stress; closer to the section for Findley
    plane = matake.critical_plane(bending)
    assert plane["shear"] == pytest.approx(BENDING / 2, rel=1e-5)
    assert abs(plane["normal"][0]) == pytest.approx(math.sqrt(0.5), abs=0.02)
    plane = findley.critical_plane(bending)
    assert abs(plane["normal"][0]) == pytest.approx(
        math.cos(0.5 * math.atan(1.0 / findley.k)), abs=5e-3
    )
    assert plane["equivalent"] == pytest.approx(
        plane["shear"] + findley.k * plane["normal_stress"], rel=1e-12
    )


def test_mean_stress_pressure_and_phase():
    laws = _laws()
    amplitude, mean = 200.0, 150.0
    history = _tension(amplitude, mean)
    largest = amplitude + mean
    crossland, sines, dang_van = laws["crossland"], laws["sines"], laws["dang_van"]
    assert crossland.equivalent(history) == pytest.approx(
        amplitude / math.sqrt(3.0) + crossland.alpha * largest / 3
    )
    assert sines.equivalent(history) == pytest.approx(
        amplitude / math.sqrt(3.0) + sines.alpha * mean / 3
    )
    assert dang_van.equivalent(history) == pytest.approx(
        amplitude / 2 + dang_van.alpha * largest / 3, rel=1e-7
    )
    # the planes of a tension: tau_a = (a / 2) sin 2θ, sigma_n,max = (max / 2) (1 + cos 2θ)
    k = laws["findley"].k
    assert laws["findley"].equivalent(history) == pytest.approx(
        0.5 * (math.hypot(amplitude, k * largest) + k * largest), rel=1e-5
    )
    k = laws["matake"].k
    assert laws["matake"].equivalent(history) == pytest.approx(
        0.5 * amplitude + 0.5 * k * largest, rel=2e-3
    )
    # a compression held lowers them all but Sines' (a mean) by its pressure
    pushed = _torsion(TORSION)
    pushed[:, :3] -= 90.0
    assert crossland.equivalent(pushed) == pytest.approx(TORSION - 90.0 * crossland.alpha)
    assert dang_van.equivalent(pushed) == pytest.approx(TORSION - 90.0 * dang_van.alpha)
    assert sines.equivalent(pushed) == pytest.approx(TORSION - 90.0 * sines.alpha)
    assert laws["findley"].equivalent(pushed) < laws["findley"].limit
    # a cycle of pressure: no shear, the largest pressure alone
    pressure = np.zeros((ANGLES.size, 6))
    pressure[:, :3] = (100.0 * np.sin(ANGLES))[:, None]
    assert crossland.equivalent(pressure) == pytest.approx(100.0 * crossland.alpha)
    assert dang_van.equivalent(pressure) == pytest.approx(100.0 * dang_van.alpha)
    assert sines.equivalent(pressure) == pytest.approx(0.0, abs=1e-9)
    # a tension and a torsion a quarter of a cycle apart
    apart = _tension(300.0) + _torsion(200.0, phase=np.pi / 2)
    assert crossland.equivalent(apart) == pytest.approx(200.0 + crossland.alpha * 100.0)


def test_criteria_do_not_depend_on_the_frame():
    rng = np.random.default_rng(3)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))

    def rotated(history):
        out = np.zeros_like(history)
        for step, stress in enumerate(history):
            tensor = np.zeros((3, 3))
            for c, (i, j) in enumerate(VOIGT):
                tensor[i, j] = tensor[j, i] = stress[c]
            turned = rotation @ tensor @ rotation.T
            out[step] = [turned[i, j] for i, j in VOIGT]
        return out

    history = _tension(250.0, 80.0) + _torsion(150.0, 20.0, phase=1.0)
    history[:, 1] = 60.0 * np.cos(2.0 * ANGLES)
    history[:, 3] = 40.0 * np.sin(3.0 * ANGLES)
    turned = rotated(history)
    for name, law in _laws().items():
        tolerance = {"findley": 1e-4, "matake": 3e-3}.get(name, 1e-7)
        assert law.equivalent(turned) == pytest.approx(law.equivalent(history), rel=tolerance)
    assert stress_amplitude(turned)[0] == pytest.approx(stress_amplitude(history)[0], rel=1e-9)
    # more planes in the first search: the same critical plane
    fine = Findley.from_limits(BENDING, TORSION, planes=1500)
    assert fine.equivalent(history) == pytest.approx(
        _laws()["findley"].equivalent(history), rel=1e-5
    )


def test_criteria_from_mappings_and_their_errors():
    law = criterion_from_config({"criterion": "crossland", "bending": BENDING, "torsion": TORSION})
    assert law == Crossland.from_limits(BENDING, TORSION)
    assert criterion_from_config({"criterion": "dang_van", "alpha": 0.4, "beta": 250}) == DangVan(
        0.4, 250.0
    )
    assert criterion_from_config(
        {"criterion": "sines", "torsion": TORSION, "repeated": REPEATED}
    ) == Sines.from_limits(TORSION, REPEATED)
    assert criterion_from_config(
        {"criterion": "matake", "bending": BENDING, "torsion": TORSION, "planes": 100}
    ) == Matake.from_limits(BENDING, TORSION, planes=100)
    assert criterion_from_config({"criterion": "findley", "k": 0.3, "limit": 270}) == Findley(
        0.3, 270.0
    )
    for bad, match in [
        ({"criterion": "goodman"}, "one of"),
        ({"bending": 1.0, "torsion": 1.0}, "one of"),
        ({"criterion": "crossland", "bending": BENDING}, "its limits"),
        ({"criterion": "crossland", "bending": 1, "torsion": 1, "alpha": 1, "beta": 1}, "limits"),
        ({"criterion": "crossland", "repeated": 1.0, "torsion": 1.0}, "unknown parameters"),
        ({"criterion": "sines", "bending": 1.0, "torsion": 1.0}, "unknown parameters"),
        ({"criterion": "crossland", "planes": 10, "alpha": 1, "beta": 1}, "unknown parameters"),
        ({"criterion": "crossland", "bending": "high", "torsion": 1.0}, "are numbers"),
        ("crossland", "a mapping"),
    ]:
        with pytest.raises(ValueError, match=match):
            criterion_from_config(bad)
    # limits for which a criterion has no meaning
    for call, match in [
        (lambda: Crossland.from_limits(400.0, 200.0), "torsion limit between"),
        (lambda: DangVan.from_limits(400.0, 190.0), "torsion limit between"),
        (lambda: Findley.from_limits(400.0, 190.0), "torsion limit between"),
        (lambda: Findley.from_limits(400.0, 400.0), "torsion limit between"),
        (lambda: Matake.from_limits(400.0, 190.0), "torsion limit between"),
        (lambda: Sines.from_limits(100.0, 400.0), "repeated tension below"),
        (lambda: Crossland(-0.1, 100.0), "alpha"),
        (lambda: Crossland(0.2, 0.0), "positive beta"),
        (lambda: Findley(0.3, 100.0, planes=5), "20 planes"),
        (lambda: Matake(-1.0, 100.0), "k of 0"),
    ]:
        with pytest.raises(ValueError, match=match):
            call()


def test_fatigue_of_points_by_phase():
    crossland = Crossland.from_limits(BENDING, TORSION)
    history = np.stack([_tension(200.0), _tension(400.0), _torsion(130.0), _torsion(390.0)], axis=1)
    res = fatigue(history, crossland)
    assert isinstance(res, FatigueResult) and res.utilization.shape == (4,)
    np.testing.assert_allclose(res.utilization, [0.5, 1.0, 0.5, 1.5], rtol=1e-9)
    assert res.largest == pytest.approx(1.5) and res.safety == pytest.approx(1 / 1.5)
    assert res.fraction() == 0.25 and res.fraction(0.4) == 1.0 and res.fraction(2.0) == 0.0
    np.testing.assert_allclose(res.amplitude[2:], [130.0, 390.0])
    np.testing.assert_allclose(res.hydrostatic_max[:2], [200.0 / 3, 400.0 / 3])
    assert set(res.components()) == {"utilization", "safety", "beyond"}
    assert set(res.fields()) == {"fatigue_utilization", "fatigue_equivalent",
                                 "fatigue_amplitude", "fatigue_hydrostatic_max"}  # fmt: skip
    # a criterion per phase, as objects or as mappings; the other phases are not judged
    phases = np.array([0, 0, 1, 2])
    weights = np.array([1.0, 3.0, 2.0, 2.0])
    laws = {0: crossland, 1: {"criterion": "dang_van", "bending": BENDING, "torsion": TORSION}}
    res = fatigue(history, laws, phases=phases, weights=weights)
    np.testing.assert_allclose(res.utilization[:3], [0.5, 1.0, 0.5], rtol=1e-9)
    assert np.isnan(res.utilization[3]) and res.largest == pytest.approx(1.0)
    assert res.fraction(0.75) == pytest.approx(3.0 / 6.0)
    scaled = fatigue(res.safety * 1.25 * history, laws, phases=phases)
    assert scaled.largest == pytest.approx(1.25)
    with pytest.raises(ValueError, match="phase of each point"):
        fatigue(history, laws)
    with pytest.raises(ValueError, match="one phase per point"):
        fatigue(history, laws, phases=phases[:2])
    with pytest.raises(ValueError, match="one weight per point"):
        fatigue(history, crossland, weights=weights[:2])
    with pytest.raises(TypeError, match="a criterion"):
        fatigue(history, 400.0)
    nobody = fatigue(history, {5: crossland}, phases=phases)
    assert np.isnan(nobody.largest) and np.isnan(nobody.fraction())


SOLID = Material(young=200e3, poisson=0.3)
#: A tension cycled between 300 and -100: four segments, the fields kept at each step
CYCLE = {"times": [0, 1, 2, 3, 4], "increments": [2, 2, 2, 2],
         "stress": {"11": [0.0, 300.0, 100.0, -100.0, 100.0]}}  # fmt: skip


@pytest.mark.parametrize("solver", ["fe", "fft"])
def test_fatigue_of_the_fields_of_a_cycle(solver):
    crossland = Crossland.from_limits(BENDING, TORSION)
    expected = (200.0 / math.sqrt(3.0) + crossland.alpha * 100.0) / TORSION
    image = VoxelImage(np.zeros((3, 3, 3), np.uint8), periodic=True)
    sample = voxel_tet_mesh(image) if solver == "fe" else image
    out = homogenize(sample, {0: SOLID}, properties=[], paths={"cycle": CYCLE}, fields="steps")
    result = out["cycle.periodic"]
    # the cycle: from the first peak on (the rise from zero is not part of it)
    res = path_fatigue(result, crossland, cycle=(1, 4))
    assert res.times[0] == 1.0 and res.times[-1] == 4.0 and res.times.size == 7
    np.testing.assert_allclose(res.utilization, expected, rtol=1e-8)
    np.testing.assert_allclose(res.amplitude, 200.0 / math.sqrt(3.0), rtol=1e-8)
    np.testing.assert_allclose(res.hydrostatic_max, 100.0, rtol=1e-8)
    np.testing.assert_allclose(res.hydrostatic_mean, 100.0 / 3.0, rtol=1e-8)
    assert res.utilization.shape == ((sample.n_elements,) if solver == "fe" else (3, 3, 3))
    assert res.largest == pytest.approx(expected) and res.safety == pytest.approx(1 / expected)
    assert res.fraction() == 0.0 and res.support is result.support
    assert (res.weights is not None) == (solver == "fe")
    # the whole path: the rise from zero widens the cycle of the pressure, not its shear
    whole = path_fatigue(result, crossland)
    assert whole.times.size == 8 and whole.largest == pytest.approx(expected)
    dang_van = path_fatigue(result, {0: {"criterion": "dang_van", "alpha": 0.45, "beta": 260.0}},
                            cycle=(1, 4))  # fmt: skip
    assert dang_van.largest == pytest.approx((100.0 + 0.45 * 100.0) / 260.0, rel=1e-7)
    with pytest.raises(ValueError, match="later one"):
        path_fatigue(result, crossland, cycle=(3, 1))
    with pytest.raises(ValueError, match="two steps"):
        path_fatigue(result, crossland, cycle=(3.2, 3.4))
    ends = homogenize(sample, {0: SOLID}, properties=[], paths={"cycle": CYCLE})
    with pytest.raises(ValueError, match="fields='steps'"):
        path_fatigue(ends["cycle.periodic"], crossland)


def test_fatigue_around_a_stiff_particle():
    """A stiff particle concentrates the stress of its matrix: the matrix is judged, by
    its own criterion; its largest utilization is above that of the matrix alone, and
    the factor of safety brings it to its limit."""
    data = np.zeros((6, 6, 6), np.uint8)
    data[2:4, 2:4, 2:4] = 1
    image = VoxelImage(data, periodic=True)
    materials = {0: SOLID, 1: Material(young=600e3, poisson=0.2)}
    crossland = Crossland.from_limits(BENDING, TORSION)
    alone = (200.0 / math.sqrt(3.0) + crossland.alpha * 100.0) / TORSION
    out = homogenize(image, materials, properties=[], paths={"cycle": CYCLE}, fields="steps")
    res = path_fatigue(out["cycle.periodic"], {0: crossland}, cycle=(1, 4))
    assert np.all(np.isnan(res.utilization[data == 1]))
    assert np.all(np.isfinite(res.utilization[data == 0]))
    assert 1.05 * alone < res.largest < 2.0 * alone
    assert res.utilization[data == 0].min() < alone
    assert 0.0 < res.fraction(alone) < 1.0
    # the stresses of an elastic cell grow with its loading: at the factor of safety,
    # with the mean stress scaled too, the largest utilization is one
    scaled = dict(CYCLE, stress={"11": [res.safety * s for s in CYCLE["stress"]["11"]]})
    out = homogenize(image, materials, properties=[], paths={"cycle": scaled}, fields="steps")
    again = path_fatigue(out["cycle.periodic"], {0: crossland}, cycle=(1, 4))
    assert again.largest == pytest.approx(1.0, rel=1e-6)


# -- configurations and the pipeline ----------------------------------------------------------


def _config(tmp_path, **fatigue) -> dict:
    """A stiff sphere in a matrix, on voxels, under the cycle of tension."""
    return {
        "model": {"type": "hardcore_spheres", "radius": 1.2, "number": 1},
        "domain": {"size": 6.0, "periodic": True},
        "seed": 1,
        "voxels": {"shape": 12},
        "homogenization": {
            "method": "fft",
            "materials": {0: {"young": 200e3, "poisson": 0.3}, 1: {"young": 600e3, "poisson": 0.2}},
            "properties": [],
            "paths": {"cycle": CYCLE},
            "fatigue": {
                "criterion": "crossland",
                "bending": BENDING,
                "torsion": TORSION,
                "cycle": [1, 4],
                **fatigue,
            },
        },
        "outputs": {"directory": str(tmp_path / "out"), "name": "s"},
    }


def test_configuration_of_the_fatigue(tmp_path):
    law = {"criterion": "crossland", "bending": BENDING, "torsion": TORSION}
    spec = load_config(_config(tmp_path)).homogenization.fatigue
    assert spec == {"path": "cycle", "cycle": (1.0, 4.0), "criteria": {0: law, 1: law}}
    one = load_config(_config(tmp_path, phases=[0])).homogenization.fatigue
    assert one["criteria"] == {0: law}
    data = _config(tmp_path)
    other = {"criterion": "dang_van", "alpha": 0.45, "beta": 260.0}
    data["homogenization"]["fatigue"] = {"criteria": {0: law, 1: other}}
    spec = load_config(data).homogenization.fatigue
    assert spec == {"path": "cycle", "cycle": None, "criteria": {0: law, 1: other}}
    # the results of the fatigue are quantities of the statistics of volume elements
    data = _config(tmp_path)
    data["domain"].pop("size")
    data["model"] = {"type": "hardcore_spheres", "radius": 1.2, "volume_fraction": 0.03}
    data["voxels"] = {"voxel_size": 0.5}
    data["rve"] = {"sizes": [6.0], "realizations": 2,
                   "properties": ["cycle.fatigue_utilization", "cycle.fatigue_safety"]}  # fmt: skip
    assert load_config(data).rve.properties == ["cycle.fatigue_utilization", "cycle.fatigue_safety"]
    data["homogenization"].pop("fatigue")
    with pytest.raises(ConfigError, match="unknown quantity 'fatigue_utilization'"):
        load_config(data)

    def changed(**fatigue):
        data = _config(tmp_path)
        data["homogenization"]["fatigue"] = fatigue
        return data

    two = _config(tmp_path)
    two["homogenization"]["paths"]["other"] = CYCLE
    porous = _config(tmp_path, phases=[1])
    porous["homogenization"]["materials"][1] = "void"
    no_path = _config(tmp_path)
    no_path["homogenization"].update(paths={}, properties=["bulk_modulus"])
    level_set = _config(tmp_path)
    level_set["homogenization"]["method"] = "xfem"
    not_a_mapping = _config(tmp_path)
    not_a_mapping["homogenization"]["fatigue"] = "crossland"
    for data, match in [
        (not_a_mapping, "a mapping"),
        (no_path, "cycle of a loading path"),
        (two, "path: one of the loading paths"),
        (_config(tmp_path, path="tension"), "one of the loading paths"),
        (_config(tmp_path, cycle=[4, 1]), "the first before the second"),
        (_config(tmp_path, cycle=[1, 9]), "two times of the path"),
        (_config(tmp_path, cycle=[1]), "times of its start and of its end"),
        (_config(tmp_path, cycle=[1, 1.5]), "two steps of the path"),
        (_config(tmp_path, phases=[3]), "not solid phases"),
        (porous, "not solid phases"),
        (_config(tmp_path, criteria={0: law}), "or 'criteria'"),
        (changed(criteria={}), "or 'criteria'"),
        (changed(criterion="goodman"), "the criterion is one of"),
        (changed(criterion="crossland", bending=400.0), "its limits"),
        (changed(criterion="crossland", bending=400.0, torsion=100.0), "torsion limit between"),
        (level_set, "method: fe or fft"),
    ]:
        with pytest.raises(ConfigError, match=match):
            load_config(data)


def test_fatigue_in_the_pipeline(tmp_path):
    data = _config(tmp_path, phases=[0])
    view = {"of": "cycle.periodic", "kind": "field"}
    data["visualization"] = {"figures": [
        {**view, "color": "fatigue_utilization"},
        {**view, "color": "fatigue_amplitude", "name": "amplitude"},
    ]}  # fmt: skip
    out = pipeline.run(load_config(data))
    names = {p.name for p in out.iterdir()}
    assert "s_0000_cycle_periodic_fatigue.vti" in names
    # the fields of the fatigue colour the figures of the path
    assert {"s_0000_cycle_periodic_fatigue_utilization.png", "s_0000_amplitude.png"} <= names
    with (out / "results.csv").open(encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    with (out / "homogenization.csv").open(encoding="utf-8") as stream:
        assert float(next(csv.DictReader(stream))["fatigue_utilization"]) == pytest.approx(
            float(row["cycle.periodic.fatigue_utilization"])
        )
    largest = float(row["cycle.periodic.fatigue_utilization"])
    alone = (200.0 / math.sqrt(3.0) + (3 * TORSION / BENDING - math.sqrt(3.0)) * 100.0) / TORSION
    assert 1.02 * alone < largest < 2.0 * alone
    assert float(row["cycle.periodic.fatigue_safety"]) == pytest.approx(1.0 / largest)
    assert float(row["cycle.periodic.fatigue_beyond"]) == 0.0
    # the curve of the path has no column of the fatigue: a result of the cycle
    with (out / "s_0000_cycle_periodic.csv").open(encoding="utf-8") as stream:
        assert not any("fatigue" in column for column in next(csv.reader(stream)))
    images = {image.name: image for image in io.read_vti(out / "s_0000_cycle_periodic_fatigue.vti")}
    assert set(images) == {"fatigue_utilization", "fatigue_equivalent", "fatigue_amplitude",
                           "fatigue_hydrostatic_max"}  # fmt: skip
    utilization = images["fatigue_utilization"].data
    assert utilization.shape == (12, 12, 12) and np.isnan(utilization).any()
    assert np.nanmax(utilization) == pytest.approx(largest, rel=1e-6)
    assert np.all(np.isfinite(images["fatigue_amplitude"].data))


@pytest.mark.slow
@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_fatigue_at_a_spherical_pore():
    """A pore far from the others under a tension that alternates: Goodier's stresses at
    its equator, ``K_t = (27 - 15 ν) / (2 (7 - 5 ν))`` along the load and ``(15 ν - 3) /
    (2 (7 - 5 ν))`` around the pore, give the largest utilization of Dang Van's
    criterion."""
    import morphogen as mg
    from morphogen.mesh.conforming import conforming_mesh

    micro = mg.models.HardcoreSpheres(radius=0.5, number=1).generate(
        mg.Domain.cube(4.0, periodic=True), seed=1
    )
    mesh = conforming_mesh(micro, size=0.5, periodic=True, phases="matrix", order=2)
    nu, far = 0.33, 40.0
    materials = {0: Material(young=72000.0, poisson=nu), 1: Material(void=True)}
    cycle = {"times": [0, 1, 2, 3], "increments": [1, 2, 2],
             "stress": {"11": [0.0, far, -far, far]}}  # fmt: skip
    out = homogenize(mesh, materials, properties=[], paths={"cycle": cycle}, fields="steps")
    law = DangVan.from_limits(110.0, 80.0)
    res = path_fatigue(out["cycle.periodic"], {0: law}, cycle=(1, 3))
    along = (27 - 15 * nu) / (2 * (7 - 5 * nu))
    around = (15 * nu - 3) / (2 * (7 - 5 * nu))
    stress = np.abs(out["cycle.periodic"].fields[1.0]["stress"][:, 0])
    assert stress.max() / far == pytest.approx(along, rel=0.06)
    # no stress across the surface of the pore: Tresca's shear is half the largest stress
    exact = (0.5 * along * far + law.alpha * (along + around) * far / 3) / law.limit
    assert res.largest == pytest.approx(exact, rel=0.06)
    assert res.safety == pytest.approx(1 / res.largest)
    homogeneous = (0.5 * far + law.alpha * far / 3) / law.limit
    assert np.nanmedian(res.utilization) == pytest.approx(homogeneous, rel=0.05)
    assert res.fraction(0.5) < 0.02  # the stress is concentrated at the pore


def test_pores_of_an_image_are_not_judged():
    """The fatigue of a voxel image with a pore: the voxels of the pore have no stress
    and are left out of the fields and of the share of the volume; the pore nearly
    doubles the utilization of the solid (Goodier: 2.05 for an isolated sphere)."""
    n = 12
    x = (np.indices((n, n, n)) + 0.5) / n - 0.5
    pore = (x**2).sum(axis=0) < 0.2**2
    image = VoxelImage((~pore).astype(np.uint8), spacing=1.0 / n)
    solid = Material(young=70000.0, poisson=0.33)
    cycle = {"times": [0, 1, 2, 3], "increments": [1, 2, 2],
             "stress": {"33": [0, 40.0, -40.0, 40.0]}}  # fmt: skip
    out = homogenize(image, {0: Material(void=True), 1: solid}, properties=[],
                     paths={"cycle": cycle}, fields="steps")  # fmt: skip
    law = Crossland.from_limits(BENDING, TORSION)
    res = path_fatigue(out["cycle.periodic"], {1: law}, cycle=(1, 3))
    assert res.utilization.shape == (n, n, n)
    assert np.isnan(res.utilization[pore]).all() and np.isfinite(res.utilization[~pore]).all()
    assert np.isnan(res.amplitude[pore]).all() and np.isnan(res.hydrostatic_max[pore]).all()
    alone = (
        law.equivalent(np.array([[0, 0, s, 0, 0, 0.0] for s in (40.0, -40.0, 40.0)])[:, None, :])[0]
        / law.limit
    )
    assert 1.25 * alone < res.largest < 2.2 * alone  # (a pore of five voxels across)
    assert res.fraction(0.0) == 1.0 and res.fraction(res.largest) == 0.0
    # one criterion for all the points, and stresses that are nowhere defined
    assert fatigue(np.stack([out["cycle.periodic"].fields[t]["stress"] for t in (1.0, 2.0, 3.0)]),
                   law).largest == pytest.approx(res.largest)  # fmt: skip
    with pytest.raises(ValueError, match="not finite"):
        fatigue(np.full((3, 4, 6), np.nan), law)
