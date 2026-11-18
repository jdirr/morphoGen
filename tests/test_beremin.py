"""The criterion of cleavage of Beremin: the stresses that drive the cleavage of a point
(the largest principal stress, the normal stress of the cleavage planes of its crystal),
the Weibull stress of a field and the probability of failure, against closed forms; along
the loading path of a cell, with its plastic zone; in a configuration."""

import csv
import math

import numpy as np
import pytest

from morphogen.config import ConfigError, load_config
from morphogen.homogenization import Beremin, Material, Plasticity, path_beremin
from morphogen.homogenization.beremin import (
    FIELD,
    SERIES,
    failure_probability,
    normal_stress,
    principal_stress,
    weibull_stress,
)
from morphogen.homogenization.fft_nonlinear import NonlinearFFTProblem
from morphogen.image import VoxelImage
from morphogen.orientations import euler_to_matrix, random_orientations


def _voigt(tensor) -> np.ndarray:
    t = np.asarray(tensor, dtype=float)
    return np.array([t[0, 0], t[1, 1], t[2, 2], t[1, 2], t[2, 0], t[0, 1]])


def test_driving_stresses():
    rng = np.random.default_rng(0)
    tensors = rng.normal(0.0, 100.0, (6, 3, 3))
    tensors = tensors + np.swapaxes(tensors, 1, 2)
    stresses = np.array([_voigt(t) for t in tensors])
    assert np.allclose(principal_stress(stresses), np.linalg.eigvalsh(tensors)[:, -1])
    assert principal_stress(stresses.reshape(2, 3, 6)).shape == (2, 3)
    # a tension along the axis 3 on crystals turned about the axis 1: the planes (001)
    # and (010) carry cos^2 and sin^2 of it
    tension = np.array([0.0, 0.0, 1000.0, 0.0, 0.0, 0.0])
    for angle in (0.0, 30.0, 45.0, 70.0):
        c2 = math.cos(math.radians(angle)) ** 2
        value = normal_stress(tension, "cube", (0.0, angle, 0.0))
        assert value == pytest.approx(1000.0 * max(c2, 1.0 - c2), rel=1e-12)
        assert normal_stress(tension, "basal", (0.0, angle, 0.0), 1.856) == pytest.approx(
            1000.0 * c2, rel=1e-12, abs=1e-9
        )
    # one orientation per point; no orientation: the axes of the sample
    several = normal_stress(np.tile(tension, (2, 1)), "cube", [[0, 30.0, 0], [0, 45.0, 0]])
    assert np.allclose(several, [750.0, 500.0])
    assert normal_stress(tension) == pytest.approx(1000.0)
    # the normal stress of a plane is at most the largest principal stress; the {111}
    # planes of a crystal pulled along [001] carry a third of the tension
    euler = random_orientations(rng, 6)
    for planes in ("cube", "dodecahedral", "octahedral"):
        assert np.all(normal_stress(stresses, planes, euler) <= principal_stress(stresses) + 1e-9)
    assert normal_stress(tension, "octahedral") == pytest.approx(1000.0 / 3.0)
    # turning the sample and the crystal together leaves it as it is
    g = euler_to_matrix(euler)  # crystal = g @ sample
    turned = np.array([_voigt(q.T @ t @ q) for q, t in zip(g, tensors, strict=True)])
    assert np.allclose(normal_stress(turned, "cube", euler), normal_stress(stresses, "cube"))
    with pytest.raises(ValueError, match="cleavage planes"):
        normal_stress(tension, "pyramidal")
    with pytest.raises(ValueError, match="Bunge angles"):
        normal_stress(np.tile(tension, (2, 1)), "cube", np.zeros((3, 3)))
    with pytest.raises(ValueError, match="6 components"):
        principal_stress(np.zeros((2, 5)))


def test_weibull_stress():
    # a uniform stress: sigma (V / V0)^(1/m)
    assert weibull_stress(np.full(50, 1500.0), np.full(50, 0.002), 22.0, 0.001) == pytest.approx(
        1500.0 * 100.0 ** (1.0 / 22.0), rel=1e-12
    )
    # two regions; compressed points and those outside the plastic zone (NaN) do not count
    sigma = np.array([800.0, 1200.0, -500.0, np.nan])
    volumes = np.array([3.0, 0.5, 10.0, 10.0])
    expected = ((800.0**4 * 3.0 + 1200.0**4 * 0.5) / 2.0) ** 0.25
    assert weibull_stress(sigma, volumes, 4.0, 2.0) == pytest.approx(expected, rel=1e-12)
    assert weibull_stress([-1.0, np.nan], [1.0, 1.0], 4.0, 2.0) == 0.0
    # one volume for all the points; large moduli do not overflow
    assert weibull_stress([2000.0, 2000.0], 0.5, 60.0, 1.0) == pytest.approx(2000.0, rel=1e-12)
    # the weakest link: 63.2 % at the scale, the product of the survivals of two volumes
    assert failure_probability(2600.0, 22.0, 2600.0) == pytest.approx(1.0 - math.exp(-1.0))
    both = weibull_stress([750.0, 1100.0], [1.0, 2.0], 10.0, 1.0)
    alone = [weibull_stress([s], [v], 10.0, 1.0) for s, v in ((750.0, 1.0), (1100.0, 2.0))]
    survival = np.prod([1.0 - failure_probability(w, 10.0, 1000.0) for w in alone])
    assert 1.0 - failure_probability(both, 10.0, 1000.0) == pytest.approx(survival, rel=1e-12)


def test_criterion():
    law = Beremin.from_config({"modulus": 22, "scale": 2600.0, "volume": 1.25e-4})
    assert law == Beremin(22.0, 2600.0, 1.25e-4) and Beremin.from_config(law) is law
    crystal = Beremin(22.0, 2600.0, 1.0, planes="cube", plastic=1e-4)
    tension = np.array([0.0, 0.0, 1000.0, 0.0, 0.0, 0.0])
    assert crystal.driving(tension, (0.0, 45.0, 0.0)) == pytest.approx(500.0)
    assert law.driving(tension) == pytest.approx(1000.0)
    for bad, match in [
        ("beremin", "a mapping"),
        ({"modulus": 22, "scale": 2600.0}, "give \\['volume'\\]"),
        ({"modulus": 22, "scale": 2600.0, "volume": 1.0, "weibull": 2}, "unknown parameters"),
        ({"modulus": 0.5, "scale": 2600.0, "volume": 1.0}, "modulus of Weibull of 1 or more"),
        ({"modulus": 22, "scale": -1.0, "volume": 1.0}, "scale, a positive number"),
        ({"modulus": 22, "scale": 1.0, "volume": 1.0, "planes": "cubic"}, "planes, one of"),
        ({"modulus": 22, "scale": 1.0, "volume": 1.0, "plastic": -1.0}, "plastic"),
    ]:
        with pytest.raises(ValueError, match=match):
            Beremin.from_config(bad)


def _cell(shape=(2, 2, 3)):
    """A homogeneous elastic-plastic cell on voxels (MPa), of volume 12."""
    steel = Material(young=200000.0, poisson=0.3, plasticity=Plasticity(
        600.0, isotropic=[(200.0, 30.0)]))  # fmt: skip
    return NonlinearFFTProblem(VoxelImage(np.zeros(shape, dtype=np.uint8)), {0: steel})


def test_path_of_a_homogeneous_cell():
    """A homogeneous cell pulled, unloaded and pulled again: its Weibull stress is its
    stress times (V / V0)^(1/m) once it has yielded, zero before (the plastic zone); the
    probability of failure is that of the largest Weibull stress reached."""
    path = {"times": [0, 1, 2, 3], "increments": [8, 4, 6],
            "strain": {"33": [0.01, 0.006, 0.014]}}  # fmt: skip
    res = _cell().run(path, fields="steps")
    stress = res.stress[1:, 2]
    law = Beremin(modulus=20.0, scale=750.0, volume=3.0, plastic=1e-5)
    found = path_beremin(res, law)
    factor = (12.0 / 3.0) ** (1.0 / 20.0)
    yielded = res.plastic[1:] > 1e-5
    assert yielded[0] == False and yielded[-1] == True  # noqa: E712
    assert np.allclose(found.times, res.times[1:])
    # (unloaded into compression, the cell is not driven to cleave)
    driven = np.where(yielded, np.maximum(stress, 0.0), 0.0)
    assert np.allclose(found.weibull, driven * factor, rtol=1e-9, atol=1e-3)
    assert stress.min() < 0.0
    largest = np.maximum.accumulate(found.weibull)
    assert np.allclose(found.probability, -np.expm1(-((largest / 750.0) ** 20.0)), rtol=1e-12)
    # unloaded, the cell is as likely to have failed as before; loaded further, more
    assert np.allclose(found.probability[8:12], found.probability[7])
    assert found.probability[-1] > found.probability[7] > 0.1
    assert found.largest == pytest.approx(found.weibull.max())
    assert found.driving.shape == (2, 2, 3)
    assert np.allclose(found.driving, res.fields[3.0]["stress"][..., 2], rtol=1e-9)
    # every point counts without a plastic zone, from the first step
    whole = path_beremin(res, {"modulus": 20.0, "scale": 750.0, "volume": 3.0})
    assert np.allclose(whole.weibull, np.maximum(stress, 0.0) * factor, rtol=1e-9, atol=1e-3)
    # the planes of a crystal at 45 degrees carry half of the tension
    planes = path_beremin(res, {**law.__dict__, "planes": "cube"}, orientations=(0.0, 45.0, 0.0))
    assert np.allclose(planes.weibull, 0.5 * found.weibull, rtol=1e-9, atol=1e-3)
    per_voxel = np.tile([0.0, 45.0, 0.0], (2, 2, 3, 1))
    again = path_beremin(res, {**law.__dict__, "planes": "cube"}, orientations=per_voxel)
    assert np.allclose(again.weibull, planes.weibull, rtol=1e-12)
    # the curves of the criterion, beside those of the path
    series = found.series(res.times)
    assert set(series) == set(SERIES) and series[SERIES[0]][0] == 0.0
    assert np.allclose(series[SERIES[1]][1:], found.probability)
    assert found.fields()[FIELD] is found.driving
    # a path without the fields of its steps
    with pytest.raises(ValueError, match="stress fields of the path"):
        path_beremin(_cell().run(path), law)
    assert path_beremin(_cell().run(path, fields=True), law).series(res.times) is None


def _config(tmp_path, **beremin) -> dict:
    return {
        "model": {"type": "voronoi", "n_cells": 6, "crystal_orientations": True},
        "domain": {"size": 0.1, "periodic": True},
        "seed": 4,
        "voxels": {"shape": 6},
        "homogenization": {
            "method": "fft",
            "materials": {
                "1": {
                    "cubic": [230000.0, 135000.0, 117000.0],
                    "crystal_plasticity": {
                        "families": ["bcc110"],
                        "tau0": 150.0,
                        "K": 10.0,
                        "n": 10.0,
                    },
                }
            },
            "properties": [],
            "paths": {"tension": {"times": [0, 1], "increments": 6, "strain": {"33": 0.012}}},
            "beremin": {"modulus": 22.0, "scale": 1200.0, "volume": 1.25e-4, **beremin},
        },
        "outputs": {"directory": str(tmp_path / "out"), "name": "steel"},
    }


def test_configuration(tmp_path):
    cfg = load_config(_config(tmp_path, planes="cube", plastic=1e-4))
    assert cfg.homogenization.beremin == {
        "path": "tension",
        "criterion": {"modulus": 22.0, "scale": 1200.0, "volume": 1.25e-4, "planes": "cube",
                      "plastic": 1e-4},
    }  # fmt: skip
    two = _config(tmp_path)
    two["homogenization"]["paths"]["shear"] = {"times": [0, 1], "strain": {"12": 0.01}}
    level_set = _config(tmp_path)
    level_set["homogenization"]["method"] = "xfem"
    no_path = _config(tmp_path)
    no_path["homogenization"]["paths"] = {}
    no_path["homogenization"]["properties"] = ["stiffness"]
    not_a_mapping = _config(tmp_path)
    not_a_mapping["homogenization"]["beremin"] = 22
    rve = _config(tmp_path)
    rve["rve"] = {"method": "ensemble", "sizes": [0.1], "realizations": 2,
                  "properties": ["tension.beremin_probability"]}  # fmt: skip
    rve["model"] = {"type": "voronoi", "intensity": 6000.0, "crystal_orientations": True}
    rve["domain"], rve["voxels"] = {"periodic": True}, {"voxel_size": 0.1 / 6}
    assert load_config(rve).rve is not None
    unknown = _config(tmp_path)
    unknown.update(model=rve["model"], domain=rve["domain"], voxels=rve["voxels"])
    unknown["rve"] = {**rve["rve"], "properties": ["tension.beremin_stress"]}
    for data, match in [
        (not_a_mapping, "a mapping"),
        (no_path, "fields of a loading path"),
        (two, "path: one of the loading paths"),
        (_config(tmp_path, path="shear"), "one of the loading paths"),
        (_config(tmp_path, modulus=0.5), "modulus of Weibull"),
        (_config(tmp_path, planes="cubic"), "planes, one of"),
        (_config(tmp_path, weibull=2.0), "unknown parameters"),
        (level_set, "method: fe or fft"),
        (unknown, "unknown quantity"),
    ]:
        with pytest.raises(ConfigError, match=match):
            load_config(data)


def test_pipeline(tmp_path):
    """A polycrystal of ferrite pulled: the curve of its path has the Weibull stress and
    the probability of failure at each step, the files the stress that drives the
    cleavage of each voxel: the normal stress of the {100} planes of its grain, below
    its largest principal stress."""
    from morphogen import io, pipeline

    out = pipeline.run(load_config(_config(tmp_path, planes="cube", plastic=1e-4)))
    with (out / "steel_0000_tension_periodic.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    weibull = np.array([float(r["beremin_weibull"]) for r in rows])
    probability = np.array([float(r["beremin_probability"]) for r in rows])
    assert weibull[0] == 0.0 and weibull[-1] > 0.0 and np.all(np.diff(probability) >= 0)
    assert probability[-1] == pytest.approx(-math.expm1(-((weibull.max() / 1200.0) ** 22.0)))
    assert 0.0 < probability[-1] < 1.0
    with (out / "results.csv").open(encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["tension.periodic.beremin_probability"]) == pytest.approx(probability[-1])
    names = {p.name for p in out.iterdir()}
    assert "steel_0000_tension_periodic_beremin.vti" in names
    (image,) = io.read_vti(out / "steel_0000_tension_periodic_beremin.vti")
    assert image.name == FIELD and image.data.shape == (6, 6, 6)
    driving = image.data
    # the cell: 0.1^3 for a reference volume of 1.25e-4; every voxel has yielded
    assert np.all(np.isfinite(driving))
    volume = 0.1**3 / 6**3
    assert weibull_stress(driving, volume, 22.0, 1.25e-4) == pytest.approx(weibull[-1], rel=1e-5)
    # against the largest principal stress of the same fields: the planes carry less
    principal = pipeline.run(load_config(_config(tmp_path / "principal", plastic=1e-4)))
    with (principal / "steel_0000_tension_periodic.csv").open(encoding="utf-8") as stream:
        last = list(csv.DictReader(stream))[-1]
    assert weibull[-1] < float(last["beremin_weibull"]) < 2.0 * weibull[-1]
