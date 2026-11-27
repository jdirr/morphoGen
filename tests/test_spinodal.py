"""Spinodal decomposition by the Cahn–Hilliard equation (models.spinodal): the fastest
fluctuations of the linear stage, the decrease of the free energy, the mean composition
kept, the coarsening as the cube root of time, the fraction of the lever rule, and the
realizations as structures of implicit functions."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import morphogen as mg
from morphogen.models.base import Microstructure
from morphogen.models.spinodal import (
    Spinodal,
    cahn_hilliard,
    free_energy,
    structure_peak,
)


def test_linear_stage_grows_the_fastest_wavelength():
    model = Spinodal(wavelength=1.0, resolution=12)
    kappa = model.kappa  # λ_m of 12 points
    rng = np.random.default_rng(0)
    c = cahn_hilliard(0.01 * rng.uniform(-1.0, 1.0, (96, 96, 96)), kappa, 3 * 4 * kappa)
    # within the width of a bin of the radial average
    assert structure_peak(c) == pytest.approx(2 * math.pi / 12, abs=2 * math.pi / 96)


def test_energy_decreases_and_the_mean_is_kept():
    model = Spinodal(volume_fraction=0.4)
    rng = np.random.default_rng(1)
    c0 = -0.2 + 0.05 * rng.uniform(-1.0, 1.0, (32, 32, 32))
    c0 += -0.2 - c0.mean()
    c, energy = cahn_hilliard(c0, model.kappa, 10 * 4 * model.kappa / 0.88**2, energy=True)
    assert np.all(np.diff(energy) <= 1e-12)
    assert energy[-1] < free_energy(c0, model.kappa)
    assert c.mean() == pytest.approx(-0.2, abs=1e-12)


@pytest.mark.slow
def test_coarsening_as_the_cube_root_of_time():
    """Lifshitz–Slyozov: the length of the pattern doubles over eight times the age."""
    model = Spinodal(resolution=12)
    kappa = model.kappa
    rng = np.random.default_rng(2)
    young = cahn_hilliard(0.05 * rng.uniform(-1.0, 1.0, (96, 96, 96)), kappa, 20 * 4 * kappa)
    old = cahn_hilliard(young, kappa, 140 * 4 * kappa)
    ratio = structure_peak(young) / structure_peak(old)
    assert ratio == pytest.approx(8 ** (1 / 3), rel=0.1)


def test_realizations_of_the_model():
    model = Spinodal(volume_fraction=0.5, wavelength=1.0, coarsening=10.0)
    micro = model.generate(mg.Domain.cube(4.0), seed=1)
    s = micro.summary()
    assert s["fraction"] == pytest.approx(0.5, abs=0.02)
    assert s["mean_composition"] == pytest.approx(0.0, abs=1e-12)
    assert 1.0 < s["pattern_length"] < 1.6  # coarsened beyond the wavelength of the start
    image = micro.voxelize(32)
    assert image.data.mean() == pytest.approx(s["fraction"], abs=0.02)
    pts = np.random.default_rng(3).random((300, 3)) * 4.0
    f = micro.phase_level_sets_at(pts, 0.5)
    assert np.all((f[0] < 0) != (f[1] < 0))
    back = Microstructure.from_dict(json.loads(json.dumps(micro.to_dict())))
    assert np.array_equal(back.voxelize(24).data, micro.voxelize(24).data)
    window = Spinodal(volume_fraction=0.4, coarsening=20.0).generate(
        mg.Domain.cube(3.0, periodic=False), seed=2
    )
    assert window.voxelize(24).data.mean() == pytest.approx(0.4, abs=0.06)
    with pytest.raises(ValueError, match="spinodal"):
        Spinodal(volume_fraction=0.15)
    th = model.theory()
    assert th["volume_fraction"] == 0.5 and th["fastest_wavelength"] == 1.0
    assert th["fastest_rate"] == pytest.approx(1.0 / (4 * model.kappa * (1 / 12) ** 2))
