"""Entangled cross-linked fibres (Chatti et al. 2018, 2020): the Gaussian girdle of their
orientations, the contacts of Poisson lines and of Boolean fibres against their exact
means, the bonded fraction, the cross-links of their own phase, seeds, periodicity."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, stats

import morphogen as mg
from morphogen.models import model_from_config
from morphogen.models.fibers import CrossLinkedFibers, excluded_volume, mean_sine
from morphogen.orientations import Girdle, orientation_from_config


def test_girdle_orientation_law():
    rng = np.random.default_rng(3)
    for sigma in (0.2, 0.4, 1.5):
        law = orientation_from_config({"type": "girdle", "sigma": sigma, "axis": [1, 0, 1]})
        assert law == Girdle(sigma, (1.0, 0.0, 1.0))
        assert orientation_from_config(law.to_config()) == law
        t = np.arccos(np.clip(law.sample(rng, 4000) @ np.array(law.axis), -1.0, 1.0))

        def density(a, sigma=sigma):
            return np.sin(a) * np.exp(-((a - 0.5 * math.pi) ** 2) / (2.0 * sigma**2))

        total = integrate.quad(density, 0.0, math.pi)[0]
        cdf = np.array([integrate.quad(density, 0.0, a)[0] for a in t]) / total
        assert stats.kstest(cdf, "uniform").pvalue > 1e-3, sigma
    with pytest.raises(ValueError, match="sigma"):
        Girdle(0.0)
    with pytest.raises(ValueError, match="at most 20"):  # its rejection would draw on and on
        Girdle(25.0)
    assert Girdle(20.0).sample(rng, 2000).shape == (2000, 3)
    with pytest.raises(ValueError, match="girdle"):
        orientation_from_config({"type": "girdle", "kappa": 1.0})


def _line_contacts(start, end):
    """Distances between the infinite lines through the segments, all pairs (i < j), and
    the middles of their closest points (closed form)."""
    d = end - start
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    i, j = np.triu_indices(len(start), 1)
    w = start[i] - start[j]
    b = np.einsum("ij,ij->i", d[i], d[j])
    e, f = np.einsum("ij,ij->i", d[i], w), np.einsum("ij,ij->i", d[j], w)
    s, t = (b * f - e) / (1.0 - b * b), (f - b * e) / (1.0 - b * b)
    p, q = start[i] + s[:, None] * d[i], start[j] + t[:, None] * d[j]
    return i, j, np.linalg.norm(p - q, axis=1), 0.5 * (p + q)


def test_contacts_of_poisson_lines():
    """Contacts per unit length of axis, 2 D L_V E|sin g| = pi D L_V / 2 (isotropic), the
    pairs being exactly the lines closer than D whose closest points are near the box."""
    model = CrossLinkedFibers(radius=1.0, volume_fraction=0.1, link_radius=1.5)
    theory = model.theory()
    lv = -math.log(0.9) / math.pi
    assert theory["contacts_per_length"] == pytest.approx(math.pi * 2.0 * lv / 2.0)
    assert theory["contact_spacing"] == pytest.approx(1.0 / (math.pi * lv))
    domain = mg.Domain.cube(60.0, periodic=False)
    rng = np.random.default_rng(0)
    start, end = model._lines(domain, rng, 2.5)
    pairs, middles = model.contacts(start, end, domain)
    lo, hi = domain.lower - 1.5, domain.upper + 1.5
    near = np.all((middles >= lo) & (middles <= hi), axis=1)
    found = {(int(a), int(b)) for a, b in pairs[near]}
    i, j, dist, centre = _line_contacts(start, end)
    true = (dist < 2.0) & np.all((centre >= lo) & (centre <= hi), axis=1)
    assert found == {(int(a), int(b)) for a, b in zip(i[true], j[true], strict=True)}
    assert len(found) > 300
    assert np.allclose(np.sort(middles[near], axis=0), np.sort(centre[true], axis=0))
    contacts, length, fibres = 0, 0.0, 0
    for seed in range(8):
        summary = model.generate(domain, seed=seed).summary()
        contacts += summary["contacts"]
        length += 2.0 * summary["contacts"] * summary["contact_spacing"]
        fibres += summary["fibres"]
    assert 2.0 * contacts / length == pytest.approx(theory["contacts_per_length"], rel=0.06)
    assert fibres / 8 == pytest.approx(model.expected_count(domain), rel=0.08)  # L_V S / 4


def test_contacts_of_boolean_fibres_and_bonds():
    """Finite fibres in a periodic box: contacts per fibre theta E[V_ex] (Poisson), with
    the girdle law; half the contacts bonded."""
    cfg = {
        "type": "cross_linked_fibers", "radius": 0.5, "length": 8.0, "volume_fraction": 0.15,
        "bonded_fraction": 0.5, "orientation": {"type": "girdle", "sigma": 0.4},
    }  # fmt: skip
    model = model_from_config(cfg)
    theory = model.theory()
    theta = -math.log(0.85) / (math.pi * 0.25 * 8.0 + 4.0 / 3.0 * math.pi * 0.125)
    sine = mean_sine(model.orientation)
    assert 0.70 < sine < 0.76  # below pi/4: gathered about a plane, not in it (2/pi)
    assert theory["contacts_per_fibre"] == pytest.approx(
        theta * excluded_volume(1.0, 8.0, 64.0, sine)
    )
    domain = mg.Domain.cube(30.0, periodic=True)
    fibres = contacts = links = 0
    for seed in range(6):
        summary = model.generate(domain, seed=seed).summary()
        fibres += summary["fibres"]
        contacts += summary["contacts"]
        links += summary["cross_links"]
    assert 2.0 * contacts / fibres == pytest.approx(theory["contacts_per_fibre"], rel=0.05)
    assert links / contacts == pytest.approx(0.5, abs=3.0 * math.sqrt(0.25 / contacts))
    assert fibres / 6 == pytest.approx(model.expected_count(domain), rel=0.05)


def test_cross_links_phase_seeds_and_periodicity():
    model = CrossLinkedFibers(radius=0.4, length=4.0, volume_fraction=0.12, link_radius=0.7)
    domain = mg.Domain.cube(12.0, periodic=True)
    micro = model.generate(domain, seed=5)
    image = micro.voxelize(shape=64)
    fractions = np.bincount(image.data.ravel(), minlength=3) / image.data.size
    assert fractions[1] == pytest.approx(model.theory()["volume_fraction"], abs=0.03)
    assert 0.002 < fractions[2] < 0.05
    assert set(np.unique(image.data)) == {0, 1, 2}
    again = model.generate(domain, seed=5).voxelize(shape=64)
    assert np.array_equal(image.data, again.data)
    assert not np.array_equal(image.data, model.generate(domain, seed=6).voxelize(64).data)
    merged = CrossLinkedFibers(
        radius=0.4, length=4.0, volume_fraction=0.12, link_radius=0.7, link_phase=1
    )
    merged = merged.generate(domain, seed=5).voxelize(shape=64)
    assert np.array_equal(merged.data == 1, image.data > 0)
    assert model.active_phases() == {0: "pores", 1: "fibres", 2: "cross-links"}
    with pytest.raises(ValueError, match="half the periodic box"):
        CrossLinkedFibers(length=6.0, volume_fraction=0.1).generate(domain)
    with pytest.raises(ValueError, match="periodic"):
        CrossLinkedFibers(volume_fraction=0.1).generate(domain)  # infinite fibres
    with pytest.raises(ValueError, match="bonded_fraction"):
        CrossLinkedFibers(volume_fraction=0.1, bonded_fraction=1.5)


def test_mezeix_distance_between_contacts():
    """Carbon fibres of 7 um at 8.5 % (Mezeix et al. 2016, eq. 16: d / 2f = 41 um between
    contacts at small fractions); cross-links on a third of the contacts, three times
    farther; the moulded girdle law (sigma = 0.4) spaces the contacts more."""
    iso = CrossLinkedFibers(radius=3.5, volume_fraction=0.085, bonded_fraction=1 / 3)
    theory = iso.theory()
    assert theory["contact_spacing"] == pytest.approx(7.0 / (2 * 0.085), rel=0.05)
    assert theory["cross_link_spacing"] == pytest.approx(3.0 * theory["contact_spacing"])
    girdle = CrossLinkedFibers(
        radius=3.5, volume_fraction=0.085, orientation=Girdle(0.4)
    ).theory()  # fmt: skip
    assert girdle["contact_spacing"] > theory["contact_spacing"]
    assert girdle["volume_fraction"] == pytest.approx(0.085)
