"""The contact packing of Keilmann, Redenbach & Willot (2026) on the bending fibres of
Altendorf & Jeulin: the shortlist of the candidates, more contacts than the packing alone
and than Toll's estimate, no overlap, seeds."""

from __future__ import annotations

import math

import numpy as np
import pytest

import morphogen as mg
from morphogen.kernels.force_biased import near_pairs
from morphogen.models.fibers import RandomWalkFibers, _shortlist, excluded_volume


def test_shortlist_keeps_one_pair_per_segment_and_fibre():
    seg_fibre = np.array([0, 0, 0, 1, 1, 2])
    s = np.array([0, 1, 1, 2, 0])
    t = np.array([3, 3, 4, 4, 5])
    d = np.array([0.3, 0.1, 0.2, 0.4, 0.9])
    # (1, 3) beats (0, 3) on segment 3 and (1, 4) on segment 1; (2, 4) loses segment 4 to
    # (1, 4), itself dropped: each pair is compared with all the candidates (eq. 18)
    keep = _shortlist(s, t, d, seg_fibre, 3)
    assert keep.tolist() == [False, True, False, False, True]


def _fibres(**kw):
    cfg = {"radius": 1.0, "length": 10.0, "volume_fraction": 0.2, "step": 2.0}
    return RandomWalkFibers(**{**cfg, **kw})


def test_contact_packing_raises_the_contacts():
    domain = mg.Domain.cube(20.0, periodic=True)
    model = _fibres(interaction_distance=0.5)
    micro = model.generate(domain, seed=2)
    summary = micro.summary()
    before, after = summary["contacts_per_fibre_before"], summary["contacts_per_fibre"]
    toll = model.theory()["toll_contacts_per_fibre"]
    assert after > 1.3 * before
    # short fibres (aspect ratio 6): the packing alone near Toll's estimate, the contact
    # packing above it (the paper's fig. 10 at 20 %)
    assert before == pytest.approx(toll, rel=0.2)
    assert after > 1.2 * toll
    assert summary["contact_pairs"] > summary["released_pairs"]
    assert summary["volume_fraction"] == pytest.approx(0.2, abs=0.01)
    # no overlap left between fibres (each segment its own pair of nodes)
    seg, m = micro.segments, 5
    x = np.stack([seg.start, seg.end], axis=1).reshape(-1, 3)
    seg_fibre = np.repeat(np.arange(len(seg) // m), m)
    hits = near_pairs(x, 2 * np.arange(len(seg)), seg_fibre, seg.radii[::m], domain.lengths, -1e-6)
    assert len(hits[0]) == 0
    # without the contact packing: the same fibres as the packing alone
    plain = _fibres().generate(domain, seed=2).summary()
    assert plain["contacts_per_fibre"] == pytest.approx(before, rel=0.05)
    assert "contact_pairs" not in plain
    again = model.generate(domain, seed=2)
    assert np.array_equal(again.segments.start, seg.start)


def test_toll_estimate_and_validation():
    model = _fibres(orientation={"type": "planar"})
    theta = 0.2 / (math.pi * 10.0 + 4.0 / 3.0 * math.pi)
    assert model.theory()["toll_contacts_per_fibre"] == pytest.approx(
        theta * excluded_volume(2.0, 10.0, 100.0, 2.0 / math.pi)
    )
    with pytest.raises(ValueError, match="packing"):
        _fibres(packing=False, interaction_distance=0.5)
    with pytest.raises(ValueError, match="contact_weight"):
        _fibres(contact_weight=1.5)


def test_contacts_counted_on_the_fibres_returned():
    """With the shrinking end step, the contacts of the summary are those of the fibres
    returned, their radii reduced: recounted from the segments of the realization."""
    domain = mg.Domain.cube(20.0, periodic=True)
    model = _fibres(interaction_distance=0.5, end_step="shrink")
    micro = model.generate(domain, seed=2)
    summary = micro.summary()
    seg, m = micro.segments, 5
    x = np.stack([seg.start, seg.end], axis=1).reshape(-1, 3)
    seg_fibre = np.repeat(np.arange(len(seg) // m), m)
    radii = seg.radii[::m]
    assert summary["radius_reduction"] > 0.0
    tol = model.contact_distance + 0.01 * radii.min()  # the summary's definition
    s, t, _ = near_pairs(x, 2 * np.arange(len(seg)), seg_fibre, radii, domain.lengths, tol)
    fibres = np.unique(np.sort(np.stack([seg_fibre[s], seg_fibre[t]], axis=1), axis=1), axis=0)
    assert summary["contacts"] == len(fibres)
    assert summary["contacts_per_fibre"] == pytest.approx(2.0 * len(fibres) / len(radii))
