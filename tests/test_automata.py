"""Cellular-automaton engine: neighbourhoods, rules, updates and reaction-diffusion."""

import numpy as np
import pytest

from morphogen.analysis.connectivity import connected_components
from morphogen.automata import (
    ActivatorInhibitorRule,
    CellularAutomaton,
    ConservedMajorityRule,
    LatticeContext,
    LifeRule,
    MajorityRule,
    ProbabilisticRule,
    interface_density,
    kernel_sum,
    make_rule,
    neighbour_counts,
    offsets,
    random_state,
    sublattices,
)
from morphogen.automata.reaction import GRAY_SCOTT_PRESETS, gray_scott

# -- lattices ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("neighbourhood", "radius", "dims", "count"),
    [
        ("von_neumann", 1, 3, 6),
        ("moore", 1, 3, 26),
        ("edges", 1, 3, 18),
        ("euclidean", 1, 3, 6),
        ("euclidean", 2**0.5, 3, 18),
        ("euclidean", 3**0.5, 3, 26),
        ("euclidean", 2, 3, 32),
        ("von_neumann", 2, 3, 24),
        ("moore", 2, 3, 124),
        ("von_neumann", 1, 2, 4),
        ("moore", 1, 2, 8),
        ("euclidean", 2, 2, 12),
    ],
)
def test_neighbourhood_sizes(neighbourhood, radius, dims, count):
    offs = offsets(neighbourhood, radius, dims=dims)
    assert len(offs) == count
    assert not np.any(np.all(offs == 0, axis=1))
    assert len(offsets(neighbourhood, radius, dims=dims, include_center=True)) == count + 1
    # symmetric: -o is a neighbour when o is
    assert {tuple(o) for o in offs} == {tuple(-o) for o in offs}
    if dims == 2:
        assert np.all(offs[:, 2] == 0)


def test_neighbourhood_errors():
    with pytest.raises(ValueError, match="neighbourhood"):
        offsets("hexagonal")
    with pytest.raises(ValueError, match="radius"):
        offsets("moore", 0.5)
    with pytest.raises(ValueError, match="dims"):
        offsets("moore", dims=1)


def _brute_counts(state, offs, periodic, boundary):
    s = state.astype(np.int64)
    pad = int(np.abs(offs).max())
    padded = np.pad(
        s,
        pad,
        mode="wrap" if periodic else "constant",
        **({} if periodic else {"constant_values": boundary}),
    )
    out = np.zeros_like(s)
    n = s.shape
    for o in offs:
        out += padded[
            pad + o[0] : pad + o[0] + n[0],
            pad + o[1] : pad + o[1] + n[1],
            pad + o[2] : pad + o[2] + n[2],
        ]
    return out


@pytest.mark.parametrize("neighbourhood", ["von_neumann", "moore", "edges", "euclidean"])
def test_neighbour_counts_match_direct_sums(rng, neighbourhood):
    state = (rng.random((7, 6, 5)) < 0.4).astype(np.uint8)
    for radius in (1, 2):
        offs = offsets(neighbourhood, radius)
        assert np.array_equal(neighbour_counts(state, offs), _brute_counts(state, offs, True, 0))
        for boundary in (0, 1):
            assert np.array_equal(
                neighbour_counts(state, offs, periodic=False, boundary=boundary),
                _brute_counts(state, offs, False, boundary),
            )


def test_kernel_sum_matches_direct_sums(rng):
    field = rng.random((9, 8, 7))
    offs = offsets("euclidean", 2.5, include_center=True)
    weights = rng.standard_normal(len(offs))
    direct = np.zeros_like(field)
    for o, w in zip(offs, weights, strict=True):
        direct += w * np.roll(field, tuple(-o), axis=(0, 1, 2))
    assert np.allclose(kernel_sum(field, offs, weights), direct)
    padded = np.pad(field, 3, constant_values=0.7)
    fixed = np.zeros_like(padded)
    for o, w in zip(offs, weights, strict=True):
        fixed += w * np.roll(padded, tuple(-o), axis=(0, 1, 2))
    got = kernel_sum(field, offs, weights, periodic=False, boundary=0.7)
    assert np.allclose(got, fixed[3:-3, 3:-3, 3:-3])


@pytest.mark.parametrize(
    ("neighbourhood", "dims", "colours"),
    [("von_neumann", 3, 2), ("moore", 3, 8), ("edges", 3, 8), ("moore", 2, 4)],
)
def test_sublattices_are_independent_sets(neighbourhood, dims, colours):
    shape = (6, 4, 8) if dims == 3 else (6, 4, 1)
    groups = sublattices(neighbourhood, shape, dims)
    assert len(groups) == colours
    assert np.array_equal(np.sum(groups, axis=0), np.ones(shape))
    for g in groups:
        for o in offsets(neighbourhood, 1, dims=dims):
            assert not np.any(g & np.roll(g, tuple(o), axis=(0, 1, 2)))


# -- rules ---------------------------------------------------------------------------------


def test_life_rule_notation():
    life = LifeRule.parse("B3/S23")
    assert life.birth == {3} and life.survival == {2, 3}
    assert LifeRule.parse("s23/b3").birth == {3}
    clouds = LifeRule.parse("B13-14,17-19/S13-26")
    assert clouds.birth == {13, 14, 17, 18, 19}
    assert clouds.survival == set(range(13, 27))
    assert clouds.describe() == "B13-14,17-19/S13-26"
    assert LifeRule.parse("B/S").birth == set()
    for bad in ("X3/S2", "B5-3/S1", "B3", "B3/S2a"):
        with pytest.raises(ValueError, match="rule"):
            LifeRule.parse(bad)
    assert make_rule("life").describe() == "B3/S23"
    # descriptions read back as the same rule, a lone count of two digits included
    for birth, survival in [({13}, {4, 12, 14, 15, 16}), ({}, {9, 10}), ({1, 3}, {26})]:
        rule = LifeRule(birth, survival)
        back = LifeRule.parse(rule.describe())
        assert back.birth == rule.birth and back.survival == rule.survival
    assert make_rule("anneal").describe() == "anneal"
    with pytest.raises(ValueError, match="volume fraction"):
        make_rule("conserved")
    assert isinstance(make_rule("conserved", fraction=0.3), ConservedMajorityRule)


def test_game_of_life_glider_and_blinker():
    state = np.zeros((8, 8, 1), dtype=np.uint8)
    for i, j in [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]:
        state[i, j, 0] = 1
    ca = CellularAutomaton("life", dims=2)
    evo = ca.run(state, 4)
    assert np.array_equal(evo.state, np.roll(state, (1, 1), axis=(0, 1)))
    assert evo.history["fraction"] == [5 / 64] * 5
    blinker = np.zeros((5, 5, 1), dtype=np.uint8)
    blinker[2, 1:4, 0] = 1
    one = ca.step(blinker, np.random.default_rng(0))
    assert np.array_equal(one[1:4, 2, 0], [1, 1, 1]) and one.sum() == 3
    assert np.array_equal(ca.step(one, np.random.default_rng(0)), blinker)
    # fixed boundary of live cells: 3 live neighbours along the edges, 5 at the corners
    walled = CellularAutomaton("life", dims=2, periodic=False, boundary=1)
    born = walled.step(np.zeros((5, 5, 1), dtype=np.uint8), np.random.default_rng(0))
    assert born[0, 2, 0] == 1 and born[0, 0, 0] == 0 and born[2, 2, 0] == 0
    assert born[..., 0].sum() == 4 * 3


def test_majority_rules(rng):
    lattice = LatticeContext(offsets("moore"))
    for value in (0, 1):
        uniform = np.full((6, 6, 6), value, dtype=np.uint8)
        assert np.array_equal(MajorityRule().next_state(uniform, lattice, rng), uniform)
        assert np.array_equal(MajorityRule(True).next_state(uniform, lattice, rng), uniform)
    lone = np.zeros((6, 6, 6), dtype=np.uint8)
    lone[3, 3, 3] = 1
    assert MajorityRule().next_state(lone, lattice, rng).sum() == 0
    # anneal inverts the votes next to a tie (13 or 14 of 27)
    state = np.zeros((6, 6, 6), dtype=np.uint8)
    state[:, :3, :] = 1  # planar interface: 9 + 9 * (cell) of 27 on each side
    total = neighbour_counts(state, lattice.offsets) + state
    out = MajorityRule(True).next_state(state, lattice, rng)
    assert np.all(out[total == 13] == 1) and np.all(out[total == 14] == 0)


def test_coarsening_rules_reduce_the_interface(rng):
    state = random_state((32, 32, 32), 0.5, rng)
    for rule in ("majority", "anneal", "clouds"):
        evo = CellularAutomaton(rule).run(state, 10, np.random.default_rng(1))
        assert evo.history["interface"][-1] < 0.6 * evo.history["interface"][0]
    evo = CellularAutomaton("conserved", fraction=0.3).run(
        random_state((32, 32, 32), 0.3, rng, exact=True), 15, np.random.default_rng(1)
    )
    n = round(0.3 * 32**3)
    assert all(f == n / 32**3 for f in evo.history["fraction"])
    assert evo.history["interface"][-1] < 0.5 * evo.history["interface"][0]


def test_probabilistic_rule(rng):
    lattice = LatticeContext(offsets("von_neumann"))
    state = random_state((8, 8, 8), 0.5, rng)
    always = ProbabilisticRule(np.ones(7), np.ones(7))
    never = ProbabilisticRule(np.zeros(7), np.zeros(7))
    assert always.next_state(state, lattice, rng).all()
    assert not never.next_state(state, lattice, rng).any()
    # survival only: state 1 kept, no birth
    keep = ProbabilisticRule([0.0], np.ones(7))
    assert np.array_equal(keep.next_state(state, lattice, rng), state)
    with pytest.raises(ValueError, match="p_birth"):
        ProbabilisticRule([1.5], [0.5])


def test_activator_inhibitor_rule(rng):
    rule = ActivatorInhibitorRule(2.0, 5.0)
    lattice = LatticeContext(offsets("moore"))
    for value in (0, 1):  # balanced weight: uniform states are steady
        uniform = np.full((12, 12, 12), value, dtype=np.uint8)
        assert np.array_equal(rule.next_state(uniform, lattice, rng), uniform)
    state = random_state((12, 12, 12), 0.5, rng)
    direct = np.zeros(state.shape)
    for o, w in zip(rule.offsets, rule.weights, strict=True):
        direct += w * np.roll(state, tuple(-o), axis=(0, 1, 2))
    assert np.allclose(rule.score(state, lattice), direct)
    kept = ActivatorInhibitorRule(2.0, 5.0, fraction=0.3)
    assert kept.next_state(state, lattice, rng).sum() == round(0.3 * state.size)
    with pytest.raises(ValueError, match="radii"):
        ActivatorInhibitorRule(3.0, 2.0)


def _peak_wavelength(state):
    a = state[..., 0].astype(float)
    a -= a.mean()
    power = np.abs(np.fft.fft2(a)) ** 2
    n = a.shape[0]
    k = np.fft.fftfreq(n) * n
    kr = np.rint(np.hypot(*np.meshgrid(k, k, indexing="ij"))).astype(int)
    radial = np.bincount(kr.ravel(), power.ravel()) / np.maximum(1, np.bincount(kr.ravel()))
    radial[0] = 0.0
    return n / np.argmax(radial[: n // 2])


@pytest.mark.parametrize(("r1", "r2"), [(2.0, 5.0), (3.0, 7.0)])
def test_activator_inhibitor_wavelength(r1, r2):
    """Young's patterns have a wavelength between one and three inhibition radii."""
    rng = np.random.default_rng(4)
    ca = CellularAutomaton(ActivatorInhibitorRule(r1, r2, dims=2), dims=2)
    evo = ca.run(random_state((128, 128, 1), 0.5, rng), 40, rng)
    assert evo.steps < 40  # reaches a steady pattern
    assert r2 <= _peak_wavelength(evo.state) <= 3 * r2
    assert 0.4 < evo.state.mean() < 0.6


# -- updates -------------------------------------------------------------------------------


def test_updates(rng):
    state = random_state((10, 10, 10), 0.5, rng)
    ca = CellularAutomaton("majority", neighbourhood="von_neumann")
    sync = ca.step(state, np.random.default_rng(0))
    full = ca.step(state, np.random.default_rng(0), update="asynchronous", async_fraction=1.0)
    assert np.array_equal(sync, full)
    half = ca.step(state, np.random.default_rng(0), update="asynchronous", async_fraction=0.5)
    changed = half != state
    assert np.all(half[changed] == sync[changed])
    assert 0 < changed.sum() < (sync != state).sum()
    seq = ca.step(state, np.random.default_rng(0), update="sequential")
    assert not np.array_equal(seq, sync)
    assert np.array_equal(seq, ca.step(state, np.random.default_rng(0), update="sequential"))
    with pytest.raises(ValueError, match="update"):
        ca.step(state, rng, update="random")
    with pytest.raises(ValueError, match="radius-1"):
        CellularAutomaton("majority", radius=2).step(state, rng, update="sequential")
    with pytest.raises(ValueError, match="synchronous"):
        CellularAutomaton("conserved", fraction=0.5).step(state, rng, update="asynchronous")
    with pytest.raises(ValueError, match="async_fraction"):
        ca.step(state, rng, update="asynchronous", async_fraction=0.0)


def test_run_history_and_stopping(rng):
    ca = CellularAutomaton("majority")
    evo = ca.run(np.zeros((6, 6, 6), dtype=np.uint8), 10)
    assert evo.steps == 1 and evo.history["changed"] == [0, 0]
    frames = []
    evo = ca.run(
        random_state((8, 8, 8), 0.5, rng),
        5,
        np.random.default_rng(2),
        stop_when_stable=False,
        callback=lambda k, s: frames.append(k),
    )
    assert evo.steps == 5 and frames == [1, 2, 3, 4, 5]
    arrays = evo.as_arrays()
    assert set(arrays) == {"step", "fraction", "changed", "interface"}
    assert arrays["step"].tolist() == list(range(6))
    assert ca.run(evo.state, 3, record=False).history["step"] == []
    with pytest.raises(ValueError, match="binary"):
        ca.run(np.full((4, 4, 4), 2), 1)
    with pytest.raises(ValueError, match="3D"):
        ca.run(np.zeros((4, 4)), 1)
    with pytest.raises(ValueError, match="2D"):
        CellularAutomaton("life", dims=2).run(np.zeros((4, 4, 2)), 1)


def test_random_state_and_interface_density(rng):
    exact = random_state((10, 10, 10), 0.37, rng, exact=True)
    assert exact.sum() == 370
    state = np.zeros((8, 8, 8), dtype=np.uint8)
    assert interface_density(state) == 0.0
    state[:4] = 1
    assert interface_density(state) == pytest.approx(0.25 / 3)
    assert interface_density(state, periodic=False) == pytest.approx(1 / 7 / 3)
    checker = np.indices((4, 4, 4)).sum(axis=0) % 2
    assert interface_density(checker) == 1.0


# -- reaction-diffusion --------------------------------------------------------------------


def test_gray_scott_patterns():
    rng = np.random.default_rng(1)
    shape = (96, 96, 1)
    spots = gray_scott(shape, rng, feed=0.030, kill=0.062, steps=4000)[1] > 0.2
    holes = gray_scott(shape, rng, feed=0.039, kill=0.058, steps=4000)[1] > 0.2
    assert connected_components(spots.astype(np.uint8), 1).n_components > 20
    assert 0.1 < spots.mean() < 0.35
    assert holes.mean() > 0.55
    assert connected_components(holes.astype(np.uint8), 1).n_components == 1
    assert set(GRAY_SCOTT_PRESETS) == {"spots", "worms", "labyrinth", "holes"}


def test_gray_scott_scheme():
    rng = np.random.default_rng(3)
    u, v = gray_scott((16, 16, 16), rng, steps=200)
    assert u.shape == v.shape == (16, 16, 16)
    assert np.all((u >= 0) & (u <= 1)) and np.all(v >= 0)
    a = gray_scott((16, 16, 1), np.random.default_rng(5), steps=50, periodic=False)
    b = gray_scott((16, 16, 1), np.random.default_rng(5), steps=50, periodic=False)
    assert np.array_equal(a[1], b[1])
    frames = []
    gray_scott((8, 8, 8), rng, steps=3, callback=lambda k, u, v: frames.append(k))
    assert frames == [1, 2, 3]
    with pytest.raises(ValueError, match="stability"):
        gray_scott((8, 8, 8), rng, du=0.2)
    assert gray_scott((8, 8, 1), rng, du=0.2, steps=1)[0].shape == (8, 8, 1)  # 2D: D <= 1/4
    with pytest.raises(ValueError, match="seed_fraction"):
        gray_scott((8, 8, 8), rng, seed_fraction=0.0)
    with pytest.raises(ValueError, match="a single cell"):
        gray_scott((1, 1, 1), rng)
