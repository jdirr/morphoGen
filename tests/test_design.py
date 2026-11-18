"""Combinatorial design: tile sets, constraint solvers, frameworks and search."""

import itertools

import numpy as np
import pytest

import morphogen as mg
from morphogen import io
from morphogen.design import (
    Framework,
    TileSet,
    TransferMatrix,
    count_assemblies,
    count_polarized_blocks,
    fit_rule,
    genetic_algorithm,
    pipe_tiles,
    polarized_blocks,
    sample_uniform,
    simulated_annealing,
    structure_descriptors,
    wave_function_collapse,
)
from morphogen.design.search import descriptor_cost
from morphogen.models import RandomLattice, TileAssembly

# -- tile sets -----------------------------------------------------------------------------


def _brute_force_count(tiles, shape, periodic):
    cells = int(np.prod(shape))
    every = np.array(list(itertools.product(range(tiles.n_tiles), repeat=cells)))
    every = every.reshape(-1, *shape)
    compat = tiles.compatibility()
    ok = np.ones(len(every), dtype=bool)
    for a in range(3):
        if periodic:
            pairs = compat[a][every, np.roll(every, -1, axis=a + 1)]
        elif shape[a] > 1:
            lo = np.take(every, range(shape[a] - 1), axis=a + 1)
            hi = np.take(every, range(1, shape[a]), axis=a + 1)
            pairs = compat[a][lo, hi]
        else:
            continue
        ok &= pairs.reshape(len(every), -1).all(axis=1)
    return int(ok.sum())


def test_polarized_blocks():
    tiles = polarized_blocks(6)
    assert tiles.n_tiles == 6 and tiles.match == "opposite"
    # strains (e_x, e_y, e_z) of the six states: never three equal signs
    strains = tiles.faces[:, ::2]
    assert np.all(np.abs(strains.sum(axis=1)) == 1)
    compat = tiles.compatibility()
    # along x, a block elongated along x (x+) must be followed by one shrinking along x
    x_plus = tiles.names.index("x+")
    followers = {tiles.names[b] for b in np.flatnonzero(compat[0, x_plus])}
    assert followers == {"x-", "y+", "z+"}
    assert tiles.voxels.shape == (6, 6, 6, 6) and tiles.voxels[0, :, 3, 3].sum() == 0
    flat = polarized_blocks(6, dims=2)
    assert flat.n_tiles == 4 and flat.voxels.shape == (4, 6, 6, 1)
    assert flat.compatibility()[2].all()


def test_pipe_tiles_and_rotations():
    tiles = pipe_tiles(8, radius=0.2)
    assert tiles.n_tiles == 64 and tiles.match == "equal"
    # the voxels of every face touch the pipe exactly when the face is open
    for t in range(tiles.n_tiles):
        v = tiles.voxels[t]
        layers = [v[0], v[-1], v[:, 0], v[:, -1], v[:, :, 0], v[:, :, -1]]
        assert [int(layer.any()) for layer in layers] == tiles.faces[t].tolist()
    one = tiles.subset([tiles.names.index("+x")])
    rotated = one.with_rotations()
    assert rotated.n_tiles == 6  # one per face
    for t in range(6):
        v = rotated.voxels[t]
        layers = [v[0], v[-1], v[:, 0], v[:, -1], v[:, :, 0], v[:, :, -1]]
        assert [int(layer.any()) for layer in layers] == rotated.faces[t].tolist()
    assert tiles.with_rotations().n_tiles == 64  # closed under rotations
    no_dead_ends = pipe_tiles(8, degrees=(0, 2, 3))
    assert set(no_dead_ends.faces.sum(axis=1)) == {0, 2, 3}
    assert pipe_tiles(8, dims=2).n_tiles == 16
    with pytest.raises(ValueError, match="degrees"):
        pipe_tiles(8, degrees=(9,))


def test_assembly_geometry_and_mismatches():
    tiles = polarized_blocks(4)
    assembly = np.zeros((2, 3, 1), dtype=int)
    voxels = tiles.assemble(assembly)
    assert voxels.shape == (8, 12, 4)
    assert np.array_equal(voxels[4:8, 8:12], tiles.voxels[0])
    # identical blocks: their common faces both move out (x) or both in (y)
    assert tiles.mismatches(assembly, periodic=False) == 3 + 2 * 2
    good = np.array([[[0]], [[1]]])
    assert tiles.is_compatible(good, periodic=False)
    with pytest.raises(ValueError, match="match"):
        TileSet(np.zeros((1, 6)), match="glue")
    with pytest.raises(ValueError, match="voxels"):
        TileSet(np.zeros((2, 6)), voxels=np.zeros((2, 3, 3, 2)))


# -- constraint solvers --------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 2, 1), (2, 2, 2), (3, 2, 1)])
@pytest.mark.parametrize("periodic", [False, True])
def test_exact_counts_agree(shape, periodic):
    """Brute force, transfer matrix and the gauge sum of polarized blocks agree."""
    tiles = polarized_blocks(2)
    expected = _brute_force_count(tiles, shape, periodic)
    assert count_assemblies(tiles, shape, periodic=periodic) == expected
    assert count_polarized_blocks(shape, periodic=periodic) == expected


def test_polarized_block_counts():
    counts = [count_polarized_blocks((n, n, n)) for n in (1, 2, 3, 4)]
    assert counts == [6, 450, 151206, 145456074]
    assert count_assemblies(polarized_blocks(2), (4, 2, 3)) == count_polarized_blocks((2, 3, 4))
    assert count_polarized_blocks((3, 3, 3), periodic=True) == 0  # lines alternate
    assert count_polarized_blocks((4, 4, 4), periodic=True) == counts[3]
    with pytest.raises(ValueError, match="too wide"):
        count_polarized_blocks((6, 6, 6))


def test_transfer_matrix_counts_wang_tiles():
    # two colours on every face, equal matching: every assembly of one tile is valid
    tiles = TileSet([[0] * 6, [1] * 6], match="equal")
    assert count_assemblies(tiles, (5, 2, 2)) == 2  # each block is all 0 or all 1
    mixed = TileSet(list(itertools.product((0, 1), repeat=6)), match="equal")
    tm = TransferMatrix(mixed, (1, 1))
    # 1 x 1 sections: free faces along y and z, chains along x share their face colours
    assert tm.count(3) == 2**4 * 2**4 * 2**4 * 2 * 2 * 2 * 2
    with pytest.raises(ValueError, match="sections"):
        TransferMatrix(mixed, (3, 3), max_states=100)


def test_uniform_sampling():
    tiles = polarized_blocks(2)
    rng = np.random.default_rng(0)
    seen = {}
    for _ in range(1800):
        a = sample_uniform(tiles, (2, 1, 1), rng)
        assert tiles.is_compatible(a, periodic=False)
        seen[a.tobytes()] = seen.get(a.tobytes(), 0) + 1
    assert len(seen) == 18  # all the assemblies, about 100 times each
    counts = np.array(list(seen.values()))
    chi2 = np.sum((counts - 100.0) ** 2 / 100.0)
    assert chi2 < 45  # 17 degrees of freedom
    periodic = sample_uniform(tiles, (4, 2, 2), rng, periodic=True)
    assert periodic.shape == (4, 2, 2) and tiles.is_compatible(periodic, periodic=True)


def test_wave_function_collapse():
    rng = np.random.default_rng(3)
    tiles = polarized_blocks(4)
    a = wave_function_collapse(tiles, (6, 6, 6), rng)
    assert a.shape == (6, 6, 6) and tiles.is_compatible(a)
    b = wave_function_collapse(tiles, (5, 7, 3), rng, periodic=False)
    assert tiles.is_compatible(b, periodic=False)
    with pytest.raises(ValueError, match="no compatible"):
        wave_function_collapse(tiles, (3, 3, 3), rng, max_backtracks=10_000)
    pipes = pipe_tiles(6, degrees=(0, 2, 3, 4))
    net = wave_function_collapse(pipes, (8, 8, 8), rng)
    assert pipes.is_compatible(net) and set(pipes.faces[net].sum(axis=-1).ravel()) <= {0, 2, 3, 4}
    # closed boxes: no pipe leaves through the outer faces
    closed = wave_function_collapse(pipes, (5, 5, 5), rng, periodic=False, boundary=0)
    assert pipes.is_compatible(closed, periodic=False)
    assert pipes.faces[closed[0], 0].sum() == 0 and pipes.faces[closed[:, -1], 3].sum() == 0
    # cells restricted by the designer
    allowed = np.ones((4, 4, 4, pipes.n_tiles), dtype=bool)
    allowed[:2] = False
    allowed[:2, ..., pipes.names.index("empty")] = True
    graded = wave_function_collapse(pipes, (4, 4, 4), rng, periodic=False, allowed=allowed)
    assert np.all(graded[:2] == pipes.names.index("empty"))
    flat = wave_function_collapse(polarized_blocks(4, dims=2), (6, 4, 1), rng)
    assert polarized_blocks(4, dims=2).is_compatible(flat)


# -- frameworks ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cell", "coordination", "floppy"),
    [
        ("cubic", 6.0, 3),
        ("bcc", 8.0, 2),
        ("octet", 12.0, 0),
        ("kelvin", 4.0, 5),
        ("diamond", 4.0, 5),
        ("pyrochlore", 6.0, 0),
    ],
)
def test_lattice_rigidity(cell, coordination, floppy):
    """Cubic and bcc lattices are bending-dominated, the octet truss is not; the Kelvin
    and diamond lattices (four struts per node) carry a hydrostatic strain only, and the
    pyrochlore lattice of tetrahedra joined by their corners carries every strain with
    six struts per node (Maxwell's count balanced: the kagome of space), though it keeps
    mechanisms."""
    for n in (1, 2):
        frame = Framework.lattice(cell, n, cell_size=2.0)
        r = frame.rigidity()
        assert frame.coordination() == coordination
        assert r["floppy_strains"] == floppy
        assert r["stretching_dominated"] == (floppy == 0)
        # Maxwell-Calladine: m - s = 3 j - 3 - b
        assert r["mechanisms"] - r["self_stresses"] == 3 * frame.n_nodes - 3 - frame.n_bars
    assert Framework.lattice("octet", 2).rigidity()["mechanisms"] == 0
    assert Framework.lattice("pyrochlore", 1).rigidity()["mechanisms"] == 9


def test_lattices_of_quarter_cells():
    """The cells whose nodes are not at the corners, the centres of the faces or of the
    cell: the Kelvin cell (12 nodes of four struts), the diamond lattice (8 nodes of four
    struts along the diagonals) and the pyrochlore lattice (16 nodes of six struts along
    the diagonals of the faces), in one cell and in several."""
    expected = {
        "kelvin": (12, 24, 4, np.sqrt(2.0) / 4),
        "diamond": (8, 16, 4, np.sqrt(3.0) / 4),
        "pyrochlore": (16, 48, 6, np.sqrt(2.0) / 4),
    }
    for cell, (nodes, bars, degree, length) in expected.items():
        for n in (1, 2):
            frame = Framework.lattice(cell, n, cell_size=3.0)
            assert (frame.n_nodes, frame.n_bars) == (nodes * n**3, bars * n**3)
            assert set(frame.degrees()) == {degree}
            assert np.linalg.norm(frame.vectors(), axis=1) == pytest.approx(3.0 * length)
            assert len(np.unique(frame.components())) == 1
    # the struts of the pyrochlore lattice lie on straight lines: two opposite ones along
    # each of three directions at every node
    frame = Framework.lattice("pyrochlore", 1)
    unit = frame.vectors() / np.linalg.norm(frame.vectors(), axis=1)[:, None]
    for node in range(frame.n_nodes):
        out = np.vstack([unit[frame.bars[:, 0] == node], -unit[frame.bars[:, 1] == node]])
        assert np.abs(out.sum(axis=0)).max() < 1e-12 and len(out) == 6


def test_dilution_and_maxwell_transition():
    base = Framework.lattice("octet", 3)
    assert base.n_nodes == 108 and base.n_bars == 648
    rng = np.random.default_rng(0)
    rigid = base.diluted(rng, coordination=7.0, min_degree=3)
    assert rigid.coordination() == pytest.approx(7.0, abs=0.05)
    assert rigid.degrees().min() >= 3
    assert rigid.rigidity()["floppy_strains"] == 0
    floppy = base.diluted(rng, coordination=5.0, min_degree=3)
    assert floppy.rigidity()["floppy_strains"] > 0  # below Maxwell's isostatic point (6)
    cut = base.diluted(rng, fraction=0.5)
    assert cut.degrees().min() >= 2 and len(np.unique(cut.components())) == 1
    moved = base.perturbed(rng, 0.1)
    assert np.array_equal(moved.bars, base.bars)
    shift = np.linalg.norm(moved.nodes - base.nodes, axis=1)
    assert 0 < shift.max() <= 0.1
    back = Framework.from_dict(moved.to_dict())
    assert np.allclose(back.vectors(), moved.vectors())
    seg = moved.segments(0.05)
    assert len(seg) == moved.n_bars
    with pytest.raises(ValueError, match="exactly one"):
        base.diluted(rng)


# -- search --------------------------------------------------------------------------------


def test_simulated_annealing_and_genetic_algorithm():
    rng = np.random.default_rng(1)
    target = rng.random(30) < 0.5

    def cost(x):
        return float(np.count_nonzero(x != target))

    def flip(x, rng):
        y = x.copy()
        i = rng.integers(len(y))
        y[i] = ~y[i]
        return y

    sa = simulated_annealing(np.zeros(30, dtype=bool), cost, flip, rng, steps=3000)
    assert sa.cost == 0 and len(sa.history["best"]) == 3000
    pop = [rng.random(30) < 0.5 for _ in range(20)]
    ga = genetic_algorithm(
        pop,
        cost,
        rng,
        mutate=flip,
        crossover=lambda a, b, r: np.where(r.random(30) < 0.5, a, b),
        generations=60,
    )
    assert ga.cost <= 2 and ga.history["best"][-1] <= ga.history["best"][0]
    assert all(np.diff(ga.history["best"]) <= 0)  # elitism
    with pytest.raises(ValueError, match="temperatures"):
        simulated_annealing(pop[0], cost, flip, rng, t_start=1.0, t_end=2.0)


def test_inverse_design_of_automaton_rules():
    rng = np.random.default_rng(5)
    target = (
        mg.automata.CellularAutomaton("anneal")
        .run(mg.automata.random_state((24, 24, 24), 0.5, rng), 15, rng)
        .state
    )
    goal = structure_descriptors(target)
    assert goal[0] == pytest.approx(target.mean()) and 0 < goal[2] < 1
    first = []
    fit = fit_rule(
        target,
        rng=np.random.default_rng(0),
        shape=(16, 16, 16),
        steps=15,
        population=16,
        generations=10,
        callback=lambda g, x, c: first.append(c),
    )
    assert fit.cost < first[0] and fit.cost < 0.6
    assert descriptor_cost(fit.descriptors, goal) == pytest.approx(fit.cost)
    assert fit.rule.startswith("B") and "/S" in fit.rule and 0 < fit.initial_fraction < 1
    # descriptors given directly
    direct = fit_rule(goal, rng=rng, shape=(12, 12, 12), steps=5, population=6, generations=1)
    assert len(direct.descriptors) == len(goal)


# -- models --------------------------------------------------------------------------------


def test_tile_assembly_model(tmp_path):
    model = TileAssembly(tiles="pipes", degrees=(0, 2, 3, 4), tile_cells=6)
    micro = model.generate(mg.Domain.cube(5.0), seed=2)
    assert micro.shape == (30, 30, 30) and micro.attributes["mismatches"] == 0
    assert model.expected_count(mg.Domain.cube(5.0)) == 125
    io.save_microstructure(micro, tmp_path / "t.json")
    back = io.load_microstructure(tmp_path / "t.json")
    assert np.array_equal(back.phases, micro.phases)
    random = TileAssembly(tiles="pipes", method="random", tile_cells=6)
    assert random.generate(mg.Domain.cube(5.0), seed=2).attributes["mismatches"] > 0
    uniform = TileAssembly(method="uniform", tile_cells=4)
    sample = uniform.generate(mg.Domain(size=(6.0, 2.0, 2.0)), seed=1)
    assert sample.attributes["mismatches"] == 0 and sample.attributes["tile_grid"] == [6, 2, 2]
    with pytest.raises(ValueError, match="even number"):
        TileAssembly().generate(mg.Domain.cube(3.0), seed=1)
    closed = TileAssembly(tiles="pipes", boundary=0, tile_cells=4)
    box = closed.generate(mg.Domain.cube(4.0, periodic=False), seed=3)
    assert box.phases[0].sum() == 0 and box.phases[:, :, -1].sum() == 0
    flat = TileAssembly(dimension=2, tile_cells=6).generate(mg.Domain(size=(4.0, 6.0, 1.0)), seed=1)
    assert flat.shape == (24, 36, 1)
    with pytest.raises(ValueError, match="tiles"):
        TileAssembly(tiles="hexagons")


def test_random_lattice_model(tmp_path):
    model = RandomLattice(cell="octet", coordination=8.0, perturbation=0.05, strut_radius=0.06)
    domain = mg.Domain.cube(3.0)
    micro = model.generate(domain, seed=1)
    s = micro.summary()
    assert s["coordination"] == pytest.approx(8.0, abs=0.05)
    assert s["floppy_strains"] == 0 and s["n_struts"] == s["n_segments"]
    assert model.theory()["isostatic_coordination"] == 6.0
    assert model.expected_count(domain) == pytest.approx(27.0)
    image = micro.voxelize(shape=36)
    assert 0.02 < image.data.mean() < 0.3
    io.save_microstructure(micro, tmp_path / "l.json")
    back = io.load_microstructure(tmp_path / "l.json")
    assert np.array_equal(back.voxelize(shape=36).data, image.data)
    assert back.summary()["coordination"] == s["coordination"]
    sparse = RandomLattice(cell="octet", coordination=5.0).generate(domain, seed=1)
    assert sparse.summary()["floppy_strains"] > 0
    window = model.generate(mg.Domain.cube(2.5, periodic=False), seed=1)
    assert window.voxelize(shape=20).data.any()
    with pytest.raises(ValueError, match="whole number"):
        model.generate(mg.Domain.cube(2.5), seed=1)
    with pytest.raises(ValueError, match="at most one"):
        RandomLattice(coordination=8.0, removal_fraction=0.2)
