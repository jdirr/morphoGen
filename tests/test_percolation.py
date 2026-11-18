"""Percolation thresholds of families of microstructures (analysis.percolation)."""

import json

import numpy as np
import pytest
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as graph_components

from morphogen import cli
from morphogen.analysis import (
    burn,
    burn_along,
    directional_percolation,
    measure,
    percolation,
    percolation_anisotropy,
    percolation_exponent,
    percolation_scaling,
    percolation_threshold,
    rigidity_threshold,
)
from morphogen.analysis.basic import as_array
from morphogen.analysis.connectivity import _structure, wrapping
from morphogen.analysis.percolation import (
    _bears,
    _bisect,
    _bisection,
    _percolates,
    _tally,
    family_from_config,
)
from morphogen.config import ConfigError, model_from_config
from morphogen.domain import Domain
from morphogen.image import VoxelImage
from morphogen.models import BooleanSpheres


def site(n):
    """Site percolation of the simple cubic lattice: each voxel of an n^3 image in the phase
    with probability ``fraction``, independently."""
    return lambda fraction, rng: rng.random((n, n, n)) < fraction


def slab(fraction, rng, n=12):
    """A slab normal to z, floor(fraction n) voxels thick: it joins the faces normal to x and
    y from a fraction of 1/n, those normal to z when it fills the box; its complement joins
    those normal to z below 1/n, those normal to x and y below 1."""
    image = np.zeros((n, n, n), dtype=bool)
    image[:, :, : int(fraction * n)] = True
    return image


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize(
    ("phase", "axis", "step", "rising"),
    [
        (1, "x", 1 / 12, True),
        (1, "any", 1 / 12, True),
        (1, "z", 1.0, True),
        (1, "all", 1.0, True),
        (0, "z", 1 / 12, False),
        (0, "y", 1.0, False),
    ],
)
def test_bisection_brackets_the_step_of_a_slab(phase, axis, step, rising, periodic):
    r = percolation_threshold(
        slab, phase=phase, axis=axis, periodic=periodic, realizations=3, tol=1e-3, seed=0
    )
    lo, hi = r.bracket
    assert lo < step <= hi
    assert hi - lo <= 1e-3
    assert r.threshold == 0.5 * (lo + hi)
    assert r.realizations == 3
    assert len(r.fractions) == 2 + 10  # [0, 1] halved down to 2**-10 < 1e-3
    assert np.all(np.diff(r.fractions) > 0)
    expected = r.fractions >= step if rising else r.fractions < step
    np.testing.assert_array_equal(r.probabilities, expected.astype(float))


def _flags(image, connectivity, periodic):
    return tuple(_percolates(image, 1, axis, connectivity, periodic) for axis in "xyz")


def _stairs(n=12):
    """A staircase that joins the faces normal to x and does not wrap around the box."""
    stairs = np.zeros((n, n, 3), dtype=bool)
    stairs[: n // 2 + 1, 0, 1] = True
    stairs[n // 2, : n // 2 + 1, 1] = True
    stairs[n // 2 :, n // 2, 1] = True
    return stairs


def _helix(n=12):
    """A step along y for each along x, through the faces: a helix that wraps around the box
    along x and y and joins no pair of faces."""
    helix = np.zeros((n, n, 3), dtype=bool)
    for i in range(n):
        helix[i, (n // 2 + i) % n, 1] = helix[i, (n // 2 + i + 1) % n, 1] = True
    return helix


def test_wrapping_around_the_periodic_box_is_not_spanning_it():
    n = 12
    stairs = _stairs(n)
    assert _flags(stairs, 6, False) == (True, False, False)
    assert _flags(stairs, 6, True) == (False, False, False)
    helix = _helix(n)
    assert _flags(helix, 6, False) == (False, False, False)
    assert _flags(helix, 6, True) == (True, True, False)
    diagonal = np.zeros((n, n, 3), dtype=bool)  # edge neighbours, through one face or two
    diagonal[np.arange(n), (n // 2 + np.arange(n)) % n, 1] = True
    assert _flags(diagonal, 6, True) == (False, False, False)
    assert _flags(diagonal, 18, False) == (False, False, False)
    assert _flags(diagonal, 18, True) == (True, True, False)
    diagonal = np.zeros((n, n, n), dtype=bool)
    diagonal[np.arange(n), np.arange(n), np.arange(n)] = True  # corner neighbours
    assert _flags(diagonal, 18, True) == (False, False, False)
    assert _flags(diagonal, 26, False) == (True, True, True)
    assert _flags(diagonal, 26, True) == (True, True, True)
    assert _flags(np.zeros((4, 4, 4), dtype=bool), 26, True) == (False, False, False)


@pytest.mark.parametrize(
    ("connectivity", "periodic", "lower", "upper", "reference", "tolerance"),
    [
        (6, False, 0.2, 0.45, 0.3116, 0.01),
        (6, True, 0.2, 0.45, 0.3116, 0.01),
        (26, False, 0.05, 0.2, 0.0976, 0.008),
    ],
)
def test_site_percolation_of_the_simple_cubic_lattice(
    connectivity, periodic, lower, upper, reference, tolerance
):
    """Thresholds 0.3116 through faces (Lorenz & Ziff 1998), 0.0976 through faces, edges and
    corners (Kurzawski & Malarz 2012). On 48^3 voxels the crossing of 1/2 along one axis lies
    above p_c by about 0.003 (0.0025 through 26 neighbours: the crossing probability at p_c
    is below 1/2) and the median of 9 realizations spreads by about 0.003 (0.002) from seed
    to seed, single realizations by 0.007: both shrink as L^(-1/nu); the tolerances are
    about twice their sum."""
    r = percolation_threshold(
        site(48),
        lower,
        upper,
        connectivity=connectivity,
        periodic=periodic,
        tol=0.002,
        seed=0,
    )
    assert abs(r.threshold - reference) < tolerance
    assert np.all(np.diff(r.probabilities) >= 0)  # common random numbers: monotone


def test_boolean_spheres_near_their_continuum_threshold():
    """Overlapping spheres percolate from a volume fraction of 0.2895 (Rintoul & Torquato
    1997). On 64^3 voxels of a periodic box 16 radii across, spheres 4 voxels in radius touch
    when closer than about a voxel, which lowers the threshold by a few hundredths (0.27
    here, 0.28 with 6 voxels per radius), and the median of 9 realizations spreads by about
    0.01."""
    domain = Domain(size=16.0, periodic=True)

    def spheres(fraction, rng):
        model = BooleanSpheres(radius=1.0, volume_fraction=fraction)
        return model.generate(domain, seed=rng).voxelize(64)

    r = percolation_threshold(spheres, 0.15, 0.45, seed=0)
    assert abs(r.threshold - 0.2895) < 0.04


def test_reproducible_from_the_seed():
    def run(seed):
        return percolation_threshold(site(16), 0.1, 0.6, realizations=3, tol=0.02, seed=seed)

    a, b = run(7), run(np.random.SeedSequence(7))
    assert a.bracket == b.bracket
    np.testing.assert_array_equal(a.probabilities, b.probabilities)
    c, d = run(np.random.default_rng(7)), run(np.random.default_rng(7))
    assert c.bracket == d.bracket
    np.testing.assert_array_equal(c.probabilities, d.probabilities)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lower": 0.5, "upper": 0.5}, "bounds"),
        ({"lower": 0.6, "upper": 0.4}, "bounds"),
        ({"lower": -0.1}, "bounds"),
        ({"upper": 1.5}, "bounds"),
        ({"tol": 0.0}, "tol"),
        ({"realizations": 0}, "realizations"),
        ({"realizations": 2.5}, "realizations"),
        ({"axis": "w"}, "axis"),
        ({"connectivity": 8}, "connectivity"),
    ],
)
def test_bad_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        percolation_threshold(site(8), **kwargs)


@pytest.mark.parametrize(("lower", "upper"), [(0.6, 0.9), (0.0, 0.1)])
def test_no_crossing_between_the_bounds(lower, upper):
    with pytest.raises(ValueError, match="does not cross 1/2"):
        percolation_threshold(site(16), lower, upper, realizations=3, seed=0)


#: A Boolean model of spheres on a coarse grid: 3 voxels per radius, a box 8 radii across
SPHERES = {
    "model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.3},
    "domain": {"size": 8.0, "periodic": True},
    "voxels": {"shape": 24},
    "seed": 3,
}


def test_family_of_a_configuration():
    """The model of a configuration with its parameter set to the fraction, voxelized on the
    grid of the configuration; a parameter the model does not have, or no grid, refused."""
    generate = family_from_config(SPHERES)
    sparse, dense = (generate(f, np.random.default_rng(0)) for f in (0.1, 0.5))
    assert sparse.shape == (24, 24, 24) and sparse.periodic
    assert np.mean(as_array(sparse) == 1) < 0.2 < 0.4 < np.mean(as_array(dense) == 1)
    with pytest.raises(ConfigError, match=r"no parameter 'porosity' to vary .*volume_fraction"):
        family_from_config(SPHERES, "porosity")
    with pytest.raises(ConfigError, match="voxel grid"):
        family_from_config({k: v for k, v in SPHERES.items() if k != "voxels"})


def test_command_line(tmp_path, capsys):
    """``morphogen percolation``: the threshold of the spheres of a configuration, its
    bracket and the probabilities tried, again from the seed of the configuration; its
    errors reported as those of the other commands."""
    path, nogrid = tmp_path / "spheres.yaml", tmp_path / "nogrid.yaml"
    path.write_text(yaml.safe_dump(SPHERES), encoding="utf-8")
    nogrid.write_text(yaml.safe_dump({**SPHERES, "voxels": None}), encoding="utf-8")
    args = ["percolation", str(path), "--lower", "0.1", "--upper", "0.6", "--realizations",
            "3", "--tol", "0.05"]  # fmt: skip
    assert cli.main([*args, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    lo, hi = out["bracket"]
    assert lo < out["threshold"] < hi and hi - lo <= 0.05 and out["realizations"] == 3
    assert out["fractions"] == sorted(out["fractions"]) and out["fractions"][0] == 0.1
    assert out["probabilities"][0] < 0.5 <= out["probabilities"][-1]
    assert cli.main(args) == 0  # the seed of the configuration: the same threshold
    assert f"volume_fraction = {out['threshold']:.4g} in" in capsys.readouterr().out
    for argv, message in [
        ([*args, "--parameter", "porosity"], "no parameter 'porosity'"),
        ([*args[:2], "--lower", "0.7", "--upper", "0.9", "--realizations", "2"], "cross 1/2"),
        (["percolation", str(nogrid)], "voxel grid"),
    ]:
        assert cli.main(argv) == 2
        assert message in capsys.readouterr().err


def _joined(mask, connectivity, wrap=()):
    """Whether a component of the graph of the phase voxels and their neighbours, wrapping
    around the box along the axes ``wrap``, joins the two faces normal to each axis: scipy's
    components of the graph built edge by edge, an argument independent of the fire and of
    the labels."""
    shape, index = mask.shape, np.arange(mask.size).reshape(mask.shape)
    rows, cols = [], []
    for d in np.argwhere(_structure(connectivity)) - 1:
        moved = np.indices(shape) + d[:, None, None, None]
        keep = mask.copy()
        for t in set(range(3)) - set(wrap):
            keep &= (moved[t] >= 0) & (moved[t] < shape[t])
        target = index[tuple(m % n for m, n in zip(moved, shape, strict=True))]
        keep &= mask.ravel()[target]
        rows.append(index[keep])
        cols.append(target[keep])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(mask.size, mask.size))
    labels = graph_components(graph, directed=False)[1].reshape(shape)
    joined = []
    for a in range(3):
        first, last = (np.take(labels, f, a)[np.take(mask, f, a)] for f in (0, -1))
        joined.append(bool(np.intersect1d(first, last).size))
    return joined


def test_the_fire_decides_as_the_components():
    """On 200 small site lattices near their thresholds (through 6, 18 or 26 neighbours:
    0.3116, 0.1373, 0.0976), the fire reaches the opposite face exactly when a component
    joins the two faces: the labels on open boundaries, the components of the graph
    wrapping across the transverse faces on periodic ones, which the wrapping of the labels
    implies; stopped there, it times its arrival as the burning of the whole clusters."""
    rng, seen = np.random.default_rng(0), set()
    for _ in range(200):
        connectivity = int(rng.choice([6, 18, 26]))
        p = {6: 0.3116, 18: 0.1373, 26: 0.0976}[connectivity] * rng.uniform(0.7, 1.6)
        image = rng.random(tuple(rng.integers(3, 10, size=3))) < p
        opened, wraps = _joined(image, connectivity), _flags(image, connectivity, True)
        for axis, name in enumerate("xyz"):
            assert opened[axis] == _percolates(image, 1, name, connectivity, False)
            spans = _joined(image, connectivity, {0, 1, 2} - {axis})[axis]
            assert spans or not wraps[axis]
            for periodic, expected in [(False, opened[axis]), (True, spans)]:
                kwargs = {"axis": name, "connectivity": connectivity, "periodic": periodic}
                fire, whole = burn(image, **kwargs), burn(image, **kwargs, full=True)
                assert fire.percolates == whole.percolates == expected
                assert fire.time == whole.time
                np.testing.assert_array_equal(fire.front, whole.front[: len(fire.front)])
                assert len(fire.front) == (fire.time + 1 if expected else len(whole.front))
                seen.add(expected)
    assert seen == {False, True}


def _bfs(mask, axis, connectivity, periodic):
    """The burning times by a breadth-first search in pure Python (-1: unburnt)."""
    shape = mask.shape
    offsets = [d for d in (np.argwhere(_structure(connectivity)) - 1).tolist() if any(d)]
    times = np.full(shape, -1)
    front = [v for v in np.ndindex(shape) if v[axis] == 0 and mask[v]]
    for v in front:
        times[v] = 0
    step = 0
    while front:
        step, burning, front = step + 1, front, []
        for v in burning:
            for d in offsets:
                w = [c + e for c, e in zip(v, d, strict=True)]
                if all(0 <= w[t] < shape[t] or (periodic and t != axis) for t in range(3)):
                    w = tuple(c % n for c, n in zip(w, shape, strict=True))
                    if mask[w] and times[w] < 0:
                        times[w] = step
                        front.append(w)
    return times


@pytest.mark.parametrize("connectivity", [6, 18, 26])
def test_burning_times_of_the_whole_clusters(connectivity):
    """The burning times of the whole clusters lit on a face are those of a breadth-first
    search, open or periodic across the axis, and the front is their histogram."""
    rng = np.random.default_rng(connectivity)
    for periodic in (False, True):
        for axis, name in enumerate("xyz"):
            image = rng.random((5, 6, 7)) < {6: 0.45, 18: 0.25, 26: 0.15}[connectivity]
            fire = burn(image, axis=name, connectivity=connectivity, periodic=periodic, full=True)
            np.testing.assert_array_equal(fire.times, _bfs(image, axis, connectivity, periodic))
            np.testing.assert_array_equal(fire.front, np.bincount(fire.times[fire.times >= 0]))


def test_burning_times_of_known_paths():
    """A straight channel n voxels long burns through in n - 1 steps, a voxel a step; a
    staircase through faces in 2n - 3 steps, through edges in n - 1; a diagonal through
    corners in n - 1 along each axis, not at all through edges; a corridor across the box
    in n + m - 2 steps, in n through its periodic faces."""
    n, m, i = 8, 10, np.arange(8)
    channel = np.zeros((n, 5, 5), dtype=bool)
    channel[:, 2, 2] = True
    for connectivity in (6, 18, 26):
        fire = burn(channel, connectivity=connectivity, full=True)
        assert fire.percolates and fire.time == n - 1
        np.testing.assert_array_equal(fire.front, np.ones(n))
        np.testing.assert_array_equal(fire.times[:, 2, 2], i)
    fire = burn(channel, axis="y")
    assert not fire.percolates and fire.time is None and len(fire.front) == 0
    stairs = np.zeros((n, n, 3), dtype=bool)
    stairs[i, i, 1] = stairs[i[1:], i[:-1], 1] = True  # (i, i) and (i + 1, i)
    assert burn(stairs).time == 2 * n - 3
    assert burn(stairs, connectivity=18).time == n - 1
    diagonal = np.zeros((n, n, n), dtype=bool)
    diagonal[i, i, i] = True
    for name in "xyz":
        assert burn(diagonal, axis=name, connectivity=26).time == n - 1
        assert not burn(diagonal, axis=name, connectivity=18).percolates
    corridor = np.zeros((n, m, 3), dtype=bool)  # along x at y = 0, along y, along x at y = m - 1
    corridor[: n // 2 + 1, 0, 1] = corridor[n // 2, :, 1] = corridor[n // 2 :, -1, 1] = True
    assert burn(corridor).time == n + m - 2
    assert burn(corridor, periodic=True).time == n
    for kwargs, match in [({"axis": "w"}, "axis"), ({"connectivity": 8}, "connectivity")]:
        with pytest.raises(ValueError, match=match):
            burn(channel, **kwargs)
    with pytest.raises(ValueError, match="3-D"):
        burn(channel[0])
    with pytest.raises(ValueError, match="method"):
        percolation_threshold(site(8), method="flood")


@pytest.mark.parametrize("periodic", [False, True])
def test_site_percolation_by_the_fire(periodic):
    """The fire decides as the labels on open boundaries, hence the same bisection and the
    estimate of 0.3116 on 48^3 voxels; on periodic images it percolates whenever they wrap,
    hence an estimate at or below theirs."""
    kwargs = {"connectivity": 6, "periodic": periodic, "tol": 0.002, "seed": 0}
    fire = percolation_threshold(site(48), 0.2, 0.45, method="fire", **kwargs)
    labels = percolation_threshold(site(48), 0.2, 0.45, **kwargs)
    assert abs(fire.threshold - 0.3116) < 0.01
    if periodic:
        assert fire.threshold <= labels.threshold
    else:
        assert fire.bracket == labels.bracket
        np.testing.assert_array_equal(fire.probabilities, labels.probabilities)


def test_command_line_by_the_fire(tmp_path, capsys):
    """``morphogen percolation --method fire``: the threshold by the forest fire."""
    path = tmp_path / "spheres.yaml"
    path.write_text(yaml.safe_dump(SPHERES), encoding="utf-8")
    args = ["percolation", str(path), "--lower", "0.1", "--upper", "0.6", "--realizations",
            "3", "--tol", "0.05", "--method", "fire"]  # fmt: skip
    assert cli.main([*args, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    lo, hi = out["bracket"]
    assert out["method"] == "fire" and lo < out["threshold"] < hi and hi - lo <= 0.05
    assert cli.main(args) == 0
    assert "3 realizations, forest fire)" in capsys.readouterr().out


def test_wrapping_flags_of_the_measures():
    """The measure ``connectivity`` of a periodic image adds to the joining of its faces its
    wrapping around the box along each axis: the helix wraps along x and y and joins no
    faces, the staircase joins those normal to x and does not wrap; an image that is not
    periodic has no wrapping flags."""

    def flags(mask, periodic=True):
        out = measure(VoxelImage(mask.astype(np.uint8), periodic=periodic), ["connectivity"])
        return {k: v for k, v in out.items() if k.startswith(("percolates_", "wraps_"))}

    assert flags(_helix()) == {"percolates_x": 0.0, "percolates_y": 0.0, "percolates_z": 0.0,
                               "wraps_x": 1.0, "wraps_y": 1.0, "wraps_z": 0.0}  # fmt: skip
    assert flags(_stairs()) == {"percolates_x": 1.0, "percolates_y": 0.0, "percolates_z": 0.0,
                                "wraps_x": 0.0, "wraps_y": 0.0, "wraps_z": 0.0}  # fmt: skip
    assert flags(_helix(), periodic=False) == {f"percolates_{a}": 0.0 for a in "xyz"}
    assert wrapping(_helix()) == {"x": True, "y": True, "z": False}
    assert wrapping(_helix(), 0) == {"x": True, "y": True, "z": True}  # the matrix around it


#: The threshold of the sites of the simple cubic lattice (Lorenz & Ziff 1998)
P_C = 0.311608


def sites(fraction, rng, size):
    """Site percolation of the simple cubic lattice in a box ``size`` sites across."""
    return rng.random((size, size, size)) < fraction


def slabs(fraction, rng, size):
    """The slab of :func:`slab` in a box ``size`` voxels across: it joins the faces normal to
    x from a fraction of exactly 1/size."""
    return slab(fraction, rng, size)


def test_finite_size_scaling_of_site_percolation():
    """The faces normal to the three axes of an open cube of sites joined (by the fire,
    through faces) from 16 to 48 sites across: over 16 seeds with 99 realizations, the
    crossings of 1/2 lie above p_c by 0.0080 +/- 0.0008 (0.0066 at least) on 48^3 sites,
    their fit extrapolates to p_c within 0.0015 (0.0030 at most) and its standard error is
    about as much (0.0015): the tolerances are three times these scatters, and the
    extrapolation is markedly closer to p_c than the largest box (in 2.4 s)."""
    sizes = [48, 16, 32, 24, 16]
    r = percolation_scaling(sites, sizes, 0.2, 0.45, axis="all", method="fire",
                            realizations=99, tol=0.002, seed=0)  # fmt: skip
    np.testing.assert_array_equal(r.sizes, [16, 24, 32, 48])
    assert r.thresholds[-1] - P_C > 0.0055
    assert abs(r.threshold - P_C) < 0.0045
    assert 0.0005 < r.stderr < 0.0045
    assert r.amplitude > 0 and r.nu == 0.8765 and r.nu_stderr is None
    np.testing.assert_array_equal(r.thresholds, [t.threshold for t in r.results])
    np.testing.assert_array_equal(r.brackets, [t.bracket for t in r.results])
    fit = r.threshold + r.amplitude * r.sizes ** (-1 / r.nu)
    np.testing.assert_allclose(r.residuals, r.thresholds - fit, atol=1e-15)
    assert {t.realizations for t in r.results} == {99}


def test_finite_size_scaling_of_a_slab():
    """The slabs cross 1/2 at exactly 1/L: the fit of p_c + a L^(-1/nu) finds p_c = 0 and
    a = 1 with nu = 1, and nu = 1 itself from 0.8765, within the tolerance of the bisection;
    each size seeded alike whatever the others."""
    sizes = [8, 12, 16, 24, 32]
    r = percolation_scaling(slabs, sizes, nu=1.0, realizations=1, tol=1e-5, seed=0)
    lo, hi = r.brackets.T
    assert np.all(lo < 1 / r.sizes) and np.all(1 / r.sizes <= hi) and np.all(hi - lo <= 1e-5)
    assert abs(r.threshold) < 1e-4 and abs(r.amplitude - 1) < 1e-2 and 0 < r.stderr < 1e-4
    fitted = percolation_scaling(slabs, sizes, fit_nu=True, realizations=1, tol=1e-5, seed=0)
    assert abs(fitted.nu - 1) < 1e-2 and 0 < fitted.nu_stderr < 1e-2
    assert abs(fitted.threshold) < 1e-3
    a = percolation_scaling(sites, [8, 12, 16], 0.05, 0.7, realizations=3, tol=0.05, seed=4)
    b = percolation_scaling(sites, [6, 8, 12], 0.05, 0.7, realizations=3, tol=0.05, seed=4)
    np.testing.assert_array_equal(a.brackets[:2], b.brackets[1:])


@pytest.mark.parametrize(
    ("sizes", "kwargs", "match"),
    [
        ([16, 24], {}, "at least 3"),
        ([16, 24, 24, 32], {"fit_nu": True}, "at least 4"),
        ([16, 24.5, 32], {}, "positive integers"),
        ([0, 16, 24], {}, "positive integers"),
        (["16", "24", "32"], {}, "positive integers"),
        ([16, 24, 32], {"nu": 0.0}, "nu"),
    ],
)
def test_scaling_bad_arguments(sizes, kwargs, match):
    with pytest.raises(ValueError, match=match):
        percolation_scaling(sites, sizes, **kwargs)


def test_sized_family_of_a_configuration():
    """With ``sized``, boxes at the voxel size of the configuration (1/3 here), the domain
    scaled to them: 24 voxels are the box of the configuration itself."""
    plain, sized = family_from_config(SPHERES), family_from_config(SPHERES, sized=True)
    small = sized(0.3, np.random.default_rng(0), 12)
    assert small.shape == (12, 12, 12) and small.periodic
    np.testing.assert_allclose(small.spacing, 1 / 3)
    same = plain(0.3, np.random.default_rng(1)), sized(0.3, np.random.default_rng(1), 24)
    np.testing.assert_array_equal(as_array(same[0]), as_array(same[1]))
    with pytest.raises(ValueError, match="size"):
        sized(0.3, np.random.default_rng(0), 2.5)


def test_command_line_scaling(tmp_path, capsys):
    """``morphogen percolation --sizes``: the thresholds of boxes 12, 18 and 24 voxels across
    at the voxel size of the configuration, extrapolated; two sizes refused."""
    path = tmp_path / "spheres.yaml"
    path.write_text(yaml.safe_dump(SPHERES), encoding="utf-8")
    args = ["percolation", str(path), "--lower", "0.1", "--upper", "0.6", "--realizations",
            "3", "--tol", "0.05", "--method", "fire", "--sizes", "12,18,24"]  # fmt: skip
    assert cli.main([*args, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["sizes"] == [12, 18, 24] and out["nu"] == 0.8765 and out["nu_stderr"] is None
    for t, (lo, hi) in zip(out["thresholds"], out["brackets"], strict=True):
        assert lo < t < hi and hi - lo <= 0.05
    assert out["stderr"] > 0 and len(out["residuals"]) == 3 and out["realizations"] == 3
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert "extrapolated from 3 box sizes" in text and f"{out['threshold']:.4g} +/- " in text
    assert len(text.splitlines()) == 2 + 3
    assert cli.main([*args[:-2], "--sizes", "12,24"]) == 2
    assert "at least 3" in capsys.readouterr().err


# -- 2-D: planar images --------------------------------------------------------------------

PC_SQUARE = 0.59274621  # sites of the square lattice, 4 neighbours (Newman & Ziff 2000)


def squares(n):
    """Site percolation of the square lattice: an n x n image, one voxel along z."""
    return lambda fraction, rng: rng.random((n, n, 1)) < fraction


def test_planar_images_percolate_in_their_plane():
    """An image one voxel thick percolates along x or y, never trivially along z: a channel
    along x crosses for x and any axis, not for y and all, by the labels and the fire, open
    or periodic; 4 neighbours are 6 there, 8 are 18 and 26 (the same burning times); 4 and
    8 refused in 3-D, the axis z and 2-D arrays refused; no z in the measure connectivity."""
    image = np.zeros((9, 7, 1), dtype=bool)
    image[:, 3] = True
    for method in ("labels", "fire"):
        for periodic in (False, True):
            flags = [_percolates(image, 1, a, 4, periodic, method) for a in ("x", "y")]
            flags += [_percolates(image, 1, a, 4, periodic, method) for a in ("any", "all")]
            assert flags == [True, False, True, False]
    u = np.random.default_rng(0).random((24, 24, 1))
    four = [burn(u < 0.6, connectivity=c, full=True).times for c in (4, 6)]
    eight = [burn(u < 0.45, axis="y", connectivity=c, full=True).times for c in (8, 18, 26)]
    np.testing.assert_array_equal(four[0], four[1])
    np.testing.assert_array_equal(eight[0], eight[1])
    np.testing.assert_array_equal(eight[0], eight[2])
    with pytest.raises(ValueError, match="x or y for a planar image"):
        burn(image, axis="z")
    with pytest.raises(ValueError, match="for a planar image"):
        _percolates(image, 1, "z", 6, False)
    with pytest.raises(ValueError, match="a planar image"):
        burn(np.ones((4, 4, 4), dtype=bool), connectivity=8)
    with pytest.raises(ValueError, match=r"\(nx, ny, 1\)"):
        burn(np.ones((4, 4), dtype=bool))
    out = measure(VoxelImage(image, periodic=True), ["connectivity"])
    assert out["percolates_x"] == out["wraps_x"] == 1 and out["percolates_y"] == 0
    assert "percolates_z" not in out and "wraps_z" not in out


def test_eight_neighbours_are_the_dual_of_four():
    """On a rectangle of the square lattice the phase joins the faces normal to x through 8
    neighbours exactly when the other phase does not join those normal to y through 4 (the
    matching lattices of Sykes & Essam 1964, whence p_c = 1 - 0.592746 through 8): on 300
    random rectangles, by the fire and the labels."""
    rng = np.random.default_rng(1)
    for _ in range(300):
        n, m = rng.integers(1, 14, size=2)
        image = rng.random((n, m, 1)) < rng.uniform(0.3, 0.7)
        crossed = burn(image, axis="x", connectivity=8).percolates
        assert crossed != burn(~image, axis="y", connectivity=4).percolates
        assert crossed == _percolates(image, 1, "x", 8, False)


@pytest.mark.parametrize(
    ("connectivity", "lower", "upper", "reference"),
    [(4, 0.5, 0.7, PC_SQUARE), (8, 0.3, 0.5, 1 - PC_SQUARE)],
)
def test_site_percolation_of_the_square_lattice(connectivity, lower, upper, reference):
    """Thresholds 0.592746 through 4 neighbours (Newman & Ziff 2000) and 0.407254 through 8.
    A square is crossed at p_c with the probability 1/2 (by duality), so that its
    1/2-crossing lies off p_c by much less than its scatter: over four seeds, 99
    realizations of 128^2 sites give p_c - 0.0004 +/- 0.0020 through 4 neighbours and
    p_c + 0.0002 +/- 0.0021 through 8 (p_c + 0.0004 +/- 0.0012 and -0.0004 +/- 0.0018 on
    256^2 sites); the tolerance is three times the scatter."""
    r = percolation_threshold(squares(128), lower, upper, connectivity=connectivity,
                              method="fire", realizations=99, tol=0.001, seed=0)  # fmt: skip
    assert abs(r.threshold - reference) < 0.006


def test_finite_size_scaling_in_two_dimensions():
    """Rectangles twice as long along x as across, joined the long way (with a probability
    below 1/2 at p_c), from 32 to 256 sites across: nu = 4/3 for planar images, the
    1/2-crossings above p_c by 0.006 on the largest (a = 0.3 to 0.5), the extrapolation
    within 0.002 of p_c over six seeds and its standard error about as much; the tolerances
    are three times these."""

    def rectangles(fraction, rng, n):
        return rng.random((2 * n, n, 1)) < fraction

    r = percolation_scaling(rectangles, [32, 64, 128, 256], 0.5, 0.75, method="fire",
                            realizations=49, tol=0.001, seed=0)  # fmt: skip
    assert r.nu == 4 / 3 and r.amplitude > 0.15
    assert r.thresholds[-1] - PC_SQUARE > 0.002
    assert abs(r.threshold - PC_SQUARE) < 0.006 and 0.0003 < r.stderr < 0.006


def test_sized_family_of_a_planar_configuration():
    """A grid one voxel thick keeps it: the boxes grow in the plane."""
    config = {"model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.5},
              "domain": {"size": [8.0, 8.0, 0.25], "periodic": False},
              "voxels": {"shape": [32, 32, 1]}}  # fmt: skip
    image = family_from_config(config, sized=True)(0.5, np.random.default_rng(0), 16)
    assert image.shape == (16, 16, 1)
    np.testing.assert_allclose(image.spacing, 0.25)


# -- along directions ------------------------------------------------------------------------


def test_cores_along_directions():
    """The core of a box along a direction: on a full image the fire crosses it in one step
    fewer than its 22 layers along the axes (32 / sqrt 2 voxels long), more along a diagonal
    through 4 neighbours and fewer through 8, and the geometric tortuosity of a full image
    is 1 along every direction. A direction and its opposite cross alike, as the image
    reversed (the fire lit on the other cap); a channel along x crosses along x only."""
    full = np.ones((32, 32, 1), dtype=bool)
    assert [burn_along(full, d).time for d in ((1, 0), (0, 1), (-1, 0), (0, -1))] == [21] * 4
    assert burn_along(full, (1, 1)).time > 21 > burn_along(full, (1, 1), connectivity=8).time
    plane = directional_percolation(full, 8)
    assert plane.percolates.all() and np.allclose(plane.tortuosity, 1)
    np.testing.assert_array_equal(plane.distances, plane.straight)
    cube = directional_percolation(np.ones((16, 16, 16), dtype=bool), connectivity=26)
    assert len(cube.directions) == 13 and np.allclose(cube.tortuosity, 1)
    rng = np.random.default_rng(2)
    for _ in range(20):
        image, n = rng.random((20, 20, 20)) < 0.32, rng.normal(size=3)
        reversed_ = image[::-1, ::-1, ::-1]
        fires = [burn_along(image, n), burn_along(image, -n), burn_along(reversed_, n)]
        assert len({(f.percolates, f.time) for f in fires}) == 1
    channel = np.zeros((32, 32, 1), dtype=bool)
    channel[:, 16] = True
    crossing = directional_percolation(channel, "axes")
    assert crossing.percolates.tolist() == [True, False]
    assert crossing.tortuosity[0] == 1 and np.isinf(crossing.tortuosity[1])
    np.testing.assert_allclose(crossing.directions, [[1, 0, 0], [0, 1, 0]])
    for direction, match in (((1, 0, 1), "plane"), ((0, 0), "zero length"), ("x", "vector")):
        with pytest.raises(ValueError, match=match):
            burn_along(full, direction)
    with pytest.raises(ValueError, match="too few"):
        burn_along(np.ones((1, 50, 1), dtype=bool), (0, 1))
    with pytest.raises(ValueError, match="directions"):
        directional_percolation(full, "diagonals")


def test_directional_percolation_measure():
    """Along the 4 lattice directions of a planar image (10, 01, 11, 1-1) and the 13 of a
    3-D one, the tortuosity of those crossed at least 1."""
    rng = np.random.default_rng(3)
    out = measure(VoxelImage(rng.random((32, 32, 1)) < 0.75), ["directional_percolation"])
    assert {f"percolates_{k}" for k in ("10", "01", "11", "1-1")} == {
        k for k in out if k.startswith("percolates_")
    }
    out = measure(VoxelImage(rng.random((16, 16, 16)) < 0.5), ["directional_percolation"])
    names = [k[len("percolates_") :] for k in out if k.startswith("percolates_")]
    assert len(names) == 13 and "-111" in names and "1-10" in names
    assert all(out[f"tortuosity_{k}"] >= 1 for k in names if out[f"percolates_{k}"])


def test_anisotropy_of_isotropic_families():
    """Site percolation of the square (64^2 sites, 25 realizations) and of the cubic (32^3
    sites, 15) lattice along their lattice directions: over three seeds the paired
    deviations lie within 2.3 standard errors (normal over 51 of them), the ratio of the
    extreme directions within 2.5 of 1. Each realization's threshold along a direction
    brackets its own: the fire crosses above it, not below, and does not depend on the
    other directions bisected (common random numbers)."""
    r = percolation_anisotropy(squares(64), 0.4, 0.8, realizations=25, tol=0.002, seed=0)
    assert np.all(np.abs(r.deviations) < 3.5 * r.deviation_stderr)
    assert abs(r.ratio - 1) < 3.5 * r.ratio_stderr + 1e-12
    np.testing.assert_allclose(r.directions[2], [2**-0.5, 2**-0.5, 0])
    assert np.all(np.abs(r.thresholds - PC_SQUARE) < 4 * r.stderr)
    cubic = percolation_anisotropy(lambda f, rng: rng.random((32, 32, 32)) < f, 0.2, 0.5,
                                   realizations=15, tol=0.004, seed=0)  # fmt: skip
    assert cubic.samples.shape == (15, 13) and np.all(np.isfinite(cubic.eigenvalue_stderr))
    assert np.all(np.abs(cubic.deviations) < 3.5 * cubic.deviation_stderr)
    seeds = _bisection(0.4, 0.8, 0.002, 25, 0)
    for i in (0, 7):
        u = np.random.default_rng(seeds[i]).random((64, 64, 1))
        for n, t in zip(r.directions, r.samples[i], strict=True):
            assert not burn_along(u < t - 0.001, n).percolates
            assert burn_along(u < t + 0.001, n).percolates
    alone = percolation_anisotropy(squares(64), 0.4, 0.8, directions=r.directions[[3, 1]],
                                   realizations=25, tol=0.002, seed=0)  # fmt: skip
    np.testing.assert_array_equal(alone.samples, r.samples[:, [3, 1]])


def test_anisotropy_of_a_stretched_lattice():
    """Sites of 2 x 1 voxels (64^2 voxels, 25 realizations): the clusters elongated along x
    cross the core sooner along x and later along y, the diagonals alike in between (the
    lattice symmetric about x), by 4 to 6 paired standard errors along x and 3 to 4.5
    along y over three seeds, 0.2 to 1 along the diagonals; the tensor smallest along x."""

    def blocks(f, rng):
        return np.repeat(rng.random((32, 64, 1)) < f, 2, axis=0)

    r = percolation_anisotropy(blocks, 0.4, 0.8, realizations=25, tol=0.002, seed=0)
    z = r.deviations / r.deviation_stderr
    assert z[0] < -3 and z[1] > 2 and np.all(np.abs(z[2:]) < 2)
    assert r.thresholds[0] < r.thresholds[2:].min() and r.thresholds[2:].max() < r.thresholds[1]
    assert r.ratio > 1 + 3 * r.ratio_stderr
    assert abs(r.principal[0, 0]) > 0.99 and r.eigenvalues[0] < r.eigenvalues[1]


def test_anisotropy_of_layers():
    """Layers of sites normal to z separated by walls of the other phase: they percolate
    along x, y and the diagonals of the layers, never along z (inf, the tensor undefined),
    nor at 45 degrees from the layers (the cores' caps out of reach of a layer)."""

    def layers(f, rng):
        image = rng.random((24, 24, 24)) < f
        image[:, :, 1::2] = False
        return image

    r = percolation_anisotropy(layers, 0.3, 0.95, realizations=5, tol=0.01, seed=0)
    in_plane = np.abs(r.directions[:, 2]) < 1e-12
    assert np.all(np.isfinite(r.thresholds[in_plane])) and np.all(r.thresholds[in_plane] < 0.65)
    assert np.isposinf(r.thresholds[2]) and np.isposinf(r.thresholds[[5, 6, 7, 8]]).all()
    assert np.isinf(r.ratio) and np.isnan(r.tensor).all() and np.isnan(r.eigenvalues).all()


def test_anisotropy_of_aligned_fibres():
    """Boolean fibres aligned along x, 4 radii long in a box of 8 (32^3 voxels, 9
    realizations): they percolate along x from a fraction about three times lower than
    across (0.11 to 0.14 against 0.37 to 0.43 over two seeds), the tensor smallest along x."""
    domain = Domain(size=8.0)
    config = {"type": "boolean_cylinders", "radius": 0.5, "length": 4.0,
              "orientation": {"type": "aligned", "direction": [1, 0, 0]}}  # fmt: skip

    def fibres(fraction, rng):
        model = model_from_config({**config, "volume_fraction": fraction})
        return model.generate(domain, seed=rng).voxelize(32)

    r = percolation_anisotropy(fibres, 0.005, 0.6, directions="axes", realizations=9, tol=0.01,
                               seed=0)  # fmt: skip
    assert r.thresholds[0] < 0.5 * r.thresholds[1:].min() and r.ratio > 2
    assert abs(r.principal[0, 0]) > 0.99 and r.eigenvalues[0] == pytest.approx(r.tensor[0, 0])


def test_anisotropy_bad_arguments():
    """A family that never crosses between the bounds, images of two grids and a direction
    out of the plane of a planar image are refused; a falling family (the other phase) is
    the rising one reversed."""
    with pytest.raises(ValueError, match="does not change"):
        percolation_anisotropy(squares(16), 0.0, 0.1, realizations=2)
    shapes = iter([(16, 16, 1), (12, 12, 1)] * 4)
    with pytest.raises(ValueError, match="one grid"):
        percolation_anisotropy(lambda f, rng: rng.random(next(shapes)) < f, realizations=2)
    with pytest.raises(ValueError, match="plane"):
        percolation_anisotropy(squares(16), directions=[(1, 0, 1)], realizations=2)
    falling = percolation_anisotropy(squares(32), 0.2, 0.6, phase=0, realizations=6, tol=0.01,
                                     seed=5)  # fmt: skip
    rising = percolation_anisotropy(lambda f, rng: 1 - rng.random((32, 32, 1)) < f, 0.4, 0.8,
                                    realizations=6, tol=0.01, seed=5)  # fmt: skip
    np.testing.assert_allclose(falling.samples, 1 - rising.samples, atol=1e-9)
    np.testing.assert_allclose(falling.thresholds, 1 - rising.thresholds, atol=1e-9)


def test_command_line_anisotropy(tmp_path, capsys):
    """``morphogen percolation --directions``: the thresholds along the axes, their tensor
    and their table; a number of directions below 1 refused."""
    path = tmp_path / "spheres.yaml"
    path.write_text(yaml.safe_dump(SPHERES), encoding="utf-8")
    args = ["percolation", str(path), "--lower", "0.1", "--upper", "0.6", "--realizations",
            "3", "--tol", "0.05", "--directions", "axes"]  # fmt: skip
    assert cli.main([*args, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["thresholds"]) == 3 and len(out["samples"]) == 3 and out["realizations"] == 3
    assert np.array(out["tensor"]).shape == (3, 3) and out["ratio"] >= 1
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert "along 3 directions" in text and "threshold tensor:" in text
    assert len(text.splitlines()) == 3 + 3
    with pytest.raises(SystemExit):
        cli.main([*args[:-1], "0"])
    assert "expected axes, lattice or a number" in capsys.readouterr().err


# -- critical exponents of the effective properties ---------------------------------------


def channel(fraction, rng, size, planar=True):
    """A straight channel of one row of voxels along x, whatever the fraction: its
    conductivity along x is exactly 1/L in 2-D (1/L^2 in 3-D), its modulus of a uniaxial
    strain E/(1 - nu^2)/L (one voxel along z, which the periodic cell holds)."""
    image = np.zeros((size, size, 1 if planar else size), dtype=bool)
    image[:, 0, 0] = True
    return image


def layers(fraction, rng, size):
    """Layers along x filling the fraction of the rows: conductivity along x exactly the
    fraction (the arithmetic mean) when it is a multiple of 1/size."""
    image = np.zeros((size, size, 1), dtype=bool)
    image[:, : round(fraction * size), 0] = True
    return image


def test_exponent_of_exact_power_laws():
    """A channel conducts as L^-1 in 2-D (t/nu = 1, t = 4/3 with the nu of 2-D) and as
    L^-2 in 3-D (t/nu = 2), its realizations alike (no statistical error, an exact fit);
    its uniaxial-strain modulus as L^-1; layers conduct as the fraction, the sweep of
    exponent 1 from a threshold at 0, where they vanish (no exponent: NaN)."""
    r = percolation_exponent(channel, [8, 16, 32], 0.5, contrast=0.0, realizations=2,
                             device="cpu", seed=0)  # fmt: skip
    assert r.nu == 4 / 3 and abs(r.ratio - 1) < 1e-10 and abs(r.exponent - 4 / 3) < 1e-10
    np.testing.assert_allclose(r.means, 1 / r.sizes, rtol=1e-10)
    assert r.stderr < 1e-8 and r.threshold_error == 0 and np.all(r.mean_stderr == 0)
    assert abs(r.amplitude - 1) < 1e-8 and np.all(np.abs(r.residuals) < 1e-10)
    assert r.samples.shape == (3, 2, 1) and r.sweep.size == 0 and r.sweep_exponents.size == 0
    assert r.fractions.tolist() == [0.5] and r.physics == "thermal" and r.realizations == 2

    def tube(f, rng, size):
        return channel(f, rng, size, planar=False)

    cubic = percolation_exponent(tube, [4, 6, 8], 0.5, nu=1.0, contrast=0.0, realizations=2,
                                 device="cpu")  # fmt: skip
    assert abs(cubic.exponent - 2) < 1e-8 and cubic.nu == 1.0
    elastic = percolation_exponent(channel, [8, 16, 32], 0.5, physics="elastic", nu=1.0,
                                   contrast=0.0, realizations=2, device="cpu")  # fmt: skip
    assert abs(elastic.exponent - 1) < 1e-8
    np.testing.assert_allclose(elastic.means, 1 / (1 - 0.3**2) / elastic.sizes, rtol=1e-8)
    sweep = percolation_exponent(layers, [8, 16, 32], 0.0, fractions=[0.25, 0.5, 0.75],
                                 contrast=0.0, realizations=2, device="cpu")  # fmt: skip
    assert np.isnan(sweep.exponent) and np.all(sweep.means == 0)
    np.testing.assert_allclose(sweep.sweep_exponents, 1, atol=1e-8)
    assert np.all(sweep.sweep_stderr < 1e-8) and np.all(sweep.sweep_threshold_error == 0)
    np.testing.assert_allclose(sweep.values[:, 1:], [[0.25, 0.5, 0.75]] * 3, atol=1e-10)


def test_exponent_of_site_percolation_in_two_dimensions():
    """Sites of the square lattice through 8 neighbours, as the nodes of the voxels join
    them, at p_c = 1 - 0.59274621 (16, 24 and 32 sites, 12 realizations): t within its error
    of 1.30, an error near 1 in boxes so small (0.7 to 1.2 over four seeds), the threshold's
    error from the same images shifted by 0.002. The images that percolate conduct more at
    the higher fraction, which adds sites to them (common random numbers); those that do not
    conduct as the contrast, to the tolerance of the solver."""
    r = percolation_exponent(squares_sized, [16, 24, 32], 1 - PC_SQUARE, threshold_stderr=0.002,
                             realizations=12, device="cpu", seed=0)  # fmt: skip
    error = np.hypot(r.stderr, r.threshold_error)
    assert 0.2 < r.stderr < 2 and r.threshold_error > 0
    assert abs(r.exponent - 1.30) < 3 * error
    assert r.fractions.size == 3 and np.all(np.diff(r.fractions) > 0)
    np.testing.assert_array_equal(r.means, r.values[:, 1])
    assert r.iterations > 0 and r.residual <= 1e-8 and r.contrast == 1e-8
    big = r.samples[:, :, 0] > 1e-4
    assert big.any() and (~big).any() and np.all(r.samples[:, :, 0][~big] < 1e-5)
    assert np.all(r.samples[:, :, 0][big] <= r.samples[:, :, 2][big] * (1 + 1e-6))


def squares_sized(fraction, rng, size):
    """Site percolation of the square lattice, ``size`` sites across."""
    return rng.random((size, size, 1)) < fraction


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        (([8, 16], 0.5), {}, "at least 3"),
        (([8, 16, 32], 0.5), {"physics": "plastic"}, "physics"),
        (([8, 16, 32], 0.5), {"axis": "any"}, "axis"),
        (([8, 16, 32], 0.5), {"axis": "z"}, "planar"),
        (([8, 16, 32], 0.5), {"fractions": [0.6, 0.7]}, "at least 3"),
        (([8, 16, 32], 0.5), {"fractions": [0.5, 0.6, 0.7]}, "above the threshold"),
        (([8, 16, 32], 0.5), {"threshold_stderr": 0.2, "fractions": [0.6, 0.7, 0.8]}, "above"),
        (([8, 16, 32], 0.5), {"threshold_stderr": -0.1}, "threshold_stderr"),
        (([8, 16, 32], np.nan), {}, "threshold"),
        (([8, 16, 32], 0.5), {"contrast": 1.0}, "contrast"),
        (([8, 16, 32], 0.5), {"realizations": 1}, "realizations"),
        (([8, 16, 32], 0.5), {"nu": 0.0}, "nu"),
        (([8, 16, 32], 0.5), {"tol": 0.0}, "tol"),
        (([8, 16, 32], 0.5), {"solver": "amg"}, "solver"),
    ],
)
def test_exponent_bad_arguments(args, kwargs, match):
    with pytest.raises(ValueError, match=match):
        percolation_exponent(channel, *args, device="cpu", **kwargs)


def test_command_line_exponent(tmp_path, capsys):
    """``morphogen percolation --exponent``: the threshold extrapolated first (through 26
    neighbours, wrapping) with its standard error, or given, then the exponent over the
    same sizes, with a sweep; the sizes required, one axis."""
    path = tmp_path / "spheres.yaml"
    path.write_text(yaml.safe_dump(SPHERES), encoding="utf-8")
    args = ["percolation", str(path), "--lower", "0.05", "--upper", "0.6", "--realizations",
            "3", "--tol", "0.05", "--sizes", "9,12,15", "--exponent", "thermal",
            "--device", "cpu"]  # fmt: skip
    assert cli.main([*args, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["physics"] == "thermal" and out["sizes"] == [9, 12, 15] and out["nu"] == 0.8765
    assert out["threshold_sizes"] == [9, 12, 15] and 0 < out["threshold_stderr"] < 0.1
    assert len(out["means"]) == 3 and len(out["fractions"]) == 3 and out["realizations"] == 3
    assert out["error"] >= out["stderr"] and out["contrast"] == 1e-8
    given = [*args, "--sizes", "7,9,11", "--threshold", "0.25", "--fractions", "0.4,0.5,0.6",
             "--exponent", "elastic"]  # fmt: skip
    assert cli.main(given) == 0
    text = capsys.readouterr().out
    assert "critical exponent of the uniaxial-strain modulus" in text and "f = " in text
    assert "(given)" in text and "sweep over volume_fraction = 0.4, 0.5, 0.6" in text
    assert len(text.splitlines()) == 3 + 3 + 1 + 3
    assert cli.main([*args[:-6], "--exponent", "thermal"]) == 2
    assert "--sizes" in capsys.readouterr().err
    assert cli.main([*args, "--axis", "any"]) == 2
    assert "one axis" in capsys.readouterr().err


# -- the rigidity threshold: where the elastic moduli vanish ---------------------------------


def hinged(fraction, rng, size):
    """Chains of pixels joined at their corners, which the trilinear voxels of the solver
    pin together, wrapping around the box along x: from a fraction of 0.3 a zigzag whose
    pins do not line up, a mechanism (its moduli vanish, with void around), from 0.6 a
    diagonal, a straight chain of pins that bears a uniaxial strain along x (C_xxxx = 0.2747
    / L exactly); no two pixels of a chain share an edge."""
    image = np.zeros((size, size, 1), dtype=bool)
    x = np.arange(size)
    if fraction >= 0.3:
        image[x, np.array([0, 1, 2, 1])[x % 4] + size // 2, 0] = True
    if fraction >= 0.6:
        image[x, x, 0] = True
    return image


def test_rigidity_threshold_of_hinged_chains():
    """The moduli vanish below 0.6 though the pixels join the faces through 8 neighbours from
    0.3: the rigidity threshold is 0.6 at every size, the threshold of the connectivity 0.3;
    each realization is solved twice per size (the zigzag at round-off, the diagonal bearing
    as 1/L), its other images decided by their connectivity or by inclusion; the elastic
    exponent of the diagonal is 1 with nu = 1; the bounds, which the bracket does not reach,
    are not tried (7 fractions per size). With a contrast the zigzag takes it times an
    enhancement, which the cut must exceed."""
    sizes = [8, 12, 16]
    r = rigidity_threshold(hinged, sizes, 0.1, 0.9, realizations=2, tol=0.01, device="cpu",
                           seed=0)  # fmt: skip
    for one in r.results:
        lo, hi = one.bracket
        assert lo < 0.6 <= hi and hi - lo <= 0.01 and one.probabilities.max() == 1.0
    assert abs(r.threshold - 0.6) < 0.01 and r.stderr < 0.01 and r.nu == 4 / 3
    assert r.solutions == 2 * 3 * 2 and r.decided == 2 * 3 * 7 - r.solutions
    assert abs(r.floppy) < 1e-12 and abs(r.rigid * 16 / 0.2747 - 1) < 1e-3
    assert r.contrast == 0 and r.cut == 1e-8 and 0 < r.iterations and r.residual <= 1e-8
    for size, bearing, moduli in zip(r.sizes, r.bearing, r.moduli, strict=True):
        assert bearing.shape == moduli.shape == (7, 2) and np.all(np.isfinite(moduli).sum(0) == 2)
        np.testing.assert_allclose(moduli[np.isfinite(moduli) & bearing] * size, 0.27474,
                                   rtol=1e-4)  # fmt: skip
    connected = percolation_scaling(hinged, sizes, 0.1, 0.9, connectivity=8, periodic=True,
                                    realizations=2, tol=0.01)  # fmt: skip
    assert np.all(np.abs(connected.thresholds - 0.3) < 0.01)
    e = percolation_exponent(hinged, sizes, 0.7, physics="elastic", nu=1.0, contrast=0.0,
                             realizations=2, device="cpu")  # fmt: skip
    assert abs(e.exponent - 1) < 1e-8 and np.allclose(e.means * e.sizes, 0.27474, rtol=1e-4)
    soft = rigidity_threshold(hinged, sizes, 0.1, 0.9, contrast=1e-8, cut=1e-6, realizations=2,
                              tol=0.01, device="cpu", seed=0)  # fmt: skip
    assert abs(soft.threshold - r.threshold) < 1e-12 and 1e-8 < soft.floppy < 1e-7


def test_rigidity_threshold_between_the_connectivities():
    """Sites of the square lattice: the realizations, those of percolation_scaling from the
    same seed, bear from fractions between those from which they join the faces through 8
    and through 4 neighbours (wrapping), so that the rigidity thresholds of the boxes lie
    between theirs (to the tolerance); the shares that bear are those of the decisions, and
    the moduli solved those of realizations joined through 8 neighbours but not 4. The
    threshold and its standard error give the elastic exponent at it."""
    sizes, kwargs = [8, 12, 16], {"realizations": 5, "tol": 0.01, "seed": 4}
    r = rigidity_threshold(squares_sized, sizes, 0.3, 0.7, device="cpu", **kwargs)
    joined, welded = (percolation_scaling(squares_sized, sizes, 0.2, 0.8, connectivity=k,
                                          periodic=True, **kwargs) for k in (8, 4))  # fmt: skip
    assert np.all(joined.thresholds <= r.thresholds + 0.01)
    assert np.all(r.thresholds <= welded.thresholds + 0.01)
    assert np.all(welded.thresholds - joined.thresholds > 0.1)
    for one, bearing, moduli in zip(r.results, r.bearing, r.moduli, strict=True):
        np.testing.assert_array_equal(one.probabilities, bearing.mean(axis=1))
        assert np.all(np.diff(bearing.astype(int), axis=0) >= 0)  # growing with the fraction
        assert np.all(moduli[np.isfinite(moduli) & bearing] > r.cut)
    assert sum(np.isfinite(m).sum() for m in r.moduli) == r.solutions > 0
    assert r.floppy < 1e-10 < r.cut < r.rigid
    e = percolation_exponent(squares_sized, sizes, r.threshold, threshold_stderr=r.stderr,
                             physics="elastic", realizations=4, device="cpu", seed=1)  # fmt: skip
    assert np.isfinite(e.exponent) and e.stderr > 0 and e.threshold_error > 0
    np.testing.assert_allclose(e.fractions, [r.threshold - r.stderr, r.threshold,
                                             r.threshold + r.stderr])  # fmt: skip
    assert r.solver == e.solver == "fft"


def test_rigidity_and_exponent_by_the_assembled_voxels():
    """``solver="fe"``: the matrix of the voxels assembled and factorized decides as the FFT
    solver (the realizations of the boxes and the thresholds the same, the moduli of those
    that bear to the tolerance), in a few iterations (1 to 4 by the machine and its
    libraries: at most 10 here) where the FFT solver takes hundreds
    near the rigidity threshold, the floppy moduli at the round-off; the exponent at the
    threshold the same, the other phase void or at the contrast 1e-8 (all its voxels
    assembled, a few more iterations); the chains of pins rigid from 0.6, the diagonal
    bearing as 0.2747 / L."""
    sizes, kwargs = [8, 12, 16], {"realizations": 5, "tol": 0.01, "seed": 4, "device": "cpu"}
    fft = rigidity_threshold(squares_sized, sizes, 0.3, 0.7, **kwargs)
    fe = rigidity_threshold(squares_sized, sizes, 0.3, 0.7, solver="fe", **kwargs)
    np.testing.assert_array_equal(fe.thresholds, fft.thresholds)
    assert fe.threshold == fft.threshold and fe.solutions == fft.solutions
    assert fe.decided == fft.decided
    for a, b, bears in zip(fft.moduli, fe.moduli, fe.bearing, strict=True):
        np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
        solved = np.isfinite(a) & bears
        np.testing.assert_allclose(b[solved], a[solved], rtol=1e-6)
    for a, b in zip(fft.bearing, fe.bearing, strict=True):
        np.testing.assert_array_equal(a, b)
    assert fe.solver == "fe" and fe.iterations <= 10 and fft.iterations > 100
    assert abs(fe.floppy) < 1e-11 and fe.rigid == pytest.approx(fft.rigid, rel=1e-6)
    for contrast in (0.0, 1e-8):
        a, b = (percolation_exponent(squares_sized, sizes, fe.threshold, physics="elastic",
                                     contrast=contrast, realizations=3, tol=1e-10,
                                     device="cpu", solver=solver, seed=1)
                for solver in ("fft", "fe"))  # fmt: skip
        np.testing.assert_allclose(b.samples, a.samples, rtol=1e-6, atol=1e-10)
        assert b.solver == "fe" and b.iterations <= 10 < a.iterations
    r = rigidity_threshold(hinged, sizes, 0.1, 0.9, realizations=2, tol=0.01, device="cpu",
                           solver="fe", seed=0)  # fmt: skip
    assert abs(r.threshold - 0.6) < 0.01 and abs(r.rigid * 16 / 0.2747 - 1) < 1e-3
    assert abs(r.floppy) < 1e-11 and r.iterations <= 10


def _chain(shape, steps):
    """Voxels joined along x, each the last moved by ``steps[x % len(steps)]`` (y, z) as x
    grows by 1, around the periodic box."""
    image = np.zeros(shape, dtype=bool)
    y = z = 0
    for x in range(shape[0]):
        image[x, y % shape[1], z % shape[2]] = True
        dy, dz = steps[x % len(steps)]
        y, z = y + dy, z + dz
    return image


@pytest.mark.parametrize(
    ("image", "bears"),
    [
        (_chain((12, 12, 1), [(1, 0)]), True),  # pins in a line (the diagonal)
        (_chain((12, 12, 1), [(1, 0), (1, 0), (-1, 0), (-1, 0)]), False),  # a zigzag of pins
        (_chain((12, 12, 1), [(0, 0)] * 5 + [(1, 0)] + [(0, 0)] * 5 + [(-1, 0)]), True),
        (_chain((12, 12, 1), [(0, 0)] * 2 + [(1, 0)] + [(0, 0)] * 2 + [(-1, 0)]), True),
        (_chain((8, 8, 8), [(1, 1)]), True),  # ball joints in a line
        (_chain((8, 8, 8), [(1, 1), (1, 1), (-1, -1), (-1, -1)]), False),  # their zigzag
        (_chain((8, 8, 2), [(1, 0), (1, 0), (-1, 0), (-1, 0)]), False),  # a zigzag of hinges
        (_chain((8, 8, 2), [(0, 0)] * 3 + [(1, 0)] + [(0, 0)] * 3 + [(-1, 0)]), True),
    ],
)
def test_the_rigidity_decisions_of_chains(image, bears):
    """Chains of voxels wrapping around the box along x, none joined to the next by a face,
    which the solver decides: pins (2-D corners), ball joints (3-D corners) or hinges (3-D
    edges) in a line along the axis bear, bars of pixels joined end to end at their corners
    included (two hinged bars, the pins on a line), the zigzags of pins, ball joints and
    hinges are mechanisms; the energies of those that bear agree with their stress averages
    and those of the mechanisms vanish to the round-off, by both solvers."""
    for solver in ("fe", "fft"):
        tally = _tally()
        decided, modulus = _bears(image, np.ones(3), "x", [], 0.0, 1e-8, 1e-10, 20000, "cpu",
                                  tally, solver)  # fmt: skip
        assert decided == bears and tally["solutions"] == 1 and tally["unresolved"] == 0
        assert modulus > 1e-4 if bears else modulus < 1e-15


def test_the_rigidity_decision_of_unresolved_moduli(monkeypatch):
    """A modulus is the energy of its solution, an upper bound: none at or under the cut; above
    it the stress average decides within half of it (the energy resolved); else the energy is
    unresolved, decided by two solutions with the element matrices perturbed at the round-off
    (the voxels factorized: bearing when the energies agree within 1 %, as in the floppy boxes
    of 256^2 pixels whose energies scatter by tens of percent) or by the stress average (the
    FFT solver)."""
    image = _chain((12, 12, 1), [(1, 0), (1, 0), (-1, 0), (-1, 0)])
    floor = percolation._FLOOR
    plain, perturbed = [(1e-8, None)], [(1e-8, None), (1e-8, 1), (1e-8, 2)]
    twice = [*perturbed, (floor, None), (floor, 1), (floor, 2)]
    cases = [  # energies (unperturbed, two perturbations), stress average, solver, bears,
        # unresolved solutions, the tolerances and perturbations of the solutions: an energy
        # of the voxels factorized that bears under 1000 times the cut is solved again at
        # the floor of the residual and decided by that solution (the same here)
        ((8e-9, 8e-9, 8e-9), 7.97e-9, "fe", False, 0, plain),
        ((1.024e-8, 1.024e-8, 1.024e-8), 7.3e-9, "fe", True, 0, [*plain, (floor, None)]),
        ((1.96e-7, 1.5e-7, 2.4e-7), -6.0e-7, "fe", False, 1, perturbed),
        ((1.0e-7, 1.003e-7, 0.998e-7), -6.0e-7, "fe", True, 2, twice),
        ((1.96e-7, 1.5e-7, 2.4e-7), -6.0e-7, "fft", False, 1, plain),
        ((1.96e-7, 1.5e-7, 2.4e-7), 4.0e-8, "fft", True, 1, plain),
    ]
    for energies, average, solver, bears, unresolved, solved in cases:
        calls = []

        def effective(*args, rounding=None, energies=energies, average=average, calls=calls):
            calls.append((args[5], rounding))
            return energies[rounding or 0], 2, 1e-11, average

        monkeypatch.setattr(percolation, "_effective", effective)
        tally = _tally()
        decided, modulus = _bears(image, np.ones(3), "x", [], 0.0, 1e-8, 1e-8, None, "cpu",
                                  tally, solver)  # fmt: skip
        assert decided == bears and modulus == energies[0]
        assert tally["unresolved"] == unresolved and tally["solutions"] == 1
        assert calls == solved and tally["confirmed"] == (bears and solver == "fe")
        assert tally["floppy" if modulus <= 1e-8 else "rigid"] == (
            modulus if bears or modulus <= 1e-8 else np.inf
        )


def test_the_rigidity_decision_confirmed_at_the_floor():
    """Sites of the cubic lattice near their rigidity threshold (32³ voxels, the voxels
    factorized): at the tolerance 1e-8 a floppy image keeps an energy above the cut, its soft
    modes unresolved, whose stress average agrees with it; solved again at the floor of the
    residual its energy falls under the cut, and it does not bear. The default tolerance of
    the voxels factorized is 1e-10, that of the FFT solver 1e-8."""
    seed = np.random.SeedSequence(20261102, spawn_key=(3, 32, 1))
    mask = np.random.default_rng(seed).random((32, 32, 32)) < 0.1305
    value, _, _, average = percolation._effective(mask, np.ones(3), 0, "elastic", 0.0, 1e-8,
                                                  2600, "cpu", "fe")  # fmt: skip
    assert 1e-12 < value < 1e-9 and abs(average - value) <= 0.5 * value
    tally = _tally()
    decided, modulus = _bears(mask, np.ones(3), "x", [], 0.0, 1e-12, 1e-8, None, "cpu", tally,
                              "fe")  # fmt: skip
    assert not decided and modulus <= 1e-12 and tally["confirmed"] == 1
    assert percolation.SOLVER_TOLS == {"fft": 1e-8, "fe": 1e-10}


def test_the_bisection_with_its_bounds_last():
    """With the direction of the probability known, the bounds are tried last and only those
    that the bracket reaches: the same bracket as the bounds tried first, in fewer
    fractions; refused when a bound reached lies on the side of the threshold."""
    for rising in (True, False):

        def probability(f, rising=rising):
            return float((f >= 0.362) == rising)

        first = _bisect(probability, 0.1, 0.9, 0.01, 1)
        last = _bisect(probability, 0.1, 0.9, 0.01, 1, rising=rising)
        assert last.bracket == first.bracket and len(last.fractions) == len(first.fractions) - 2
        assert 0.1 not in last.fractions and 0.9 not in last.fractions
        near = _bisect(probability, 0.36, 0.9, 0.01, 1, rising=rising)
        assert near.bracket[0] == 0.36 and 0.36 in near.fractions and 0.9 not in near.fractions
    with pytest.raises(ValueError, match=r"does not cross 1/2 between 0\.1 and 0\.9"):
        _bisect(lambda f: 1.0, 0.1, 0.9, 0.01, 1, "bearing", rising=True)


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        (([8, 16], 0.1, 0.9), {}, "at least 3"),
        (([8, 12, 16], 0.1, 0.9), {"fit_nu": True}, "at least 4"),
        (([8, 12, 16], 0.1, 0.9), {"axis": "any"}, "axis"),
        (([8, 12, 16], 0.1, 0.9), {"axis": "z"}, "planar"),
        (([8, 12, 16], 0.1, 0.9), {"cut": 0.0}, "cut"),
        (([8, 12, 16], 0.1, 0.9), {"contrast": 1e-6}, "cut"),
        (([8, 12, 16], 0.1, 0.9), {"contrast": 1.0}, "contrast"),
        (([8, 12, 16], 0.1, 0.9), {"solver_tol": 0.0}, "solver_tol"),
        (([8, 12, 16], 0.1, 0.9), {"maxiter": 0}, "maxiter"),
        (([8, 12, 16], 0.1, 0.9), {"nu": 0.0}, "nu"),
        (([8, 12, 16], 0.1, 0.9), {"solver": "direct"}, "solver"),
        (([8, 12, 16], 0.9, 0.1), {}, "bounds"),
        (([8, 12, 16], 0.65, 0.9), {}, "bearing probability does not cross 1/2"),
    ],
)
def test_rigidity_bad_arguments(args, kwargs, match):
    with pytest.raises(ValueError, match=match):
        rigidity_threshold(hinged, *args, realizations=1, device="cpu", **kwargs)


DISCS = {  # the sections of spheres of 2 pixels of radius, in boxes the solver makes periodic
    "model": {"type": "boolean_spheres", "radius": 0.5, "volume_fraction": 0.5},
    "domain": {"size": [8.0, 8.0, 0.25], "periodic": False},
    "voxels": {"shape": [32, 32, 1]},
    "seed": 4,  # 3 realizations of 12 to 20 pixels: the same thresholds, within the bounds
}


def test_command_line_rigidity(tmp_path, capsys):
    """``morphogen percolation --rigidity``: the threshold of the moduli of the discs
    extrapolated from the sizes; with ``--exponent elastic`` and no threshold, the exponent
    at it; refused without sizes, along any axis or for the conductivity."""
    path = tmp_path / "discs.yaml"
    path.write_text(yaml.safe_dump(DISCS), encoding="utf-8")
    args = ["percolation", str(path), "--lower", "0.2", "--upper", "0.95", "--realizations",
            "3", "--tol", "0.05", "--sizes", "12,16,20", "--device", "cpu"]  # fmt: skip
    assert cli.main([*args, "--rigidity", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["criterion"] == "rigidity" and out["sizes"] == [12, 16, 20] and out["cut"] == 1e-8
    assert 0.2 < out["threshold"] < 0.95 and out["nu"] == 4 / 3 and out["solutions"] > 0
    assert out["cut"] < out["rigid"] and out["unresolved"] == 0
    assert len(out["probabilities"]) == len(out["fractions"]) == 3
    assert out["floppy"] == "nan" or out["floppy"] < out["cut"]  # none solved: nan
    assert cli.main([*args, "--rigidity"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("rigidity threshold of phase 1 of boolean_spheres along x")
    assert "FFT solutions" in text and "0 energies above the cut unresolved" in text
    assert len(text.splitlines()) == 2 + 1 + 3
    assert cli.main([*args, "--exponent", "elastic", "--json"]) == 0
    exponent = json.loads(capsys.readouterr().out)
    assert exponent["threshold_criterion"] == "rigidity"
    assert exponent["threshold"] == out["threshold"] and exponent["physics"] == "elastic"
    assert cli.main([*args[:-4], "--rigidity"]) == 2
    assert "--sizes" in capsys.readouterr().err
    assert cli.main([*args, "--rigidity", "--axis", "any"]) == 2
    assert "one axis" in capsys.readouterr().err
    assert cli.main([*args, "--rigidity", "--exponent", "thermal"]) == 2
    assert "for --exponent elastic" in capsys.readouterr().err
    assert out["solver"] == "fft"
    assert cli.main([*args, "--rigidity", "--solver", "fe", "--json"]) == 0
    fe = json.loads(capsys.readouterr().out)
    assert fe["solver"] == "fe" and fe["thresholds"] == out["thresholds"]
    assert fe["solutions"] == out["solutions"] and fe["iterations"] <= 10 < out["iterations"]
    assert cli.main([*args, "--rigidity", "--solver", "fe"]) == 0
    assert "voxel FE solutions" in capsys.readouterr().out
    assert cli.main([*args, "--exponent", "elastic", "--solver", "fe", "--contrast", "0",
                     "--json"]) == 0  # fmt: skip
    exponent = json.loads(capsys.readouterr().out)
    assert exponent["solver"] == "fe" and exponent["threshold"] == out["threshold"]
    assert exponent["iterations"] <= 10 and exponent["contrast"] == 0
