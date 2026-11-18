"""Torquato's approaches to random heterogeneous materials: dense packings of hard
spheres grown by the Lubachevsky–Stillinger algorithm, penetrable spheres with
impenetrable cores (the cherry-pit model), and the stochastic reconstruction of
Yeong & Torquato from the two-point probability and the lineal-path function."""

import itertools
import json
import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

import morphogen as mg
from morphogen import analysis as an
from morphogen.models import Microstructure
from morphogen.models.hardcore import grown_packing

CUBE = mg.Domain.cube(1.0, periodic=True)


def _gaps(centers, radii, box):
    """The gaps between the surfaces of the neighbouring spheres (periodic box)."""
    tree = cKDTree(np.mod(centers, box), boxsize=box)
    pairs = tree.query_pairs(3.0 * radii.max(), output_type="ndarray")
    d = centers[pairs[:, 0]] - centers[pairs[:, 1]]
    d -= box * np.round(d / box)
    return np.linalg.norm(d, axis=1) - radii[pairs[:, 0]] - radii[pairs[:, 1]], pairs


@pytest.mark.slow
def test_lubachevsky_stillinger_packings():
    """Equal spheres grown to 0.6, beyond the RSA limit: no overlap, the fraction exact on
    voxels, the contacts of the radial distribution, the same packing for the same seed;
    the minimal gap; jamming near 0.64; polydisperse spheres and non-periodic samples."""
    model = mg.models.HardcoreSpheres(radius=0.06, volume_fraction=0.6,
                                      method="lubachevsky_stillinger")  # fmt: skip
    micro = model.generate(CUBE, seed=0)
    s = micro.spheres
    gaps, _ = _gaps(s.centers, s.radii, np.ones(3))
    assert gaps.min() > -1e-12 * 0.06
    assert 4 / 3 * math.pi * np.sum(s.radii**3) == pytest.approx(0.6, abs=0.6 / len(s.radii))
    assert micro.voxelize((96, 96, 96)).data.mean() == pytest.approx(0.6, abs=0.01)
    r, g = an.radial_distribution(s.centers, (1, 1, 1), r_max=0.3, bins=60)
    assert np.all(g[r < 0.118] == 0) and g[(r > 0.12) & (r < 0.13)].max() > 3.0
    again = model.generate(CUBE, seed=0).spheres
    assert np.array_equal(again.centers, s.centers)
    spaced = mg.models.HardcoreSpheres(
        radius=0.06, volume_fraction=0.5, min_gap=0.01, method="lubachevsky_stillinger"
    ).generate(CUBE, seed=1)
    gaps, _ = _gaps(spaced.spheres.centers, spaced.spheres.radii, np.ones(3))
    assert gaps.min() > 0.01 - 1e-12
    rng = np.random.default_rng(2)
    with pytest.raises(RuntimeError, match="jammed"):
        n = 200
        grown_packing(np.full(n, 1.0), np.full(3, (n * 4 / 3 * math.pi / 0.7) ** (1 / 3)), rng)
    poly = mg.models.HardcoreSpheres(
        radius={"type": "lognormal", "mean": 0.05, "std": 0.01},
        volume_fraction=0.62,
        method="lubachevsky_stillinger",
    )
    s = poly.generate(CUBE, seed=3).spheres
    gaps, _ = _gaps(s.centers, s.radii, np.ones(3))
    assert gaps.min() > -1e-12 * 0.05
    window = model.generate(mg.Domain.cube(0.6, periodic=False), seed=4)
    assert window.voxelize((60, 60, 60)).data.mean() == pytest.approx(0.6, abs=0.03)
    with pytest.raises(ValueError, match="method"):
        mg.models.HardcoreSpheres(radius=0.1, volume_fraction=0.3, method="jammed")


def test_packings_place_the_largest_spheres_first():
    """The radii of the template ``dense_packing``: in the order of the draws, the
    largest sphere of seed 2 came after 975 others and found no room left in the initial
    random sequential addition; placed first, it does (the growth rate, faster here,
    does not change the initial positions)."""
    model = mg.models.HardcoreSpheres(
        radius={"type": "lognormal", "mean": 0.5, "std": 0.1},
        volume_fraction=0.6,
        method="lubachevsky_stillinger",
        growth_rate=0.05,
    )
    s = model.generate(mg.Domain.cube(10.0, periodic=True), seed=2).spheres
    gaps, _ = _gaps(s.centers, s.radii, np.full(3, 10.0))
    assert gaps.min() > -1e-12 * 0.5
    assert 4 / 3 * math.pi * np.sum(s.radii**3) / 1000.0 == pytest.approx(0.6, abs=0.005)


def test_penetrable_spheres():
    """The cherry-pit model: hard spheres (``λ = 1``, fraction ``η``), fully penetrable
    spheres (``1 - e^{-η}``), the cores of ``λ = 0.5`` apart and the fraction between
    both; cores packed beyond the RSA limit by the Lubachevsky–Stillinger algorithm."""
    eta = 0.3
    hard = mg.models.PenetrableSpheres(radius=0.05, impenetrability=1.0, reduced_density=eta)
    assert hard.theory()["volume_fraction"] == eta
    img = hard.generate(CUBE, seed=0).voxelize((96, 96, 96))
    assert img.data.mean() == pytest.approx(eta, abs=0.01)
    soft = mg.models.PenetrableSpheres(radius=0.05, impenetrability=0.0, reduced_density=eta)
    assert soft.theory()["volume_fraction"] == pytest.approx(1 - math.exp(-eta))
    img = soft.generate(CUBE, seed=0).voxelize((96, 96, 96))
    assert img.data.mean() == pytest.approx(1 - math.exp(-eta), abs=0.015)
    cherry = mg.models.PenetrableSpheres(radius=0.05, impenetrability=0.5, reduced_density=eta)
    micro = cherry.generate(CUBE, seed=1)
    s = micro.spheres
    gaps, _ = _gaps(s.centers, 0.5 * s.radii, np.ones(3))
    assert gaps.min() >= -1e-12  # the cores do not overlap
    fraction = micro.voxelize((96, 96, 96)).data.mean()
    t = cherry.theory()
    assert t["volume_fraction_lower"] < fraction < t["volume_fraction_upper"]
    assert t["core_fraction"] == pytest.approx(eta / 8) and "volume_fraction" not in t
    assert len(s.radii) == round(eta / (4 / 3 * math.pi * 0.05**3))
    dense = mg.models.PenetrableSpheres(radius=0.05, impenetrability=0.95, reduced_density=0.55,
                                        method="lubachevsky_stillinger")  # fmt: skip
    s = dense.generate(CUBE, seed=2).spheres
    gaps, _ = _gaps(s.centers, 0.95 * s.radii, np.ones(3))
    assert gaps.min() > -1e-12 and dense.theory()["core_fraction"] > 0.46  # beyond RSA
    with pytest.raises(ValueError, match="impenetrability"):
        mg.models.PenetrableSpheres(radius=0.05, impenetrability=1.5, reduced_density=0.1)


def test_reconstruction():
    """Debye random media: the two-point probability of the target along the three axes,
    the volume fraction exact, the theory; a Boolean model: ``S_2`` alone leaves its
    lineal-path function, which its weight brings to the target; a planar lattice; the
    round trips and the errors."""
    debye = mg.models.Reconstruction(correlation_length=0.06, volume_fraction=0.3,
                                     cell_size=1 / 24, sweeps=40)  # fmt: skip
    micro = debye.generate(CUBE, seed=0)
    img = micro.voxelize((24, 24, 24))
    target, _, p = debye.targets(CUBE, seed=0)
    assert img.data.mean() == pytest.approx(0.3, abs=0.5 / 24**3) and p == 0.3
    cov = an.covariance(img, 1, max_lag=6)
    for a, name in enumerate("xyz"):
        np.testing.assert_allclose(cov.values[name], target[a], atol=1e-3)
    assert micro.attributes["energy"] < 1e-3 * micro.attributes["initial_energy"]
    t = debye.theory()
    assert t["specific_surface"] == pytest.approx(4 * 0.3 * 0.7 / 0.06)
    assert t["integral_range"] == pytest.approx(8 * math.pi * 0.06**3)
    reference = mg.models.BooleanSpheres(radius=0.08, volume_fraction=0.3).to_config()
    errors = {}
    for weight in (0.0, 1.0):
        model = mg.models.Reconstruction(reference=reference, cell_size=1 / 24, sweeps=40,
                                         lineal_path=weight)  # fmt: skip
        img = model.generate(CUBE, seed=1).voxelize((24, 24, 24))
        s2, lp, _ = model.targets(CUBE, seed=1)
        cov = an.covariance(img, 1, max_lag=6)
        line = an.lineal_path(img, 1, max_lag=6)
        assert max(np.abs(cov.values[n] - s2[a]).max() for a, n in enumerate("xyz")) < 2e-3
        errors[weight] = max(np.abs(line.values[n] - lp[a]).max() for a, n in enumerate("xyz"))
    assert errors[1.0] < 2e-3 < errors[0.0]
    planar = mg.models.Reconstruction(correlation_length=0.08, volume_fraction=0.4,
                                      cell_size=1 / 32, dimension=2, sweeps=40)  # fmt: skip
    img = planar.generate(CUBE, seed=2).voxelize((32, 32, 1))
    assert img.data.mean() == pytest.approx(0.4, abs=1 / 32**2)
    for model in (debye, planar):
        assert mg.models.model_from_config(json.loads(json.dumps(model.to_config()))) == model
    micro = planar.generate(CUBE, seed=2)
    back = Microstructure.from_dict(json.loads(json.dumps(micro.to_dict())))
    assert np.array_equal(back.voxelize((32, 32, 1)).data, img.data)
    for bad, match in [
        ({"correlation_length": 0.1}, "volume_fraction"),
        ({"correlation_length": 0.1, "volume_fraction": 0.3, "reference": reference}, "one target"),
        ({"correlation_length": 0.1, "volume_fraction": 0.3, "lineal_path": 1.0}, "lineal_path"),
        ({"reference": reference, "volume_fraction": 0.3}, "volume_fraction"),
    ]:
        with pytest.raises(ValueError, match=match):
            mg.models.Reconstruction(**bad)
    with pytest.raises(ValueError, match="max_lag"):
        mg.models.Reconstruction(
            correlation_length=0.1, volume_fraction=0.3, cell_size=0.25, max_lag=3
        ).generate(CUBE, seed=0)


def test_reconstruction_from_an_image(tmp_path):
    """The targets of an image file at its own voxels, interpolated at the lags of a
    coarser lattice of a larger domain."""
    model = mg.models.BooleanSpheres(radius=0.1, volume_fraction=0.25)
    image = model.generate(CUBE, seed=5).voxelize((48, 48, 48))
    mg.io.save_image(image, tmp_path / "sample.vti")
    rec = mg.models.Reconstruction(image=str(tmp_path / "sample.vti"), cell_size=1 / 16,
                                   sweeps=20)  # fmt: skip
    big = mg.Domain.cube(1.5, periodic=True)
    s2, _, p = rec.targets(big)
    cov = an.covariance(mg.io.load_image(tmp_path / "sample.vti"), 1)  # as the file says
    assert p == pytest.approx(image.data.mean())
    h = np.arange(s2.shape[1]) / 16
    np.testing.assert_allclose(s2[0], np.interp(h, cov.lags["x"], cov.values["x"]))
    micro = rec.generate(big, seed=0)
    assert micro.phases.shape == (24, 24, 24)
    assert micro.phases.mean() == pytest.approx(p, abs=1 / 24**3)


def test_pairs_of_the_packings_are_neighbours():
    """The cell lists of the packings find every pair closer than a diameter (brute force
    over the periodic images)."""
    micro = mg.models.HardcoreSpheres(
        radius=0.08, volume_fraction=0.5, method="lubachevsky_stillinger"
    ).generate(CUBE, seed=6)
    c, r = micro.spheres.centers, micro.spheres.radii
    worst = math.inf
    for i, j in itertools.combinations(range(len(r)), 2):
        d = c[i] - c[j]
        d -= np.round(d)
        worst = min(worst, float(np.linalg.norm(d)) - r[i] - r[j])
    assert worst > -1e-12


@pytest.mark.slow
def test_stealthy_hyperuniform_germs():
    """Stealthy hyperuniform patterns (collective coordinates): ``S(k)`` zero on the
    constrained wave vectors, ``χ`` at most the one asked; the Voronoi cells of such germs,
    far more regular than those of Poisson germs; a rectangular box; the errors."""
    from morphogen.kernels.collective import stealthy_points, stealthy_wave_vectors

    rng = np.random.default_rng(0)
    pts, k, chi = stealthy_points(150, (1.0, 1.0, 1.0), 0.3, rng)
    phase = pts @ k.T
    s = (np.cos(phase).sum(0) ** 2 + np.sin(phase).sum(0) ** 2) / len(pts)
    assert (
        s.max() < 1e-10
        and 0.25 < chi <= 0.3
        and len(k) == len(stealthy_wave_vectors(150, (1, 1, 1), 0.3))
    )
    pts, _, _ = stealthy_points(120, (1.0, 1.5, 0.8), 0.2, rng)
    assert np.all((pts >= 0) & (pts < [1.0, 1.5, 0.8]))
    volumes = {}
    for germs in ("poisson", "stealthy"):
        tess = mg.models.Voronoi(n_cells=200, germs=germs).generate(CUBE, seed=1)
        v = tess.cell_complex().cell_volumes(tess.n_cells)
        volumes[germs] = v.std() / v.mean()
    assert volumes["poisson"] == pytest.approx(math.sqrt(0.1797), abs=0.06)
    assert volumes["stealthy"] < 0.25 * volumes["poisson"]
    t = mg.models.Voronoi(n_cells=200, germs="stealthy").theory(CUBE)
    assert t == {"intensity": 200.0, "mean_cell_volume": 1 / 200}
    for bad, match in [({"germs": "lattice"}, "germs"), ({"stealthiness": 0.6}, "stealthiness")]:
        with pytest.raises(ValueError, match=match):
            mg.models.Voronoi(n_cells=10, **bad)
    with pytest.raises(ValueError, match="chi"):
        stealthy_points(10, (1, 1, 1), 0.7, rng)
