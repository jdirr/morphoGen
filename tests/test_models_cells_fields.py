"""Tessellations, Gaussian excursions, TPMS and strut lattices."""

import math

import numpy as np
import pytest
from scipy.spatial import Voronoi as QhullVoronoi

import morphogen as mg
from morphogen import analysis, theory
from morphogen.models import (
    TPMS,
    GaussianExcursion,
    Laguerre,
    StrutLattice,
    Tessellation,
    Voronoi,
)
from morphogen.models.lattice import lattice_fraction, unit_cell_struts

# -- tessellations ------------------------------------------------------------------------


def _brute_force_cells(points, germs, weights=None, box=None):
    diff = points[:, None] - germs[None]
    if box is not None:
        diff -= box * np.round(diff / box)
    power = np.sum(diff**2, axis=2)
    if weights is not None:
        power = power - weights[None]
    return np.argmin(power, axis=1)


def test_voronoi_and_laguerre_cells_match_brute_force(rng):
    pts = rng.random((3000, 3))
    v = Voronoi(n_cells=200).generate(mg.Domain.cube(1.0), seed=1)
    assert np.array_equal(v.cell_of(pts), _brute_force_cells(pts, v.germs, box=1.0))
    lag = Laguerre(n_cells=150, radius={"type": "uniform", "low": 0.02, "high": 0.1}).generate(
        mg.Domain.cube(1.0), seed=2
    )
    assert np.array_equal(lag.cell_of(pts), _brute_force_cells(pts, lag.germs, lag.weights, 1.0))
    nv = Voronoi(intensity=300.0).generate(mg.Domain.cube(1.0, periodic=False), seed=3)
    assert np.array_equal(nv.cell_of(pts), _brute_force_cells(pts, nv.germs))
    # equal weights give the Voronoi tessellation
    same = Tessellation(v.domain, v.germs, np.full(v.n_cells, 0.01))
    assert np.array_equal(same.cell_of(pts), v.cell_of(pts))


def test_poisson_voronoi_cell_statistics():
    model = Voronoi(n_cells=2000)
    domain = mg.Domain.cube(1.0)
    micro = model.generate(domain, seed=11)
    labels = micro.labels(shape=128)
    vol = analysis.cell_volumes(labels, micro.n_cells)
    assert vol.sum() == pytest.approx(1.0)
    cv2 = vol.var() / vol.mean() ** 2
    assert cv2 == pytest.approx(theory.VORONOI_VOLUME_CV2, abs=0.025)
    th = model.theory(domain)
    assert th["integral_range"] == pytest.approx(1.179 / 2000, rel=1e-3)
    boundary = analysis.boundary_density(labels)
    assert boundary == pytest.approx(th["boundary_surface_density"], rel=0.06)
    # topology of the germs: mean number of faces of Poisson-Voronoi cells
    shifts = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T
    allpts = np.concatenate([micro.germs + s for s in shifts])
    qv = QhullVoronoi(allpts)
    n = micro.n_cells
    centre = int(np.flatnonzero((shifts == 0).all(axis=1))[0])
    faces = np.bincount(qv.ridge_points.ravel(), minlength=len(allpts))
    mean_faces = faces[centre * n : (centre + 1) * n].mean()
    assert mean_faces == pytest.approx(theory.VORONOI_MEAN_FACES, abs=0.25)


def test_tessellation_phases_and_orientations():
    model = Voronoi(n_cells=3000, phase_fractions=(0.3, 0.7), crystal_orientations=True)
    micro = model.generate(mg.Domain.cube(1.0), seed=5)
    assert micro.cell_phases.mean() == pytest.approx(0.7, abs=4 * math.sqrt(0.21 / 3000))
    e = micro.orientations
    assert e.shape == (3000, 3) and e[:, 1].max() <= 180 and e[:, [0, 2]].max() < 360
    img = micro.voxelize(shape=48)
    frac = micro.voxelize(shape=48, supersampling=2)
    assert frac.data.dtype == np.float32
    assert abs(frac.data.mean() - img.data.mean()) < 0.01
    table = micro.cell_table()
    assert set(table) >= {"id", "x", "y", "z", "phase", "phi1", "Phi", "phi2"}
    neighbours = analysis.cell_neighbors(micro.labels(shape=48), n_cells=3000)
    assert 12 < neighbours[neighbours > 0].mean() < 16


def test_tessellation_parameters():
    with pytest.raises(ValueError):
        Voronoi()
    with pytest.raises(ValueError):
        Voronoi(n_cells=10, phase_fractions=(0.5, 0.6))
    with pytest.raises(ValueError, match="domain"):
        Voronoi(n_cells=10).theta(None)


# -- Gaussian excursions ---------------------------------------------------------------------


def test_gaussian_field_evaluation_paths_agree():
    micro = GaussianExcursion(correlation_length=1.0, volume_fraction=0.3).generate(
        mg.Domain.cube(10.0), seed=1
    )
    grid = micro.grid(shape=40)
    m, amp = micro.modes()
    x0 = grid.origin + 0.5 * grid.spacing - micro.box_lower
    np.testing.assert_allclose(
        micro._field_fft(grid, m, amp, x0), micro._field_separable(grid, m, amp, x0), atol=1e-5
    )
    # one realization, any resolution: coarse voxel centres are fine voxel centres
    coarse = micro.field(shape=20).data
    fine = micro.field(shape=60).data
    np.testing.assert_allclose(coarse, fine[1::3, 1::3, 1::3], atol=1e-5)


def test_gaussian_excursion_statistics():
    model = GaussianExcursion(correlation_length=1.0, volume_fraction=0.3)
    th = model.theory()
    assert th["threshold"] == pytest.approx(0.5244, abs=1e-4)
    domain = mg.Domain.cube(20.0)
    vf, sv, cov = [], [], []
    for seed in range(4):
        img = model.generate(domain, seed=seed).voxelize(voxel_size=0.125)
        vf.append(analysis.volume_fraction(img))
        sv.append(analysis.specific_surface(img))
        cov.append(analysis.covariance(img, max_lag=24).mean()[1])
    sigma = math.sqrt(0.21 * th["integral_range"] / domain.volume / 4)
    assert abs(np.mean(vf) - 0.3) < 4 * sigma
    assert np.mean(sv) == pytest.approx(th["specific_surface"], rel=0.05)  # Rice formula
    lags = np.arange(25) * 0.125
    np.testing.assert_allclose(np.mean(cov, axis=0), model.covariance(lags), atol=4 * sigma)


def test_gaussian_excursion_non_periodic_and_level_set():
    micro = GaussianExcursion(correlation_length=0.5, threshold=0.0).generate(
        mg.Domain(size=(4.0, 3.0, 2.0), periodic=False), seed=3
    )
    img = micro.voxelize(voxel_size=0.1)
    ls = micro.level_set(grid=img.grid())
    assert np.array_equal(ls.data < 0, img.data == 1)
    assert 0.3 < img.data.mean() < 0.7


# -- TPMS -------------------------------------------------------------------------------------


@pytest.mark.parametrize("surface", ["gyroid", "schwarz_p", "schwarz_d"])
def test_tpms_symmetric_level_gives_half(surface):
    micro = TPMS(surface=surface, cell_size=1.0, level=0.0).generate(mg.Domain.cube(1.0))
    assert analysis.volume_fraction(micro.voxelize(shape=64)) == pytest.approx(0.5, abs=2e-3)


@pytest.mark.parametrize("variant", ["network", "sheet"])
def test_tpms_target_fraction_and_periodicity(variant):
    model = TPMS(surface="schwarz_d", cell_size=2.0, variant=variant, volume_fraction=0.25)
    assert model.theory()["volume_fraction"] == pytest.approx(0.25, abs=2e-3)
    one = model.generate(mg.Domain.cube(2.0)).voxelize(shape=32).data
    two = model.generate(mg.Domain.cube(4.0)).voxelize(shape=64).data
    assert np.array_equal(np.tile(one, (2, 2, 2)), two)
    assert one.mean() == pytest.approx(0.25, abs=0.01)


def test_tpms_level_set_and_errors():
    micro = TPMS(surface="gyroid", cell_size=1.0, volume_fraction=0.3).generate(mg.Domain.cube(1.0))
    img = micro.voxelize(shape=40)
    ls = micro.level_set(shape=40)
    assert np.array_equal(ls.data <= 0, img.data == 1)
    pts = img.grid().center_points()
    assert np.array_equal(micro.contains(pts).reshape(img.shape), img.data == 1)
    with pytest.raises(ValueError, match="whole number"):
        TPMS(cell_size=0.7, level=0.0).generate(mg.Domain.cube(1.0))
    with pytest.raises(ValueError):
        TPMS(surface="lidinoid", level=0.0)


# -- strut lattices ---------------------------------------------------------------------------


def test_lattice_cells():
    counts = {c: len(unit_cell_struts(c)) for c in ("cubic", "bcc", "fcc", "octet")}
    assert counts == {"cubic": 12, "bcc": 8, "fcc": 24, "octet": 36}


def test_cubic_lattice_matches_closed_form():
    model = StrutLattice(cell="cubic", cell_size=1.0, strut_radius=0.12)
    r = 0.12
    exact = 3 * math.pi * r * r - 8 * math.sqrt(2) * r**3
    assert model.theory()["volume_fraction"] == pytest.approx(exact)
    micro = model.generate(mg.Domain.cube(2.0))
    img = micro.voxelize(shape=80, supersampling=3)
    assert analysis.volume_fraction(img) == pytest.approx(exact, rel=0.01)
    assert lattice_fraction("cubic", r) == pytest.approx(exact, rel=0.02)


def test_lattice_target_density_and_non_periodic_domain():
    model = StrutLattice(cell="octet", cell_size=1.0, volume_fraction=0.2)
    assert 0.0 < model.radius < 0.5
    img = model.generate(mg.Domain.cube(2.0)).voxelize(shape=100, supersampling=2)
    assert analysis.volume_fraction(img) == pytest.approx(0.2, abs=0.005)
    micro = model.generate(mg.Domain(size=(1.5, 1.0, 2.5), periodic=False))
    lo, hi = micro.segments.bounds()
    assert np.all(hi >= micro.domain.lower) and np.all(lo <= micro.domain.upper)
    with pytest.raises(ValueError, match="whole number"):
        model.generate(mg.Domain.cube(1.5))
