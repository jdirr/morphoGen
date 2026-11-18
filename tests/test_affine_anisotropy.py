"""Affine anisotropy of the morphological models (:mod:`morphogen.models.affine`): the
configuration, the exact map of grids in periodic boxes, the anisotropy measured on the
images (mean intercept lengths, fabric tensor, covariance) against the prescription, the
rotated principal axes, and the isotropic default unchanged."""

import numpy as np
import pytest

import morphogen as mg
from morphogen import analysis as an
from morphogen.gui.api import examples, models
from morphogen.models.affine import AffineStructure, Anisotropy, reference_domain
from morphogen.models.base import Microstructure, model_from_config

#: The models given the affine anisotropy by the generic map, with overrides of their
#: example parameters and stretches whose preimage boxes hold whole cells
MAPPED = {
    "boolean_spheres": ({}, (1, 2, 4)),
    "hardcore_spheres": ({}, (1, 2, 4)),
    "penetrable_spheres": ({}, (1, 2, 4)),
    "voronoi": ({}, (1, 2, 4)),
    "laguerre": ({}, (1, 2, 4)),
    "tpms": ({}, (1, 2, 4)),
    "strut_lattice": ({}, (1, 3, 9)),
    "random_lattice": ({}, (1, 2, 4)),
    "dead_leaves": ({}, (1, 2, 4)),
    "transparent_dead_leaves": ({}, (1, 2, 4)),
    "nucleation_growth": ({}, (1, 2, 4)),
    "aggregation": ({"n_seeds": 4, "volume_fraction": 0.02}, (1, 2, 4)),
    "cox_spheres": ({}, (1, 2, 4)),
}
ALL = [*MAPPED, "gaussian_excursion", "open_foam"]


def example(name, **params):
    e = examples()[name]
    size = float(np.ravel(e["domain"].get("size", 1.0))[0])
    return model_from_config({**e["params"], **params, "type": name}), size


def axis_mils(image, phase=1):
    """The mean intercept lengths of ``phase`` along x, y and z."""
    dirs, mil = an.mean_intercept_length(image, phase)
    return np.array([mil[np.flatnonzero(np.all(np.isclose(np.abs(dirs), e), axis=1))[0]]
                     for e in np.eye(3)])  # fmt: skip


def correlation_lengths(image, phase=1):
    """The lags where the normalized covariance of ``phase`` falls to 1/2, along x, y, z."""
    cov = an.covariance(image, phase, max_lag=image.shape[0] // 2)
    p = cov.volume_fraction
    out = []
    for name in "xyz":
        rho, h = (cov.values[name] - p * p) / (p - p * p), cov.lags[name]
        i = int(np.flatnonzero(rho < 0.5)[0])
        out.append(h[i - 1] + (rho[i - 1] - 0.5) / (rho[i - 1] - rho[i]) * (h[i] - h[i - 1]))
    return np.array(out)


def test_configuration_of_the_anisotropy():
    a = Anisotropy(stretch=(1, 1, 2))
    assert np.prod(a.principal) == pytest.approx(1.0) and a.principal[2] / a.principal[0] == 2
    np.testing.assert_array_equal(a.box_stretch(), a.principal)  # no rotation: exact
    m = mg.models.BooleanSpheres(radius=0.05, volume_fraction=0.3, anisotropy=[1, 1, 2, 0, 90, 0])
    assert m.anisotropy == Anisotropy((1, 1, 2), (0, 90, 0))
    np.testing.assert_allclose(m.anisotropy.box_stretch(), a.principal[[0, 2, 1]], atol=1e-15)
    cfg = m.to_config()
    assert cfg["anisotropy"] == {"stretch": [1.0, 1.0, 2.0], "euler": [0.0, 90.0, 0.0]}
    assert model_from_config(cfg) == m
    assert model_from_config({**cfg, "anisotropy": 2}).anisotropy.to_config() == 2.0
    assert (
        "anisotropy" not in mg.models.BooleanSpheres(radius=0.05, volume_fraction=0.3).to_config()
    )
    for bad in ({"stretch": [1, 0, 1]}, {"stretch": [1, 1, 2], "axis": 2}, [1, 2], True):
        with pytest.raises(ValueError, match="anisotropy"):
            mg.models.BooleanSpheres(radius=0.05, volume_fraction=0.3, anisotropy=bad)
    turned = mg.models.BooleanSpheres(
        radius=0.05, volume_fraction=0.3, anisotropy=[1, 1, 2, 0, 45, 0]
    )
    with pytest.raises(ValueError, match="periodicity"):
        turned.generate(mg.Domain.cube(1.0, periodic=True), seed=1)
    # the same field in every form of the GUI
    fields = {
        m["name"]: next(p for p in m["parameters"] if p["name"] == "anisotropy")
        for m in models()
        if m["name"] in ALL
    }
    assert set(fields) == set(ALL)
    assert {(f["kind"], f["doc"], f["default"]) for n, f in fields.items() if n != "open_foam"} == {
        ("anisotropy", fields["voronoi"]["doc"], None)
    }
    assert fields["open_foam"]["kind"] == "anisotropy" and fields["open_foam"]["default"] == 1.0
    # mapped cells are voxelized, not meshed grain by grain (the identity changes nothing)
    assert not mg.models.Voronoi(n_cells=10, anisotropy=[1, 1, 2]).meshed_by_grains
    assert mg.models.Voronoi(n_cells=10, anisotropy=[2, 2, 2]).meshed_by_grains


@pytest.mark.parametrize("name", list(MAPPED))
def test_periodic_images_are_those_of_the_preimage_box(name):
    params, stretch = MAPPED[name]
    iso, size = example(name, **params)
    model = model_from_config({**iso.to_config(), "anisotropy": {"stretch": list(stretch)}})
    domain = mg.Domain.cube(size, periodic=True)
    s = model.generate(domain, seed=4)
    assert isinstance(s, AffineStructure) and s.inner.domain.periodic
    t = model.anisotropy.box_stretch()
    np.testing.assert_allclose(s.inner.domain.lengths * t, domain.lengths, rtol=1e-14)
    image = s.voxelize((24, 24, 24))
    reference = iso.generate(reference_domain(model.anisotropy, domain), seed=4)
    np.testing.assert_array_equal(image.data, reference.voxelize((24, 24, 24)).data)
    assert image.periodic and np.allclose(image.spacing, size / 24)
    if s.has_labels:  # grains: the labels and their data follow
        np.testing.assert_array_equal(
            s.labels((24, 24, 24)).data, reference.labels((24, 24, 24)).data
        )
    again = Microstructure.from_dict(s.to_dict())
    np.testing.assert_array_equal(again.voxelize((24, 24, 24)).data, image.data)
    assert set(model.theory(domain)) <= set(iso.theory(domain))  # the characteristics kept


@pytest.mark.parametrize(
    "name, params, shape, phase, chords",
    [
        ("boolean_spheres", {"radius": 0.06, "volume_fraction": 0.3}, 96, 1, True),
        ("hardcore_spheres", {"radius": 0.06, "volume_fraction": 0.3, "min_gap": 0.0}, 96, 1, True),
        ("voronoi", {"n_cells": 1000, "phase_fractions": [0.5, 0.5]}, 80, 1, True),
        ("dead_leaves", {"radius": 0.08, "phase_fractions": [0.5, 0.5]}, 80, 1, False),
        ("gaussian_excursion", {"correlation_length": 0.05, "volume_fraction": 0.4}, 80, 1, True),
    ],
)  # fmt: skip
def test_measured_anisotropy_follows_the_stretch(name, params, shape, phase, chords):
    iso = model_from_config({**params, "type": name})
    domain = mg.Domain.cube(1.0, periodic=True)
    ani = model_from_config({**params, "type": name, "anisotropy": [1, 1, 2]})
    a = iso.generate(domain, seed=2).voxelize((shape,) * 3)
    b = ani.generate(domain, seed=2).voxelize((shape,) * 3)
    # correlation lengths along the axes in the ratios 1 : 1 : 2, those of the isotropic
    # model equal (sampling of the covariance at its half decay: 1-2 %)
    lengths = correlation_lengths(b, phase)
    np.testing.assert_allclose(lengths / lengths[0], [1.0, 1.0, 2.0], rtol=0.04)
    lengths = correlation_lengths(a, phase)
    np.testing.assert_allclose(lengths / lengths.mean(), 1.0, rtol=0.04)
    if chords:  # mean chords too (digitization of the shortest ones, 4-6 voxels: 2-3 %);
        # the visible parts of dead leaves have slivers thinner than the compressed voxels,
        # whose crossings are missed (1.83, 1.87, 1.90 on 64, 96, 128 voxels: not asserted)
        mil = axis_mils(b, phase)
        np.testing.assert_allclose(mil / mil[0], [1.0, 1.0, 2.0], rtol=0.05)
    fabric = an.fabric_tensor(b, phase)
    assert abs(fabric.axes[2, 0]) > 0.99  # the longest MIL along z
    # volume fraction: that of the model (random fractions of cells and leaves: ~0.03)
    expected = ani.theory(domain).get("volume_fraction", (a.data == phase).mean())
    assert (b.data == phase).mean() == pytest.approx(expected, abs=0.03)


def test_gaussian_covariance_follows_the_metric_in_periodic_boxes():
    params = {"correlation_length": 0.05, "volume_fraction": 0.3}
    iso = mg.models.GaussianExcursion(**params)
    domain = mg.Domain.cube(1.0, periodic=True)
    ani = mg.models.GaussianExcursion(**params, anisotropy={"stretch": [1, 1, 2]})
    image = ani.generate(domain, seed=3).voxelize((80, 80, 80))
    cov = an.covariance(image, 1, max_lag=20)
    s = ani.anisotropy.principal
    for axis, name in enumerate("xyz"):  # C(h e_i) = C_iso(h / s_i)
        h = cov.lags[name]
        np.testing.assert_allclose(cov.values[name], iso.covariance(h / s[axis]), atol=0.012)
    # principal axes turned about z by 45 degrees, in the periodic box: the MIL ellipsoid
    turned = mg.models.GaussianExcursion(
        **params, anisotropy={"stretch": [1, 2, 1], "euler": [45, 0, 0]}
    )
    fabric = an.fabric_tensor(turned.generate(domain, seed=3).voxelize((80, 80, 80)), 1)
    longest = fabric.axes[:, 0] * np.sign(fabric.axes[1, 0])
    np.testing.assert_allclose(longest, [-np.sqrt(0.5), np.sqrt(0.5), 0.0], atol=0.05)
    assert fabric.ratios[0] == pytest.approx(2.0, rel=0.06)
    assert set(ani.theory()) == {
        "threshold",
        "volume_fraction",
        "euler_density",
        "integral_range",
        "variance_exponent",
    }


def test_rotated_principal_axes_in_a_non_periodic_box():
    params = {"radius": 0.06, "volume_fraction": 0.3}
    domain = mg.Domain.cube(1.0, periodic=False)
    ani = mg.models.BooleanSpheres(
        **params, anisotropy={"stretch": [1, 1, 2], "euler": [30, 45, 0]}
    )
    s = ani.generate(domain, seed=6)
    image = s.voxelize((80, 80, 80))
    assert not image.periodic and (image.data == 1).mean() == pytest.approx(0.3, abs=0.02)
    fabric = an.fabric_tensor(image, 1)
    axis = ani.anisotropy.axes[:, 2]  # the principal axis of the largest stretch
    assert abs(fabric.axes[:, 0] @ axis) > 0.99
    # the fit of the 13 MILs of a turned ellipsoid on voxels: 1.98 aligned, 1.89 turned
    assert fabric.ratios[0] == pytest.approx(2.0, rel=0.08)
    np.testing.assert_allclose(fabric.ratios[1:], 1.0, atol=0.03)
    # two phases: the voxels are the point queries at their centres; grains (labels)
    # sample a finer preimage grid, interfaces within half a voxel
    grid = domain.grid((80, 80, 80))
    np.testing.assert_array_equal(
        s.contains(grid.center_points()).reshape(grid.shape), image.data == 1
    )
    cells = mg.models.Voronoi(n_cells=200, anisotropy={"stretch": [1, 1, 2], "euler": [30, 45, 0]})
    c = cells.generate(domain, seed=6)
    coarse = domain.grid((40, 40, 40))
    labels = c.labels(grid=coarse).data
    exact = c.inner.cell_of(coarse.center_points() @ np.linalg.inv(c.matrix).T)
    assert (labels == exact.reshape(coarse.shape)).mean() > 0.93  # 0.955 measured
    assert len(np.unique(labels)) > 100 and c.orientations is c.inner.orientations


def test_isotropic_default_and_the_open_foam():
    domain = mg.Domain.cube(1.0, periodic=True)
    plain = mg.models.HardcoreSpheres(radius=0.08, volume_fraction=0.2)
    unit = mg.models.HardcoreSpheres(radius=0.08, volume_fraction=0.2, anisotropy=[2, 2, 2])
    a, b = plain.generate(domain, seed=1), unit.generate(domain, seed=1)
    assert not isinstance(a, AffineStructure) and not isinstance(b, AffineStructure)
    np.testing.assert_array_equal(a.voxelize(32).data, b.voxelize(32).data)  # identity
    # the open foam: its elongation along an axis, or the common schema along the box axes
    legacy = mg.models.OpenFoam(n_cells=40, strut_radius=0.02, anisotropy=2.0, axis=0)
    common = mg.models.OpenFoam(n_cells=40, strut_radius=0.02, anisotropy={"stretch": [2, 1, 1]})
    assert (
        legacy.to_config()["anisotropy"] == 2.0
        and mg.models.OpenFoam(n_cells=40, strut_radius=0.02).to_config()["anisotropy"] == 1.0
    )
    x, y = (
        legacy.generate(domain, seed=3).voxelize(48),
        common.generate(domain, seed=3).voxelize(48),
    )
    assert (x.data == y.data).mean() > 0.999
    assert common.theory(domain)["edge_length_density"] == pytest.approx(
        legacy.theory(domain)["edge_length_density"], rel=1e-12
    )
    with pytest.raises(ValueError, match="box axes"):
        mg.models.OpenFoam(
            n_cells=40, strut_radius=0.02, anisotropy={"stretch": [2, 1, 1], "euler": [30, 0, 0]}
        )
