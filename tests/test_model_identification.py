"""Models from images (morphology.segmentation, morphology.identification, models.image):
thresholds against their criteria, particles and their ellipsoids, the scores of a
segmentation, the image as a structure of implicit functions, and the model of a volume
or of a section found among the candidates, with its parameters."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import morphogen as mg
from morphogen.models.base import Microstructure
from morphogen.models.image import ImageModel, example_image, synthetic_tomography
from morphogen.morphology.identification import CANDIDATES, describe, fit, identify
from morphogen.morphology.segmentation import (
    equalize,
    multi_otsu,
    otsu,
    particle_shapes,
    scores,
    segment,
    stack,
)


def test_thresholds_maximize_the_variance_between_classes():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(50, 5, 4000), rng.normal(120, 8, 2000)])
    t = otsu(values)
    assert 60 < t < 100  # in the gap between the classes
    # the criterion by brute force over the bins: its maximum (the empty bins between the
    # classes all give it, the same classes)
    counts, edges = np.histogram(values, bins=256)
    centres = 0.5 * (edges[1:] + edges[:-1])

    def criterion(j):
        return (counts[:j] @ centres[:j]) ** 2 / counts[:j].sum() + (
            counts[j:] @ centres[j:]
        ) ** 2 / counts[j:].sum()

    best = max(criterion(j) for j in range(1, 256))
    assert criterion(int(np.searchsorted(edges, t))) == pytest.approx(best, rel=1e-12)
    three = np.concatenate([values, rng.normal(200, 6, 3000)])
    t1, t2 = multi_otsu(three, 3)
    assert 60 < t1 < 100 and 140 < t2 < 190
    eq = equalize(three.reshape(-1, 1))
    assert eq.min() >= 0.0 and eq.max() == pytest.approx(1.0)


def test_segmentation_of_a_noisy_volume_and_its_scores():
    img = synthetic_tomography()
    truth = mg.models.HardcoreSpheres(radius=1.2, volume_fraction=0.2).generate(
        mg.Domain.cube(12.0, periodic=True), seed=7
    ).voxelize(48)  # fmt: skip
    seg = segment(img, sigma=1.0)
    assert seg.metadata["thresholds"] and seg.data.max() == 1
    s = scores(seg, truth)
    assert s["iou"] > 0.8 and s["dice"] > 0.9 and s["accuracy"] > 0.95
    assert scores(truth, truth)["iou"] == 1.0
    sections = [img.data[:, :, k] for k in range(4)]
    assert stack(sections, (0.25, 0.25, 0.25)).data.shape == (48, 48, 4)


def test_particles_and_their_ellipsoids():
    micro = mg.models.HardcoreSpheres(radius=3.0, volume_fraction=0.15).generate(
        mg.Domain.cube(48.0), seed=1
    )
    shapes = particle_shapes(micro.voxelize(48))
    assert shapes.moments(1) == pytest.approx(3.0, rel=0.03)
    assert np.median(shapes.semi_axes[:, 0]) == pytest.approx(3.0, rel=0.08)
    assert np.median(shapes.aspect_ratios[:, 1]) > 0.85
    assert shapes.density == pytest.approx(0.15 / (4 / 3 * math.pi * 27), rel=0.05)
    law = shapes.radius_law("lognormal")
    assert law.moment(1) == pytest.approx(shapes.moments(1), rel=1e-6)
    # an ellipse in a section: the semi-axes of its moments
    x, y = np.meshgrid(np.arange(200) + 0.5, np.arange(200) + 0.5, indexing="ij")
    ellipse = ((((x - 100) / 60) ** 2 + ((y - 100) / 25) ** 2) <= 1).astype(np.uint8)
    one = particle_shapes(ellipse[:, :, None])
    assert one.count == 1 and one.planar
    assert one.semi_axes[0, :2] == pytest.approx([60.0, 25.0], rel=0.01)


def test_image_model_implicit_functions_and_windows(tmp_path):
    model = ImageModel()
    assert model.path() == example_image()
    img = model.image()
    whole = model.generate(mg.Domain.cube(12.0, periodic=False), seed=0)
    # the phases at the centres of the voxels of the image are its thresholded levels
    phases = whole.voxelize(grid=mg.Grid(whole.domain, (48, 48, 48))).data
    assert np.array_equal(phases, (img.data >= whole.thresholds[0]).astype(np.uint8))
    # the functions at points: a partition, and the same at the centres
    pts = np.random.default_rng(1).random((500, 3)) * 12.0
    f = whole.phase_level_sets_at(pts, 1.0)
    assert np.all((f[0] < 0) != (f[1] < 0))
    # the packaged image is the one its function makes
    np.testing.assert_array_equal(img.data, synthetic_tomography().data)
    # random windows, the round trip of a realization
    windows = ImageModel(window="random")
    a = windows.generate(mg.Domain.cube(8.0, periodic=False), seed=1)
    b = windows.generate(mg.Domain.cube(8.0, periodic=False), seed=2)
    assert not np.allclose(a.corner, b.corner)
    back = Microstructure.from_dict(json.loads(json.dumps(a.to_dict())))
    assert np.array_equal(back.voxelize(24).data, a.voxelize(24).data)
    with pytest.raises(ValueError, match="larger than the image"):
        model.generate(mg.Domain.cube(20.0, periodic=False))
    with pytest.raises(ValueError, match="whole image"):
        model.generate(mg.Domain.cube(8.0, periodic=True))
    # an image of phases: its own phases, the functions half-way between the voxels
    labels = (img.data >= whole.thresholds[0]).astype(np.uint8)
    path = tmp_path / "phases.npy"
    np.save(path, labels)
    own = ImageModel(file=str(path), spacing=0.25).generate(mg.Domain.cube(12.0, periodic=False))
    grid = mg.Grid(own.domain, (48, 48, 48))
    assert np.array_equal(own.voxelize(grid=grid).data, labels)
    assert own.lipschitz == pytest.approx(math.sqrt(3.0))


def test_identification_of_volumes():
    """Each model found first among the candidates, from an image of one of its
    realizations: the Boolean spheres, the hard spheres and the Gaussian excursion with a
    score less than half that of the next, the Voronoi mosaic first too."""
    cases = {
        "boolean_spheres": (mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3), 0.5),
        "hardcore_spheres": (mg.models.HardcoreSpheres(radius=1.0, volume_fraction=0.25), 0.5),
        "gaussian_excursion": (
            mg.models.GaussianExcursion(correlation_length=0.8, volume_fraction=0.35),
            0.5,
        ),
        "voronoi": (mg.models.Voronoi(n_cells=150, phase_fractions=(0.6, 0.4)), 1.0),
    }
    found = {}
    for name, (model, ratio) in cases.items():
        image = model.generate(mg.Domain.cube(12.0), seed=11).voxelize(64)
        result = identify(image)
        assert result.best.name == name, result.table()
        assert result.best.total < ratio * result.candidates[1].total, result.table()
        found[name] = result.best.model
    assert found["boolean_spheres"].radius.mean() == pytest.approx(1.0, rel=0.05)
    assert found["hardcore_spheres"].radius.mean() == pytest.approx(1.0, rel=0.05)
    assert found["gaussian_excursion"].correlation_length == pytest.approx(0.8, rel=0.05)


def test_identification_of_a_section():
    """Stereology: the model of a volume from one of its sections."""
    model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3)
    micro = model.generate(mg.Domain.cube(25.6), seed=4)
    section = micro.voxelize(grid=mg.Grid(micro.domain, (256, 256, 1)))
    section = mg.VoxelImage(section.data, spacing=(0.1, 0.1, 0.1), periodic=True)
    desc = describe(section)
    assert desc.planar
    # the section's own fraction, and S_V = 4 L_A / π of the volume
    assert desc.specific_surface == pytest.approx(model.theory()["specific_surface"], rel=0.15)
    result = identify(section)
    assert result.best.name == "boolean_spheres", result.table()
    assert result.best.model.radius.mean() == pytest.approx(1.0, rel=0.12)


def test_candidates_fit_their_closed_forms():
    model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3)
    image = model.generate(mg.Domain.cube(12.0), seed=3).voxelize(48)
    desc = describe(image)
    for name in CANDIDATES:
        fitted = fit(name, desc)
        if fitted is None:
            continue
        th = fitted.theory(mg.Domain.cube(12.0))
        fraction = th.get("volume_fraction", th.get("volume_fraction_1"))
        assert fraction == pytest.approx(desc.volume_fraction, rel=1e-6), name
        if name in ("boolean_spheres", "gaussian_excursion"):
            assert th["specific_surface"] == pytest.approx(desc.specific_surface, rel=1e-6)


def test_command_identify(tmp_path, monkeypatch, capsys):
    """``morphogen identify``: the ranking printed, the best model written as a
    configuration that runs."""
    import yaml

    from morphogen import cli
    from morphogen.io import write_npz

    monkeypatch.chdir(tmp_path)
    image = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3).generate(
        mg.Domain.cube(8.0), seed=2
    ).voxelize(32)  # fmt: skip
    write_npz(tmp_path / "scan.npz", image)
    code = cli.main(["identify", "scan.npz", "--candidates", "boolean_spheres",
                     "hardcore_spheres", "--realizations", "1", "-o", "model.yaml"])  # fmt: skip
    assert code == 0
    assert "boolean_spheres" in capsys.readouterr().out
    config = yaml.safe_load((tmp_path / "model.yaml").read_text(encoding="utf-8"))
    assert config["model"]["type"] == "boolean_spheres"
    assert config["domain"]["size"] == [8.0, 8.0, 8.0] and config["domain"]["periodic"]
    assert cli.main(["run", "model.yaml", "-o", "out", "-q"]) == 0
