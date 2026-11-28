"""Microstructures of cellular automata and reaction-diffusion (lattice fields)."""

import numpy as np
import pytest

import morphogen as mg
from morphogen import io
from morphogen.models import (
    ActivatorInhibitorModel,
    CellularAutomatonModel,
    LatticeField,
    ReactionDiffusionModel,
)
from morphogen.models.lattice_field import lattice_shape

CUBE = mg.Domain.cube(16.0)


@pytest.fixture(scope="module")
def structures():
    """One realization of each model (small lattices)."""
    models = [
        CellularAutomatonModel(steps=10),
        CellularAutomatonModel(rule="conserved", volume_fraction=0.3, steps=10, cell_size=0.5),
        CellularAutomatonModel(rule="B5-7/S6-26", initial_fraction=0.45, update="asynchronous"),
        ActivatorInhibitorModel(activation_radius=1.5, inhibition_radius=3.5, steps=8),
        ReactionDiffusionModel(pattern="spots", steps=1500, volume_fraction=0.2, cell_size=0.5),
    ]
    return [(m, m.generate(CUBE, seed=7)) for m in models]


def test_lattice_structures(structures):
    for model, micro in structures:
        assert isinstance(micro, LatticeField)
        shape = lattice_shape(CUBE, model.cell_size)
        assert micro.shape == shape
        image = micro.voxelize()
        assert image.shape == shape and image.data.dtype == np.uint8
        assert np.array_equal(image.data, micro.phases)
        assert 0.05 < image.data.mean() < 0.95
        # the level set is negative exactly in phase 1 at the cell centres
        ls = micro.level_set()
        assert np.array_equal(ls.data < 0, micro.phases == 1)
        assert micro.summary()["lattice"] == "x".join(map(str, shape))
        assert "steps" in micro.summary()


def test_off_lattice_resolutions(structures):
    for _, micro in structures:
        f = micro.phases.mean()
        fine = micro.voxelize(shape=40)
        assert fine.shape == (40, 40, 40)
        assert abs(fine.data.mean() - f) < 0.05
        pv = micro.voxelize(shape=20, supersampling=3)
        assert pv.data.dtype == np.float32 and abs(pv.data.mean() - fine.data.mean()) < 0.03
        pts = fine.grid().center_points()
        assert np.array_equal(micro.contains(pts), fine.data.ravel() == 1)


def test_exact_fractions_and_theory(structures):
    conserved = structures[1][1]
    assert conserved.phases.mean() == round(0.3 * 32**3) / 32**3
    assert conserved.model.theory() == {"volume_fraction": 0.3}
    assert structures[0][0].theory() == {}
    spots = structures[4][1]
    assert abs(spots.phases.mean() - 0.2) < 1e-3
    assert spots.field is not None
    assert np.array_equal(spots.phases == 1, spots.field > spots.threshold)
    history = conserved.attributes["history"]
    assert history["interface"][-1] < history["interface"][0]
    ai = ActivatorInhibitorModel(volume_fraction=0.4, steps=3).generate(CUBE, seed=1)
    assert ai.phases.mean() == pytest.approx(0.4, abs=1e-3)


def test_json_round_trip(structures, tmp_path):
    for model, micro in structures:
        path = tmp_path / f"{model.name}.json"
        io.save_microstructure(micro, path)
        back = io.load_microstructure(path)
        assert isinstance(back, LatticeField) and back.model == model
        assert np.array_equal(back.phases, micro.phases)
        assert np.array_equal(back.voxelize(shape=24).data, micro.voxelize(shape=24).data)
        assert back.attributes.get("steps") == micro.attributes.get("steps")


def test_reproducible_and_seeded():
    model = CellularAutomatonModel(steps=5)
    a = model.generate(CUBE, seed=3).phases
    assert np.array_equal(a, model.generate(CUBE, seed=3).phases)
    assert not np.array_equal(a, model.generate(CUBE, seed=4).phases)


def test_windows_and_planar_structures():
    window = mg.Domain.cube(16.0, periodic=False)
    micro = CellularAutomatonModel(steps=6).generate(window, seed=2)
    assert micro.shape == (16, 16, 16)
    assert micro.voxelize(shape=32).shape == (32, 32, 32)
    rd = ReactionDiffusionModel(steps=300).generate(window, seed=2)
    assert rd.shape == (16, 16, 16)
    plate = mg.Domain(size=(24.0, 16.0, 3.0))
    flat = CellularAutomatonModel(dimension=2, steps=8, rule="majority").generate(plate, seed=1)
    assert flat.shape == (24, 16, 1)
    image = flat.voxelize(shape=(24, 16, 6))
    assert all(np.array_equal(image.data[..., k], flat.phases[..., 0]) for k in range(6))
    gs = ReactionDiffusionModel(dimension=2, diffusion_u=0.2, steps=200).generate(plate, seed=1)
    assert gs.shape == (24, 16, 1)


def test_reaction_diffusion_needs_room_for_the_pattern():
    with pytest.raises(ValueError, match="died out"):
        ReactionDiffusionModel(pattern="spots", steps=1500).generate(mg.Domain.cube(8.0), seed=1)
    # a box of a single cell (the default cell_size in a unit box): said so, not a division
    with pytest.raises(ValueError, match=r"a single cell .* smaller cell_size"):
        ReactionDiffusionModel().generate(mg.Domain.cube(1.0, periodic=True), seed=1)


def test_smoothing_rounds_the_interface():
    micro = CellularAutomatonModel(steps=10, smoothing=1.0).generate(CUBE, seed=5)
    sharp = CellularAutomatonModel(steps=10).generate(CUBE, seed=5)
    assert np.array_equal(micro.phases, sharp.phases)
    fine, coarse = micro.voxelize(shape=48), sharp.voxelize(shape=48)
    surface = mg.analysis.specific_surface
    assert surface(fine) < surface(coarse)
    assert abs(fine.data.mean() - coarse.data.mean()) < 0.03


@pytest.mark.parametrize(
    ("model", "kwargs", "message"),
    [
        (CellularAutomatonModel, {"rule": "conserved"}, "volume_fraction"),
        (CellularAutomatonModel, {"volume_fraction": 0.3}, "conserved"),
        (CellularAutomatonModel, {"rule": "B3"}, "rule"),
        (CellularAutomatonModel, {"neighbourhood": "hex"}, "neighbourhood"),
        (CellularAutomatonModel, {"radius": 0.5}, "radius"),
        (CellularAutomatonModel, {"update": "random"}, "update"),
        (CellularAutomatonModel, {"dimension": 1}, "dimension"),
        (CellularAutomatonModel, {"cell_size": 0.0}, "cell_size"),
        (ActivatorInhibitorModel, {"inhibition_radius": 1.5}, "inhibition_radius"),
        (ActivatorInhibitorModel, {"activation_radius": 0.5}, "activation_radius"),
        (ReactionDiffusionModel, {"feed": 0.03}, "kill"),
        (ReactionDiffusionModel, {"pattern": "zebra"}, "pattern"),
        (ReactionDiffusionModel, {"diffusion_u": 0.2}, "stable"),
        (ReactionDiffusionModel, {"steps": 0}, "steps"),
    ],
)
def test_parameter_checks(model, kwargs, message):
    with pytest.raises(ValueError, match=message):
        model(**kwargs)


def test_configs_and_registry():
    for name in ("cellular_automaton", "activator_inhibitor", "reaction_diffusion"):
        cls = mg.models.get_model(name)
        model = cls()
        assert mg.models.model_from_config(model.to_config()) == model
        assert {p["name"] for p in cls.parameters()} >= {"cell_size", "dimension", "smoothing"}
    rd = mg.models.model_from_config({"type": "reaction_diffusion", "feed": 0.03, "kill": 0.06})
    assert rd.rates() == (0.03, 0.06)


def test_multiphase_and_labelled_lattice_fields(tmp_path):
    phases = np.zeros((4, 4, 4), dtype=np.uint8)
    phases[2:] = 2
    phases[:, 2:] = 1
    labels = np.arange(64, dtype=np.int32).reshape(4, 4, 4) // 8
    labels[0, 0, 0] = -1
    micro = LatticeField(
        mg.Domain.cube(4.0),
        phases,
        labels=labels,
        orientations=np.zeros((8, 3)),
        attributes={"n_particles": 3},
    )
    assert not micro.binary and micro.has_labels
    image = micro.voxelize(shape=8)
    assert np.array_equal(image.data[::2, ::2, ::2], phases)
    with pytest.raises(NotImplementedError):
        micro.level_set()
    with pytest.raises(ValueError, match="partial volumes"):
        micro.voxelize(shape=8, supersampling=2)
    lab = micro.labels(shape=8)
    assert lab.data.dtype == np.int32 and np.array_equal(lab.data[1::2, 1::2, 1::2], labels)
    assert micro.summary()["n_grains"] == 8 and micro.summary()["n_particles"] == 3
    io.save_microstructure(micro, tmp_path / "m.json")
    back = io.load_microstructure(tmp_path / "m.json")
    assert np.array_equal(back.cell_labels, labels) and back.orientations.shape == (8, 3)
    vol = mg.analysis.cells.cell_volumes(lab)
    assert len(vol) == 8 and vol.sum() == pytest.approx(64 - 1)
    with pytest.raises(ValueError, match="labels"):
        LatticeField(mg.Domain.cube(4.0), phases, labels=labels[:2])
    plain = LatticeField(mg.Domain.cube(4.0), phases == 1)
    assert not plain.has_labels
    with pytest.raises(ValueError, match="no labels"):
        plain.labels()


def test_conforming_mesh_of_an_automaton():
    from morphogen.mesh.conforming import conforming_mesh, mmg_executable

    try:
        mmg_executable()
    except ImportError:  # pragma: no cover - depends on the environment
        pytest.skip("MMG (pymmg) is not installed")
    model = CellularAutomatonModel(
        rule="conserved", volume_fraction=0.4, steps=20, smoothing=0.75, cell_size=1.0
    )
    micro = model.generate(mg.Domain.cube(8.0), seed=2)
    mesh = conforming_mesh(micro, size=1.0, background=0.5)
    fractions = mesh.phase_fractions()
    exact = np.mean(micro.level_set(shape=64).data < 0)
    assert abs(fractions[1] - exact) < 0.02
    assert mesh.periodic and mesh.quality().min() > 0.0
