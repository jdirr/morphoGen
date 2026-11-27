"""The computations of morphoGen for VIMSEO (interop.vimseo): the simulations as plain
functions (a configuration run with its entries as inputs, the exact densities of a model
of spheres and their closed forms), and, where VIMSEO is installed (Python 3.12, ``pip
install "vimseo>=0.1.7"``), the models of VIMSEO and its two tools of verification on
them."""

from __future__ import annotations

import numpy as np
import pytest

from morphogen.interop import vimseo as mv

CONFIG = {
    "model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.3},
    "domain": {"size": 4.0, "periodic": True},
    "seed": 1,
    "voxels": {"voxel_size": 0.25},
    "homogenization": {
        "method": "fft",
        "materials": {
            0: {"young": 1.0, "poisson": 0.3, "conductivity": 1.0},
            1: {"young": 10.0, "poisson": 0.3, "conductivity": 10.0},
        },
        "properties": ["conductivity"],
    },
}
SPHERES = {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.3}


def test_entries_of_configurations():
    data = {"a": {"b": 1.0}, "materials": {1: {"young": 2.0}}}
    assert mv.get_entry(data, "a.b") == 1.0
    assert mv.get_entry(data, "materials.1.young") == 2.0  # an integer key
    mv.set_entry(data, "materials.1.young", 3.0)
    mv.set_entry(data, "c.d", 4)
    assert data["materials"][1]["young"] == 3.0 and data["c"]["d"] == 4
    with pytest.raises(KeyError):
        mv.get_entry(data, "a.x")


def test_simulation_of_a_configuration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sim = mv.configuration(CONFIG, {"h": "voxels.voxel_size", "seed": "seed"},
                           name="rve")  # fmt: skip
    assert sim.inputs == {"h": 0.25, "seed": 1.0} and sim.integers == ("seed",)
    assert sim.outputs == ("conductivity.periodic", "volume_fraction")
    out = sim()
    assert 1.0 < out["conductivity.periodic"] < 10.0
    assert out["volume_fraction"] == pytest.approx(0.3, abs=0.1)
    again = sim(h=0.25, seed=1.0)
    assert again == out  # the same realization
    coarse = sim(h=0.5)
    assert coarse["conductivity.periodic"] != out["conductivity.periodic"]
    with pytest.raises(ValueError, match="unknown inputs"):
        sim(radius=2.0)
    with pytest.raises(ValueError, match="no entry"):
        mv.configuration(CONFIG, {"x": "mesh.size"})
    with pytest.raises(ValueError, match="both inputs and outputs"):
        mv.configuration(CONFIG, {"volume_fraction": "model.volume_fraction"})


def test_exact_morphology_and_its_theory():
    sim = mv.exact_morphology(SPHERES, 12.0)
    assert set(sim.inputs) == {"model_radius", "model_volume_fraction", "seed"}
    assert sim.outputs == mv.DENSITIES
    points = {
        "model_volume_fraction": np.array([0.2, 0.3, 0.4]),
        "model_radius": np.ones(3),
        "seed": np.arange(3.0),
    }
    theory = mv.theory_of(SPHERES, points)
    for i in range(3):
        out = sim(**{k: float(v[i]) for k, v in points.items()})
        # one realization of a cube of 12: within its fluctuation of the closed forms
        assert out["volume_fraction"] == pytest.approx(theory["volume_fraction"][i], abs=0.03)
        assert out["specific_surface"] == pytest.approx(theory["specific_surface"][i], rel=0.08)
    assert theory["volume_fraction"] == pytest.approx([0.2, 0.3, 0.4])


def test_models_and_tools_of_vimseo(tmp_path, monkeypatch):
    """With VIMSEO: the exact densities verified against Miles' formulae, and the solution of
    a configuration against the size of its voxels (Richardson)."""
    pytest.importorskip("vimseo")
    monkeypatch.chdir(tmp_path)
    options = {"directory_archive_root": "archive", "directory_scratch_root": "scratch"}
    model = mv.model(mv.exact_morphology(SPHERES, 12.0), **options)
    out = model.execute({"model_volume_fraction": np.array([0.4]), "seed": np.array([3.0])})
    assert out["error_code"][0] == 0 and 0.3 < out["volume_fraction"][0] < 0.5
    points = {
        "model_volume_fraction": np.array([0.2, 0.3, 0.4]),
        "model_radius": np.ones(3),
        "seed": np.arange(3.0),
    }
    refs = mv.dataset(points, mv.theory_of(SPHERES, points))
    tool = mv.code_verification(model, refs, output_names=["volume_fraction"],
                                working_directory="cv")  # fmt: skip
    assert tool.result.integrated_metrics["RelativeErrorMetric"]["volume_fraction"] < 0.05
    sim = mv.configuration(CONFIG, {"h": "voxels.voxel_size"}, ["conductivity.periodic"])
    tool = mv.solution_verification(mv.model(sim, **options), "h", [0.4, 0.2, 0.1, 0.05],
                                    "conductivity.periodic", working_directory="sv")  # fmt: skip
    ext = tool.result.extrapolation
    assert ext["q_converged_method"] == "richardson" and 0.5 < ext["beta"] < 1.5
