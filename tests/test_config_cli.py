"""Configuration files, pipeline runs and the command-line interface."""

import json
import subprocess
import sys

import numpy as np
import pytest
import yaml

import morphogen as mg
from morphogen import cli, pipeline
from morphogen.config import ConfigError, load_config
from morphogen.optimization import OBJECTIVES, load_optimization_config


def _small(name: str, tmp_path) -> dict:
    """Template configuration scaled down for tests."""
    text = cli._templates()[name]
    cfg = yaml.safe_load(text)
    cfg["outputs"]["directory"] = str(tmp_path / name)
    return cfg


def test_all_templates_are_valid(tmp_path):
    templates = cli._templates()
    assert {
        "boolean_spheres",
        "poisson_fibers",
        "voronoi",
        "tpms",
        "rve_poisson_fibers",
        "rve_voronoi_windows",
    } <= set(templates)
    for name, text in templates.items():
        data = yaml.safe_load(text)
        if "laser_sheet" in data:
            from morphogen.laser.run import load_laser_config

            cfg = load_laser_config(data, base_dir=tmp_path)
            assert cfg.case.material.name, name
            continue
        if "optimization" in data:
            opt = load_optimization_config(data, base_dir=tmp_path)
            assert opt.optimization.objective in OBJECTIVES, name
            with pytest.raises(ConfigError, match="morphogen optimize"):
                load_config(data)
            continue
        cfg = load_config(data, base_dir=tmp_path)
        assert cfg.model.name in mg.list_models(), name


def test_config_formats_and_errors(tmp_path):
    base = {
        "model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.2},
        "domain": {"size": 5.0},
    }
    (tmp_path / "c.json").write_text(json.dumps(base))
    (tmp_path / "c.toml").write_text(
        '[model]\ntype = "boolean_spheres"\nradius = 1.0\nvolume_fraction = 0.2\n'
        "[domain]\nsize = 5.0\n"
    )
    for ext in ("json", "toml"):
        cfg = load_config(tmp_path / f"c.{ext}")
        assert cfg.model == mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.2)
        assert cfg.outputs.directory == tmp_path / "morphogen_output"
    bad_cases = [
        ({**base, "colour": 1}, "unknown keys"),
        ({"domain": {"size": 1}}, "missing keys"),
        ({**base, "model": {"type": "nope"}}, "unknown model"),
        ({**base, "analysis": ["volume_fraction"]}, "voxels"),
        ({**base, "voxels": {"shape": 8, "voxel_size": 0.1}}, "exactly one"),
        ({**base, "outputs": {"image": ["png"]}}, "unknown value"),
        ({**base, "seed": -3}, "seed"),
        ({**base, "rve": {"sizes": [1, 2], "realizations": 2}}, "domain.size"),
        (
            {
                "model": base["model"],
                "rve": {"sizes": [1, 2], "realizations": [2]},
                "voxels": {"voxel_size": 0.1},
            },
            "one per size",
        ),
        (
            {
                "model": base["model"],
                "rve": {"sizes": [1, 2], "realizations": 2},
                "voxels": {"shape": 8},
            },
            "voxel_size",
        ),
        (
            {
                "model": {"type": "voronoi", "n_cells": 50},
                "rve": {"sizes": [1, 2], "realizations": 2},
                "voxels": {"voxel_size": 0.1},
            },
            "density",
        ),
    ]
    for data, message in bad_cases:
        with pytest.raises(ConfigError, match=message):
            load_config(data, base_dir=tmp_path)


def test_batch_run_outputs(tmp_path):
    cfg = _small("boolean_spheres", tmp_path)
    cfg["domain"]["size"] = 6.0
    cfg["voxels"] = {"shape": 24}
    cfg["outputs"]["mesh"] = {"formats": ["inp", "msh"], "phases": [1]}
    cfg["outputs"]["periodic_pairs"] = True
    outdir = pipeline.run(load_config(cfg))
    names = {p.name for p in outdir.iterdir()}
    for r in (0, 1):
        assert {
            f"spheres_000{r}.json",
            f"spheres_000{r}_spheres.csv",
            f"spheres_000{r}.vti",
            f"spheres_000{r}.inp",
            f"spheres_000{r}.msh",
            f"spheres_000{r}_periodic.csv",
        } <= names
    rows = pipeline.read_rows(outdir / "results.csv")
    assert [r["realization"] for r in rows] == [0, 1]
    assert 0.15 < rows[0]["volume_fraction"] < 0.45 and "specific_surface" in rows[0]
    manifest = json.loads((outdir / "run.json").read_text(encoding="utf-8"))
    assert manifest["theory"]["volume_fraction"] == pytest.approx(0.3)
    assert manifest["versions"]["morphogen"] == mg.__version__
    # the realization written to JSON regenerates the image exactly
    micro = mg.io.load_microstructure(outdir / "spheres_0001.json")
    img = mg.io.load_image(outdir / "spheres_0001.vti")
    assert np.array_equal(micro.voxelize(shape=24).data, img.data)


def test_split_runs_equal_single_run(tmp_path):
    cfg = _small("rve_poisson_fibers", tmp_path)
    cfg["rve"]["sizes"] = [6, 8]
    cfg["rve"]["realizations"] = [5, 4]
    whole = pipeline.collect_results(pipeline.run(load_config(cfg)))
    cfg["outputs"]["directory"] = str(tmp_path / "split")
    for task in range(3):
        pipeline.run(load_config(cfg), task_id=task, n_tasks=3)
    parts = pipeline.collect_results(tmp_path / "split")
    assert len(parts) == len(whole) == 9

    def key(r):
        return (r["size"], r["realization"])

    for a, b in zip(sorted(whole, key=key), sorted(parts, key=key), strict=True):
        assert a["volume_fraction"] == b["volume_fraction"]
    report = json.loads(
        (tmp_path / "rve_poisson_fibers" / "rve_report.json").read_text(encoding="utf-8")
    )
    vf = report["properties"]["volume_fraction"]
    # two sizes and a few realizations: the fit exists but is not meaningful; no RVE
    # size is given when the variance does not decrease with the volume (gamma <= 0)
    gamma = vf["fit"]["gamma"]
    assert np.isfinite(gamma) and len(vf["statistics"]) == 2
    assert bool(vf["rve"]) == (gamma > 0)
    with pytest.raises(ValueError):
        pipeline.run(load_config(cfg), task_id=3, n_tasks=3)


def test_window_study(tmp_path):
    cfg = _small("rve_voronoi_windows", tmp_path)
    cfg["rve"]["size"] = 12.0
    cfg["rve"]["window_sizes"] = [1, 2, 3]
    cfg["rve"]["min_volume"] = None
    outdir = pipeline.run(load_config(cfg))
    report = json.loads((outdir / "rve_report.json").read_text(encoding="utf-8"))
    rep = report["properties"]["volume_fraction"]
    assert report["method"] == "windows" and rep["fit"]["gamma_fixed"]
    assert 0.2 < rep["integral_range"] < 1.5  # 1.179 asymptotically (finite windows: less)
    assert (
        (outdir / "rve_report.md")
        .read_text(encoding="utf-8")
        .startswith("# RVE report (windows method)")
    )


def test_lattice_models_in_the_pipeline(tmp_path):
    cfg = _small("nucleation_growth", tmp_path)
    cfg["domain"]["size"] = 24.0
    cfg["model"]["nucleation_rate"] = 1e-3
    cfg["voxels"] = {"shape": 24}
    outdir = pipeline.run(load_config(cfg))
    row = pipeline.read_rows(outdir / "results.csv")[0]
    assert row["n_cells_present"] == row["n_grains"] > 5
    labels = mg.io.load_image(outdir / "grains_0000_labels.vti")
    micro = mg.io.load_microstructure(outdir / "grains_0000.json")
    assert np.array_equal(labels.data, micro.cell_labels)
    assert micro.orientations.shape == (row["n_grains"], 3)
    manifest = json.loads((outdir / "run.json").read_text(encoding="utf-8"))
    assert manifest["theory"]["avrami_exponent"] == 4.0
    cfg = _small("cellular_automaton", tmp_path)
    cfg["domain"]["size"] = 4.0
    cfg["voxels"] = {"shape": 16}
    row = pipeline.read_rows(pipeline.run(load_config(cfg)) / "results.csv")[0]
    assert row["steps"] <= 40 and 0.3 < row["volume_fraction"] < 0.7
    cfg["outputs"]["labels"] = ["vti"]
    with pytest.raises(ValueError, match="cell labels"):
        pipeline.run(load_config(cfg))


def test_cli_commands(tmp_path, capsys):
    cfg = _small("voronoi", tmp_path)
    cfg["model"]["n_cells"] = 20
    cfg["voxels"] = {"shape": 16}
    path = tmp_path / "v.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert cli.main(["run", str(path), "--seed", "4", "-q"]) == 0
    outdir = tmp_path / "voronoi"
    assert (outdir / "grains_0000_labels.vti").exists()
    assert (
        cli.main(
            ["analyze", str(outdir / "grains_0000_labels.vti"), "--measure", "cells", "--json"]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out.split("results written to")[-1].split("\n", 1)[1])
    assert out["n_cells_present"] <= 20
    assert cli.main(["models"]) == 0 and "boolean_spheres" in capsys.readouterr().out
    assert cli.main(["models", "tpms", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "tpms"
    assert cli.main(["template", "tpms", "-o", str(tmp_path / "t.yaml")]) == 0
    assert load_config(tmp_path / "t.yaml").model.name == "tpms"
    assert cli.main(["info"]) == 0 and "numba" in capsys.readouterr().out
    assert cli.main(["run", str(tmp_path / "missing.yaml")]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_fit_rule(tmp_path, capsys):
    """Image -> automaton rule -> configuration generating look-alike structures."""
    micro = mg.models.CellularAutomatonModel(steps=15).generate(mg.Domain.cube(24.0), seed=3)
    mg.io.save_image(micro.voxelize(), tmp_path / "target.vti")
    args = [
        "fit-rule",
        str(tmp_path / "target.vti"),
        "--lattice",
        "12",
        "--population",
        "8",
        "--generations",
        "3",
        "-o",
        str(tmp_path / "fitted.yaml"),
        "--json",
    ]
    assert cli.main(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["rule"].startswith("B") and out["cost"] >= 0
    assert set(out["target"]) == set(out["fitted"]) >= {"volume_fraction", "correlation_1"}
    cfg = load_config(tmp_path / "fitted.yaml")
    assert cfg.model.rule == out["rule"] and cfg.voxels.shape == (24, 24, 24)
    cfg.outputs.directory = tmp_path / "fitted"
    rows = pipeline.read_rows(pipeline.run(cfg) / "results.csv")
    assert 0.0 <= rows[0]["volume_fraction"] <= 1.0
    assert cli.main(["fit-rule", str(tmp_path / "missing.vti")]) == 2


def test_cli_fit_rule_on_tilings(tmp_path, capsys):
    """Rules of automata on other tilings (--tiling): the counts of their neighbourhoods,
    configurations on non-periodic domains for quasi-periodic tilings."""
    model = mg.models.CellularAutomatonModel(
        lattice="hexagonal", dimension=2, rule="majority", steps=6
    )
    micro = model.generate(mg.Domain(size=(20, 20, 1)), seed=1)
    mg.io.save_image(micro.voxelize(shape=(20, 20, 1)), tmp_path / "target.vti")
    for tiling, most in (("hexagonal", 6), ("ammann_beenker", 12)):
        out_file = tmp_path / f"{tiling}.yaml"
        args = ["fit-rule", str(tmp_path / "target.vti"), "--tiling", tiling, "--lattice", "16",
                "--population", "6", "--generations", "1", "--steps", "4", "-o", str(out_file),
                "--json"]  # fmt: skip
        assert cli.main(args) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["tiling"] == tiling and out["cost"] >= 0
        rule = mg.automata.LifeRule.parse(out["rule"])  # counts of the vertex neighbours
        assert max(rule.birth | rule.survival, default=0) <= most
        cfg = load_config(out_file)
        assert cfg.model.lattice == tiling and cfg.model.rule == out["rule"]
        assert cfg.domain.periodic == (tiling == "hexagonal")
        grown = cfg.model.generate(cfg.domain, seed=1)
        assert isinstance(grown, mg.models.TilingField)
    assert cli.main(["fit-rule", str(tmp_path / "target.vti"), "--tiling", "fcc"]) == 2
    assert "three-dimensional" in capsys.readouterr().err


def test_cli_fit_of_external_results(tmp_path, capsys):
    rng = np.random.default_rng(0)
    rows = []
    for volume in (1e3, 8e3, 27e3, 64e3):
        sd = np.sqrt(0.5 * volume**-0.8)
        rows += [{"volume": volume, "k_app": 10 + sd * rng.standard_normal()} for _ in range(40)]
    pipeline.write_rows(tmp_path / "fe.csv", rows)
    assert (
        cli.main(
            [
                "fit",
                str(tmp_path / "fe.csv"),
                "--eps",
                "0.05",
                "--n",
                "1",
                "-o",
                str(tmp_path / "rep"),
            ]
        )
        == 0
    )
    report = json.loads((tmp_path / "rep" / "rve_report.json").read_text(encoding="utf-8"))
    fit = report["properties"]["k_app"]["fit"]
    assert fit["gamma"] == pytest.approx(0.8, abs=0.2)
    assert "| eps_rel |" in capsys.readouterr().out


def test_python_dash_m_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "morphogen", "--version"], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == f"morphogen {mg.__version__}"
