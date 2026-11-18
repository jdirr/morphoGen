"""Parallel runs: processes, split tasks, records, merging, resuming and job scripts."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from morphogen import cli, parallel, pipeline
from morphogen.config import load_config


def study(tmp_path, name, **extra):
    data = {
        "model": {"type": "boolean_spheres", "radius": 0.6, "volume_fraction": 0.3},
        "domain": {"periodic": True},
        "seed": 11,
        "voxels": {"voxel_size": 0.25},
        "homogenization": {
            "materials": {0: {"young": 1.0, "poisson": 0.3}, 1: {"young": 10.0, "poisson": 0.3}},
            "properties": ["bulk_modulus"],
        },
        "rve": {"sizes": [2.0, 3.0], "realizations": [4, 3], "bootstrap": 20},
        "outputs": {"directory": str(tmp_path / name), "name": "s"},
        **extra,
    }
    return data


def same_rows(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert {k: v for k, v in x.items() if k != "time_s"} == {
            k: v for k, v in y.items() if k != "time_s"
        }


def statistics(outdir):
    report = json.loads((outdir / "rve_report.json").read_text(encoding="utf-8"))
    return report["properties"]["bulk_modulus.periodic"]["statistics"]


@pytest.mark.slow
def test_processes_and_split_tasks_give_the_same_results(tmp_path):
    serial = pipeline.run(load_config(study(tmp_path, "serial")))
    rows = pipeline.read_rows(serial / "results.csv")
    assert len(rows) == 7 and len(list((serial / "records").glob("*.json"))) == 7
    # two processes
    jobs = pipeline.run(load_config(study(tmp_path, "jobs")), jobs=2)
    same_rows(rows, pipeline.read_rows(jobs / "results.csv"))
    assert statistics(jobs) == statistics(serial)
    # three tasks, merged
    data = study(tmp_path, "split")
    for task in (0, 2):
        pipeline.run(load_config(data), task_id=task, n_tasks=3)
    split = tmp_path / "split"
    info = parallel.status(split)
    assert info["expected"] == 7 and info["done"] == 5 and len(info["missing"]) == 2
    with pytest.raises(ValueError, match="no record yet"):
        parallel.merge(split)
    pipeline.run(load_config(data), task_id=1, n_tasks=3)
    parallel.merge(split)
    same_rows(rows, pipeline.read_rows(split / "results.csv"))
    assert statistics(split) == statistics(serial)
    assert pipeline.collect_results(split) == pipeline.read_rows(split / "results.csv")
    # a lost realization is computed again by its task, the others are read back
    lost = sorted((split / "records").glob("*.json"))[3]
    lost.unlink()
    assert parallel.status(split)["done"] == 6
    pipeline.run(load_config(data), task_id=0, n_tasks=3, resume=True)
    pipeline.run(load_config(data), task_id=1, n_tasks=3, resume=True)
    assert lost.exists() and parallel.status(split)["done"] == 7
    # split runs need a seed shared by the tasks
    data["seed"] = None
    with pytest.raises(ValueError, match="seed"):
        pipeline.run(load_config(data), task_id=0, n_tasks=2)


@pytest.mark.slow
def test_cli_of_parallel_runs(tmp_path, capsys, monkeypatch):
    config = tmp_path / "study.yaml"
    config.write_text(json.dumps(study(tmp_path, "out")))
    out = tmp_path / "out"
    monkeypatch.setenv("SGE_TASK_ID", "1")  # 1-based: task 0
    assert cli.main(["run", str(config), "--task-id", "auto", "--n-tasks", "2", "-q"]) == 0
    assert cli.main(["status", str(out)]) == 1
    assert "4 of 7" in capsys.readouterr().out
    monkeypatch.delenv("SGE_TASK_ID")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    assert (
        cli.main(["run", str(config), "--task-id", "auto", "--n-tasks", "2", "-q", "-j", "2"]) == 0
    )
    assert cli.main(["merge", str(out), "-q"]) == 0
    capsys.readouterr()
    assert cli.main(["status", str(out), "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["done"] == 7 and info["merged"]
    assert (out / "rve_report.json").exists()
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID")
    assert cli.main(["run", str(config), "--task-id", "auto", "--n-tasks", "2"]) == 2


@pytest.mark.parametrize("scheduler", sorted(parallel.SCHEDULERS))
def test_job_scripts(tmp_path, scheduler):
    config = tmp_path / "study.yaml"
    config.write_text(json.dumps(study(tmp_path, "out")))
    opts = parallel.JobOptions(
        tasks=8, jobs=2, threads=4, time="02:30:00", memory="8G", setup=["module load x"]
    )
    array, merging = parallel.job_script(scheduler, config, tmp_path / "out", opts, seed=5)
    index = parallel.SCHEDULERS[scheduler]["index"]
    assert f'--task-id "{index}"' in array and "--n-tasks 8 --jobs 2 --threads 8" in array
    assert "--resume" in array and "--seed 5" in array and "module load x" in array
    assert " merge " in merging and "--task-id" not in merging
    assert {"slurm": "--array=0-7", "pbs": "-J 0-7", "torque": "-t 0-7", "sge": "-t 1-8",
            "oar": "--array 8", "lsf": "[1-8]"}[scheduler] in array  # fmt: skip
    if scheduler == "lsf":
        assert "#BSUB -W 2:30" in array
    if sys.platform != "win32" and shutil.which("bash"):  # on Windows, bash may be WSL's
        for text in (array, merging):
            path = tmp_path / "script.sh"
            path.write_text(text)
            assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    single, _ = parallel.job_script(scheduler, config, tmp_path, parallel.JobOptions(tasks=1))
    assert '--task-id "0"' in single
    with pytest.raises(ValueError):
        parallel.job_script(scheduler, config, tmp_path, parallel.JobOptions(tasks=0))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell scripts")
def test_submission_chains_the_merging_job(tmp_path, capsys, monkeypatch):
    """A fake sbatch records its arguments: the merging job waits for the array."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.txt"
    fake = bin_dir / "sbatch"
    fake.write_text(f'#!/bin/sh\necho "$@" >> {log}\necho "4242;cluster"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    config = tmp_path / "study.yaml"
    data = study(tmp_path, "out")
    data["seed"] = None
    config.write_text(json.dumps(data))
    args = ["submit", str(config), "--scheduler", "slurm", "--tasks", "4", "--submit"]
    assert cli.main(args) == 0
    assert "job array 4242" in capsys.readouterr().out
    calls = log.read_text(encoding="utf-8").splitlines()
    assert calls[0].endswith("slurm_array.sh") and "--dependency=afterok:4242" in calls[1]
    script = (tmp_path / "out" / "slurm_array.sh").read_text(encoding="utf-8")
    assert "--seed " in script  # a seed drawn once for all the tasks
    assert parallel._job_id("sge", "8812.1-4:1") == "8812"
    assert parallel._job_id("lsf", "Job <77> is submitted to queue <normal>.") == "77"
    assert parallel._job_id("oar", "[ADMISSION RULE] ...\nOAR_ARRAY_ID=31\n") == "31"
