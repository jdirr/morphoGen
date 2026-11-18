"""Parallel runs: processes on one machine, job arrays on clusters, MPI.

A run is a list of realizations (items), each with its own seed derived from the seed
of the run and its indices (:mod:`morphogen.seeding`): the results do not depend on how
the items are spread over processes, tasks or nodes.

* Every realization leaves a *record* (``records/<name>.json``: its row of results and
  its rows of apparent properties) as soon as it is done. A run started again with
  ``resume`` skips the realizations that have a record, so that an interrupted task (a
  time limit on a cluster) loses at most the realizations in progress.
* ``jobs`` processes run the realizations of a task on one machine (``spawn`` start
  method, on every platform); the threads (``threads``, by default all the cores) are
  shared between them, so that the compiled kernels and the linear algebra do not
  oversubscribe the cores.
* A run split in ``n_tasks`` tasks (``task_id`` = 0 ... n - 1) gives task ``i`` the items
  ``i, i + n, i + 2n...``: a job array of a scheduler, or the ranks of an MPI program
  (:func:`run_mpi`). :func:`merge` gathers the records of all the tasks into the results
  tables and writes the RVE report; :func:`status` counts them.
* :func:`job_script` writes the script of a job array for SLURM, PBS Pro, Torque, SGE,
  OAR or LSF, and the script of the merging job.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from morphogen._utils import to_jsonable
from morphogen.stopping import Stopped, checkpoint

log = logging.getLogger("morphogen")

#: Environment variables bounding the threads of the linear algebra libraries.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class Item:
    """One realization of a run: its position, seed key, file stem and domain size
    (``None``: the domain of the configuration), and the columns identifying its rows."""

    index: int
    key: tuple[int, ...]
    stem: str
    size: float | None
    ident: dict = field(default_factory=dict, hash=False, compare=False)


def items(cfg) -> list[Item]:
    """Realizations of a run (a batch, or the sizes and realizations of an ensemble
    RVE study)."""
    if cfg.rve is None:
        width = max(4, len(str(cfg.realizations - 1)))
        return [
            Item(r, (r,), f"{cfg.outputs.name}_{r:0{width}d}", None, {"realization": r})
            for r in range(cfg.realizations)
        ]
    if cfg.rve.method != "ensemble":
        raise ValueError("the window method of RVE studies runs one realization")
    out: list[Item] = []
    for s, n in enumerate(cfg.rve.realizations):
        size = float(cfg.rve.sizes[s])
        ident = {"size_index": s, "size": size, "volume": size**3}
        for r in range(n):
            stem = f"{cfg.outputs.name}_L{size:g}_{r:04d}"
            out.append(Item(len(out), (s, r), stem, size, {**ident, "realization": r}))
    return out


def task_items(all_items: list[Item], task_id: int, n_tasks: int) -> list[Item]:
    """The items of a task among ``n_tasks`` (``task_id`` from 0): one item in ``n_tasks``, so that
    the tasks share them evenly."""
    if n_tasks < 1 or not 0 <= task_id < n_tasks:
        raise ValueError(f"task id {task_id} must lie in [0, {n_tasks})")
    return all_items[task_id::n_tasks]


# -- records -----------------------------------------------------------------------------


def records_dir(outdir) -> Path:
    """The folder of the records of the items of a run, in its output folder."""
    return Path(outdir) / "records"


def record_path(outdir, item: Item) -> Path:
    """The record of an item of a run (``records/<item>.json`` in the output folder)."""
    return records_dir(outdir) / f"{item.stem}.json"


def write_record(outdir, item: Item, record: dict) -> Path:
    """Write a record atomically (a partial file never looks complete)."""
    path = record_path(outdir, item)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(to_jsonable(record)), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_records(outdir) -> list[dict]:
    """All the records of a run directory, in the order of the run."""
    folder = records_dir(outdir)
    out = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            log.warning("unreadable record %s (skipped)", path)
    return sorted(out, key=lambda r: r["index"])


# -- execution -----------------------------------------------------------------------------


def realize(cfg, item: Item, outdir) -> dict:
    """Generate and process one realization; returns its record."""
    from morphogen.domain import Domain
    from morphogen.pipeline import _process
    from morphogen.seeding import derive_seed

    t0 = time.time()
    domain = cfg.domain
    if item.size is not None:
        domain = Domain(
            size=(item.size,) * 3,
            origin=cfg.domain.origin,
            periodic=cfg.domain.periodic,
            unit=cfg.domain.unit,
        )
    micro = cfg.model.generate(domain, derive_seed(cfg.seed, *item.key))
    measures = cfg.analysis if cfg.rve is None else cfg.rve.properties
    values, fe = _process(micro, cfg, item.stem, Path(outdir), measures, item.index)
    if cfg.rve is None:
        row = {**item.ident, "seed": cfg.seed if cfg.seed is not None else "", **micro.summary()}
    else:
        row = {**item.ident, **micro.summary()}
    row.update(values)
    row["time_s"] = round(time.time() - t0, 4)
    record = {
        "index": item.index,
        "stem": item.stem,
        "row": row,
        "fe_rows": [{**item.ident, **f} for f in fe],
    }
    return json.loads(json.dumps(to_jsonable(record)))  # as read back from a record


def _worker_init(threads: int, level: int) -> None:
    from morphogen import settings
    from morphogen._platform import full_speed

    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    log.setLevel(level)
    settings.set_num_threads(threads)
    full_speed()  # Windows: not inherited from the parent process


def _worker(cfg, item: Item, outdir) -> dict:
    record = realize(cfg, item, outdir)
    write_record(outdir, item, record)
    return record


def threads_per_job(jobs: int, threads: int | None = None) -> int:
    """Threads of each of ``jobs`` processes sharing ``threads`` (default: the cores)."""
    total = threads if threads is not None else (os.cpu_count() or 1)
    return max(1, int(total) // max(1, int(jobs)))


@contextmanager
def process_pool(jobs: int, threads: int) -> Iterator[ProcessPoolExecutor]:
    """Pool of ``jobs`` processes (``spawn``) with ``threads`` threads each."""
    import multiprocessing

    env = {name: str(threads) for name in THREAD_VARIABLES}
    saved = {name: os.environ.get(name) for name in env}
    os.environ.update(env)  # inherited by the processes: read when they import NumPy
    try:
        with ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init,
            initargs=(threads, log.getEffectiveLevel()),
        ) as pool:
            yield pool
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def execute(cfg, todo: list[Item], outdir, *, jobs: int = 1, resume: bool = False) -> list[dict]:
    """Records of the items ``todo`` (in their order), computed by ``jobs`` processes;
    with ``resume``, the items that have a record are not computed again. Each
    realization is a checkpoint of :mod:`morphogen.stopping` (a stop cancels those the
    processes have not started)."""
    outdir = Path(outdir)
    records: dict[int, dict] = {}
    pending = []
    for item in todo:
        path = record_path(outdir, item)
        if resume and path.exists():
            try:
                records[item.index] = json.loads(path.read_text(encoding="utf-8"))
                continue
            except ValueError:
                pass
        pending.append(item)
    if resume and records:
        log.info("resuming: %d of %d realizations already done", len(records), len(todo))
    if jobs <= 1 or len(pending) <= 1:
        for item in pending:
            checkpoint()
            t0 = time.time()
            records[item.index] = _worker(cfg, item, outdir)
            log.info("%s done in %.2f s", item.stem, time.time() - t0)
    else:
        threads = threads_per_job(jobs, cfg.threads)
        log.info("%d processes of %d threads", jobs, threads)
        with process_pool(jobs, threads) as pool:
            futures = {pool.submit(_worker, cfg, item, outdir): item for item in pending}
            try:
                for future in as_completed(futures):
                    item = futures[future]
                    records[item.index] = future.result()
                    log.info("%s done (%d/%d)", item.stem, len(records), len(todo))
                    checkpoint()
            except Stopped:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
    return [records[item.index] for item in todo]


def rows_of(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Results rows and homogenization rows of records."""
    rows = [r["row"] for r in records]
    fe_rows = [f for r in records for f in r["fe_rows"]]
    return rows, fe_rows


# -- MPI -------------------------------------------------------------------------------------


def run_mpi(cfg, *, jobs: int = 1, resume: bool = False) -> Path:
    """Run with the ranks of an MPI program (``mpirun -n N morphogen run config --mpi``):
    rank ``i`` of ``N`` is task ``i``; rank 0 merges the results when all are done."""
    try:
        from mpi4py import MPI
    except ImportError:  # pragma: no cover - depends on the environment
        raise ImportError("MPI runs need the optional 'mpi4py' package") from None
    from morphogen import pipeline

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if cfg.seed is None:  # one seed for all the ranks
        cfg.seed = comm.bcast(np.random.SeedSequence().entropy if rank == 0 else None, root=0)
    outdir = pipeline.run(cfg, task_id=rank, n_tasks=size, jobs=jobs, resume=resume)
    comm.Barrier()
    if rank == 0 and size > 1:
        merge(outdir)
    comm.Barrier()
    return outdir


# -- merging and status ------------------------------------------------------------------


def _manifest_value(outdir: Path, key: str):
    """An entry of the manifest of a run (``run.json``, or that of one of its tasks)."""
    for path in [outdir / "run.json", *sorted(outdir.glob("run_task*of*.json"))]:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")).get(key)
    return None


def _run_config(outdir: Path):
    """Configuration of a run directory (from its run.json or run_task*.json)."""
    from morphogen.config import load_config

    files = [outdir / "run.json", *sorted(outdir.glob("run_task*of*.json"))]
    for path in files:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            source = dict(data["config"])
            if data.get("seed") is not None:
                source["seed"] = data["seed"]
            cfg = load_config(source, base_dir=outdir)
            cfg.outputs.directory = outdir
            return cfg
    raise ValueError(f"{outdir}: no run.json (not a morphoGen run directory)")


def merge(outdir, *, report: bool = True) -> Path:
    """Gather the records of all the tasks of a run into ``results.csv`` and
    ``homogenization.csv``, and write the RVE report of an ensemble study."""
    from morphogen import pipeline

    outdir = Path(outdir)
    cfg = _run_config(outdir)
    expected = items(cfg)
    records = read_records(outdir)
    done = {r["stem"] for r in records}
    missing = [it.stem for it in expected if it.stem not in done]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(expected)} realizations have no record yet (e.g. "
            f"{missing[0]}); run the missing tasks again with --resume, see 'morphogen status'"
        )
    rows, fe_rows = rows_of(records)
    pipeline.write_rows(outdir / "results.csv", rows)
    if fe_rows:
        pipeline.write_rows(outdir / "homogenization.csv", fe_rows)
    if report and cfg.rve is not None and rows:
        pipeline.write_report(rows, outdir, cfg.rve, cfg, fe_rows=fe_rows)
    database = _manifest_value(outdir, "database")
    if database is not None:
        from morphogen.results import record_run

        record_run(outdir, database)
    log.info("merged %d realizations in %s", len(records), outdir)
    return outdir


def status(outdir) -> dict:
    """Realizations done and missing in a run directory."""
    outdir = Path(outdir)
    cfg = _run_config(outdir)
    expected = items(cfg)
    records = read_records(outdir)
    done = {r["stem"] for r in records}
    times = [r["row"].get("time_s", 0.0) for r in records]
    missing = [it.stem for it in expected if it.stem not in done]
    return {
        "directory": str(outdir),
        "expected": len(expected),
        "done": len(expected) - len(missing),
        "missing": missing,
        "mean_time_s": float(np.mean(times)) if times else float("nan"),
        "merged": (outdir / "results.csv").exists(),
    }


# -- job scripts -------------------------------------------------------------------------------

#: Array directive, task index (0-based) and submission command of each scheduler.
SCHEDULERS = {
    "slurm": {
        "prefix": "#SBATCH",
        "array": "--array=0-{last}",
        "index": "${SLURM_ARRAY_TASK_ID}",
        "name": "--job-name={name}",
        "cores": "--cpus-per-task={cores}",
        "time": "--time={time}",
        "memory": "--mem={memory}",
        "queue": "--partition={queue}",
        "account": "--account={account}",
        "log": ["--output={logs}/%x_%A_%a.log"],
        "submit": ["sbatch", "--parsable"],
        "after": ["--dependency=afterok:{job}"],
    },
    "pbs": {
        "prefix": "#PBS",
        "array": "-J 0-{last}",
        "index": "${PBS_ARRAY_INDEX}",
        "name": "-N {name}",
        "cores": "-l select=1:ncpus={cores}",
        "time": "-l walltime={time}",
        "memory": "-l mem={memory}",
        "queue": "-q {queue}",
        "account": "-A {account}",
        "log": ["-o {logs}/", "-j oe"],
        "submit": ["qsub"],
        "after": ["-W", "depend=afterok:{job}"],
    },
    "torque": {
        "prefix": "#PBS",
        "array": "-t 0-{last}",
        "index": "${PBS_ARRAYID}",
        "name": "-N {name}",
        "cores": "-l nodes=1:ppn={cores}",
        "time": "-l walltime={time}",
        "memory": "-l mem={memory}",
        "queue": "-q {queue}",
        "account": "-A {account}",
        "log": ["-o {logs}/", "-j oe"],
        "submit": ["qsub"],
        "after": ["-W", "depend=afterokarray:{job}"],
    },
    "sge": {
        "prefix": "#$",
        "array": "-t 1-{count}",
        "index": "$((SGE_TASK_ID - 1))",
        "name": "-N {name}",
        "cores": "-pe smp {cores}",
        "time": "-l h_rt={time}",
        "memory": "-l h_vmem={memory}",
        "queue": "-q {queue}",
        "account": "-A {account}",
        "log": ["-o {logs}/", "-j y"],
        "submit": ["qsub", "-terse"],
        "after": ["-hold_jid", "{job}"],
    },
    "oar": {
        "prefix": "#OAR",
        "array": "--array {count}",
        "index": "$((OAR_ARRAY_INDEX - 1))",
        "name": "-n {name}",
        "cores": "-l /nodes=1/core={cores},walltime={time}",
        "time": None,  # with the cores
        "memory": None,
        "queue": "-q {queue}",
        "account": "--project {account}",
        "log": ["-O {logs}/%jobid%.log", "-E {logs}/%jobid%.log"],
        "submit": ["oarsub", "-S"],
        "after": ["--anterior={job}"],
    },
    "lsf": {
        "prefix": "#BSUB",
        "array": None,  # in the job name
        "index": "$((LSB_JOBINDEX - 1))",
        "name": "-J {name}[1-{count}]",
        "cores": "-n {cores}",
        "time": "-W {time}",
        "memory": "-M {memory}",
        "queue": "-q {queue}",
        "account": "-P {account}",
        "log": ["-o {logs}/%J_%I.log"],
        "submit": ["bsub"],
        "after": ["-w", "done({job})"],
    },
}


@dataclass
class JobOptions:
    """Resources of the tasks of a job array."""

    tasks: int
    jobs: int = 1
    threads: int = 1
    time: str = "01:00:00"
    memory: str | None = None
    queue: str | None = None
    account: str | None = None
    name: str = "morphogen"
    setup: list[str] = field(default_factory=list)  # shell lines (modules, environments)

    @property
    def cores(self) -> int:
        return self.jobs * self.threads


def _directives(scheduler: str, opts: JobOptions, logs: Path, *, array: bool) -> list[str]:
    spec = SCHEDULERS[scheduler]
    time_limit = opts.time
    if scheduler == "lsf":  # LSF: [hours:]minutes
        parts = [int(x) for x in time_limit.split(":")]
        minutes = parts[0] * 60 + parts[1] + (1 if len(parts) > 2 and parts[2] else 0)
        time_limit = f"{minutes // 60}:{minutes % 60:02d}" if len(parts) > 2 else time_limit
    values = {
        "name": opts.name if array else f"{opts.name}-merge",
        "last": opts.tasks - 1,
        "count": opts.tasks,
        "cores": opts.cores if array else 1,
        "time": time_limit,
        "memory": opts.memory,
        "queue": opts.queue,
        "account": opts.account,
        "logs": logs,
    }
    array = array and opts.tasks > 1
    name = spec["name"] if (array or scheduler != "lsf") else "-J {name}"
    lines = [name]
    if array and spec["array"]:
        lines.append(spec["array"])
    lines += [spec["cores"], *spec["log"]]
    for key in ("time", "memory", "queue", "account"):
        if spec[key] and values[key] is not None:
            lines.append(spec[key])
    return [f"{spec['prefix']} {line.format(**values)}" for line in lines]


def job_script(
    scheduler: str,
    config,
    outdir,
    opts: JobOptions,
    *,
    seed: int | None = None,
    executable: str | None = None,
) -> tuple[str, str]:
    """Scripts of the job array running a configuration in ``opts.tasks`` tasks (each
    with ``opts.jobs`` processes of ``opts.threads`` threads, resumable), and of the job
    merging the results."""
    if scheduler not in SCHEDULERS:
        raise ValueError(f"scheduler: one of {sorted(SCHEDULERS)}")
    if opts.tasks < 1 or opts.jobs < 1 or opts.threads < 1:
        raise ValueError("tasks, jobs and threads: positive integers")
    outdir = Path(outdir).resolve()
    logs = outdir / "logs"
    exe = executable or f"{shlex.quote(sys.executable)} -m morphogen"
    index = SCHEDULERS[scheduler]["index"] if opts.tasks > 1 else "0"
    command = [
        exe,
        "run",
        shlex.quote(str(Path(config).resolve())),
        "-o",
        shlex.quote(str(outdir)),
        "--task-id",
        f'"{index}"',
        "--n-tasks",
        str(opts.tasks),
        "--jobs",
        str(opts.jobs),
        "--threads",
        str(opts.cores),
        "--resume",
        "--quiet",
    ]
    if seed is not None:
        command += ["--seed", str(seed)]
    header = ["#!/bin/bash"]
    body = [
        "set -euo pipefail",
        *opts.setup,
        f"mkdir -p {shlex.quote(str(logs))}",
    ]
    array = "\n".join(
        [*header, *_directives(scheduler, opts, logs, array=True), *body, " ".join(command)]
    )
    merge_cmd = f"{exe} merge {shlex.quote(str(outdir))}"
    merging = "\n".join(
        [*header, *_directives(scheduler, opts, logs, array=False), *body, merge_cmd]
    )
    return array + "\n", merging + "\n"


def _job_id(scheduler: str, output: str) -> str:
    """Identifier of a submitted job in the output of the submission command."""
    text = output.strip()
    last = text.splitlines()[-1].strip() if text else ""
    if scheduler == "slurm":
        return last.split(";")[0]
    if scheduler in ("pbs", "torque"):
        return last
    if scheduler == "lsf":
        match = re.search(r"<(\d+)>", text)
    elif scheduler == "oar":
        match = re.search(r"OAR_ARRAY_ID=(\d+)", text) or re.search(r"OAR_JOB_ID=(\d+)", text)
    else:
        match = re.match(r"\d+", last)
    if match is None:
        raise RuntimeError(f"no job id in the output of the submission: {text!r}")
    return match.group(1) if match.groups() else match.group(0)


def submit(scheduler: str, array_script: Path, merge_script: Path) -> list[str]:
    """Submit the job array, then the merging job after it; returns their ids."""
    spec = SCHEDULERS[scheduler]

    def call(extra, script):
        if scheduler == "lsf":  # bsub reads the directives of a script from its input
            with open(script, encoding="utf-8") as f:
                res = subprocess.run(
                    [*spec["submit"], *extra], stdin=f, capture_output=True, text=True, check=True
                )
        else:
            res = subprocess.run(
                [*spec["submit"], *extra, str(script)], capture_output=True, text=True, check=True
            )
        return _job_id(scheduler, res.stdout)

    for script in (array_script, merge_script):
        Path(script).chmod(0o755)
    first = call([], array_script)
    after = [part.format(job=first) for part in spec["after"]]
    return [first, call(after, merge_script)]
