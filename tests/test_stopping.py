"""Cooperative stops (:mod:`morphogen.stopping`): the checkpoints of the long loops, and
the jobs of the interface stopped from the page (tools, simulations, optimizations)."""

import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from morphogen import stopping
from morphogen.domain import Domain
from morphogen.gui import api, tools
from morphogen.gui.server import GuiServer
from morphogen.gui.tools import ToolJob
from morphogen.homogenization.solvers import conjugate_gradient
from morphogen.models.base import model_from_config
from morphogen.stopping import Stopped, checkpoint, stoppable


def test_checkpoints():
    """A checkpoint raises only in the thread whose stop is set (the innermost), and a stop
    is not an error: the handlers that fall back on another method let it through."""
    checkpoint()  # no stop: nothing
    event = threading.Event()
    with stoppable(event):
        checkpoint()
        event.set()
        assert stopping.requested()
        with pytest.raises(Stopped):
            checkpoint()
        seen = []
        other = threading.Thread(target=lambda: seen.append(stopping.requested()))
        other.start()
        other.join()
        assert seen == [False]  # another thread has its own stop
        with stoppable(threading.Event()):
            checkpoint()  # the inner stop, not set
        with pytest.raises(Stopped):
            checkpoint()  # the outer one again
    checkpoint()
    assert not issubclass(Stopped, Exception)
    with stoppable(event), pytest.raises(Stopped):
        try:
            checkpoint()
        except Exception:  # a fallback
            pytest.fail("a stop caught as an error")


def test_solvers_stop_at_an_iteration():
    a = np.diag(np.arange(1.0, 101.0))
    rhs = np.ones(100)
    x, _, res = conjugate_gradient(lambda v: a @ v, lambda r: r.copy(), rhs, 1e-10, 500)
    assert res < 1e-10 and np.allclose(a @ x, rhs)
    event = threading.Event()
    event.set()
    with stoppable(event), pytest.raises(Stopped):
        conjugate_gradient(lambda v: a @ v, lambda r: r.copy(), rhs, 1e-10, 500)


def _endless(started):
    """A tool that makes a morphology, then reports its progress until it is stopped."""

    def run(p, ctx):
        ctx.add("morphology", "half made", micro=None)
        started.set()
        for k in range(10_000):
            ctx.progress(k / 10_000, f"step {k}")
            time.sleep(0.01)
        return {}

    return run


def test_a_tool_stopped_keeps_nothing(tmp_path):
    session = api.Session(tmp_path)
    started = threading.Event()
    tools._load()
    tools.tool("demo_endless", "statistics", "Endless", "")(_endless(started))
    try:
        job = ToolJob({"tool": "demo_endless", "params": {}}, session).start()
        assert started.wait(60)
        assert len(session.workspace()) == 1  # the morphology it made
        job.stop()
        job.thread.join(timeout=60)
    finally:
        tools.TOOLS.pop("demo_endless")
    status = job.status()
    assert status["state"] == "stopped" and "result" not in status and "error" not in status
    assert session.workspace() == []  # neither the run nor what it made


def test_a_simulation_stopped_keeps_nothing(tmp_path):
    """Stopped at the first iteration of its solver, a simulation leaves the workspace as
    it was."""
    session = api.Session(tmp_path)
    model = model_from_config({"type": "boolean_spheres", "radius": 0.5, "volume_fraction": 0.3})
    micro = model.generate(Domain(size=2.0, periodic=True), 3)
    key, _ = session.folder()
    session.add(key, "morphology", "spheres", micro=micro, model=model.to_config(), seed=3,
                domain={"size": 2.0, "periodic": True})  # fmt: skip
    job = api.SimulationJob(
        {"source": key, "method": "fft", "shape": 12,
         "materials": {"0": {"young": 1.0, "poisson": 0.3}, "1": {"young": 10.0, "poisson": 0.3}}},
        session,
    )  # fmt: skip
    job.stop()
    job._run()  # synchronously
    assert job.status()["state"] == "stopped"
    assert [i["key"] for i in session.workspace()] == [key]


def test_an_optimization_stopped_keeps_its_design(tmp_path):
    session = api.Session(tmp_path)
    job = api.OptimizationJob(
        {"grid": [16, 16, 1], "volume_fraction": 0.4, "settings": {"iterations": 200}}, session
    )
    job.stop()
    job._run()  # synchronously: stopped at the end of its first iteration
    status = job.status()
    assert status["state"] == "done", status.get("error")
    result = status["result"]
    assert result["stopped"] and not result["converged"] and result["iterations"] == 1
    assert result["item"]["kind"] == "morphology"  # the design so far, in the workspace


def test_jobs_stopped_from_the_page(tmp_path):
    server = GuiServer(("127.0.0.1", 0), tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path, data):
        req = urllib.request.Request(
            base + path,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    started = threading.Event()
    tools._load()
    tools.tool("demo_endless", "statistics", "Endless", "")(_endless(started))
    try:
        job = post("/api/tool", {"tool": "demo_endless", "params": {}})["job"]
        assert started.wait(60)
        assert post("/api/job/stop", {"job": job}) == {"stopping": job}
        server.jobs[job].thread.join(timeout=60)
        with urllib.request.urlopen(f"{base}/api/job/{job}", timeout=60) as r:
            assert json.loads(r.read())["state"] == "stopped"
        with pytest.raises(urllib.error.HTTPError) as err:
            post("/api/job/stop", {"job": "nothing"})
        assert err.value.code == 400 and "no job" in json.loads(err.value.read())["error"]
    finally:
        tools.TOOLS.pop("demo_endless")
        server.shutdown()
        server.server_close()
