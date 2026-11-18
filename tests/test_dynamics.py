"""Explicit dynamics with OpenRadioss (morphogen.dynamics): the installation found (the
places, the executables, the environment), the runs and their results read back with a
stand-in of OpenRadioss (fake_openradioss: its Starter, Engine and converters write the
outputs shaped like OpenRadioss's for a bar of the apparent modulus E phi), their stops
and failures, the installation of a release from a GitHub API served locally, and the
command line."""

from __future__ import annotations

import io as stdio
import json
import os
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from fake_openradioss import fake_installation
from morphogen import dynamics
from morphogen.domain import Domain
from morphogen.models.base import model_from_config
from morphogen.stopping import Stopped, stoppable

posix = pytest.mark.skipif(sys.platform == "win32", reason="the stand-in runs by its shebang")
ALUMINIUM = {
    "young": 70e3,
    "poisson": 0.3,
    "density": 2.7e-9,
    "plasticity": {"yield": 200.0, "hardening": 500.0},
}


@pytest.fixture
def gyroid():
    spec = {"type": "tpms", "surface": "gyroid", "volume_fraction": 0.3, "cell_size": 5.0}
    return model_from_config(spec).generate(Domain(size=10.0), 1)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No installation of OpenRadioss found but those of the test."""
    for name in dynamics.openradioss.ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dynamics.openradioss, "default_home", lambda: tmp_path / "home")
    monkeypatch.setenv("PATH", str(tmp_path / "no-bin"))
    return tmp_path


def test_installation_found(clean_env, monkeypatch):
    root = fake_installation(clean_env / "OpenRadioss")
    with pytest.raises(FileNotFoundError, match="morphogen radioss install"):
        dynamics.find_openradioss()
    assert not dynamics.has_openradioss()
    for place in (root, root / "exec", root / "exec" / "starter_linux64_gf"):
        found = dynamics.find_openradioss(place)
        assert found.root == root.resolve() and found.suffix == "linux64_gf"
        assert found.engine.name == "engine_linux64_gf" and found.th_to_csv is not None
    with pytest.raises(FileNotFoundError, match="no installation"):
        dynamics.find_openradioss(clean_env / "nowhere")
    monkeypatch.setenv("OPENRADIOSS_PATH", str(root))
    assert dynamics.find_openradioss().root == root.resolve()
    monkeypatch.delenv("OPENRADIOSS_PATH")
    unpacked = clean_env / "home"  # an archive unpacked into the default folder
    fake_installation(unpacked / "OpenRadioss")
    (unpacked / dynamics.openradioss.RELEASE_FILE).write_text(json.dumps({"tag": "latest-1"}))
    found = dynamics.find_openradioss()
    assert found.root == (unpacked / "OpenRadioss").resolve() and found.version == "latest-1"
    env = found.environment(3, base={"LD_LIBRARY_PATH": "/x"})
    assert env["RAD_CFG_PATH"] == str(found.root / "hm_cfg_files")
    assert env["OMP_NUM_THREADS"] == "3" and env["RAD_H3D_PATH"].endswith("linux64")
    if sys.platform.startswith("linux"):
        parts = env["LD_LIBRARY_PATH"].split(os.pathsep)
        assert parts[-1] == "/x" and any(
            p.endswith(str(Path("hm_reader") / "linux64")) for p in parts
        )
    assert found.describe()["release"] == "latest-1"


@posix
def test_permissions_restored(clean_env):
    root = fake_installation(clean_env / "OpenRadioss")
    starter = root / "exec" / "starter_linux64_gf"
    starter.chmod(0o644)  # an archive unpacked without its permission bits
    assert os.access(dynamics.find_openradioss(root).starter, os.X_OK)


@posix
def test_explicit_test_end_to_end(clean_env, gyroid):
    root = fake_installation(clean_env / "OpenRadioss")
    seen = []
    test = dynamics.explicit_test(
        gyroid,
        {0: "void", 1: ALUMINIUM},
        {"kind": "compression", "strain": 0.3, "strain_rate": 1000},
        clean_env / "run",
        resolution=10,
        frames=6,
        history=40,
        installation=root,
        progress=lambda f, m: seen.append(f),
    )
    assert seen[0] == 0.0 and seen[-1] == 1.0 and np.all(np.diff(seen) >= -1e-12)
    result = test.result
    assert result.source == "time history" and len(result.frames) == 7
    m = result.manifest
    phi = m["solid_volume"] / m["volume"]
    c = result.curves
    assert c["strain"][-1] == pytest.approx(0.3)
    assert c["stress_upper"][-1] == pytest.approx(70e3 * phi * 0.3, rel=1e-6)  # the bar's
    assert c["stress_lower"] == pytest.approx(c["stress_upper"], rel=1e-6, abs=1e-6)
    s = dynamics.summary(result)
    assert s["absorbed energy per volume"] == pytest.approx(0.5 * 70e3 * phi * 0.3**2, rel=0.02)
    assert s["face stress gap (second half)"] < 1e-6 and s["peak stress"] == c["stress_upper"][-1]
    assert set(result.energies) >= {"time", "internal", "kinetic"}
    frame = result.frames[-1]
    assert np.allclose(frame.displacement[:, 2].min(), -3.0)
    assert np.allclose(frame.cells["von_mises"], 70e3 * phi * 0.3) and frame.status.min() == 1
    # the forces from the reactions of the animations when the time history is missing
    (test.directory / f"{m['name']}T01.csv").unlink()
    again = dynamics.read_explicit_run(test.directory)
    assert again.source == "animations" and again.energies == {}
    assert again.curves["stress_upper"][-1] == pytest.approx(c["stress_upper"][-1], rel=1e-6)
    # the frames for the 3D views: tetrahedra (six per brick) with the fields
    from morphogen.viz.surfaces import vtu_surface

    files = dynamics.write_frames(result, clean_env / "view")
    assert len(files) == 7
    surface = vtu_surface(files[-1])
    assert surface.n_triangles > 0
    tables = dynamics.write_tables(result, clean_env / "view")
    assert [p.name for p in tables] == ["curves.csv", "energies.csv"]


@posix
def test_shear_impact_and_tension(clean_env, gyroid):
    root = fake_installation(clean_env / "OpenRadioss")
    materials = {0: "void", 1: {"young": 70e3, "poisson": 0.3, "density": 2.7e-9}}
    shear = dynamics.explicit_test(
        gyroid,
        materials,
        {"kind": "shear", "strain": 0.1, "strain_rate": 500},
        clean_env / "shear",
        resolution=10,
        frames=3,
        history=10,
        installation=root,
    )
    assert shear.result.curves["strain"][-1] == pytest.approx(0.1)
    impact = dynamics.explicit_test(
        gyroid,
        materials,
        {"kind": "impact", "velocity": 1e4, "strain": 0.2},
        clean_env / "impact",
        resolution=10,
        frames=3,
        history=10,
        installation=root,
    )
    result = impact.result
    assert result.source == "momentum" and "force_wall" in result.curves
    s = dynamics.summary(result)
    assert s["initial kinetic energy"] == pytest.approx(0.5 * result.manifest["mass"] * 1e8)
    assert s["peak force on the wall"] > 0


@posix
def test_failures_and_stops(clean_env, gyroid):
    root = fake_installation(clean_env / "OpenRadioss")
    test = dynamics.explicit_test(
        gyroid,
        {0: "void", 1: ALUMINIUM},
        "compression",
        clean_env / "run",
        resolution=10,
        run=False,
    )
    assert test.run is None and test.result is None
    starter = test.inputs.starter
    starter.write_text(starter.read_text().replace("/END", "# FAKE_STARTER_ERROR\n/END"))
    with pytest.raises(RuntimeError, match="MATERIAL DEFINITION"):
        dynamics.run_openradioss(test.directory, installation=root)
    starter.write_text(starter.read_text().replace("# FAKE_STARTER_ERROR\n", ""))
    event = threading.Event()
    event.set()
    with stoppable(event), pytest.raises(Stopped):
        dynamics.run_openradioss(test.directory, installation=root)
    run = dynamics.run_openradioss(test.directory, installation=root, convert=False)
    assert run.frames == [] and run.history is None
    bare = fake_installation(clean_env / "bare", converters=False)
    converted = dynamics.convert_outputs(run, bare)
    assert converted.frames == [] and any("anim_to_vtk" in w for w in converted.warnings)
    assert any("th_to_csv" in w for w in converted.warnings)
    converted = dynamics.convert_outputs(run, root)
    assert len(converted.frames) == run_frames(test) and converted.history.suffix == ".csv"
    with pytest.raises(RuntimeError, match="MPI"):
        dynamics.run_openradioss(test.directory, installation=root, mpi=2)


def run_frames(test) -> int:
    return int(test.inputs.manifest["frames"]) + 1


class _Releases(BaseHTTPRequestHandler):
    """A GitHub API of one release, with the archive of a stand-in of OpenRadioss."""

    archive = b""

    def log_message(self, *args):
        pass

    def do_GET(self):
        host = f"http://127.0.0.1:{self.server.server_address[1]}"
        if self.path.endswith("/releases/latest") or self.path.endswith("/releases/tags/t1"):
            assets = [
                {"name": n, "size": len(self.archive), "browser_download_url": f"{host}/{n}"}
                for n in (
                    "OpenRadioss_win64.zip",
                    "OpenRadioss_linuxa64.zip",
                    "OpenRadioss_linux64.zip",
                )
            ]
            body = json.dumps({"tag_name": "t1", "name": "release t1", "assets": assets}).encode()
        elif self.path.endswith(".zip"):
            body = self.archive
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_install_a_release(clean_env):
    src = fake_installation(clean_env / "src" / "OpenRadioss")
    buffer = stdio.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for path in sorted(src.rglob("*")):
            name = str(Path("OpenRadioss") / path.relative_to(src)).replace(os.sep, "/")
            info = zipfile.ZipInfo(name + ("/" if path.is_dir() else ""))
            info.external_attr = (0o755 if "exec" in path.parts else 0o644) << 16
            zf.writestr(info, b"" if path.is_dir() else path.read_bytes())
    _Releases.archive = buffer.getvalue()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Releases)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    api = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        release = dynamics.release_info(api=api)
        assert release["tag"] == "t1" and len(release["assets"]) == 3
        assert (
            dynamics.release_asset(release, "Linux", "x86_64")["name"] == "OpenRadioss_linux64.zip"
        )
        assert (
            dynamics.release_asset(release, "Linux", "aarch64")["name"]
            == "OpenRadioss_linuxa64.zip"
        )
        assert (
            dynamics.release_asset(release, "Windows", "AMD64")["name"] == "OpenRadioss_win64.zip"
        )
        with pytest.raises(RuntimeError, match="no archive for Darwin"):
            dynamics.release_asset(release, "Darwin", "arm64")
        with pytest.raises(RuntimeError, match="no release of OpenRadioss is published"):
            dynamics.release_asset(release, "Linux", "riscv64")
        seen = []
        target = clean_env / "installed"
        target.mkdir()
        (target / "stale").write_text("an earlier installation")
        install = dynamics.install_openradioss(
            target,
            tag="t1",
            api=api,
            system="Linux",
            machine="x86_64",
            progress=lambda f, m: seen.append(m),
        )
        assert install.version == "t1" and not (target / "stale").exists()
        assert install.root == (target / "OpenRadioss").resolve() and "unpacking" in " ".join(seen)
        if sys.platform != "win32":
            assert os.access(install.engine, os.X_OK)
        bad = stdio.BytesIO()
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("../escape.txt", b"outside")
        _Releases.archive = bad.getvalue()
        with pytest.raises(RuntimeError, match="leaves the folder"):
            dynamics.install_openradioss(
                clean_env / "other", api=api, system="Linux", machine="x86_64"
            )
        assert not (clean_env / "escape.txt").exists()
    finally:
        server.shutdown()


@posix
def test_command_line(clean_env, gyroid, capsys):
    from morphogen import io
    from morphogen.cli import main

    root = fake_installation(clean_env / "OpenRadioss")
    io.save_microstructure(gyroid, clean_env / "gyroid.json")
    (clean_env / "mats.yaml").write_text(
        "0: void\n1: {young: 7.0e4, poisson: 0.3, density: 2.7e-9}\n"
    )
    assert main(["radioss", "info"]) == 1
    assert "not found" in capsys.readouterr().out
    assert main(["radioss", "info", "--path", str(root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["found"] is True
    out = clean_env / "run"
    assert (
        main(
            [
                "radioss",
                "export",
                str(clean_env / "gyroid.json"),
                "--materials",
                str(clean_env / "mats.yaml"),
                "--voxels",
                "10",
                "--strain",
                "0.1",
                "--rate",
                "500",
                "-o",
                str(out),
                "--frames",
                "4",
                "--json",
            ]
        )
        == 0
    )
    written = json.loads(capsys.readouterr().out)
    assert written["name"] == "tpms" and (out / "tpms_0000.rad").is_file()
    assert (
        main(
            [
                "radioss",
                "run",
                str(out),
                "--path",
                str(root),
                "--threads",
                "1",
                "--json",
                "--frames-vtu",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["final strain"] == pytest.approx(0.1) and summary["frames written"] == 5
    assert (out / "curves.csv").is_file() and len(list((out / "frames").glob("*.vtu"))) == 5
    assert main(["radioss", "read", str(out), "--no-convert", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["forces from"] == "time history"
    assert main(["radioss", "export", str(clean_env / "gyroid.json")]) == 2
    assert "--materials" in capsys.readouterr().err
