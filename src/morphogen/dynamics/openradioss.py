"""OpenRadioss on this machine: where it is installed (:func:`find_openradioss`), the
environment its programs need, its runs (:func:`run_openradioss`: the Starter, then the
Engine, their progress followed in their listings, stopped at a checkpoint of
:mod:`morphogen.stopping`), the conversion of its outputs (``anim_to_vtk`` for the
animations, ``th_to_csv`` for the time histories) and the installation of a release that
the OpenRadioss project publishes on GitHub (:func:`install_openradioss`).

OpenRadioss (https://github.com/OpenRadioss/OpenRadioss, AGPL-3.0) is a separate
program: morphoGen writes its inputs, runs its executables and reads their outputs, and
neither ships nor links it. An installation is the folder of a release (its ``exec``,
``hm_cfg_files`` and ``extlib`` folders), found from, in order: the path given, the
environment variables ``MORPHOGEN_OPENRADIOSS`` and ``OPENRADIOSS_PATH`` (that of
OpenRadioss's documentation), the folder where :func:`install_openradioss` puts it
(``~/morphogen/openradioss``), and a Starter on the ``PATH``."""

from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from morphogen._version import __version__
from morphogen.stopping import Stopped, checkpoint

#: The environment variables of the folder of an installation, in order
ENVIRONMENT = ("MORPHOGEN_OPENRADIOSS", "OPENRADIOSS_PATH")
#: The repository of OpenRadioss, whose releases :func:`install_openradioss` installs
REPOSITORY = "OpenRadioss/OpenRadioss"
GITHUB_API = "https://api.github.com"
#: The file that records the release installed in its folder
RELEASE_FILE = "morphogen_release.json"
#: The words of the names of the archives of the releases, by system and machine
ASSETS = {
    ("Linux", "x86_64"): ("linux64",),
    ("Linux", "aarch64"): ("linuxa64", "linux_arm64", "aarch64", "arm64"),
    ("Linux", "arm64"): ("linuxa64", "linux_arm64", "aarch64", "arm64"),
    ("Windows", "AMD64"): ("win64",),
    ("Windows", "x86_64"): ("win64",),
    ("Darwin", "arm64"): ("macos", "mac", "darwin"),
    ("Darwin", "x86_64"): ("macos", "mac", "darwin"),
}


def default_home() -> Path:
    """The folder where :func:`install_openradioss` installs OpenRadioss by default."""
    return Path.home() / "morphogen" / "openradioss"


@dataclass
class OpenRadioss:
    """An installation of OpenRadioss: its folder (``root``), the Starter, the Engine
    (and its MPI version), the converters of its outputs when it has them, the suffix of
    the names of its executables (``linux64_gf``, ``win64``...) and the release
    installed (when :func:`install_openradioss` installed it)."""

    root: Path
    starter: Path
    engine: Path
    engine_mpi: Path | None = None
    anim_to_vtk: Path | None = None
    th_to_csv: Path | None = None
    suffix: str = ""
    release: dict = field(default_factory=dict)

    @property
    def version(self) -> str | None:
        return self.release.get("tag")

    def library_folders(self) -> list[Path]:
        """The folders of the shared libraries of its programs (the reader of the
        inputs, H3D, the runtimes of the compilers), for the platform of the executables."""
        extlib = self.root / "extlib"
        if not extlib.is_dir():
            return []
        names = {self.suffix.split("_")[0], self.suffix}
        found = []
        for folder in sorted(extlib.rglob("*")):
            if folder.is_dir() and folder.name in names:
                found.append(folder)
        return found

    def environment(self, threads: int | None = None, base: dict | None = None) -> dict:
        """The environment of its programs: ``OPENRADIOSS_PATH``, ``RAD_CFG_PATH`` (the
        configuration of the reader of the inputs), ``RAD_H3D_PATH``, the stack of the
        OpenMP threads, their number, and the folders of its libraries on the search path
        of the system."""
        env = dict(os.environ if base is None else base)
        env["OPENRADIOSS_PATH"] = str(self.root)
        env["RAD_CFG_PATH"] = str(self.root / "hm_cfg_files")
        folders = self.library_folders()
        h3d = [f for f in folders if "h3d" in f.parts]
        if h3d:
            env["RAD_H3D_PATH"] = str(h3d[0])
        env.setdefault("OMP_STACKSIZE", "400m")
        env.setdefault("KMP_STACKSIZE", "400m")
        if threads:
            env["OMP_NUM_THREADS"] = str(int(threads))
        if folders:
            name = {"win32": "PATH", "darwin": "DYLD_LIBRARY_PATH"}.get(
                sys.platform, "LD_LIBRARY_PATH"
            )
            paths = [str(f) for f in folders]
            if env.get(name):
                paths.append(env[name])
            env[name] = os.pathsep.join(paths)
        return env

    def describe(self) -> dict:
        """Its folder, executables and release, for the page and the command line."""
        out = {
            "root": str(self.root),
            "starter": str(self.starter),
            "engine": str(self.engine),
            "engine (MPI)": str(self.engine_mpi) if self.engine_mpi else "none",
            "anim_to_vtk": str(self.anim_to_vtk) if self.anim_to_vtk else "none",
            "th_to_csv": str(self.th_to_csv) if self.th_to_csv else "none",
        }
        if self.version:
            out["release"] = self.version
        return out


def _executable(path: Path) -> bool:
    return path.is_file() and (sys.platform == "win32" or os.access(path, os.X_OK))


def _make_executable(folder: Path) -> None:
    """Restore the permission to run the programs of a folder (an archive unpacked
    without its permission bits)."""
    if sys.platform == "win32" or not folder.is_dir():
        return
    for path in folder.iterdir():
        if path.is_file() and not os.access(path, os.X_OK):
            try:
                path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            except OSError:
                pass


def _exec_folder(path: Path) -> Path | None:
    """The folder of the executables of the installation at ``path``: ``path/exec``, a
    folder holding a Starter, the folder of a Starter given, or that of the only
    installation in ``path`` (an archive unpacked into it)."""
    path = Path(path).expanduser()
    if path.is_file():
        return path.parent if path.name.lower().startswith("starter_") else None
    if not path.is_dir():
        return None
    for candidate in (path / "exec", path):
        if candidate.is_dir() and any(
            p.name.lower().startswith("starter_") for p in candidate.iterdir()
        ):
            return candidate
    inner = [p for p in path.iterdir() if p.is_dir() and (p / "exec").is_dir()]
    return inner[0] / "exec" if len(inner) == 1 else None


def _pick(folder: Path, prefix: str, suffix: str, variants=("",)) -> Path | None:
    """The program ``<prefix><suffix><variant>`` of a folder (with ``.exe`` on Windows),
    the first variant found."""
    for variant in variants:
        for ext in ("", ".exe"):
            path = folder / f"{prefix}{suffix}{variant}{ext}"
            if path.is_file():
                return path
    return None


def _installation(folder: Path) -> OpenRadioss:
    folder = Path(folder).resolve()
    _make_executable(folder)
    starters = sorted(
        p
        for p in folder.iterdir()
        if p.name.lower().startswith("starter_")
        and p.suffix.lower() in ("", ".exe")
        and not p.stem.endswith("_sp")
    )
    if not starters:
        raise FileNotFoundError(f"no Starter of OpenRadioss in {folder}")
    # the double precision Starter of the machine: the shortest name (no _sp, no variant)
    starter = min(starters, key=lambda p: len(p.stem))
    suffix = starter.stem[len("starter_") :]
    engine = _pick(folder, "engine_", suffix)
    if engine is None:
        raise FileNotFoundError(f"no Engine of OpenRadioss in {folder} (engine_{suffix})")
    root = folder.parent if folder.name == "exec" else folder
    release = {}
    for place in (root, root.parent):
        record = place / RELEASE_FILE
        if record.is_file():
            try:
                release = json.loads(record.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                release = {}
            break
    return OpenRadioss(
        root=root,
        starter=starter,
        engine=engine,
        engine_mpi=_pick(folder, "engine_", suffix, ("_ompi", "_impi", "_mpi")),
        anim_to_vtk=_pick(folder, "anim_to_vtk_", suffix),
        th_to_csv=_pick(folder, "th_to_csv_", suffix),
        suffix=suffix,
        release=release,
    )


def find_openradioss(path=None) -> OpenRadioss:
    """The installation of OpenRadioss at ``path`` (its folder, its ``exec`` folder or
    its Starter), else the first found in the places of the module; raises
    FileNotFoundError, saying how to install it, when there is none."""
    tried = []
    places = [(str(path), "the path given")] if path not in (None, "") else []
    places += [(os.environ[v], f"${v}") for v in ENVIRONMENT if os.environ.get(v)]
    places.append((str(default_home()), "the folder of morphogen radioss install"))
    for place, origin in places:
        folder = _exec_folder(Path(place))
        if folder is not None:
            return _installation(folder)
        tried.append(f"{origin} ({place})")
        if origin == "the path given":
            raise FileNotFoundError(f"no installation of OpenRadioss at {place}")
    for name in ("starter_linux64_gf", "starter_win64", "starter_linuxa64", "starter_macos64"):
        exe = shutil.which(name)
        if exe:
            return _installation(Path(exe).resolve().parent)
    raise FileNotFoundError(
        "OpenRadioss not found (tried " + ", ".join(tried) + " and the PATH): install a "
        "release with `morphogen radioss install` (or the tool OpenRadioss installation of the "
        "interface), or set OPENRADIOSS_PATH to the folder of one (https://github.com/OpenRadioss/"
        "OpenRadioss/releases)"
    )


def has_openradioss(path=None) -> bool:
    """Whether an installation of OpenRadioss is found."""
    try:
        find_openradioss(path)
    except FileNotFoundError:
        return False
    return True


# -- runs ----------------------------------------------------------------------------------


@dataclass
class RadiossRun:
    """A run of OpenRadioss in its ``directory``: its animations converted to legacy VTK
    files (``frames``, in their order), its time history converted to CSV (``history``,
    None without ``th_to_csv``), its listings and the time it took (s)."""

    directory: Path
    name: str
    frames: list[Path]
    history: Path | None
    listings: dict[str, Path]
    elapsed: float = 0.0
    warnings: list[str] = field(default_factory=list)


def _wait(
    process: subprocess.Popen,
    listing: Path | None,
    end_time: float | None,
    progress: Callable | None,
    span: tuple[float, float],
    label: str,
    timeout: float | None,
) -> int:
    """Wait for a program, its progress read in its listing (the time of its last cycle
    over ``end_time``, mapped on ``span``), stopped at a checkpoint (it is killed)."""
    from morphogen.io.radioss import read_radioss_listing

    started = time.monotonic()
    last = -1.0
    while True:
        try:
            code = process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            code = None
        try:
            checkpoint()
        except Stopped:
            _kill(process)
            raise
        if timeout is not None and time.monotonic() - started > timeout:
            _kill(process)
            raise RuntimeError(f"{label}: stopped after {timeout:g} s (timeout)")
        if progress is not None and listing is not None and end_time and listing.is_file():
            try:
                now = read_radioss_listing(listing).time
            except OSError:
                now = last
            if now > last:
                last = now
                share = min(max(now / end_time, 0.0), 1.0)
                progress(
                    span[0] + share * (span[1] - span[0]),
                    f"{label}: t = {now:.4g} of {end_time:.4g}",
                )
        if code is not None:
            return int(code)


def _kill(process: subprocess.Popen) -> None:
    try:
        process.kill()
        process.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _check(listing: Path, code: int, label: str, log: Path) -> None:
    """Raise RuntimeError when a program failed: its exit code, its listing's errors or
    its abnormal termination (the end of its log otherwise)."""
    from morphogen.io.radioss import read_radioss_listing

    state = read_radioss_listing(listing) if listing.is_file() else None
    errors = state.errors if state is not None else []
    if code == 0 and state is not None and state.normal is not False and not errors:
        return
    if code == 0 and state is None:
        return
    detail = "; ".join(errors[:3])
    if not detail and log.is_file():
        tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-8:]
        detail = " | ".join(s.strip() for s in tail if s.strip())
    where = listing.name if listing.is_file() else log.name
    raise RuntimeError(f"{label} failed (exit code {code}; see {where}): {detail or 'no detail'}")


def run_openradioss(
    directory,
    name: str | None = None,
    *,
    threads: int | None = None,
    mpi: int | None = None,
    installation=None,
    progress: Callable | None = None,
    convert: bool = True,
    timeout: float | None = None,
) -> RadiossRun:
    """Run the inputs written in ``directory`` (:func:`morphogen.io.write_radioss`): the
    Starter, then the Engine with ``threads`` OpenMP threads (all the cores by default)
    or on ``mpi`` processes (``mpiexec`` and the Engine built with MPI), then the
    conversion of the animations and of the time history (``convert``).

    ``progress(fraction, message)`` is called as the run advances (the time of the
    last cycle printed in the listing of the Engine); a stop asked by
    :mod:`morphogen.stopping` kills the programs. Raises RuntimeError when a program
    fails (the errors of its listing); its logs stay in the directory."""
    from morphogen.io.radioss import read_manifest

    directory = Path(directory).resolve()
    manifest = read_manifest(directory)
    name = name or manifest["name"]
    install = (
        installation if isinstance(installation, OpenRadioss) else find_openradioss(installation)
    )
    threads = int(threads or os.cpu_count() or 1)
    env = install.environment(threads)
    started = time.monotonic()
    end_time = float(manifest.get("end_time") or 0.0) or None
    domains = int(mpi or 1)
    listings = {"starter": directory / f"{name}_0000.out", "engine": directory / f"{name}_0001.out"}
    for stale in listings.values():
        stale.unlink(missing_ok=True)
    if progress:
        progress(0.0, "OpenRadioss Starter")
    log = directory / "starter.log"
    with open(log, "w", encoding="utf-8") as out:
        process = subprocess.Popen(
            [str(install.starter), "-i", f"{name}_0000.rad", "-np", str(domains)],
            cwd=directory,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
        code = _wait(process, None, None, None, (0.0, 0.05), "Starter", timeout)
    _check(listings["starter"], code, "the Starter of OpenRadioss", log)
    if domains > 1:
        mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
        if install.engine_mpi is None or mpiexec is None:
            raise RuntimeError(
                f"{domains} MPI processes: an Engine built with MPI "
                f"({'found' if install.engine_mpi else 'none in ' + str(install.root)}) and "
                f"mpiexec ({'found' if mpiexec else 'not on the PATH'})"
            )
        command = [mpiexec, "-n", str(domains), str(install.engine_mpi)]
        env["OMP_NUM_THREADS"] = str(max(1, threads // domains))
    else:
        command = [str(install.engine)]
    command += ["-i", f"{name}_0001.rad", "-nt", env.get("OMP_NUM_THREADS", str(threads))]
    log = directory / "engine.log"
    with open(log, "w", encoding="utf-8") as out:
        process = subprocess.Popen(
            command, cwd=directory, env=env, stdout=out, stderr=subprocess.STDOUT
        )
        code = _wait(
            process, listings["engine"], end_time, progress, (0.05, 0.9), "OpenRadioss", timeout
        )
    _check(listings["engine"], code, "the Engine of OpenRadioss", log)
    run = RadiossRun(directory, name, [], None, listings, time.monotonic() - started)
    if convert:
        convert_outputs(run, install, env=env, progress=progress)
    run.elapsed = time.monotonic() - started
    return run


def animation_files(directory, name: str) -> list[Path]:
    """The animation files of a run (``<name>A001``, ``<name>A002``...), in order."""
    found = []
    for path in Path(directory).glob(f"{name}A*"):
        tail = path.name[len(name) + 1 :]
        if tail.isdigit():
            found.append(path)
    return sorted(found, key=lambda p: int(p.name[len(name) + 1 :]))


def convert_outputs(run: RadiossRun, installation=None, *, env=None, progress=None) -> RadiossRun:
    """Convert the outputs of a run: each animation to a legacy VTK file (``anim_to_vtk``,
    ``<name>A001.vtk``...) and the time history to CSV (``th_to_csv``); the outputs
    already converted are kept. A converter missing from the installation is skipped
    with a warning of the run."""
    install = (
        installation if isinstance(installation, OpenRadioss) else find_openradioss(installation)
    )
    env = env or install.environment()
    directory, name = Path(run.directory), run.name
    animations = animation_files(directory, name)
    frames = []
    if animations and install.anim_to_vtk is None:
        run.warnings.append("no anim_to_vtk in the installation: the animations are not read")
    for k, path in enumerate(animations):
        target = path.with_name(path.name + ".vtk")
        if install.anim_to_vtk is not None and not target.is_file():
            checkpoint()
            if progress:
                progress(
                    0.9 + 0.08 * k / len(animations), f"animation {k + 1} of {len(animations)}"
                )
            with open(target, "wb") as out:
                done = subprocess.run(
                    [str(install.anim_to_vtk), path.name],
                    cwd=directory,
                    env=env,
                    stdout=out,
                    stderr=subprocess.PIPE,
                )
            if done.returncode != 0 or target.stat().st_size == 0:
                target.unlink(missing_ok=True)
                run.warnings.append(
                    f"anim_to_vtk failed on {path.name}: "
                    f"{done.stderr.decode(errors='replace').strip()[:200]}"
                )
                continue
        if target.is_file():
            frames.append(target)
    run.frames = frames
    history = directory / f"{name}T01"
    if history.is_file():
        found = sorted(directory.glob(f"{name}T01*.csv"))
        if not found and install.th_to_csv is not None:
            checkpoint()
            done = subprocess.run(
                [str(install.th_to_csv), history.name], cwd=directory, env=env, capture_output=True
            )
            found = sorted(directory.glob(f"{name}T01*.csv"))
            if not found and done.returncode == 0 and done.stdout.strip():
                target = directory / f"{name}T01.csv"  # a converter writing to its output
                target.write_bytes(done.stdout)
                found = [target]
            if not found:
                run.warnings.append(
                    f"th_to_csv gave no CSV file: "
                    f"{done.stderr.decode(errors='replace').strip()[:200]}"
                )
        elif not found:
            run.warnings.append("no th_to_csv in the installation: the time history is not read")
        run.history = found[0] if found else None
    return run


# -- installation of a release -------------------------------------------------------------


def _request(url: str, timeout: float):
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json, application/octet-stream;q=0.9",
            "User-Agent": f"morphoGen/{__version__}",
        },
    )
    return urllib.request.urlopen(request, timeout=timeout)


def release_info(tag: str | None = None, *, api: str = GITHUB_API, timeout: float = 60) -> dict:
    """The release ``tag`` of OpenRadioss on GitHub (the latest by default): its name,
    tag and archives (``assets``: name, size, URL)."""
    path = f"tags/{tag}" if tag else "latest"
    with _request(f"{api.rstrip('/')}/repos/{REPOSITORY}/releases/{path}", timeout) as res:
        data = json.loads(res.read().decode("utf-8"))
    assets = [
        {
            "name": a.get("name", ""),
            "size": int(a.get("size") or 0),
            "url": a.get("browser_download_url", ""),
        }
        for a in data.get("assets", [])
    ]
    return {
        "tag": data.get("tag_name", tag or ""),
        "name": data.get("name", ""),
        "published": data.get("published_at", ""),
        "assets": assets,
    }


def release_asset(release: dict, system: str | None = None, machine: str | None = None) -> dict:
    """The archive of a release (:func:`release_info`) for this system and machine."""
    system = system or platform.system()
    machine = machine or platform.machine()
    words = ASSETS.get((system, machine))
    if words is None:
        raise RuntimeError(
            f"no release of OpenRadioss is published for {system} on {machine}: build it from "
            "its sources (https://github.com/OpenRadioss/OpenRadioss) and set OPENRADIOSS_PATH"
        )
    archives = [
        a
        for a in release.get("assets", [])
        if a["name"].lower().endswith((".zip", ".tar.gz", ".tgz"))
    ]
    for word in words:
        for asset in archives:
            name = asset["name"].lower()
            if word in name and "src" not in name and "source" not in name:
                if word == "linux64" and ("linuxa64" in name or "arm" in name):
                    continue
                return asset
    raise RuntimeError(
        f"the release {release.get('tag')} of OpenRadioss has no archive for {system} on "
        f"{machine} (archives: {', '.join(a['name'] for a in archives) or 'none'})"
    )


def _extract(archive: Path, target: Path) -> None:
    """Unpack a zip or tar archive into ``target``, refusing the members that would land
    outside it, with the permission bits of the zip members kept."""
    target = target.resolve()
    if archive.name.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                path = (target / member.filename).resolve()
                if target != path and target not in path.parents:
                    raise RuntimeError(
                        f"{archive.name}: the member {member.filename} leaves the folder"
                    )
                zf.extract(member, target)
                mode = (member.external_attr >> 16) & 0o777
                if mode and not member.is_dir():
                    path.chmod(mode | stat.S_IRUSR | stat.S_IWUSR)
        return
    import tarfile

    with tarfile.open(archive) as tf:
        tf.extractall(target, filter="data")


def install_openradioss(
    target=None,
    *,
    tag: str | None = None,
    api: str = GITHUB_API,
    progress: Callable | None = None,
    timeout: float = 120,
    system: str | None = None,
    machine: str | None = None,
) -> OpenRadioss:
    """Download the release ``tag`` of OpenRadioss (the latest by default) for this system
    from GitHub and unpack it into ``target`` (``~/morphogen/openradioss`` by default),
    which replaces an earlier installation there; returns the installation. Releases are
    published for Linux (x86-64, ARM) and Windows; elsewhere OpenRadioss is built from its
    sources. ``progress(fraction, message)`` follows the download."""
    target = Path(target).expanduser() if target else default_home()
    if progress:
        progress(0.0, "the release of OpenRadioss")
    release = release_info(tag, api=api, timeout=timeout)
    asset = release_asset(release, system, machine)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".openradioss-", dir=target.parent) as tmp:
        archive = Path(tmp) / Path(asset["name"]).name
        with _request(asset["url"], timeout) as res, open(archive, "wb") as out:
            total = int(res.headers.get("Content-Length") or asset.get("size") or 0)
            done = 0
            while True:
                checkpoint()
                block = res.read(1 << 20)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if progress and total:
                    progress(
                        0.9 * done / total,
                        f"{asset['name']}: {done / 2**20:.0f} of {total / 2**20:.0f} MB",
                    )
        if progress:
            progress(0.92, f"unpacking {asset['name']}")
        unpacked = Path(tmp) / "unpacked"
        unpacked.mkdir()
        _extract(archive, unpacked)
        folder = _exec_folder(unpacked)
        if folder is None:
            raise RuntimeError(f"{asset['name']}: no Starter of OpenRadioss in the archive")
        record = {
            "tag": release["tag"],
            "name": release["name"],
            "asset": asset["name"],
            "url": asset["url"],
            "installed": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (unpacked / RELEASE_FILE).write_text(json.dumps(record, indent=1), encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(unpacked), str(target))
    if progress:
        progress(1.0, f"OpenRadioss {release['tag']} in {target}")
    return find_openradioss(target)
