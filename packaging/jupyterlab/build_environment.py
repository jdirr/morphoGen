"""The environment of the JupyterLab installers: a Python of its own (a relocatable build of
CPython, as uv installs them), morphoGen with its extras, JupyterLab and the packages of
everyday notebooks, in one folder that runs wherever it is copied.

    python packaging/jupyterlab/build_environment.py [--morphogen checkout|X.Y.Z]
                                                     [--python 3.13] [--out build/jupyterlab]

The folder holds ``python/`` (the interpreter and its packages), ``launcher.py`` (which opens
JupyterLab on the user's notebooks), ``welcome.ipynb`` (the first notebook), ``VERSION`` (that
of morphoGen) and ``environment.txt`` (every package installed, with its version). morphoGen
comes from the checkout of the repository, or from PyPI at the version given. The packages
are installed with their bytecode compiled. The launchers that pip and uv write for the
programs of the packages name the interpreter of the build: they are replaced by programs
that find the interpreter beside them (``relocatable_programs``). The launcher runs JupyterLab
through ``python -m``, and the kernel of the notebooks is the ``python`` that Jupyter resolves
to its own interpreter; the kernel opts out of the power throttling of Windows before it
starts (``kernel/launch_kernel.py``), and JupyterLab neither fetches news nor checks for
updates (``overrides.json``).

Needs uv (``pip install uv``) and an internet connection; builds for the system that runs it.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
#: The packages beside morphoGen and its extras
PACKAGES = ["jupyterlab>=4.4", "ipykernel>=6.29", "ipywidgets>=8", "pandas>=2.2"]


def _uv() -> list[str]:
    exe = shutil.which("uv")
    return [exe] if exe else [sys.executable, "-m", "uv"]


def _run(*command, capture: bool = False) -> str:
    words = [str(w) for w in command]
    print("+", " ".join(words), flush=True)
    done = subprocess.run(words, check=True, text=True, capture_output=capture)
    return done.stdout if capture else ""


def interpreter(python: Path) -> Path:
    """The interpreter of a relocatable CPython folder."""
    return python / "python.exe" if os.name == "nt" else python / "bin" / "python3"


def fetch_python(version: str, work: Path) -> Path:
    """A relocatable CPython of ``version`` (python-build-standalone), as uv installs it:
    the folder of its full version (uv also links the minor version to it)."""
    store = work / "pythons"
    _run(*_uv(), "python", "install", version, "--install-dir", store, "--no-bin")
    full = re.compile(rf"cpython-{re.escape(version)}\.\d+-[^+]*$")
    found = sorted(p for p in store.iterdir() if full.match(p.name))
    if not found:
        raise SystemExit(f"no CPython {version} in {store}: {sorted(os.listdir(store))}")
    return found[-1]


def kernel_folder(python: Path) -> Path:
    """The folder of the Python kernel that ipykernel installs in the environment."""
    found = list(python.glob("share/jupyter/kernels/python3"))
    if not found:
        raise SystemExit(f"no kernel python3 under {python}/share/jupyter")
    return found[0]


def _entry_points(python: Path) -> dict[str, str]:
    """The programs of the installed packages: name -> ``module:attribute``."""
    found = {}
    for listing in python.glob("**/site-packages/*.dist-info/entry_points.txt"):
        section = ""
        for line in listing.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("["):
                section = line.strip("[]")
            elif "=" in line and section in ("console_scripts", "gui_scripts"):
                name, target = (part.strip() for part in line.split("=", 1))
                found[name] = target.split("[")[0].strip()
    return found


def relocatable_programs(python: Path) -> int:
    """Make the programs of the packages (``morphogen``, ``jupyter``, ``pip``...) run from
    wherever the environment is copied, in place of the launchers that pip and uv wrote with
    the path of the interpreter of the build: on Windows, a ``.cmd`` that runs the entry point
    with the interpreter of its own folder (and no ``.exe``); elsewhere, a script whose first
    lines find the interpreter beside it (``/bin/sh`` reads them as an ``exec``, Python as a
    string). Returns the number of programs."""
    count = 0
    for name, target in _entry_points(python).items():
        module, _, attribute = target.partition(":")
        first = attribute.split(".")[0]
        if os.name == "nt":
            code = (
                f"import sys; from {module} import {first}; sys.argv[0] = '{name}'; "
                f"sys.exit({attribute}())"
            )
            scripts = python / "Scripts"
            (scripts / f"{name}.cmd").write_text(
                f'@"%~dp0..\\python.exe" -c "{code}" %*\n', encoding="ascii", newline="\r\n"
            )
            (scripts / f"{name}.exe").unlink(missing_ok=True)
        else:
            script = python / "bin" / name
            script.unlink(missing_ok=True)
            script.write_text(
                "#!/bin/sh\n"
                '\'\'\'exec\' "$(dirname -- "$0")/python3" "$0" "$@"\n'
                "' '''\n"
                "import sys\n"
                f"from {module} import {first}\n\n"
                'if __name__ == "__main__":\n'
                f"    sys.argv[0] = {name!r}\n"
                f"    sys.exit({attribute}())\n",
                encoding="utf-8",
                newline="\n",
            )
            script.chmod(0o755)
        count += 1
    return count


def build(morphogen: str, version: str, out: Path) -> None:
    work = out.parent / f"{out.name}-work"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    work.mkdir(parents=True, exist_ok=True)
    python = out / "python"
    shutil.copytree(fetch_python(version, work), python, symlinks=True)
    for marker in python.rglob("EXTERNALLY-MANAGED"):  # uv's mark: this Python is ours
        marker.unlink()
    exe = interpreter(python)
    if morphogen == "checkout":  # the wheel of this checkout
        wheels = work / "wheels"
        shutil.rmtree(wheels, ignore_errors=True)
        _run(*_uv(), "build", "--wheel", "--out-dir", wheels, ROOT)
        spec = f"morphogen[all] @ {next(wheels.glob('morphogen-*.whl')).as_uri()}"
    else:
        spec = f"morphogen[all]=={morphogen}"
    # binary wheels only: where the latest version of a dependency has none for this system
    # (llvmlite on macOS Intel), the latest one that has
    _run(
        *_uv(),
        "pip",
        "install",
        "--python",
        exe,
        "--only-binary",
        ":all:",
        "--link-mode",
        "copy",
        "--compile-bytecode",
        spec,
        *PACKAGES,
    )
    print(f"{relocatable_programs(python)} programs made relocatable")
    kernel = kernel_folder(python)
    for name in ("kernel.json", "launch_kernel.py"):
        shutil.copy2(HERE / "kernel" / name, kernel / name)
    settings = python / "share" / "jupyter" / "lab" / "settings"
    settings.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / "overrides.json", settings / "overrides.json")
    for name in ("launcher.py", "welcome.ipynb"):
        shutil.copy2(HERE / name, out / name)
    shutil.copy2(ROOT / "LICENSE", out / "LICENSE.txt")
    release = _run(exe, "-c", "import morphogen; print(morphogen.__version__)", capture=True)
    (out / "VERSION").write_text(release.strip() + "\n", encoding="utf-8")
    frozen = _run(*_uv(), "pip", "freeze", "--python", exe, capture=True)
    (out / "environment.txt").write_text(frozen, encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(f"morphoGen {release.strip()} and {len(frozen.splitlines())} packages in {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--morphogen", default="checkout", help="'checkout' (this repository) or a version on PyPI"
    )
    parser.add_argument("--python", default="3.13", help="the minor version of CPython")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "jupyterlab")
    args = parser.parse_args()
    build(args.morphogen, args.python, args.out.resolve())


if __name__ == "__main__":
    main()
