"""The installers of the JupyterLab environment, checked as a user meets them, on the system
that runs this script:

* Windows: ``morphoGen-JupyterLab-<version>-windows-x64-setup.exe`` is installed silently for
  the current user in a folder of its own; its shortcut in the Start menu and its entry among
  the installed applications are found; the environment it installs is checked
  (``check_environment.py``, with its own Python); then it is uninstalled, and its folder,
  shortcut and entry are gone;
* macOS: ``morphoGen-JupyterLab-<version>-macos-<arch>.pkg`` is installed in the user's
  Applications folder (``installer -target CurrentUserHomeDirectory``, as "Install for me
  only" does); the application is checked (its property list, its executable, the command of
  Terminal), then the environment inside it; then it is removed.

    python packaging/jupyterlab/check_installer.py dist/morphoGen-JupyterLab-<version>-<system>...

Exits with 0 when all went well (the workflow jupyterlab.yml runs it on each build).
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "morphoGen JupyterLab"
#: The key of the installed application (the AppId of morphogen-jupyterlab.iss, with Inno
#: Setup's suffix)
UNINSTALL_KEY = (
    r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
    r"\{CA5891F7-4A6F-4217-A6C7-34AA214EBFAE}_is1"
)


def environment_checked(environment: Path, python: Path) -> list[str]:
    check = subprocess.run(
        [str(python), str(HERE / "check_environment.py"), str(environment)], timeout=3600
    )
    return [] if check.returncode == 0 else [f"the environment of {environment} (see above)"]


def _entry() -> dict | None:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as key:
            return {
                name: winreg.QueryValueEx(key, name)[0]
                for name in ("DisplayName", "DisplayVersion", "InstallLocation")
            }
    except FileNotFoundError:
        return None


def windows(setup: Path) -> list[str]:
    folder = Path(tempfile.mkdtemp(prefix="morphogen_jupyterlab_")) / NAME
    log = folder.parent / "install.log"
    shortcut = Path(os.environ["APPDATA"], f"Microsoft/Windows/Start Menu/Programs/{NAME}.lnk")
    python = folder / "python" / "python.exe"
    flags = ["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"]
    start = time.perf_counter()
    subprocess.run(
        [str(setup), *flags, "/CURRENTUSER", f"/DIR={folder}", f"/LOG={log}"],
        check=True,
        timeout=3600,
    )
    print(f"installed in {time.perf_counter() - start:.0f} s")
    entry = _entry()
    problems = [
        message
        for ok, message in (
            (python.is_file(), f"{python} is not installed (see {log})"),
            (shortcut.is_file(), f"the shortcut {shortcut} is missing"),
            (entry is not None, f"{NAME} is not among the installed applications"),
        )
        if not ok
    ]
    if not problems:
        print(f"installed: {entry['DisplayName']} in {entry['InstallLocation']}")
        problems += environment_checked(folder, python)
    if python.is_file():
        # the uninstaller runs from a copy of itself in the temporary folder: wait for it
        subprocess.run([str(folder / "unins000.exe"), *flags], check=True, timeout=1200)
        for _ in range(1200):
            if not folder.exists() and not shortcut.exists() and _entry() is None:
                break
            time.sleep(0.5)
        left = [path for path in (python, shortcut, folder) if path.exists()]
        if left or _entry() is not None:
            problems.append(f"the uninstaller left {left or 'its entry'}")
        else:
            print("uninstalled: the folder, the shortcut and the entry gone")
    if not problems:
        shutil.rmtree(folder.parent, ignore_errors=True)
    return problems


def macos(package: Path) -> list[str]:
    app = Path.home() / "Applications" / f"{NAME}.app"
    if app.exists():
        return [f"{app} is there before the installation"]
    start = time.perf_counter()
    subprocess.run(
        ["installer", "-pkg", str(package), "-target", "CurrentUserHomeDirectory"],
        check=True,
        timeout=3600,
    )
    print(f"installed in {time.perf_counter() - start:.0f} s")
    contents = app / "Contents"
    resources = contents / "Resources"
    python = resources / "python" / "bin" / "python3"
    problems = [
        message
        for ok, message in (
            (python.is_file(), f"{python} is not installed"),
            (
                os.access(contents / "MacOS" / "launcher", os.X_OK),
                "the executable of the application",
            ),
            (os.access(resources / "start.command", os.X_OK), "the command that Terminal runs"),
            ((resources / "morphogen.icns").is_file(), "the icon of the application"),
        )
        if not ok
    ]
    lint = subprocess.run(["plutil", "-lint", str(contents / "Info.plist")])
    if lint.returncode:
        problems.append("the property list of the application")
    if not problems:
        problems += environment_checked(resources, python)
    shutil.rmtree(app, ignore_errors=True)
    return problems


def main(installer: str) -> int:
    path = Path(installer).resolve()
    problems = windows(path) if sys.platform == "win32" else macos(path)
    for problem in problems:
        print("problem:", problem)
    print(
        f"{path.name}: " + ("all checks passed" if not problems else f"{len(problems)} problem(s)")
    )
    return 1 if problems else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
