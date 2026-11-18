"""The Windows installer of the stand-alone application, checked as a user meets it: it is
installed silently for the current user in a folder of its own, the program it installs
is checked without a window (``morphoGen.exe --check``), its shortcut in the Start menu and
its entry among the installed applications are found, then it is uninstalled and they are
gone.

    python packaging/desktop/check_installer.py dist/morphoGen-1.0.0-setup.exe

Exits with 0 when all went well (the workflow desktop.yml runs it on each build)."""

import os
import subprocess
import sys
import tempfile
import time
import winreg
from pathlib import Path

#: The key of the installed application (the AppId of morphogen.iss, with Inno Setup's suffix)
UNINSTALL_KEY = (
    r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
    r"\{A466015B-5487-4768-874A-77F8CDE59D3F}_is1"
)


def _entry() -> dict | None:
    """The entry of morphoGen among the applications installed for the current user."""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as key:
            return {
                name: winreg.QueryValueEx(key, name)[0]
                for name in ("DisplayName", "DisplayVersion", "InstallLocation")
            }
    except FileNotFoundError:
        return None


def main(setup: str) -> int:
    folder = Path(tempfile.mkdtemp(prefix="morphogen_installer_")) / "morphoGen"
    log = folder.parent / "install.log"
    shortcut = Path(os.environ["APPDATA"], "Microsoft/Windows/Start Menu/Programs/morphoGen.lnk")
    program = folder / "morphoGen.exe"
    flags = ["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"]
    subprocess.run(
        [setup, *flags, "/CURRENTUSER", f"/DIR={folder}", f"/LOG={log}"], check=True, timeout=1800
    )
    entry = _entry()
    problems = [
        message
        for ok, message in (
            (program.is_file(), f"{program} is not installed (see {log})"),
            (shortcut.is_file(), f"the shortcut {shortcut} is missing"),
            (entry is not None, "morphoGen is not among the installed applications"),
        )
        if not ok
    ]
    if not problems:
        print(f"installed: {entry['DisplayName']} in {entry['InstallLocation']}")
        check = subprocess.run([str(program), "--check"], timeout=1800)
        if check.returncode:
            problems.append(f"morphoGen.exe --check exited with {check.returncode}")
        else:
            print("morphoGen.exe --check: the page, the models, the tools and a realization")
    if program.is_file():
        # the uninstaller runs from a copy of itself in the temporary folder: wait for it
        subprocess.run([str(folder / "unins000.exe"), *flags], check=True, timeout=600)
        for _ in range(600):
            if not program.exists() and not shortcut.exists() and _entry() is None:
                break
            time.sleep(0.5)
        left = [path for path in (program, shortcut, folder) if path.exists()]
        if left or _entry() is not None:
            problems.append(f"the uninstaller left {left or 'its entry'}")
        else:
            print("uninstalled: the folder, the shortcut and the entry gone")
    for problem in problems:
        print("problem:", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
