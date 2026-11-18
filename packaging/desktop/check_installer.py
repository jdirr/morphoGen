"""The Windows installer of the stand-alone application, checked as a user meets it: it is
installed silently for the current user in a folder of its own, the program it installs
is checked without a window (``morphoGen.exe --check``), its shortcut in the Start menu and
its entry among the installed applications are found, then it is uninstalled and they are
gone. With ``--window``, the program is also started as from its shortcut: its window
must open, and it is closed as a user closes it.

    python packaging/desktop/check_installer.py dist/morphoGen-1.0.0-setup.exe [--window]

Exits with 0 when all went well (the workflow desktop.yml runs it on each build)."""

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
import winreg
from ctypes import wintypes
from pathlib import Path

#: The key of the installed application (the AppId of morphogen.iss, with Inno Setup's suffix)
UNINSTALL_KEY = (
    r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
    r"\{A466015B-5487-4768-874A-77F8CDE59D3F}_is1"
)
WM_CLOSE = 0x0010


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


class _Process(ctypes.Structure):  # PROCESSENTRY32W
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


def _family(pid: int) -> set[int]:
    """The process ``pid`` and its descendants (a launcher runs the program in a child)."""
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snapshot = kernel32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    parents = {}
    entry = _Process()
    entry.dwSize = ctypes.sizeof(_Process)
    try:
        more = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while more:
            parents[entry.th32ProcessID] = entry.th32ParentProcessID
            more = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    family, grown = {pid}, True
    while grown:
        grown = False
        for child, parent in parents.items():
            if parent in family and child not in family:
                family.add(child)
                grown = True
    return family


def _windows(pid: int) -> list[tuple[int, str]]:
    """The visible windows of the process ``pid`` and of its descendants that have a title:
    (handle, title)."""
    user32 = ctypes.windll.user32
    family = _family(pid)
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value in family and user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            title = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, title, length + 1)
            if title.value:
                found.append((hwnd, title.value))
        return True

    user32.EnumWindows(visit, 0)
    return found


def check_window(program: Path, timeout: float = 180.0) -> str | None:
    """Start the program as from its shortcut, wait for its window and close it: None when
    the window opened and the program ended when it closed, what went wrong otherwise."""
    process = subprocess.Popen([str(program)])
    try:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if process.poll() is not None:
                return f"morphoGen.exe ended with {process.returncode} before its window opened"
            windows = [w for w in _windows(process.pid) if w[1].startswith("morphoGen")]
            if windows:
                hwnd, title = windows[0]
                print(f"window: '{title}', after {time.monotonic() - start:.0f} s")
                ctypes.windll.user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
                try:
                    code = process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    return "morphoGen.exe did not end when its window closed"
                return None if code == 0 else f"morphoGen.exe ended with {code} once closed"
            time.sleep(0.5)
        return f"no window of morphoGen.exe after {timeout:.0f} s"
    finally:
        if process.poll() is None:
            process.kill()


def main(setup: str, window: bool = False) -> int:
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
        if window and (problem := check_window(program)):
            problems.append(problem)
    if program.is_file():
        # the uninstaller runs from a copy of itself in the temporary folder: wait for it
        subprocess.run([str(folder / "unins000.exe"), *flags], check=True, timeout=600)
        for _ in range(600):
            # its folder goes last, once emptied
            if not folder.exists() and not shortcut.exists() and _entry() is None:
                break
            time.sleep(0.5)
        left = [path for path in (program, shortcut, folder) if path.exists()]
        if left or _entry() is not None:
            problems.append(f"the uninstaller left {left or 'its entry'}")
        else:
            print("uninstalled: the folder, the shortcut and the entry gone")
    for problem in problems:
        print("problem:", problem)
    if problems:
        print(f"the log of the installation: {log}")
    else:
        shutil.rmtree(folder.parent, ignore_errors=True)
    return 1 if problems else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the Windows installer of morphoGen.")
    parser.add_argument("setup", help="the installer, morphoGen-<version>-setup.exe")
    parser.add_argument("--window", action="store_true", help="also open and close the window")
    args = parser.parse_args()
    sys.exit(main(args.setup, window=args.window))
