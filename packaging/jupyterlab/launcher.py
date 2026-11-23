"""Opens JupyterLab on the user's notebooks, with the Python of the environment beside it.

The notebooks live in a folder of the user's documents, ``Documents/morphoGen`` (the
environment variable ``MORPHOGEN_NOTEBOOKS`` names another one), made at the first start
with a first notebook (``welcome.ipynb``), which that start opens. JupyterLab opens in the
web browser; the window that runs this program (a console on Windows, a Terminal window on
macOS) shows its messages, and closing it stops JupyterLab. The arguments are passed on to
JupyterLab (``--no-browser``, ``--port``...).
"""

import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def documents() -> Path:
    """The user's documents: on Windows, where the system has them (OneDrive may move
    them), ``~/Documents`` elsewhere."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class Guid(ctypes.Structure):
                _fields_ = [
                    ("data1", wintypes.DWORD),
                    ("data2", wintypes.WORD),
                    ("data3", wintypes.WORD),
                    ("data4", ctypes.c_ubyte * 8),
                ]

            # FOLDERID_Documents, {FDD39AD0-238F-46AF-ADB4-6C85480369C7}
            folder = Guid(
                0xFDD39AD0,
                0x238F,
                0x46AF,
                (ctypes.c_ubyte * 8)(0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7),
            )
            path = ctypes.c_wchar_p()
            shell32 = ctypes.WinDLL("shell32")
            if shell32.SHGetKnownFolderPath(ctypes.byref(folder), 0, None, ctypes.byref(path)) == 0:
                found = Path(path.value)
                ctypes.WinDLL("ole32").CoTaskMemFree(path)
                return found
        except (OSError, AttributeError, ValueError):
            pass
    return Path.home() / "Documents"


def notebooks() -> tuple[Path, bool]:
    """The folder of the notebooks, and whether this start made it (with the first
    notebook)."""
    folder = Path(os.environ.get("MORPHOGEN_NOTEBOOKS") or documents() / "morphoGen")
    if folder.exists():
        return folder, False
    folder.mkdir(parents=True)
    shutil.copy2(HERE / "welcome.ipynb", folder / "welcome.ipynb")
    return folder, True


def main() -> int:
    folder, first = notebooks()
    # the terminals of JupyterLab and the shell commands of the notebooks find this Python
    # and the programs of its packages first
    python = Path(sys.executable).parent
    programs = [python, python / "Scripts"] if sys.platform == "win32" else [python]
    os.environ["PATH"] = os.pathsep.join([*map(str, programs), os.environ.get("PATH", "")])
    version = (HERE / "VERSION").read_text(encoding="utf-8").strip()
    title = f"morphoGen {version} - JupyterLab"
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetConsoleTitleW(title)
    else:
        print(f"\033]0;{title}\007", end="")
    print(
        f"{title}\n\nJupyterLab opens in the web browser, on the folder\n    {folder}\n"
        "Keep this window open while you work: closing it stops JupyterLab.\n",
        flush=True,
    )
    from jupyterlab.labapp import main as lab

    options = [f"--ServerApp.root_dir={folder}"]
    if first:  # the first start opens the first notebook
        options.append("--LabApp.default_url=/lab/tree/welcome.ipynb")
    sys.argv = ["jupyter-lab", *options, *sys.argv[1:]]
    return lab()


if __name__ == "__main__":
    sys.exit(main())
