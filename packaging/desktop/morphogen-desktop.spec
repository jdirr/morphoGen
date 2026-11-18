# PyInstaller's recipe of the stand-alone application of morphoGen: a folder (or a macOS
# application) that runs without Python. From the root of the repository, in an
# environment with morphoGen and its extras installed:
#
#     python -m pip install ".[all,desktop]" pyinstaller
#     pyinstaller packaging/desktop/morphogen-desktop.spec --noconfirm
#
# gives dist/morphoGen/morphoGen (morphoGen.exe on Windows; dist/morphoGen.app on macOS).
# `morphoGen --check` checks a bundle without opening a window. See packaging/desktop/README.md.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

here = Path(SPECPATH)
# the icon of the package: a Windows icon for the program of Windows, a PNG elsewhere
icon = here / "../../src/morphogen/gui" / ("icon.ico" if sys.platform == "win32" else "icon.png")

hidden = collect_submodules("morphogen")  # the tools of the interface, found at run time
hidden += ["webview", "webview.platforms.qt"] if sys.platform.startswith("linux") else ["webview"]

a = Analysis(
    [str(here / "morphogen_desktop.py")],
    datas=collect_data_files("morphogen"),  # the page, its 3D view, the templates, the icon
    hiddenimports=hidden,
    excludes=["tkinter", "IPython", "pytest", "sphinx", "skfem"],
    # the sources beside the compiled modules: Numba caches its kernels against them
    module_collection_mode={"morphogen": "pyz+py"},
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="morphoGen",
    console=False,
    icon=str(icon),
)
coll = COLLECT(exe, a.binaries, a.datas, name="morphoGen")
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="morphoGen.app",
        icon=str(icon),
        bundle_identifier="io.github.jdirr.morphogen",
        info_plist={"NSHighResolutionCapable": True},
    )
