# The stand-alone application

morphoGen's graphical interface runs in the web browser (`morphogen gui`) or as a
stand-alone application, in a window of its own (`morphogen desktop`, or the program
`morphogen-desktop` once morphoGen is installed with `pip install "morphogen[desktop]"`).
This folder builds a bundle of the application that runs without Python, with
[PyInstaller](https://pyinstaller.org):

| File | What it is |
|---|---|
| `morphogen-desktop.spec` | PyInstaller's recipe: a folder `dist/morphoGen` (and `dist/morphoGen.app` on macOS) |
| `morphogen_desktop.py` | the program the recipe freezes (`morphogen.gui.desktop.main`) |
| `morphogen.ico` | the icon of the Windows program (the PNG of the package elsewhere) |
| `morphogen.desktop` | the entry of the menus of Linux desktops |

## Build

From the root of the repository, in an environment with morphoGen, the extras its users
need and PyInstaller (Python 3.12 to 3.14):

```bash
python -m pip install ".[all,desktop]" pyinstaller
pyinstaller packaging/desktop/morphogen-desktop.spec --noconfirm
dist/morphoGen/morphoGen --check        # Windows: dist\morphoGen\morphoGen.exe --check
```

`--check` runs the bundle without a window: its server serves the page, the catalog of
the tools and a realization of a model (which compiles the kernels of Numba); it prints
what it checked and exits with 0. The bundle weighs about 1 GB on Linux (Qt WebEngine,
SciPy, Numba and LLVM, matplotlib), less on Windows and macOS, whose web engines are the
system's; zip it, or wrap it in an installer (Inno Setup, a DMG). Build it on each
system for that system: PyInstaller does not cross-compile. The workflow
`.github/workflows/desktop.yml` builds and checks the three on demand.

The window needs the web engine of the system: WebView2 on Windows (part of Windows 10
and 11), WebKit on macOS, and on Linux Qt WebEngine (bundled) with the EGL and X
libraries of the system (`libegl1`, `libxkbcommon-x11-0`, `libxcb-cursor0` on Debian and
Ubuntu). Numba caches the compiled kernels in the user's cache folder, so that the
second start is faster.

## Install on Linux

```bash
cp -r dist/morphoGen ~/.local/opt/morphoGen
ln -s ~/.local/opt/morphoGen/morphoGen ~/.local/bin/morphogen-desktop
cp packaging/desktop/morphogen.desktop ~/.local/share/applications/
cp src/morphogen/gui/icon.png ~/.local/share/icons/hicolor/256x256/apps/morphogen.png
```

With morphoGen installed by pip instead, the program `morphogen-desktop` is already on
the `PATH`: copy the `.desktop` file and the icon alone.

## OpenRadioss

The explicit dynamic simulations (the tab Explicit dynamics) run OpenRadioss, which the
bundle does not contain: install it from the tab (the tool OpenRadioss installation
downloads its latest release from GitHub into `~/morphogen/openradioss`), with
`morphogen radioss install`, or set `OPENRADIOSS_PATH` to the folder of an installation.
