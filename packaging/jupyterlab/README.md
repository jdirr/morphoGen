# The JupyterLab installers

A ready-to-use environment of notebooks, for those who have no Python, or who do not want to
manage one: a Python of its own, morphoGen with its extras (figures, conforming meshes, TIFF
and STL files), JupyterLab and the usual packages of notebooks (NumPy, SciPy, matplotlib,
pandas, ipywidgets), installed with a shortcut that opens JupyterLab on the user's notebooks.
No administrator rights, and no connection once it is downloaded.

| File | What it is |
|---|---|
| `build_environment.py` | builds the environment in one folder (`build/jupyterlab`): a relocatable CPython (python-build-standalone, through uv), the packages installed with their bytecode compiled, their programs made to find the interpreter beside them |
| `launcher.py` | what the shortcut runs: JupyterLab on `Documents/morphoGen` (made at the first start, with `welcome.ipynb`), in the browser, its messages in a window that stops it when closed |
| `welcome.ipynb` | the first notebook: the first study of the README, and where to read on |
| `kernel/` | the Python kernel of the notebooks, which opts out of the power throttling of Windows 11 (EcoQoS) before it starts |
| `overrides.json` | JupyterLab without news nor checks for updates |
| `morphogen-jupyterlab.iss` | the Windows installer (Inno Setup 6): `dist/morphoGen-JupyterLab-<version>-windows-x64-setup.exe` |
| `build_pkg.sh` | the macOS package: `morphoGen JupyterLab.app` in `dist/morphoGen-JupyterLab-<version>-macos-<arch>.pkg` |
| `check_environment.py` | the checks of an installed environment, run with its own Python |
| `check_installer.py` | the check of an installer: installed, checked, removed |

## Build

From the root of the repository, with uv (`python -m pip install uv`), on the system to build
for (the environment holds compiled packages of its system):

```bash
python packaging/jupyterlab/build_environment.py                     # morphoGen of this checkout
python packaging/jupyterlab/build_environment.py --morphogen 1.0.2   # or a version of PyPI
```

The folder `build/jupyterlab` weighs about 800 MB (CPython, NumPy, SciPy, Numba and LLVM,
matplotlib, scikit-image, JupyterLab); it runs wherever it is copied. Then, on Windows, with
[Inno Setup 6](https://jrsoftware.org/isinfo.php):

```bash
iscc /DAppVersion=1.0.2 packaging\jupyterlab\morphogen-jupyterlab.iss
python packaging/jupyterlab/check_installer.py dist/morphoGen-JupyterLab-1.0.2-windows-x64-setup.exe
```

and on macOS (Apple Silicon or Intel, for that architecture):

```bash
sh packaging/jupyterlab/build_pkg.sh build/jupyterlab dist/morphoGen-JupyterLab-1.0.2-macos-arm64.pkg
python packaging/jupyterlab/check_installer.py dist/morphoGen-JupyterLab-1.0.2-macos-arm64.pkg
```

The workflow `.github/workflows/jupyterlab.yml` builds and checks the three (Windows x64, macOS
arm64 and x86_64) when their recipe changes, and attaches them to each release (to an earlier
one on demand: its input `release`, the version of morphoGen then taken from PyPI).

## What the user meets

**Windows.** The installer installs for the current user, in
`%LOCALAPPDATA%\Programs\morphoGen JupyterLab` (an administrator may choose all users), with a
shortcut *morphoGen JupyterLab* in the Start menu (and on the desktop if asked) and an
uninstaller among the installed applications; a later version replaces the installed one.
Its wizard speaks English or French, after the language of Windows. It is not signed: on its
first start, Windows SmartScreen tells that its publisher is unknown, and *More info*, then
*Run anyway*, starts it.

**macOS.** The package installs `morphoGen JupyterLab.app` in the Applications folder of the
user (*Install for me only*, no administrator rights) or of all the users. It is not signed by
an Apple developer: macOS first refuses to open it; *System Settings*, *Privacy & Security*,
then *Open Anyway* opens it (on macOS 14 and earlier, a right click on the package, then
*Open*). The application opens a window of Terminal that runs JupyterLab; macOS may ask whether
Terminal may access the Documents folder. The package of an architecture installs on Macs of
that architecture: arm64 for Apple Silicon, x86_64 for Intel.

**Both.** The shortcut opens JupyterLab in the web browser, on the folder
`Documents/morphoGen` (made at the first start, with the first notebook, which that start
opens; the environment variable `MORPHOGEN_NOTEBOOKS` names another folder). The window of its
messages (a console, a window of Terminal) stays open while JupyterLab runs; closing it stops
JupyterLab. The terminals of JupyterLab find the Python of the environment and its programs
first (`morphogen gui`, `morphogen info`, `pip`); `%pip install <package>` in a notebook adds a
package to the environment. Uninstalling removes the environment, not the notebooks.
