# Installation

morphoGen needs Python 3.12 or newer, on Windows, macOS (Apple Silicon) or Linux. All its
dependencies are installed from binary wheels: no compiler is needed.

## From PyPI

```bash
python -m pip install morphogen            # the core: NumPy, SciPy, Numba and PyYAML
python -m pip install "morphogen[all]"     # with figures, conforming meshes, TIFF and STL
```

preferably in an environment of its own.

**With uv** (it can also install Python itself):

```bash
uv venv --python 3.13
uv pip install "morphogen[all]"
```

Activate the environment with `.venv\Scripts\activate` (Windows) or
`source .venv/bin/activate` (macOS, Linux).

**With pip and venv:**

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS, Linux
python -m pip install "morphogen[all]"
```

**With conda** (Miniforge or Anaconda):

```bash
conda create -n morphogen python=3.13
conda activate morphogen
python -m pip install "morphogen[all]"
```

## From the sources

To change the code, or to run the examples and the tests of the repository:

```bash
git clone https://github.com/jdirr/morphoGen.git
cd morphoGen
python -m pip install -e ".[dev]"      # editable, with the tests and the tools of development
```

The page [Development](development.md) gives the checks of a change.

## Optional components

The core needs only NumPy, SciPy, Numba and PyYAML; the graphical interface needs
nothing else. The optional extras are `surface` (STL surfaces), `io` (TIFF and other
mesh formats), `plot` (figures and animations, with matplotlib), `mesh` (the MMG
mesher, from the `pymmg` wheels: conforming meshes for the finite element solver) and
`video` (3D views and mp4 videos: VTK and ffmpeg); `all` gathers the first four. Where
pymmg has no wheel (e.g. Linux on ARM), build `mmg3d` from
<https://github.com/MmgTools/mmg> and set `MORPHOGEN_MMG3D` to its path;
`morphogen info` shows the mesher found.

**GPU.** On a machine with an NVIDIA GPU, `python -m pip install "morphogen[gpu]"`
adds CuPy and the CUDA libraries it loads (the runtime, NVRTC and cuFFT, about 270 MB;
neither the CUDA toolkit nor a compiler is needed, only a driver of CUDA 13). With a
driver of CUDA 12, install
`python -m pip install cupy-cuda12x "cuda-toolkit[cudart,nvrtc,cufft]==12.*"` instead.
The FFT solvers then run on the GPU with `device: gpu` (or `auto`: the GPU when there is
one); `morphogen info` shows the GPU found, or why none is usable. These libraries are
proprietary (NVIDIA's licence, free of charge), and morphoGen never requires them.

**Stand-alone application.** `python -m pip install "morphogen[desktop]"` adds
pywebview, which shows the interface in a window of the web engine of the system
(WebView2 on Windows, WebKit on macOS; on Linux, Qt WebEngine from its wheels, with the
system's EGL and X libraries, e.g. `apt install libegl1 libxkbcommon-x11-0
libxcb-cursor0`); `morphogen desktop` (or the program `morphogen-desktop`) opens it.
On Windows, the installer `morphoGen-<version>-setup.exe` of the
[releases](https://github.com/jdirr/morphoGen/releases) installs the application without
Python and without administrator rights, with a shortcut in the Start menu and an
uninstaller; it is not signed, so that Windows may first tell that its publisher is
unknown (*More info*, then *Run anyway*). A folder that runs without Python
(`morphoGen.exe` on Windows, `morphoGen.app` on macOS) is built with PyInstaller by the
recipe of `packaging/desktop` in the repository, which also makes the installer.

**OpenRadioss.** The explicit dynamic simulations run OpenRadioss, a
separate program: `morphogen radioss install` downloads its latest release from GitHub
(Linux x86-64 and ARM, Windows) into `~/morphogen/openradioss`, where morphoGen finds it
(or set `OPENRADIOSS_PATH` to the folder of an installation, e.g. one built from its
sources on macOS); `morphogen radioss info` shows the installation found. Without it,
morphoGen writes the inputs and the scripts that run them elsewhere.

**MFront behaviours.** MGIS and MFront are not on PyPI: install them from
conda-forge in morphoGen's environment, with a C++ compiler for the behaviours:

```bash
conda install -c conda-forge mgis mfront cxx-compiler
```

On Windows, MFront needs the C++ compiler of Visual Studio (its Build Tools); WSL with
Ubuntu and Miniforge is simpler, and is where morphoGen's MFront tests run.
`morphogen info` shows MGIS and the version of MFront found.

## Check the installation

```bash
morphogen info
```

shows the version of morphoGen, of Python and of its dependencies, the optional
components it finds (the mesher, OpenRadioss, the GPU, MGIS and MFront) and the models.
The first call of each kernel compiles it (a few seconds);
compiled code is cached. In a clone of the repository, `python -m pytest` runs the tests.
