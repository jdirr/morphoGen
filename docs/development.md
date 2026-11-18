# Development

The environment of development, from a clone of the repository, and the checks of a
change:

```bash
python -m pip install -e ".[dev]"
python -m pip install vtk gmsh   # optional: independent readers of the exported files
python -m pytest -n auto         # 1900+ tests in parallel (pytest-xdist): about 25 minutes
                                 # on 4 cores
python -m pytest -n auto -m "not slow"   # the fast tier: about 10 minutes on 4 cores
python tests/test_public_api.py  # the list of the public interface written again,
                                 # after a change of it made on purpose
ruff check src tests examples docs && ruff format --check src tests examples docs
python -m sphinx -b html -W docs site   # the documentation site, without warnings
tools/ci_local.sh                # the CI jobs of this machine's system, locally
CI_LOCAL_PACKAGES="vtk gmsh" tools/ci_local.sh   # the same with the VTK and Gmsh checks
```

The statistical tests compare every model with its theory, using fixed seeds and
tolerances derived from that theory. The homogenization tests check exact solutions
(homogeneous media, laminates, the bulk modulus of cubic polycrystals), the ordering of
the boundary conditions and the agreement with scikit-fem. The phase-3 tests check automata against known patterns
(gliders, blinkers), growth against the Avrami law and exact distances, exact counts
against brute force, rigidity against Maxwell–Calladine, sensitivities against finite
differences and optimized cells against the Hashin–Shtrikman bounds. Optional checks
with VTK and Gmsh (their Python packages: VTK reads the `.vti` files, Gmsh the `.msh`
meshes and meshes the `.geo` scripts) run when those packages are installed; the GitHub
workflow does not install them. The GPU tests compare the solutions of both devices
where CuPy finds a GPU, and the MFront tests morphoGen's behaviours with their MFront
twins where MGIS and MFront are installed (in WSL on the development machine).
Continuous integration runs the tests on Linux, Windows and macOS
(`.github/workflows/ci.yml`); `tools/ci_local.sh` runs the jobs of the development
machine's system (an environment per Python version, the tests against the built wheel,
the command-line smoke test, the documentation site), for when GitHub Actions is not
available. The Python versions run at once, each on its share of the cores
(`CI_LOCAL_WORKERS` test processes per version); the slowest tests (marker `slow`,
two thirds of the time) run under the last version only (`CI_LOCAL_SLOW`), which gets
three times the cores of each other version, and the GitHub workflow runs them in its
job Ubuntu with Python 3.14. An environment is kept from one run to the next while
`pyproject.toml` does not change (`CI_LOCAL_FRESH=1` makes new ones). It runs in Git
Bash on Windows. With uv it finds or installs
Python 3.12 and 3.14; without uv it uses venv and pip, and takes the versions
(`pythonX.Y` on the PATH or `py -X.Y`) or the paths of the interpreters:
`tools/ci_local.sh /path/to/python3.12 /path/to/python3.14`. Changes go through pull
requests to `main`.

## The documentation site

The site (https://jdirr.github.io/morphoGen/) is built at each push to `main` from the
pages of `docs/` and the API reference written from the docstrings; locally:
`python -m pip install ".[docs]"`, then `python -m sphinx -b html docs site` (open
`site/index.html`). Its pages copy their code blocks at a click, and its gallery filters
its cards by words. The continuous integration builds it too and keeps it as the
artifact `site` of each run. The figures of the tutorials are written by
`docs/tutorials/make_figures.py` (which runs the snippets) and
`docs/tutorials/make_screenshots.py` (which drives the interface), the cards of the
gallery by `docs/gallery/make_gallery.py`.

How to report a problem or propose a change, and what a pull request checks:
[CONTRIBUTING.md](https://github.com/jdirr/morphoGen/blob/main/CONTRIBUTING.md); what
changed from one version to the next: the [change log](changelog.md).
