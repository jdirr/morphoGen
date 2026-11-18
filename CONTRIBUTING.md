# Contributing to morphoGen

Questions, bug reports and proposals are welcome as issues of the repository: for a bug,
the version (`morphogen info`), the system, the smallest script or configuration that
shows it and what it gives against what you expected.

## Changes

Changes go through pull requests to `main`. The page *Development* of the documentation
([docs/development.md](docs/development.md)) gives the environment
(`python -m pip install -e ".[dev]"`) and the checks; before a pull request:

* `ruff check src tests examples docs` and `ruff format --check src tests examples docs`
  pass;
* `python -m pytest -n auto` passes (`-m "not slow"`, the fast tier, checks in about a
  third of the time), and the new code has its tests, those of more than about 15 s
  marked `@pytest.mark.slow`. morphoGen's tests compare with exact solutions, closed
  forms or independent codes (theory of the models, laminates, Hashin–Shtrikman bounds,
  scikit-fem, MFront), with tolerances that follow from them, rather than with the
  code's own earlier outputs; the tests of the graphical interface check its plumbing on
  small problems, the physics being checked by those of the library;
* `tools/ci_local.sh` runs the jobs of the continuous integration of your system, in its
  own environments, when GitHub Actions does not;
* the documentation follows: the docstrings (Sphinx roles, ``double backquotes`` for
  code), the formulas and their references in `docs/theory.md`, an entry of
  `CHANGELOG.md`; `python -m sphinx -b html -W docs site` builds the site without
  warnings;
* the pages of the documentation stay runnable: the Python snippets of the tutorials
  (`docs/tutorial*.md`), of the recipes (`docs/howto.md`), of the landing page and of
  the guide to the physics follow each other in one session, and the tests run them
  (`tests/test_tutorial.py`, `test_howto.py`, `test_physics_guide.py`), at smaller
  sizes where they would take long. Their figures are written by scripts, never by
  hand (`docs/tutorials/make_figures.py` and `make_screenshots.py`,
  `docs/gallery/make_gallery.py`), and a new template has its card in the gallery
  (`FAMILIES` in that last script);
* a change of the public interface ([docs/stability.md](docs/stability.md)) also
  changes `tests/public_api.txt` (`python tests/test_public_api.py` writes it again)
  and says so in the change log;
* a change of a Numba kernel that another module compiles into its own (`cp_small` and
  `cp_finite` of `homogenization/crystal.py` in `update_points` of `behaviours.py` and
  `update_points_finite` of `finite.py`) comes with a change of the revision `_KERNELS`
  of that module: Numba dates the cache of a compiled function by the file that defines
  it alone, and a cache left from the earlier version would run the old kernel without
  a word.

Code, comments and documentation are in English. Where morphoGen meets another code
(Z-set, MFront, Abaqus, AMITEX_FFTP), the order of the tensor components of each side
is stated and tested with distinct shear moduli.

## Support

morphoGen is maintained on a best-effort basis by its authors: issues are read and
answered as time allows, without a guaranteed delay. The interface follows semantic
versioning from version 1.0 ([docs/stability.md](docs/stability.md)).
