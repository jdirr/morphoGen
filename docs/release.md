# Releases

The steps of a public release, version 1.0 first. They are the author's: the files they
use are ready in the repository (`CITATION.cff`, `paper/`, `packaging/conda-forge/`,
`.readthedocs.yaml`), but nothing is published before these steps.

## Before the first public release

1. **Rights.** In France the economic rights to software written by employees in the
   course of their duties belong to the employer (Code de la propriété intellectuelle,
   art. L113-9, extended to people with research missions), so that the holder of the
   rights and the licence had to be confirmed with the institution. Done on 2026-11-17:
   the author holds the rights, and the institution confirmed it (decision D1 of the
   [specification](specification.md)). Done on 2026-11-18: the copyright lines
   (`README.md`, `docs/conf.py`) name the holder, Justin Dirrenberger, as do
   `CITATION.cff` and the authors of `pyproject.toml`.
2. **Identifiers.** The ORCID of each author in `CITATION.cff` (`orcid:`) and in
   `paper/paper.md`, and if wished the ROR identifier of the affiliation. Done on
   2026-11-18: 0000-0002-3964-305X (Justin Dirrenberger); no ROR identifier.
3. **Branch.** The development branch merged into `main`. Done on 2026-11-01: pull
   request #2 merged the development line, up to `91d8987`, into `main` (merge commit
   `9a6b660`, morphoGen 0.6).
4. **Repository.** Made public, with its issue tracker open: JOSS asks that anyone can
   read the code and file issues without registering for anything else. Done on
   2026-11-18.

## Each release

1. **Version.** `src/morphogen/_version.py`; the section of `CHANGELOG.md`
   (`## X.Y.Z — date`); `version` and `date-released` in `CITATION.cff`. For 1.0, the
   classifier `Development Status :: 5 - Production/Stable` in `pyproject.toml`, and
   the classifier `Private :: Do Not Upload` removed (PyPI refuses any upload while it
   is there).
2. **Checks.** `tools/ci_local.sh` and the GitHub workflow pass;
   `python -m build` writes the source archive and the wheel, which
   `python -m twine check dist/*` accepts.
3. **Tag.** `git tag -a vX.Y.Z -m "morphoGen X.Y.Z"`, `git push origin vX.Y.Z`, and a
   GitHub release of the tag with the section of the change log.
4. **DOI.** Zenodo's GitHub integration, switched on for the repository before the
   release (zenodo.org, GitHub, the repository), archives each GitHub release with the
   metadata of `CITATION.cff` and gives it a DOI. The DOI of all versions (the concept
   DOI) then goes in `CITATION.cff` (`doi:`), in the README and in
   [](citing.md).
5. **PyPI.** `python -m twine upload dist/*`, with an API token of the project (or
   trusted publishing from a GitHub workflow).
6. **conda-forge.** For the first release, a pull request to
   `conda-forge/staged-recipes` that adds `recipes/morphogen/recipe.yaml`, copied from
   `packaging/conda-forge/recipe.yaml` with the version and the sha256 of the source
   archive on PyPI; the bot of the feedstock then follows the next releases on PyPI.
7. **Documentation site.** Read the Docs builds `docs/` with `.readthedocs.yaml` for
   each version and for `main` once the public repository is imported there; or
   GitHub Pages from `python -m sphinx -b html docs site`.

## Paper for JOSS

`paper/paper.md` and `paper/paper.bib` are a draft for the
[Journal of Open Source Software](https://joss.theoj.org), in the form its guidelines ask
(750 to 1750 words; Summary, Statement of need, State of the field, Software design,
Research impact statement, AI usage disclosure). JOSS reviews software whose repository
has been public for more than six months, with active development over that period,
that is used for research, ideally by others, and, with a single author, that shows its
open practice (public history, releases and change log, tests and continuous
integration, documentation, contribution guide, stated support). The submission
therefore comes six months after the repository is made public at the earliest; the
research impact statement grows meanwhile with the papers that use morphoGen. The AI
usage disclosure names the tools and their versions, where and how they were used, and
states that the human authors reviewed and validated what they produced and made the
design decisions; a disclosure found incomplete may lead to a rejection. JOSS's Docker
image (`openjournals/inara`) checks that the paper compiles, and the accepted paper
needs the DOI of the reviewed version (step 4 above).
