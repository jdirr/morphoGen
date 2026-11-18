# Citing morphoGen

The metadata of the software are in `CITATION.cff`, at the root of the repository, which
GitHub shows as "Cite this repository" and from which Zenodo writes the record of each
release. Each public release will have its DOI (Zenodo); until the first one, cite the
version and the repository:

```bibtex
@software{morphogen,
  author  = {Dirrenberger, Justin},
  title   = {morphoGen: morphological modelling and generation of virtual microstructures},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/jdirr/morphoGen},
  license = {GPL-3.0-or-later},
}
```

Cite also the papers of the methods you use: the models, measures and solvers of
morphoGen follow the references given with them in the [theory notes](theory.md) and
in the [references](references.md).
