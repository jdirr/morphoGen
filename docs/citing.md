# Citing morphoGen

The metadata of the software are in `CITATION.cff`, at the root of the repository, which
GitHub shows as "Cite this repository" and from which Zenodo writes the record of each
release. Each public release has its DOI on Zenodo, and the DOI of all the versions,
[10.5281/zenodo.24340315](https://doi.org/10.5281/zenodo.24340315), leads to the latest
(10.5281/zenodo.24340316 for 1.0.0):

```bibtex
@software{morphogen,
  author  = {Dirrenberger, Justin},
  title   = {morphoGen: morphological modelling and generation of virtual microstructures},
  version = {1.1.0},
  year    = {2026},
  doi     = {10.5281/zenodo.24340315},
  url     = {https://github.com/jdirr/morphoGen},
  license = {GPL-3.0-or-later},
}
```

morphoGen also calls libraries that have their own references. With the model `microgen` or
`morphogen.interop.microgen`, cite Microgen: K. Marchais, Y. Chemisky, R. d'Esparbès,
M. R. Guevara, Y. Legerstee et al., *3MAH/microgen: microstructure generation and meshing*,
Zenodo, [10.5281/zenodo.6793573](https://doi.org/10.5281/zenodo.6793573) ([github.com/3MAH/microgen](https://github.com/3MAH/microgen)).

Cite also the papers of the methods you use: the models, measures and solvers of
morphoGen follow the references given with them in the [theory notes](theory.md) and
in the [references](references.md).
