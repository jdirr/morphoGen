# Stability of the interface

From version 1.0, morphoGen follows [semantic versioning](https://semver.org): a
version `MAJOR.MINOR.PATCH` changes its `PATCH` for fixes, its `MINOR` for additions
that leave the existing uses working, and its `MAJOR` for the changes that break some.
Until then (versions 0.x) the interface may still change from one version to the next;
the [change log](changelog.md) says how.

## What is public

* The names of the `__all__` of `morphogen` and of its packages (`morphogen.models`,
  `morphogen.homogenization`, ...), and the documented names of the modules of the
  [API reference](api.md), with their parameters and their documented results.
* The commands of `morphogen` and their options, the keys of the configuration files
  and the files that the runs write (their names, formats and columns).
* The formats read and written for other codes (Z-set, MFront, AMITEX_FFTP, Abaqus,
  FEniCSx, Neper, DAMASK, VTK, XDMF) and for the orientation maps of the EBSD systems
  (`.ang`, `.ctf`).

## What is not

* The names that start with an underscore, in any module, and the modules whose name
  does.
* The graphical interface: the page, the requests of its server and the Python modules
  behind them (`morphogen.gui`). Its workspaces are folders of ordinary files, which
  stay readable.
* The compiled kernels (`morphogen.kernels` and the Numba and CUDA functions of the
  other modules) and the functions behind the commands (`morphogen.cli`).
* The digits of the numerical results beyond the tolerances of their tests: a fix or a
  better algorithm may change them in a `MINOR` version; the change log says which.

## Changes of the public interface

A public name removed or changed incompatibly within a `MAJOR` version first warns
(`FutureWarning`, with what replaces it) for at least one `MINOR` version, then goes in
the next `MAJOR` one. The test `tests/test_public_api.py` compares the public names of
the packages and of the modules of the API reference, with the parameters of their
functions and classes, to the list kept in `tests/public_api.txt`: a change of that
list is a change of the interface, made on purpose, with its entry in the change log.
`python tests/test_public_api.py` writes the list again from the installed package.
