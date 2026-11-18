"""Global run-time settings (number of threads)."""

from __future__ import annotations

import functools
import sys
from collections.abc import Iterator
from contextlib import contextmanager

_threads: int | None = None
#: The functions that read and set the number of threads of the OpenBLAS libraries of the
#: process (found once; see :func:`blas_on_one_thread`)
_openblas_threads: list | None = None


def set_num_threads(n: int | None) -> None:
    """Number of threads used by compiled kernels and SciPy (``None``: all cores)."""
    global _threads
    import numba

    if n is None:
        _threads = None
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
        return
    n = int(n)
    if n < 1:
        raise ValueError("number of threads must be positive")
    numba.set_num_threads(min(n, numba.config.NUMBA_NUM_THREADS))
    _threads = n


def workers() -> int:
    """``workers`` argument for SciPy routines."""
    return -1 if _threads is None else _threads


def _openblas() -> list:
    """The functions ``(get, set)`` of the number of threads of each OpenBLAS library
    that NumPy and SciPy have loaded (their wheels carry one each; an installation may
    share one): the libraries mapped by the process on Linux, those of the folders of
    the wheels elsewhere. Empty with another BLAS, or where none is found."""
    global _openblas_threads
    if _openblas_threads is not None:
        return _openblas_threads
    import ctypes
    import glob
    import os

    import numpy
    import numpy.linalg  # (the libraries are loaded with the modules that call them)
    import scipy
    import scipy.linalg

    paths: list[str] = []
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/maps") as maps:  # (the path: what follows five fields)
                mapped = [line.split(None, 5) for line in maps if "openblas" in line.lower()]
            paths = sorted({fields[5].strip() for fields in mapped if len(fields) == 6})
        except OSError:
            paths = []
    else:
        for package in (numpy, scipy):
            name = package.__name__
            root = os.path.dirname(os.path.dirname(os.path.abspath(package.__file__)))
            for folder in (f"{name}.libs", os.path.join(name, ".dylibs")):
                paths += sorted(glob.glob(os.path.join(root, folder, "*openblas*")))
    found = []
    for path in paths:
        try:
            library = ctypes.CDLL(path)
        except OSError:
            continue
        for prefix, suffix in (("scipy_", "64_"), ("scipy_", ""), ("", "64_"), ("", "")):
            get = getattr(library, f"{prefix}openblas_get_num_threads{suffix}", None)
            put = getattr(library, f"{prefix}openblas_set_num_threads{suffix}", None)
            if get is not None and put is not None:
                get.restype = ctypes.c_int
                put.argtypes = [ctypes.c_int]
                put.restype = None
                found.append((get, put))
                break
    _openblas_threads = found
    return found


@contextmanager
def blas_on_one_thread() -> Iterator[None]:
    """Within the block, the BLAS of NumPy and SciPy (OpenBLAS) on one thread; the
    number of its threads is restored after it.

    The compiled kernels run on their own threads. A product of BLAS between two of
    them (a norm, a small factorization) wakes the threads of OpenBLAS, which then stay
    busy for a tenth of a second, waiting for the next product, and take the cores of the
    kernels that follow: a loading path or a Krylov iteration is slowed by a third to a
    half. The dense algebra within those loops is small (6 x 6 systems, the Hessenberg
    matrices of GMRES): one thread serves it as well. Without OpenBLAS (another BLAS, or
    none found), the block runs as it is."""
    libraries = _openblas()
    before = [get() for get, _ in libraries]
    for _, put in libraries:
        put(1)
    try:
        yield
    finally:
        for (_, put), count in zip(libraries, before, strict=True):
            put(count)


def without_blas_threads(function):
    """Decorator: the function runs within :func:`blas_on_one_thread`."""

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with blas_on_one_thread():
            return function(*args, **kwargs)

    return wrapped
