"""The program of the stand-alone application of morphoGen, as PyInstaller freezes it
(morphogen-desktop.spec). Numba caches the compiled kernels of a frozen application in
the user's cache folder (``~/.cache/numba``, ``%LOCALAPPDATA%\\numba``...) when the bundle
is read-only, against the sources the recipe collects beside the modules."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()  # the parallel runs in a frozen program
    from morphogen.gui.desktop import main

    sys.exit(main())
