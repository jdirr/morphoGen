"""Compiled (Numba) kernels.

Kernels take and return plain NumPy arrays so that they can be replaced by other
implementations (GPU, Rust...) without changing the rest of the package.
"""
