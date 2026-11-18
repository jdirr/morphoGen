"""Optional GPU backend (NVIDIA GPUs, through CuPy): the choice of the device.

The FFT solvers of voxel images run their conjugate gradients on the GPU with
``device="gpu"`` (or ``"auto"``: the GPU when there is one): the stiffness operators of
the voxels, applied matrix-free, become CUDA kernels and the Fourier preconditioner cuFFT
transforms (:mod:`morphogen.homogenization.gpu`). Along loading paths the integration
points stay on the GPU, where their behaviours are integrated; the Newton iterations of
the periodic element stay on the CPU. CuPy and the CUDA libraries it needs (the runtime,
NVRTC and cuFFT, about 270 MB; no CUDA toolkit is needed) are optional::

    pip install cupy-cuda13x "cuda-toolkit[cudart,nvrtc,cufft]==13.*"

(``cupy-cuda12x`` and ``cuda-toolkit[...]==12.*`` for drivers older than CUDA 13).
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

DEVICES = ("cpu", "gpu", "auto")
INSTALL = 'pip install cupy-cuda13x "cuda-toolkit[cudart,nvrtc,cufft]==13.*"'

#: The libraries kept loaded for the life of the process (see :func:`_keep_nvrtc_builtins`)
_KEPT: list = []


@lru_cache(maxsize=1)
def _probe() -> tuple[bool, str]:
    """Whether a CUDA device can be used through CuPy, and why not otherwise."""
    try:
        import cupy
    except ImportError:
        return False, f"CuPy is not installed ({INSTALL})"
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False, "no CUDA device"
        cupy.zeros(1).sum()  # the runtime and the kernels work
    except Exception as exc:  # the driver, the runtime or NVRTC is missing
        return False, f"CUDA is not usable: {exc}"
    _keep_nvrtc_builtins(cupy)
    return True, ""


def _keep_nvrtc_builtins(cupy) -> None:
    """NVRTC's library of builtins kept loaded (Windows). NVRTC opens it by its name at
    each compilation and closes it after; once another library of the process has narrowed
    the search path of the DLLs (Gmsh's does, when it opens a file), the next compilation
    fails ("failed to open nvrtc-builtins64_134.dll"), unless the library is already
    loaded. It lies next to NVRTC's own, which CuPy has loaded."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    try:
        major, minor = cupy.cuda.nvrtc.getVersion()
    except Exception:  # an NVRTC that CuPy cannot reach: nothing to keep
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    nvrtc = kernel32.GetModuleHandleW(f"nvrtc64_{major}0_0.dll")
    if not nvrtc:
        return
    name = ctypes.create_unicode_buffer(32768)
    if not kernel32.GetModuleFileNameW(nvrtc, name, len(name)):
        return
    builtins = os.path.join(os.path.dirname(name.value), f"nvrtc-builtins64_{major}{minor}.dll")
    if os.path.isfile(builtins):
        _KEPT.append(ctypes.WinDLL(builtins))


def available() -> bool:
    """Whether the GPU backend can run (CuPy, a CUDA device and its runtime)."""
    return _probe()[0]


def device_of(name) -> str:
    """The device that a choice ``"cpu"``, ``"gpu"`` or ``"auto"`` designates: ``"auto"``
    is the GPU when it is available; ``"gpu"`` without one is an error."""
    if name not in DEVICES:
        raise ValueError(f"device: one of {DEVICES}, got {name!r}")
    if name == "auto":
        return "gpu" if available() else "cpu"
    if name == "gpu" and not available():
        raise RuntimeError(f"device 'gpu': {_probe()[1]}")
    return name


def info() -> dict:
    """The GPU and CuPy, or why the backend is not available."""
    ok, why = _probe()
    if not ok:
        return {"gpu": None, "reason": why}
    import cupy

    props = cupy.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    return {
        "gpu": name.decode() if isinstance(name, bytes) else str(name),
        "memory_gb": round(props["totalGlobalMem"] / 2**30, 2),
        "compute_capability": f"{props['major']}.{props['minor']}",
        "cupy": cupy.__version__,
        "cuda_runtime": cupy.cuda.runtime.runtimeGetVersion(),
    }
