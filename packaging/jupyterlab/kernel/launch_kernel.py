"""The Python kernel of the notebooks of the environment.

On Windows the kernel first opts out of the power throttling (EcoQoS) that Windows 11
applies to the processes it deems in the background, as a kernel is while the web browser
shows its notebook: throttled, compiled kernels run several times slower on the efficiency
cores (``morphogen._platform.full_speed`` does the same; the processes a kernel starts make
the setting themselves). Then it starts as ipykernel's own launcher does.
"""

import sys
from pathlib import Path


def full_speed() -> None:
    """Never throttled (PROCESS_POWER_THROTTLING_STATE: control of the execution speed,
    no throttling); nothing on the other systems, or where Windows does not know it."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class State(ctypes.Structure):
            _fields_ = [
                ("version", wintypes.ULONG),
                ("control", wintypes.ULONG),
                ("state", wintypes.ULONG),
            ]

        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        state = State(1, 1, 0)
        kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(),
            4,  # ProcessPowerThrottling
            ctypes.byref(state),
            ctypes.sizeof(state),
        )
    except (OSError, AttributeError):
        pass


if __name__ == "__main__":
    full_speed()
    # the folder of this script is not that of the notebook, which the kernel adds itself
    if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
        del sys.path[0]
    from ipykernel import kernelapp

    kernelapp.launch_new_instance()
