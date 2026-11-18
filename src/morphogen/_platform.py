"""Settings of the process that depend on the operating system."""

from __future__ import annotations

import sys


def full_speed() -> bool:
    """Opt the process out of the power throttling of Windows (EcoQoS).

    Windows 11 moves a process that it deems in the background, such as one started by
    an agent, to the efficiency cores at a low clock after a few seconds of computing,
    where the compiled kernels run several times slower. A process whose execution speed
    is set to "never throttled" stays on the performance cores. The processes that it
    starts do not inherit the setting: each one makes it (the command line, the
    processes of :func:`morphogen.parallel.process_pool`; the processes of the tests in
    ``tests/conftest.py``, the build of the documentation in ``docs/conf.py``).

    Returns whether the setting was made: ``False`` on the other systems, where it does
    nothing, and on the versions of Windows without it (before Windows 10 1709). Never
    raises.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class State(ctypes.Structure):  # PROCESS_POWER_THROTTLING_STATE
            _fields_ = (
                ("Version", wintypes.ULONG),
                ("ControlMask", wintypes.ULONG),
                ("StateMask", wintypes.ULONG),
            )

        kernel32 = ctypes.WinDLL("kernel32")  # an instance of its own: its argtypes
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        kernel32.SetProcessInformation.restype = wintypes.BOOL
        # version 1; the execution speed (bit 1) controlled and not throttled
        state = State(1, 1, 0)
        done = kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(),
            4,  # ProcessPowerThrottling
            ctypes.byref(state),
            ctypes.sizeof(state),
        )
        return bool(done)
    except Exception:  # left as it is (an older Windows, a restricted process)
        return False
