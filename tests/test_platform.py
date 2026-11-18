"""Settings of the process that depend on the system: the power throttling of Windows
(EcoQoS) turned off, nothing done elsewhere."""

import ctypes
import sys

import pytest

from morphogen._platform import full_speed


def power_throttling() -> tuple[int, int]:
    """The power throttling set for this process: its control and state masks."""
    from ctypes import wintypes

    class State(ctypes.Structure):  # PROCESS_POWER_THROTTLING_STATE
        _fields_ = (
            ("Version", wintypes.ULONG),
            ("ControlMask", wintypes.ULONG),
            ("StateMask", wintypes.ULONG),
        )

    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessInformation.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    kernel32.GetProcessInformation.restype = wintypes.BOOL
    state = State(1, 0, 0)
    process = kernel32.GetCurrentProcess()
    assert kernel32.GetProcessInformation(process, 4, ctypes.byref(state), ctypes.sizeof(state))
    return state.ControlMask, state.StateMask


@pytest.mark.skipif(sys.platform != "win32", reason="the power throttling of Windows")
def test_full_speed_on_windows():
    assert full_speed() is True
    control, state = power_throttling()
    assert control & 1 and not state & 1  # the execution speed controlled, not throttled


def test_full_speed_does_nothing_elsewhere(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert full_speed() is False


def test_full_speed_never_raises(monkeypatch):
    # a Windows without the function, or the ctypes of another system
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delattr(ctypes, "WinDLL", raising=False)
    assert full_speed() is False
