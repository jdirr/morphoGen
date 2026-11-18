"""Cooperative stops of long computations.

A thread that runs a computation it may have to stop (a job of the interface) runs it
under :func:`stoppable` with an event; the long loops of morphoGen (the iterations of the
conjugate gradients, the realizations of a run, the steps the tools of the interface
report) call :func:`checkpoint`, which raises :class:`Stopped` once the event is set. A
stop thus takes effect at the next checkpoint, never in the middle of a step; elsewhere
(no event, another thread) :func:`checkpoint` does nothing.

:class:`Stopped` derives from :class:`BaseException`, as :class:`KeyboardInterrupt` does,
so that the handlers of :class:`Exception` that fall back on another method (a GPU on the
CPU, a solver on another) let it through."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_local = threading.local()


class Stopped(BaseException):
    """A computation asked to stop (raised by :func:`checkpoint`)."""


@contextmanager
def stoppable(event: threading.Event) -> Iterator[threading.Event]:
    """Run the block with ``event`` as the stop of the computations of this thread (the
    innermost when nested)."""
    previous = getattr(_local, "event", None)
    _local.event = event
    try:
        yield event
    finally:
        _local.event = previous


def requested() -> bool:
    """Whether the computation of this thread was asked to stop."""
    event = getattr(_local, "event", None)
    return event is not None and event.is_set()


def checkpoint() -> None:
    """Raise :class:`Stopped` if the computation of this thread was asked to stop."""
    if requested():
        raise Stopped("stopped on request")
