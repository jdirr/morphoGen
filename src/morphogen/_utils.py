"""Small validation and conversion helpers shared by the package."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def as_triplet(value, name: str, *, positive: bool = False, integer: bool = False) -> tuple:
    """Return ``value`` as a tuple of three numbers (a scalar is repeated)."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name}: expected a number or a list of 3 numbers, got {value!r}"
        ) from None
    if arr.ndim == 0:
        arr = np.repeat(arr, 3)
    if arr.shape != (3,):
        raise ValueError(f"{name}: expected a number or a list of 3 numbers, got {value!r}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: values must be finite, got {value!r}")
    if positive and np.any(arr <= 0):
        raise ValueError(f"{name}: values must be positive, got {value!r}")
    if integer:
        if np.any(arr != np.round(arr)):
            raise ValueError(f"{name}: values must be integers, got {value!r}")
        return tuple(int(v) for v in arr)
    return tuple(float(v) for v in arr)


def check_fraction(value: float, name: str, *, open_interval: bool = True) -> float:
    """Check that ``value`` lies in (0, 1) (or [0, 1])."""
    v = float(value)
    ok = 0.0 < v < 1.0 if open_interval else 0.0 <= v <= 1.0
    if not ok:
        bounds = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name}: must lie in {bounds}, got {value!r}")
    return v


def check_positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0 or (v == 0 and not allow_zero):
        sign = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name}: must be a finite {sign} number, got {value!r}")
    return v


def exactly_one(name_a: str, a, name_b: str, b) -> None:
    """Raise unless exactly one of two alternative parameters is given (two ways of setting
    the same quantity); the message says whether both or neither were given."""
    if a is not None and b is not None:
        raise ValueError(
            f"give exactly one of '{name_a}' and '{name_b}', not both: they set the same "
            f"quantity in two ways (got {name_a}={a!r} and {name_b}={b!r})"
        )
    if a is None and b is None:
        raise ValueError(f"give exactly one of '{name_a}' and '{name_b}' (neither is given)")


def unit_vector(v: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name}: expected 3 components, got {v!r}")
    norm = np.linalg.norm(arr)
    if not norm > 0:
        raise ValueError(f"{name}: must be a non-zero vector, got {v!r}")
    return arr / norm


def to_jsonable(value):
    """Convert numpy scalars/arrays (recursively) to plain Python types."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value
