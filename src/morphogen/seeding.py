"""Reproducible random number generation.

Every random draw in morphoGen goes through a :class:`numpy.random.Generator` built
from a seed. The realizations of a study get seeds derived from the study seed and
their indices (``SeedSequence(seed, spawn_key=(size_index, realization_index))``), so
a result does not depend on how the study was split across processes or cluster jobs.
"""

from __future__ import annotations

import numpy as np

SeedLike = int | np.random.SeedSequence | np.random.Generator | None


def derive_seed(base: int | None, *key: int) -> np.random.SeedSequence:
    """Seed of the item ``key`` (e.g. ``(size_index, realization_index)``) of a study."""
    return np.random.SeedSequence(entropy=base, spawn_key=tuple(int(k) for k in key))


def make_rng(seed: SeedLike = None) -> tuple[np.random.Generator, dict | None]:
    """Return a generator and a JSON-able description that reproduces it.

    ``seed`` may be an integer, a :class:`~numpy.random.SeedSequence`, an existing
    generator (used as is, not reproducible from the description) or ``None`` (fresh
    entropy, recorded so that the realization can be regenerated).
    """
    if isinstance(seed, np.random.Generator):
        return seed, None
    if isinstance(seed, np.random.SeedSequence):
        ss = seed
    elif seed is None:
        ss = np.random.SeedSequence()
    else:
        if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
            raise ValueError(f"seed: expected a non-negative integer, got {seed!r}")
        ss = np.random.SeedSequence(int(seed))
    return np.random.default_rng(ss), describe_seed(ss)


def describe_seed(ss: np.random.SeedSequence) -> dict:
    """The entropy and spawn key of a seed sequence, which recreate it (recorded with the
    results)."""
    return {"entropy": ss.entropy, "spawn_key": list(ss.spawn_key)}


def seed_from_description(desc: dict | int | None) -> np.random.SeedSequence | int | None:
    """Inverse of :func:`describe_seed` (also accepts a plain integer)."""
    if desc is None or isinstance(desc, int):
        return desc
    return np.random.SeedSequence(entropy=desc["entropy"], spawn_key=tuple(desc["spawn_key"]))
