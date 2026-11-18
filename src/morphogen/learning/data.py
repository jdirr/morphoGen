"""Training data: random strain paths and the responses of a behaviour along them.

A path starts from the natural state and joins ``segments`` random strains linearly, each
segment in ``increments`` steps: a random walk (each strain the previous one plus a
random step, held within the amplitude) or independent strains (each component uniform
within the amplitude). The volumetric part of the strains is scaled by ``volumetric``:
the volumetric strains of metals are elastic and small, those of porous media are not.
The responses come from a material point (:func:`~morphogen.homogenization.nonlinear.
simulate_point`) or from any solver of the paths of a microstructure (the homogenized
response of :class:`~morphogen.homogenization.fft_nonlinear.NonlinearFFTProblem` or of
:class:`~morphogen.homogenization.nonlinear.NonlinearFEProblem`), at the planned steps
of each path (the steps that a solver divides are dropped).

In small strains the data are the strain (engineering shears) and the stress; in finite
strains the Green-Lagrange strain ``E = (F^T F - I) / 2`` (engineering shears) and the
second Piola-Kirchhoff stress ``S = J F^-1 sigma F^-T``, objective measures conjugate to
each other, whatever the rotation of the gradient (the upper triangular gradients of the
paths, their Cauchy stresses).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from morphogen.homogenization.paths import LoadingPath, PathResult, _index

#: The kinds of random paths
KINDS = ("walk", "linear")
#: The measures of the data: small strains (the strain, the stress) or finite strains (the
#: Green-Lagrange strain, the second Piola-Kirchhoff stress)
MEASURES = ("small", "green-lagrange")
#: The Voigt pairs (11, 22, 33, 23, 31, 12)
PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))
_TRACE = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
_ENGINEERING = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])


def random_paths(
    n: int,
    *,
    segments: int = 4,
    increments: int = 10,
    amplitude: float = 0.01,
    volumetric: float = 0.2,
    kind: str = "walk",
    components=None,
    free=False,
    rate: float | None = None,
    seed: int = 0,
) -> list[LoadingPath]:
    """``n`` random strain paths of ``segments`` linear segments of ``increments`` steps.

    The strains at the ends of the segments: with ``kind="walk"`` a random walk (steps of
    tensor components uniform, of standard deviation ``amplitude / sqrt(segments)``, the
    components held within ``amplitude``), with ``kind="linear"`` independent strains
    (tensor components uniform within ``amplitude``); their volumetric part scaled by
    ``volumetric``. ``components``: the components driven (names ``"11"`` ... or Voigt
    indices; by default all six), the others held at zero strain but those of ``free``
    (components, or True for all of them), left free (zero stress): a tension-torsion
    test drives 11 and 12 and frees 22 and 33. A model learns the responses to the
    components of its data only: the paths it runs must hold the others likewise.
    ``rate``: the largest strain rate of a component on each segment (the segments then
    last the time of their strain), by default a unit time each."""
    if kind not in KINDS:
        raise ValueError(f"kind: one of {KINDS}, got {kind!r}")
    if int(n) < 1 or int(segments) < 1 or int(increments) < 1:
        raise ValueError("paths, segments and increments: at least one")
    if not (amplitude > 0 and volumetric >= 0):
        raise ValueError("amplitude: positive; volumetric: non-negative")
    if rate is not None and not rate > 0:
        raise ValueError("rate: positive")
    driven = list(range(6)) if components is None else sorted({_component(c) for c in components})
    mask = np.zeros(6, dtype=bool)
    mask[driven] = True
    released = np.zeros(6, dtype=bool)
    if free is True:
        released = ~mask
    elif free:
        released[[_component(c) for c in free]] = True
        if np.any(released & mask):
            raise ValueError("free: components not driven")
    rng = np.random.default_rng(seed)
    step = amplitude / np.sqrt(segments) * np.sqrt(3.0)  # uniform of that deviation
    out = []
    for _ in range(int(n)):
        tensor = np.zeros(6)  # tensor components (half the engineering shears)
        ends = [np.zeros(6)]
        for _ in range(int(segments)):
            if kind == "walk":
                tensor = np.clip(tensor + rng.uniform(-step, step, 6), -amplitude, amplitude)
            else:
                tensor = rng.uniform(-amplitude, amplitude, 6)
            strain = np.where(mask, tensor, 0.0)
            mean = strain[:3].mean() if mask[:3].all() else 0.0
            strain = strain - (1.0 - volumetric) * mean * _TRACE
            ends.append(strain * _ENGINEERING)
        values = np.array(ends)
        if rate is None:
            times = np.arange(len(ends), dtype=float)
        else:
            span = np.abs(np.diff(values, axis=0)).max(axis=1)
            times = np.concatenate([[0.0], np.cumsum(np.maximum(span, 1e-3 * amplitude) / rate)])
        controls = ~released
        out.append(LoadingPath(times, np.where(controls, values, 0.0), controls,
                               increments=(int(increments),)))  # fmt: skip
    return out


def _component(c) -> int:
    if isinstance(c, (int, np.integer)):
        if not 0 <= int(c) < 6:
            raise ValueError(f"component index {c}: from 0 to 5")
        return int(c)
    return _index(str(c))


def _tensor(v, engineering: bool) -> np.ndarray:
    """Symmetric tensors (n, 3, 3) of Voigt components (n, 6)."""
    v = np.asarray(v, dtype=float)
    t = np.empty((*v.shape[:-1], 3, 3))
    for k, (i, j) in enumerate(PAIRS):
        x = v[..., k] * (0.5 if engineering and i != j else 1.0)
        t[..., i, j] = x
        t[..., j, i] = x
    return t


def _voigt(t, engineering: bool) -> np.ndarray:
    """Voigt components (n, 6) of symmetric tensors (n, 3, 3)."""
    out = np.stack([t[..., i, j] for i, j in PAIRS], axis=-1)
    if engineering:
        out[..., 3:] *= 2.0
    return out


def gradient_tensor(h) -> np.ndarray:
    """The deformation gradients ``F`` (n, 3, 3) of the components 11, 22, 33, 23, 13, 12 of
    the upper triangular displacement gradients of the paths in finite strains (n, 6)."""
    h = np.asarray(h, dtype=float)
    f = np.zeros((*h.shape[:-1], 3, 3))
    f[..., [0, 1, 2], [0, 1, 2]] = 1.0 + h[..., :3]
    f[..., 1, 2], f[..., 0, 2], f[..., 0, 1] = h[..., 3], h[..., 4], h[..., 5]
    return f


def green_lagrange(f) -> np.ndarray:
    """The Green-Lagrange strains (n, 6; engineering shears) of gradients ``F`` (n, 3, 3)."""
    c = np.einsum("...ki,...kj->...ij", f, f)
    return _voigt(0.5 * (c - np.eye(3)), engineering=True)


def second_piola(f, cauchy) -> np.ndarray:
    """The second Piola-Kirchhoff stresses (n, 6) of the gradients ``F`` (n, 3, 3) and the
    Cauchy stresses (n, 6)."""
    inv = np.linalg.inv(f)
    s = np.linalg.det(f)[..., None, None] * inv @ _tensor(cauchy, False) @ np.swapaxes(inv, -1, -2)
    return _voigt(s, engineering=False)


@dataclass
class Dataset:
    """Responses along paths: the ``times`` (paths x points), the ``strain`` and the
    ``stress`` (paths x points x 6, Voigt; engineering shears), the first point of each
    path the natural state, in small strains or (``measure="green-lagrange"``) the
    Green-Lagrange strain and the second Piola-Kirchhoff stress; ``info``: how they were
    made (the behaviour, the paths)."""

    times: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    measure: str = "small"
    info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times = np.atleast_2d(np.asarray(self.times, dtype=float))
        self.strain = np.asarray(self.strain, dtype=float)
        self.stress = np.asarray(self.stress, dtype=float)
        n, m = self.times.shape
        if self.strain.shape != (n, m, 6) or self.stress.shape != (n, m, 6) or m < 2:
            raise ValueError("data: times (paths x points), strains and stresses (x 6)")
        if self.measure not in MEASURES:
            raise ValueError(f"measure: one of {MEASURES}, got {self.measure!r}")
        if not (np.all(np.isfinite(self.strain)) and np.all(np.isfinite(self.stress))):
            raise ValueError("data: finite strains and stresses")

    @property
    def n_paths(self) -> int:
        return self.times.shape[0]

    @property
    def n_points(self) -> int:
        return self.times.shape[1]

    def subset(self, index) -> Dataset:
        """The paths of ``index``."""
        index = np.asarray(index)
        return Dataset(self.times[index], self.strain[index], self.stress[index], self.measure,
                       dict(self.info))  # fmt: skip

    def split(self, validation: float = 0.2, *, seed: int = 0) -> tuple[Dataset, Dataset]:
        """The paths shuffled (``seed``) into a training set and a validation set of the
        fraction ``validation`` of them (at least one path each)."""
        if not 0.0 < validation < 1.0 or self.n_paths < 2:
            raise ValueError("split: a fraction between 0 and 1 of at least two paths")
        order = np.random.default_rng(seed).permutation(self.n_paths)
        k = min(max(round(validation * self.n_paths), 1), self.n_paths - 1)
        return self.subset(np.sort(order[k:])), self.subset(np.sort(order[:k]))

    def save(self, path) -> Path:
        """The data in an ``.npz`` file (their ``info`` as JSON)."""
        path = Path(path)
        np.savez_compressed(path, times=self.times, strain=self.strain, stress=self.stress,
                            measure=np.array(self.measure),
                            info=np.array(json.dumps(self.info)))  # fmt: skip
        return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")

    @classmethod
    def load(cls, path) -> Dataset:
        with np.load(path, allow_pickle=False) as f:
            return cls(f["times"], f["strain"], f["stress"], str(f["measure"]),
                       json.loads(str(f["info"])))  # fmt: skip


def _planned(result: PathResult, path: LoadingPath) -> np.ndarray:
    """The rows of a response at the start and at the planned steps of its path."""
    planned = np.concatenate([[path.times[0]], path.schedule()])
    times = np.asarray(result.times)
    k = np.clip(np.searchsorted(times, planned - 1e-9 * max(1.0, abs(planned[-1]))), 0,
                len(times) - 1)  # fmt: skip
    if not np.allclose(times[k], planned, rtol=1e-9, atol=1e-12):
        raise ValueError(f"path {result.name!r}: the response misses planned steps")
    return k


def measures(result: PathResult, rows=None) -> tuple[np.ndarray, np.ndarray, str]:
    """The strain and the stress of a response (its ``rows``) in the measure of its
    kinematics: small strains, or the Green-Lagrange strain and the second
    Piola-Kirchhoff stress (finite strains)."""
    rows = slice(None) if rows is None else rows
    if not result.finite:
        return np.asarray(result.strain)[rows], np.asarray(result.stress)[rows], "small"
    f = gradient_tensor(np.asarray(result.strain)[rows])
    return green_lagrange(f), second_piola(f, np.asarray(result.cauchy)[rows]), "green-lagrange"


def responder(
    source, kinematics: str = "small", orientation=None
) -> Callable[[LoadingPath], PathResult]:
    """The response to a path of ``source``: a material (a
    :class:`~morphogen.homogenization.materials.Material` or its configuration) at a
    material point (turned by the Bunge angles ``orientation`` of a crystal), or a callable
    ``path -> PathResult`` (a microstructure)."""
    if callable(source) and not isinstance(source, dict):
        return source
    from morphogen.homogenization.materials import Material
    from morphogen.homogenization.nonlinear import simulate_point

    material = Material.from_config(source)
    return lambda path: simulate_point(material, path, kinematics=kinematics,
                                       orientation=orientation)  # fmt: skip


def generate(
    source,
    paths,
    *,
    kinematics: str = "small",
    info: dict | None = None,
    callback: Callable[[int, int], None] | None = None,
) -> Dataset:
    """The responses of ``source`` (see :func:`responder`: a material at a material point,
    or a callable ``path -> PathResult`` of a microstructure) along ``paths`` (loading
    paths or their specifications, all of the same numbers of steps), at their planned
    steps. ``callback(done, total)`` after each path (it may raise
    :class:`~morphogen.stopping.Stopped`)."""
    respond = responder(source, kinematics)
    paths = [LoadingPath.from_spec(p) for p in paths]
    if not paths:
        raise ValueError("generate: at least one path")
    times, strains, stresses, measure = [], [], [], "small"
    for k, path in enumerate(paths):
        result = respond(path)
        rows = _planned(result, path)
        e, s, measure = measures(result, rows)
        times.append(np.asarray(result.times)[rows])
        strains.append(e)
        stresses.append(s)
        if callback is not None:
            callback(k + 1, len(paths))
    if len({len(t) for t in times}) != 1:
        raise ValueError("generate: paths of the same numbers of steps")
    return Dataset(np.array(times), np.array(strains), np.array(stresses), measure,
                   dict(info or {}))  # fmt: skip


@dataclass(frozen=True)
class Scales:
    """The normalization of the data: the root mean squares of the strain, of the stress,
    of the strain increments and the mean time step of a training set. The networks see
    the strains and the increments divided by theirs and give stresses divided by
    theirs: scalars, which keep the isotropy of the data and the potential of the
    stresses (:class:`~morphogen.learning.networks.Potential`)."""

    strain: float
    stress: float
    increment: float
    time: float

    @classmethod
    def of(cls, data: Dataset) -> Scales:
        def rms(a) -> float:
            return float(np.sqrt(np.mean(np.square(a))))

        strain, stress = rms(data.strain), rms(data.stress)
        if not (strain > 0 and stress > 0):
            raise ValueError("data: some strains and stresses")
        increment = rms(np.diff(data.strain, axis=1)) or strain
        dt = float(np.mean(np.diff(data.times, axis=1)))
        return cls(strain, stress, increment, dt if dt > 0 else 1.0)

    def to_dict(self) -> dict:
        return {"strain": self.strain, "stress": self.stress, "increment": self.increment,
                "time": self.time}  # fmt: skip
