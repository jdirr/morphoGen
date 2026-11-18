"""Training of the models: Adam on mini-batches, early stopping on a validation set.

The parameters follow Adam (Kingma & Ba 2015) on the gradients of mini-batches of the
training set (points, or whole paths for a recurrent model), shuffled at each epoch;
the global norm of a gradient is clipped to ``clip``, the learning rate falls
geometrically from ``learning_rate`` to ``final`` times it over the epochs. The
parameters kept are those of the least loss of the validation set (of the training set
without one); the training stops after ``patience`` epochs without a better one. The
seeds of the initial parameters (the model's) and of the shuffles (the training's)
make it reproducible. The mini-batches may be turned at each epoch by the symmetries of
the behaviour: uniform rotations (isotropic behaviours) or the 24 rotations of the cube (a
cubic crystal in its axes), data that the behaviour gives without more responses.
"""

from __future__ import annotations

import copy
import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from morphogen.learning.data import Dataset, _tensor, _voigt
from morphogen.learning.networks import Model


@dataclass
class Training:
    """A training: the model (its parameters those of the best epoch), the losses of the
    training and validation sets at each epoch (mean squares of the gaps of the
    normalized stresses), the best epoch, whether it stopped early, its time."""

    model: Model
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    stopped: bool = False
    time_s: float = 0.0

    @property
    def epochs(self) -> int:
        return len(self.train_loss)

    @property
    def best_loss(self) -> float:
        return float(self.validation_loss[self.best_epoch])

    def summary(self) -> dict:
        return {"epochs": self.epochs, "best_epoch": self.best_epoch,
                "train_loss": float(self.train_loss[self.best_epoch]),
                "validation_loss": self.best_loss, "stopped": self.stopped,
                "time_s": round(self.time_s, 3)}  # fmt: skip


def rotations(n: int, rng) -> np.ndarray:
    """``n`` rotation matrices drawn uniformly (n, 3, 3): the QR decompositions of Gaussian
    matrices, the signs of the diagonals of R taken out, the determinants made positive."""
    q, r = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    q = q * np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
    q[:, :, 0] *= np.sign(np.linalg.det(q))[:, None]
    return q


def cubic_rotations() -> np.ndarray:
    """The 24 rotations of the cube (24, 3, 3): the signed permutation matrices of
    determinant 1, the proper symmetries of a cubic crystal in its axes."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3))
            m[range(3), perm] = signs
            if np.linalg.det(m) > 0:
                out.append(m)
    return np.array(out)


#: The symmetries the mini-batches may be turned by (``rotate`` of :func:`train`)
SYMMETRIES = ("isotropic", "cubic")


def _symmetry(rotate):
    """The rotations of ``rotate``: None (none), "isotropic" (uniform: True) or the group
    of the cube (``"cubic"``)."""
    if rotate in (False, None, "none"):
        return None
    if rotate is True or rotate == "isotropic":
        return "isotropic"
    if rotate == "cubic":
        return cubic_rotations()
    raise ValueError(f"rotate: false, true or one of {', '.join(SYMMETRIES)}, got {rotate!r}")


def voigt_rotations(rot) -> tuple[np.ndarray, np.ndarray]:
    """The 6 x 6 matrices (n, 6, 6) that turn the Voigt components of strains (engineering
    shears) and of stresses by rotations ``rot`` (n, 3, 3): ``R t R^T``."""
    eye = np.eye(6)
    out = []
    for engineering in (True, False):
        basis = _tensor(eye, engineering)  # (6, 3, 3): the tensors of the components
        turned = np.einsum("nij,kjl,nml->nkim", rot, basis, rot)
        out.append(np.swapaxes(_voigt(turned, engineering), 1, 2))
    return out[0], out[1]


def rotated(data: Dataset, rng, points: bool = False) -> Dataset:
    """The paths of ``data`` each turned by a random rotation (each point, with
    ``points``): the responses of an isotropic behaviour along the turned paths."""
    n = data.n_paths * (data.n_points if points else 1)
    te, ts = voigt_rotations(rotations(n, rng))
    shape = (data.n_paths, data.n_points if points else 1, 6, 6)
    te, ts = te.reshape(shape), ts.reshape(shape)
    strain = np.einsum("ptij,ptj->pti", te, data.strain)
    stress = np.einsum("ptij,ptj->pti", ts, data.stress)
    return Dataset(data.times, strain, stress, data.measure, data.info)


def _turned(model: Model, arrays, rng, group="isotropic") -> tuple:
    """A mini-batch of normalized arrays (:meth:`~morphogen.learning.networks.Model.
    prepare`) turned by a random rotation per sample (a point, or a path): uniform, or of
    the rotations ``group`` (n, 3, 3); the scales are scalars, the rotations act on the
    normalized strains, increments and stresses."""
    n = arrays[0].shape[0]
    rot = rotations(n, rng) if isinstance(group, str) else group[rng.integers(len(group), size=n)]
    te, ts = voigt_rotations(rot)
    if not model.recurrent:
        x, y = arrays
        return np.einsum("nij,nj->ni", te, x), np.einsum("nij,nj->ni", ts, y)
    if model.kind == "standard":  # the strains, the time steps, the stresses
        x, dt, y = arrays
        return np.einsum("nij,ntj->nti", te, x), dt, np.einsum("nij,ntj->nti", ts, y)
    u, x, y = arrays
    u = u.copy()
    u[..., :6] = np.einsum("nij,ntj->nti", te, u[..., :6])
    u[..., 6:12] = np.einsum("nij,ntj->nti", te, u[..., 6:12])
    return u, np.einsum("nij,ntj->nti", te, x), np.einsum("nij,ntj->nti", ts, y)


def train(
    model: Model,
    data: Dataset,
    validation: Dataset | None = None,
    *,
    epochs: int = 200,
    batch: int = 32,
    learning_rate: float = 0.01,
    final: float = 0.1,
    patience: int | None = None,
    clip: float = 1.0,
    rotate: bool | str = False,
    seed: int = 0,
    callback: Callable[[int, float, float], None] | None = None,
) -> Training:
    """Train ``model`` on ``data`` (see the module), set up on them first when it has no
    parameters; ``validation``: the paths of the early stopping. With ``rotate`` (true or
    ``"isotropic"``: an isotropic behaviour), each mini-batch of paths is turned by random
    rotations at each epoch (:func:`rotated`): the data of every direction, without more
    responses; with ``"cubic"`` (a cubic crystal, its data in its axes), by random
    rotations of the cube (:func:`cubic_rotations`). ``callback(epoch, train_loss,
    validation_loss)`` after each epoch (it may raise :class:`~morphogen.stopping.
    Stopped`)."""
    if int(epochs) < 1 or int(batch) < 1 or not learning_rate > 0 or not 0 < final <= 1:
        raise ValueError("epochs, batch: at least one; learning_rate positive; final in (0, 1]")
    group = _symmetry(rotate)
    if data.measure != model.measure and model.ready:
        raise ValueError(f"data in {data.measure} strains, the model in {model.measure}")
    t0 = time.perf_counter()
    if not model.ready:
        model.setup(data)
    arrays = model.prepare(data)
    checks = model.prepare(validation) if validation is not None else None
    n = arrays[0].shape[0]
    rng = np.random.default_rng(seed)
    first = {k: np.zeros_like(v) for k, v in model.params.items()}
    second = {k: np.zeros_like(v) for k, v in model.params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    out = Training(model)
    best, kept, wait, count = np.inf, copy.deepcopy(model.params), 0, 0
    for epoch in range(int(epochs)):
        rate = learning_rate * final ** (epoch / max(int(epochs) - 1, 1))
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, int(batch)):
            idx = order[start : start + int(batch)]
            sub = tuple(a[idx] for a in arrays)
            if group is not None:
                sub = _turned(model, sub, rng, group)
            loss, grads = model.objective(sub)
            total += loss * idx.size
            norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
            factor = min(1.0, clip / norm) if clip and norm > 0 else 1.0
            count += 1
            c1, c2 = 1.0 - beta1**count, 1.0 - beta2**count
            for k, g in grads.items():
                g = g * factor
                first[k] = beta1 * first[k] + (1.0 - beta1) * g
                second[k] = beta2 * second[k] + (1.0 - beta2) * g * g
                model.params[k] = model.params[k] - rate * (first[k] / c1) / (
                    np.sqrt(second[k] / c2) + eps
                )
        out.train_loss.append(total / n)
        check = model.objective(checks, grad=False)[0] if checks is not None else total / n
        out.validation_loss.append(float(check))
        if callback is not None:
            callback(epoch, out.train_loss[-1], float(check))
        if check < best:
            best, kept, wait, out.best_epoch = check, copy.deepcopy(model.params), 0, epoch
        else:
            wait += 1
            if patience is not None and wait >= int(patience):
                out.stopped = True
                break
    model.params = kept
    out.time_s = time.perf_counter() - t0
    model.metadata["training"] = out.summary()
    return out


def errors(model: Model, data: Dataset) -> dict:
    """The gaps of the stresses of ``model`` to those of ``data``: relative to the root
    mean square of the stresses of each path (``paths``: one per path; their ``mean`` and
    ``max``) and of all (``relative``), and the root mean square of the gaps (``rmse``,
    in the units of the stresses). A recurrent model runs each path from its natural
    state."""
    if model.recurrent:
        predicted = model.predict(data.strain, data.times)
    else:
        predicted = model.predict(data.strain)
    gap = np.sum((predicted - data.stress) ** 2, axis=-1)
    norm = np.sum(data.stress**2, axis=-1)
    paths = np.sqrt(gap.sum(axis=1) / np.maximum(norm.sum(axis=1), 1e-300))
    return {"relative": float(np.sqrt(gap.sum() / max(norm.sum(), 1e-300))),
            "mean": float(paths.mean()), "max": float(paths.max()),
            "rmse": float(np.sqrt(gap.mean() / 6.0)), "paths": paths.tolist()}  # fmt: skip
