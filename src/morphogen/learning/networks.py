"""The models of learned behaviours, with their gradients written out (no automatic
differentiation): the stress as a function of the strain, a potential of the stress, and
a recurrent network over the strain increments.

Every model sees the strain divided by the scale of the strains of its training set and
gives the stress divided by that of the stresses (:class:`~morphogen.learning.data.
Scales`); its linear term starts from the least-squares fit of the training set, which
the networks then correct. The objective of the training is the mean square of the gap
of the normalized stresses.

* :class:`FeedForward`: ``y = L x + N(x)``, ``N`` a perceptron of hidden layers of
  ``tanh``. Path-independent, its tangent the Jacobian of the network (not symmetric in
  general).
* :class:`Potential`: ``y = dW/dx (x) - dW/dx (0)``, ``W(x) = x.A x / 2 + w . z_K`` an
  input-convex network (Amos, Xu & Kolter 2017): ``z_1 = g(U_1 x + b_1)``, ``z_k = g(W_k
  z_{k-1} + U_k x + b_k)`` with ``W_k >= 0``, ``w >= 0`` (softplus of free parameters),
  ``g`` the softplus (convex, increasing) and ``A = L L^T``. The potential of the stress
  ``psi = s_sigma s_eps (W(x) - W(0) - x . dW/dx(0))`` is convex, zero with its gradient
  at the natural state: the stress derives from it (hyperelasticity), its tangent
  ``d2 psi / d eps2`` is symmetric and positive semi-definite. The training fits the
  gradient of ``W``: the gradients of the objective with respect to the parameters are
  those of a derivative of ``W`` along the gaps, a forward derivative (tangents of the
  layers) differentiated backward.
* :class:`Recurrent`: a gated recurrent unit (Cho et al. 2014) over the steps, whose
  input is the strain at the end of the step and the increment of the step (and the time
  step, with ``rate``) and whose state ``h`` (``hidden`` values: the internal variables
  of the law, learned) gives the stress ``y = V h + Q x + c``; path-dependent, trained by
  backpropagation through time over whole paths. Its consistent tangent is the
  derivative of the stress of a step with respect to the strain at its end, the state at
  its start held.

In finite strains (data of the Green-Lagrange strain ``E`` and the second Piola-Kirchhoff
stress ``S``) the option ``strain`` of every model (:data:`STRAINS`) chooses what it sees:
``logarithmic`` (the default), the logarithmic strain ``H = 1/2 ln C`` (and its increments,
for the recurrent network), its output the stress ``T`` conjugate to ``H``, whence ``S =
J^T T`` and ``dS/dE = J^T (dT/dH) J + T : d2H/dE2`` (:mod:`morphogen.learning.logarithmic`;
the potential ``psi(H)`` of ``T`` is that of ``S``, ``psi(H(E))``); or ``green-lagrange``,
``E`` and ``S`` themselves. ``J = dH/dE`` does not depend on the parameters: the gaps of
``S`` reach ``T`` through it, ``J r``. Either way a law is a function of ``C`` (``H`` an
isotropic function of ``C``), objective. The files written before the option load as
laws of ``E``.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
from numba import njit

from morphogen.learning.data import Dataset, Scales
from morphogen.learning.logarithmic import _measures, _pullback

#: The kinds of models (``standard``: :class:`~morphogen.learning.standard.Standard`)
KINDS = ("feedforward", "potential", "recurrent", "standard")
#: The strains of the models in finite strains (their option ``strain``): the logarithmic
#: strain ``1/2 ln C`` (their stress ``T`` conjugate to it, ``S = J^T T``) or the
#: Green-Lagrange strain ``E`` (their stress ``S``)
STRAINS = ("logarithmic", "green-lagrange")


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _softplus(x):
    return np.logaddexp(0.0, x)


def _inverse_softplus(y):
    return np.log(np.expm1(y))


def _strain(strain) -> str:
    """The option ``strain`` of a model, checked."""
    if strain not in STRAINS:
        raise ValueError(f"strain: that of the law in finite strains, one of {STRAINS}, "
                         f"got {strain!r}")  # fmt: skip
    return str(strain)


class Model:
    """A learned behaviour: its parameters (``params``, arrays), the normalization of its
    training set (``scales``), the measure of its data (``"small"`` or
    ``"green-lagrange"``), its options and its ``metadata`` (the data, the training). In
    finite strains its option ``strain`` (:data:`STRAINS`) chooses the strain it sees,
    ``1/2 ln C`` or ``E`` (see the module)."""

    kind = ""
    recurrent = False
    #: Whether its tangent is symmetric (a potential): symmetrized to the bit in ``ln C``
    _symmetric = False

    def __init__(self, **options) -> None:
        self.options = options
        self.params: dict[str, np.ndarray] = {}
        self.scales: Scales | None = None
        self.measure = "small"
        self.metadata: dict = {}
        self.path: Path | None = None

    # -- set-up ------------------------------------------------------------------------

    @property
    def n_state(self) -> int:
        """The internal variables of a point (the state of a recurrent network)."""
        return 0

    @property
    def ready(self) -> bool:
        return bool(self.params) and self.scales is not None

    @property
    def logarithmic(self) -> bool:
        """Whether the model sees the logarithmic strains of the Green-Lagrange strains of
        its data and gives the stresses conjugate to them (finite strains,
        ``strain="logarithmic"``)."""
        return self.measure == "green-lagrange" and self.options.get("strain") == "logarithmic"

    def setup(self, data: Dataset) -> None:
        """The scales of the training set ``data``, and the parameters initialized
        (seeded by the option ``seed``)."""
        self.scales = Scales.of(data)
        self.measure = data.measure
        rng = np.random.default_rng(int(self.options.get("seed", 0)))
        self.params = self._initial(rng, *self.prepare(data))

    def _initial(self, rng, *arrays) -> dict:
        raise NotImplementedError

    def prepare(self, data: Dataset) -> tuple:
        """The normalized arrays of the objective, their first axis the samples (the
        points of the paths, or the paths of a recurrent model)."""
        s = self._scales()
        x = data.strain.reshape(-1, 6) / s.strain
        y = data.stress.reshape(-1, 6) / s.stress
        return x, y

    def _scales(self) -> Scales:
        if self.scales is None:
            raise ValueError(f"{self.kind} model: not trained (no scales)")
        return self.scales

    def _conjugates(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """The normalized strains and stresses (n x 6) that the model sees and gives, of
        those of its data: themselves, or (``ln C``) the logarithmic strains and the
        stresses ``T = J^-T S`` conjugate to them."""
        if not self.logarithmic:
            return x, y
        h, jac = _measures(np.ascontiguousarray(x), self._scales().strain)
        return h, np.linalg.solve(np.swapaxes(jac, 1, 2), y[..., None])[..., 0]

    def _pulled(self, x) -> tuple[np.ndarray, np.ndarray]:
        """The normalized stresses ``S = J^T T`` and tangents ``dS/dE`` of a law of ``ln C``
        at normalized Green-Lagrange strains ``x`` (n x 6): ``T`` and ``dT/dH`` those of
        the model at their logarithmic strains."""
        x = np.ascontiguousarray(x)
        scale = self._scales().strain
        h = _measures(x, scale)[0]
        return _pullback(x, np.ascontiguousarray(self._output(h)),
                         np.ascontiguousarray(self._jacobian(h)), scale,
                         self._symmetric)  # fmt: skip

    def objective(self, arrays, grad: bool = True) -> tuple[float, dict | None]:
        """The mean square of the gaps of the normalized stresses, and its gradients with
        respect to the parameters (with ``grad``)."""
        raise NotImplementedError

    # -- use ---------------------------------------------------------------------------

    def predict(self, strain, times=None) -> np.ndarray:
        """The stresses of strains (..., 6); of a recurrent model, of paths (paths x
        points x 6, the first point the natural state, at the ``times``). In finite
        strains: the second Piola-Kirchhoff stresses of Green-Lagrange strains."""
        s = self._scales()
        x = np.asarray(strain, dtype=float) / s.strain
        if self.logarithmic:  # T of the logarithmic strains, S = J^T T
            h, jac = _measures(np.ascontiguousarray(x.reshape(-1, 6)), s.strain)
            return s.stress * np.einsum("nki,nk->ni", jac, self._output(h)).reshape(x.shape)
        return s.stress * self._output(x.reshape(-1, 6)).reshape(x.shape)

    def tangent(self, strain) -> np.ndarray:
        """The tangents ``d sigma / d eps`` (..., 6, 6) at strains (..., 6); ``dS/dE`` in
        finite strains."""
        s = self._scales()
        x = np.asarray(strain, dtype=float) / s.strain
        if self.logarithmic:
            j = self._pulled(x.reshape(-1, 6))[1]
        else:
            j = self._jacobian(x.reshape(-1, 6))
        return (s.stress / s.strain * j).reshape(*x.shape, 6)

    def step(self, e_new, e_old, h_old, dt):
        """A step of points: the stresses (n x 6) and tangents (n x 6 x 6) at the strains
        ``e_new`` at its end, from the strains ``e_old`` and the states ``h_old`` (n x
        :attr:`n_state`) at its start over the time ``dt``, and the states at its end."""
        e_new = np.asarray(e_new, dtype=float)
        if self.logarithmic:  # the stresses and their tangents at once
            s = self._scales()
            y, c = self._pulled(e_new / s.strain)
            return s.stress * y, s.stress / s.strain * c, np.zeros((e_new.shape[0], 0))
        return self.predict(e_new), self.tangent(e_new), np.zeros((e_new.shape[0], 0))

    def _output(self, x):
        raise NotImplementedError

    def _jacobian(self, x):
        raise NotImplementedError

    # -- files -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """The metadata of a file of the model (JSON)."""
        from morphogen._version import __version__

        return {"kind": self.kind, "options": self.options, "measure": self.measure,
                "scales": self._scales().to_dict(), "metadata": self.metadata,
                "morphogen": __version__}  # fmt: skip

    def save(self, path) -> Path:
        """The parameters in an ``.npz`` file, with the metadata (JSON) of the model."""
        if not self.ready:
            raise ValueError(f"{self.kind} model: not trained, nothing to save")
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        meta = json.dumps(self.to_dict(), default=float)
        np.savez(path, metadata=np.array(meta), **{f"param_{k}": v for k, v in self.params.items()})
        self.path = path
        return path


def load_model(path) -> Model:
    """A model saved by :meth:`Model.save`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["metadata"]))
        params = {k.removeprefix("param_"): np.array(f[k]) for k in f.files if k != "metadata"}
    cls = _classes().get(meta.get("kind"))
    if cls is None:
        raise ValueError(f"{path}: not a model of morphogen.learning (kind {meta.get('kind')!r})")
    # the files written before the option strain: laws of E in finite strains
    model = cls(**{"strain": "green-lagrange", **meta["options"]})
    model.params = params
    model.scales = Scales(**meta["scales"])
    model.measure = meta["measure"]
    model.metadata = meta.get("metadata", {})
    model.path = path
    return model


def _classes() -> dict:
    """The classes of the models by their kinds."""
    from morphogen.learning.standard import Standard

    return {c.kind: c for c in (FeedForward, Potential, Recurrent, Standard)}


def make_model(kind: str, **options) -> Model:
    """A model of ``kind`` (:data:`KINDS`) with its options (``hidden``, ``seed``,
    ``strain`` in finite strains; ``rate`` for a recurrent or standard model;
    ``internal``, ``plastic`` and ``smoothing`` for a standard model)."""
    classes = _classes()
    if kind not in classes:
        raise ValueError(f"model: one of {KINDS}, got {kind!r}")
    return classes[kind](**options)


def _sizes(hidden) -> list[int]:
    sizes = [int(hidden)] if np.ndim(hidden) == 0 else [int(h) for h in hidden]
    if not sizes or min(sizes) < 1:
        raise ValueError("hidden: the sizes of the hidden layers, positive")
    return sizes


def _linear_fit(x, y) -> np.ndarray:
    """The matrix ``L`` of the least-squares fit ``y = L x``."""
    return np.linalg.lstsq(x, y, rcond=None)[0].T


class FeedForward(Model):
    """The stress of the strain by a perceptron (see the module): ``hidden``, the sizes
    of its hidden layers; ``strain``, the strain it sees in finite strains
    (:data:`STRAINS`; ``T(1/2 ln C)``, the default, or ``S(E)``); ``seed``, that of its
    initial parameters."""

    kind = "feedforward"

    def __init__(self, hidden=(32, 32), *, strain: str = "logarithmic", seed: int = 0) -> None:
        super().__init__(hidden=_sizes(hidden), strain=_strain(strain), seed=int(seed))

    def _initial(self, rng, x, y) -> dict:
        sizes = [6, *self.options["hidden"]]
        p = {}
        for k, (a, b) in enumerate(itertools.pairwise(sizes)):
            p[f"W{k}"] = rng.normal(0.0, 1.0 / math.sqrt(a), (b, a))
            p[f"b{k}"] = np.zeros(b)
        p["Wout"] = rng.normal(0.0, 0.1 / math.sqrt(sizes[-1]), (6, sizes[-1]))
        p["bout"] = np.zeros(6)
        p["L"] = _linear_fit(*self._conjugates(x, y))
        return p

    def _layers(self, x) -> list:
        p, z = self.params, [x]
        for k in range(len(self.options["hidden"])):
            z.append(np.tanh(z[-1] @ p[f"W{k}"].T + p[f"b{k}"]))
        return z

    def _output(self, x):
        p = self.params
        return self._layers(x)[-1] @ p["Wout"].T + p["bout"] + x @ p["L"].T

    def objective(self, arrays, grad=True):
        x, y = arrays
        p = self.params
        jac = None
        if self.logarithmic:  # the network of the logarithmic strains, its output T
            x, jac = _measures(np.ascontiguousarray(x), self._scales().strain)
        z = self._layers(x)
        out = z[-1] @ p["Wout"].T + p["bout"] + x @ p["L"].T
        if jac is not None:  # S = J^T T
            out = np.einsum("nki,nk->ni", jac, out)
        r = out - y
        loss = float(np.mean(r * r))
        if not grad:
            return loss, None
        dy = 2.0 * r / r.size
        if jac is not None:  # the gaps of S reach T through J
            dy = np.einsum("nik,nk->ni", jac, dy)
        g = {"Wout": dy.T @ z[-1], "bout": dy.sum(0), "L": dy.T @ x}
        dz = dy @ p["Wout"]
        for k in reversed(range(len(self.options["hidden"]))):
            da = dz * (1.0 - z[k + 1] ** 2)
            g[f"W{k}"] = da.T @ z[k]
            g[f"b{k}"] = da.sum(0)
            dz = da @ p[f"W{k}"]
        return loss, g

    def _jacobian(self, x):
        p = self.params
        z = self._layers(x)
        t = np.broadcast_to(np.eye(6), (x.shape[0], 6, 6))
        for k in range(len(self.options["hidden"])):
            t = (1.0 - z[k + 1] ** 2)[:, :, None] * np.einsum("ij,njd->nid", p[f"W{k}"], t)
        return np.einsum("ij,njd->nid", p["Wout"], t) + p["L"]


class Potential(Model):
    """The stress derived from a learned convex potential (see the module): ``hidden``,
    the sizes of the hidden layers of its input-convex network; ``strain``, the strain of
    the potential in finite strains (:data:`STRAINS`; ``psi(1/2 ln C)``, the default, which
    represents Hencky's elasticity with its quadratic term, or ``psi(E)``); ``seed``, that
    of its initial parameters."""

    kind = "potential"
    _symmetric = True

    def __init__(self, hidden=(32, 32), *, strain: str = "logarithmic", seed: int = 0) -> None:
        super().__init__(hidden=_sizes(hidden), strain=_strain(strain), seed=int(seed))

    def _initial(self, rng, x, y) -> dict:
        sizes = self.options["hidden"]
        p = {}
        for k, h in enumerate(sizes):
            p[f"U{k}"] = rng.normal(0.0, 1.0 / math.sqrt(6.0), (h, 6))
            p[f"b{k}"] = np.zeros(h)
            if k:  # W_k = softplus(V_k) about 1 / h_{k-1}: the mean of the layer below
                base = _inverse_softplus(1.0 / sizes[k - 1])
                p[f"V{k}"] = base + rng.normal(0.0, 0.3, (h, sizes[k - 1]))
        p["v"] = _inverse_softplus(0.1 / sizes[-1]) + rng.normal(0.0, 0.3, sizes[-1])
        # A: the symmetric positive part of the least-squares fit (of T = J^-T S to the
        # logarithmic strain in ln C: Hencky's elasticity exactly)
        fit = _linear_fit(*self._conjugates(x, y))
        values, vectors = np.linalg.eigh(0.5 * (fit + fit.T))
        values = np.maximum(values, 1e-3 * max(values.max(), 1e-12))
        p["L"] = vectors * np.sqrt(values)
        return p

    def _weights(self) -> tuple[list, np.ndarray, np.ndarray]:
        p = self.params
        w = [None] + [_softplus(p[f"V{k}"]) for k in range(1, len(self.options["hidden"]))]
        return w, _softplus(p["v"]), p["L"] @ p["L"].T

    def _primal(self, x, w) -> tuple[list, list]:
        """The pre-activations ``a_k`` and the layers ``z_k`` of the points ``x``."""
        p, a, z = self.params, [], []
        for k in range(len(self.options["hidden"])):
            ak = x @ p[f"U{k}"].T + p[f"b{k}"]
            if k:
                ak = ak + z[-1] @ w[k].T
            a.append(ak)
            z.append(_softplus(ak))
        return a, z

    def _gradient(self, x, hessian: bool = False):
        """``dW/dx`` at the points ``x`` (n x 6), and its Jacobian (n x 6 x 6)."""
        p = self.params
        w, wout, amat = self._weights()
        a, _ = self._primal(x, w)
        s = [_sigmoid(ak) for ak in a]
        n_layers = len(a)
        deltas = [None] * n_layers  # dW / da_k
        d = wout * s[-1]
        deltas[-1] = d
        for k in range(n_layers - 1, 0, -1):
            d = (d @ w[k]) * s[k - 1]
            deltas[k - 1] = d
        grad = sum(deltas[k] @ p[f"U{k}"] for k in range(n_layers)) + x @ amat
        if not hessian:
            return grad, None
        ds = [sk * (1.0 - sk) for sk in s]
        tang = []  # da_k / dx (n x h_k x 6)
        for k in range(n_layers):
            t = np.broadcast_to(p[f"U{k}"], (x.shape[0], *p[f"U{k}"].shape))
            if k:
                t = t + np.einsum("ij,njd->nid", w[k], s[k - 1][:, :, None] * tang[-1])
            tang.append(t)
        dd = (wout * ds[-1])[:, :, None] * tang[-1]
        hess = np.einsum("nid,ij->njd", dd, p[f"U{n_layers - 1}"])
        for k in range(n_layers - 1, 0, -1):
            back = np.einsum("nid,ij->njd", dd, w[k])
            dd = (
                back * s[k - 1][:, :, None]
                + ((deltas[k] @ w[k]) * ds[k - 1])[:, :, None] * tang[k - 1]
            )
            hess = hess + np.einsum("nid,ij->njd", dd, p[f"U{k - 1}"])
        hess = hess + amat
        return grad, 0.5 * (hess + np.swapaxes(hess, 1, 2))

    def _output(self, x):
        return self._gradient(x)[0] - self._gradient(np.zeros((1, 6)))[0]

    def _jacobian(self, x):
        return self._gradient(x, hessian=True)[1]

    def _energy(self, x):
        """``W(x) - W(0) - x . dW/dx(0)`` at the points ``x``."""
        w, wout, amat = self._weights()

        def value(pts):
            _, z = self._primal(pts, w)
            return z[-1] @ wout + 0.5 * np.einsum("ni,ij,nj->n", pts, amat, pts)

        zero = np.zeros((1, 6))
        return value(x) - value(zero)[0] - x @ self._gradient(zero)[0][0]

    def energy(self, strain) -> np.ndarray:
        """The potential of the stress (energy per unit volume) at strains (..., 6;
        Green-Lagrange strains in finite strains)."""
        s = self._scales()
        x = np.asarray(strain, dtype=float) / s.strain
        flat = x.reshape(-1, 6)
        if self.logarithmic:  # psi(H(E))
            flat = _measures(np.ascontiguousarray(flat), s.strain)[0]
        return s.stress * s.strain * self._energy(flat).reshape(x.shape[:-1])

    def objective(self, arrays, grad=True):
        x, y = arrays
        jac = None
        if self.logarithmic:  # the potential of the logarithmic strains, its gradient T
            x, jac = _measures(np.ascontiguousarray(x), self._scales().strain)
        out = self._output(x)
        if jac is not None:  # S = J^T T
            out = np.einsum("nki,nk->ni", jac, out)
        r = out - y
        loss = float(np.mean(r * r))
        if not grad:
            return loss, None
        rr = 2.0 * r / r.size
        if jac is not None:  # the gaps of S reach T through J
            rr = np.einsum("nik,nk->ni", jac, rr)
        # the gaps at the points, minus their sum at the natural state (dW/dx(0) subtracted)
        pts = np.vstack([x, np.zeros((1, 6))])
        return loss, self._directional(pts, np.vstack([rr, -rr.sum(0, keepdims=True)]))

    def _directional(self, x, r) -> dict:
        """The gradients with respect to the parameters of ``G = sum_n r_n . dW/dx(x_n)``:
        ``G`` is the sum of the derivatives of ``W`` along ``r_n`` (forward tangents of
        the layers ``a'``, ``z'``), differentiated backward."""
        p = self.params
        w, wout, _ = self._weights()
        n_layers = len(self.options["hidden"])
        a, z = self._primal(x, w)
        s = [_sigmoid(ak) for ak in a]
        at_list, zt = [], []
        for k in range(n_layers):
            at = r @ p[f"U{k}"].T
            if k:
                at = at + zt[-1] @ w[k].T
            at_list.append(at)
            zt.append(s[k] * at)
        g = {"v": zt[-1].sum(0) * _sigmoid(p["v"])}
        zbar = np.zeros_like(z[-1])
        ztbar = np.broadcast_to(wout, zt[-1].shape)
        for k in reversed(range(n_layers)):
            sk = s[k]
            atbar = sk * ztbar
            abar = sk * (1.0 - sk) * at_list[k] * ztbar + sk * zbar
            g[f"U{k}"] = abar.T @ x + atbar.T @ r
            g[f"b{k}"] = abar.sum(0)
            if k:
                wbar = abar.T @ z[k - 1] + atbar.T @ zt[k - 1]
                g[f"V{k}"] = wbar * _sigmoid(p[f"V{k}"])
                zbar = abar @ w[k]
                ztbar = atbar @ w[k]
        lmat = p["L"]
        g["L"] = x.T @ (r @ lmat) + r.T @ (x @ lmat)
        return g


#: The parameters of the recurrent network, in the order of :func:`_gru_gradients`
_GRU = ("Wi", "Wh", "b", "bn", "V", "Q", "c")


@njit(cache=True)
def _gru_gradients(u, x, y, jac, log, wi, wh, b, bn, v, q, c):
    """The mean square of the gaps of the stresses of a mini-batch of paths (time-major:
    steps x paths x ...) and its gradients with respect to the parameters of the gated
    recurrent unit (backpropagation through time; the products of a step by BLAS). With
    ``log`` the network gives the stress ``T`` conjugate to the logarithmic strain ``x``
    and the stresses ``y`` are ``S = J^T T`` (``jac``: ``J = dH/dE`` at each step of each
    path): the gaps of ``S`` reach ``T`` as ``J r``."""
    steps, n, n_in = u.shape
    h = wh.shape[0]
    states = np.zeros((steps + 1, n, h))
    gates = np.empty((4, steps, n, h))
    for t in range(steps):
        gi = np.dot(u[t], wi)
        gh = np.dot(states[t], wh)
        for i in range(n):
            for j in range(h):
                r = 1.0 / (1.0 + math.exp(-(gi[i, j] + b[j] + gh[i, j])))
                z = 1.0 / (1.0 + math.exp(-(gi[i, h + j] + b[h + j] + gh[i, h + j])))
                hn = gh[i, 2 * h + j] + bn[j]
                cand = math.tanh(gi[i, 2 * h + j] + b[2 * h + j] + r * hn)
                states[t + 1, i, j] = cand + z * (states[t, i, j] - cand)
                gates[0, t, i, j] = r
                gates[1, t, i, j] = z
                gates[2, t, i, j] = cand
                gates[3, t, i, j] = hn
    total = steps * n * 6
    loss = 0.0
    dy = np.empty((steps, n, 6))
    gv, gq, gc = np.zeros((h, 6)), np.zeros((6, 6)), np.zeros(6)
    qt = np.ascontiguousarray(q.T)
    gap = np.empty(6)
    for t in range(steps):
        out = np.dot(states[t + 1], v) + np.dot(x[t], qt)
        for i in range(n):
            if log:  # the gaps of S = J^T T, and their derivatives with respect to T
                for k in range(6):
                    s = 0.0
                    for m in range(6):
                        s += jac[t, i, m, k] * (out[i, m] + c[m])
                    gap[k] = s - y[t, i, k]
                    loss += gap[k] * gap[k]
                for m in range(6):
                    s = 0.0
                    for k in range(6):
                        s += jac[t, i, m, k] * gap[k]
                    dy[t, i, m] = 2.0 * s / total
                    gc[m] += dy[t, i, m]
                continue
            for k in range(6):
                res = out[i, k] + c[k] - y[t, i, k]
                loss += res * res
                dy[t, i, k] = 2.0 * res / total
                gc[k] += dy[t, i, k]
        gv += np.dot(np.ascontiguousarray(states[t + 1].T), dy[t])
        gq += np.dot(np.ascontiguousarray(dy[t].T), x[t])
    vt, wht = np.ascontiguousarray(v.T), np.ascontiguousarray(wh.T)
    dh = np.zeros((n, h))
    dgi = np.empty((steps, n, 3 * h))
    dgh = np.empty((n, 3 * h))
    gwi, gwh = np.zeros((n_in, 3 * h)), np.zeros((h, 3 * h))
    gb, gbn = np.zeros(3 * h), np.zeros(h)
    for t in range(steps - 1, -1, -1):
        dh += np.dot(dy[t], vt)
        for i in range(n):
            for j in range(h):
                r, z = gates[0, t, i, j], gates[1, t, i, j]
                cand, hn = gates[2, t, i, j], gates[3, t, i, j]
                d = dh[i, j]
                dan = d * (1.0 - z) * (1.0 - cand * cand)
                dar = dan * hn * r * (1.0 - r)
                daz = d * (states[t, i, j] - cand) * z * (1.0 - z)
                dgi[t, i, j], dgi[t, i, h + j], dgi[t, i, 2 * h + j] = dar, daz, dan
                dgh[i, j], dgh[i, h + j], dgh[i, 2 * h + j] = dar, daz, dan * r
                gbn[j] += dan * r
                dh[i, j] = d * z
        dh += np.dot(dgh, wht)
        gwh += np.dot(np.ascontiguousarray(states[t].T), dgh)
        gwi += np.dot(np.ascontiguousarray(u[t].T), dgi[t])
        for i in range(n):
            for j in range(3 * h):
                gb[j] += dgi[t, i, j]
    return loss / total, gwi, gwh, gb, gbn, gv, gq, gc


class Recurrent(Model):
    """A recurrent network over the steps of the paths (see the module): ``hidden``, the
    size of its state; ``rate``, whether the time step is an input (rate-dependent
    behaviours); ``strain``, the strain it sees in finite strains (:data:`STRAINS`: the
    logarithmic strain and its increments, its output ``T``, the default; or ``E`` and its
    increments, its output ``S``); ``seed``, that of its initial parameters."""

    kind = "recurrent"
    recurrent = True

    def __init__(self, hidden: int = 32, *, rate: bool = False, strain: str = "logarithmic",
                 seed: int = 0) -> None:  # fmt: skip
        if isinstance(hidden, bool) or not isinstance(hidden, (int, np.integer)) or hidden < 1:
            raise ValueError(f"hidden: the size of the state, a positive integer, got {hidden!r}")
        super().__init__(hidden=int(hidden), rate=bool(rate), strain=_strain(strain),
                         seed=int(seed))  # fmt: skip

    @property
    def n_state(self) -> int:
        return self.options["hidden"]

    @property
    def n_inputs(self) -> int:
        return 13 if self.options["rate"] else 12

    def _inputs(self, strain, times):
        """The inputs (paths x steps x inputs) and the normalized strains that the network
        sees at the ends of the steps (paths x steps x 6) of paths, and (``ln C``: the
        logarithmic strains, their increments in the inputs) their ``J = dH/dE`` (paths x
        steps x 6 x 6; None otherwise)."""
        s = self._scales()
        strain = np.asarray(strain, dtype=float)
        x = strain / s.strain
        jac = None
        if self.logarithmic:
            h, jac = _measures(np.ascontiguousarray(x.reshape(-1, 6)), s.strain)
            x, jac = h.reshape(strain.shape), jac.reshape(*strain.shape, 6)[:, 1:]
            parts = [x[:, 1:], np.diff(x, axis=1) * (s.strain / s.increment)]
        else:
            parts = [x[:, 1:], np.diff(strain, axis=1) / s.increment]
        if self.options["rate"]:
            if times is None:
                raise ValueError("recurrent model of a rate: give the times of the paths")
            dt = np.diff(np.atleast_2d(np.asarray(times, dtype=float)), axis=1)
            parts.append(np.broadcast_to(dt[..., None], (*x[:, 1:].shape[:2], 1)) / s.time)
        return np.concatenate(parts, axis=-1), x[:, 1:], jac

    def prepare(self, data: Dataset) -> tuple:
        """The inputs, the normalized strains at the ends of the steps and the normalized
        stresses of the paths (``ln C``: the Green-Lagrange strains, which the rotations of
        the training turn, their logarithmic strains in the inputs)."""
        u, x, _ = self._inputs(data.strain, data.times)
        if self.logarithmic:
            x = data.strain[:, 1:] / self._scales().strain
        return u, x, data.stress[:, 1:] / self._scales().stress

    def _initial(self, rng, u, x, y) -> dict:
        h, n_in = self.options["hidden"], self.n_inputs
        wh = np.concatenate([np.linalg.qr(rng.normal(size=(h, h)))[0] for _ in range(3)], axis=1)
        b = np.zeros(3 * h)
        b[h : 2 * h] = 1.0  # the update gate starts open to the state: a memory
        return {
            "Wi": rng.normal(0.0, 1.0 / math.sqrt(n_in), (n_in, 3 * h)),
            "Wh": wh,
            "b": b,
            "bn": np.zeros(h),
            "V": rng.normal(0.0, 0.1 / math.sqrt(h), (h, 6)),
            "Q": _linear_fit(*self._conjugates(x.reshape(-1, 6), y.reshape(-1, 6))),
            "c": np.zeros(6),
        }

    def _run(self, u, x):
        """The normalized stresses (paths x steps x 6) of the inputs ``u`` and the states
        (paths x steps + 1 x hidden)."""
        p, h = self.params, self.options["hidden"]
        n, steps = u.shape[:2]
        gi = u @ p["Wi"] + p["b"]
        state = np.zeros((n, h))
        states = np.zeros((n, steps + 1, h))
        wh, bn = p["Wh"], p["bn"]
        for t in range(steps):
            gh = state @ wh
            rz = _sigmoid(gi[:, t, : 2 * h] + gh[:, : 2 * h])
            r, z = rz[:, :h], rz[:, h:]
            cand = np.tanh(gi[:, t, 2 * h :] + r * (gh[:, 2 * h :] + bn))
            state = cand + z * (state - cand)
            states[:, t + 1] = state
        return states[:, 1:] @ p["V"] + x @ p["Q"].T + p["c"], states

    def objective(self, arrays, grad=True):
        u, x, y = arrays
        jac = None
        if self.logarithmic:  # the logarithmic strains of the (turned) Green-Lagrange ones
            h, jac = _measures(np.ascontiguousarray(x.reshape(-1, 6)), self._scales().strain)
            x, jac = h.reshape(x.shape), jac.reshape(*x.shape, 6)
        if not grad:
            out = self._run(u, x)[0]
            if jac is not None:  # S = J^T T
                out = np.einsum("ptki,ptk->pti", jac, out)
            res = out - y
            return float(np.mean(res * res)), None
        p = self.params
        # time-major arrays: the products of a step on contiguous blocks
        major = [np.ascontiguousarray(np.swapaxes(a, 0, 1)) for a in (u, x, y)]
        log = jac is not None
        jac = np.ascontiguousarray(np.swapaxes(jac, 0, 1)) if log else np.zeros((1, 1, 6, 6))
        loss, *grads = _gru_gradients(*major, jac, log,
                                      *(np.ascontiguousarray(p[k]) for k in _GRU))  # fmt: skip
        return float(loss), dict(zip(_GRU, grads, strict=True))

    def predict(self, strain, times=None) -> np.ndarray:
        strain = np.asarray(strain, dtype=float)
        single = strain.ndim == 2
        strain = strain[None] if single else strain
        if times is not None and single:
            times = np.asarray(times, dtype=float)[None]
        u, x, jac = self._inputs(strain, times)
        out = self._run(u, x)[0]
        if jac is not None:  # S = J^T T
            out = np.einsum("ptki,ptk->pti", jac, out)
        out = out * self._scales().stress
        out = np.concatenate([np.zeros((strain.shape[0], 1, 6)), out], axis=1)
        return out[0] if single else out

    def tangent(self, strain):
        raise ValueError("a recurrent model has the tangents of its steps (step)")

    def step(self, e_new, e_old, h_old, dt):
        p, h, s = self.params, self.options["hidden"], self._scales()
        e_new, e_old = np.asarray(e_new, dtype=float), np.asarray(e_old, dtype=float)
        n = e_new.shape[0]
        x = e_new / s.strain
        seen = x
        if self.logarithmic:  # the logarithmic strain at the end of the step, its increment
            seen = _measures(np.ascontiguousarray(x), s.strain)[0]
            start = _measures(np.ascontiguousarray(e_old / s.strain), s.strain)[0]
            parts = [seen, (seen - start) * (s.strain / s.increment)]
        else:
            parts = [x, (e_new - e_old) / s.increment]
        if self.options["rate"]:
            parts.append(np.full((n, 1), float(dt) / s.time))
        u = np.concatenate(parts, axis=1)
        gi = u @ p["Wi"] + p["b"]
        gh = h_old @ p["Wh"]
        rz = _sigmoid(gi[:, : 2 * h] + gh[:, : 2 * h])
        r, z = rz[:, :h], rz[:, h:]
        hn = gh[:, 2 * h :] + p["bn"]
        cand = np.tanh(gi[:, 2 * h :] + r * hn)
        state = cand + z * (h_old - cand)
        y = state @ p["V"] + seen @ p["Q"].T + p["c"]
        # tangents: the inputs of the strain at the end, then of the increment (of H in ln C)
        du = np.zeros((self.n_inputs, 6))
        du[:6] = np.eye(6) / s.strain
        du[6:12] = np.eye(6) / s.increment
        dgi = (p["Wi"].T @ du)[None]  # (1, 3 h, 6)
        dr = (r * (1.0 - r))[:, :, None] * dgi[:, :h]
        dz = (z * (1.0 - z))[:, :, None] * dgi[:, h : 2 * h]
        dcand = (1.0 - cand * cand)[:, :, None] * (dgi[:, 2 * h :] + dr * hn[:, :, None])
        dstate = (1.0 - z)[:, :, None] * dcand + dz * (h_old - cand)[:, :, None]
        dy = np.einsum("nhd,hk->nkd", dstate, p["V"]) + p["Q"] / s.strain
        if self.logarithmic:  # S = J^T T, dS/dE = J^T (dT/dH) J + T : d2H/dE2
            y, dy = _pullback(np.ascontiguousarray(x), y, np.ascontiguousarray(s.strain * dy),
                              s.strain, False)  # fmt: skip
            return s.stress * y, s.stress / s.strain * dy, state
        return s.stress * y, s.stress * dy, state
