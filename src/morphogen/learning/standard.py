"""A learned generalized standard material (Halphen & Nguyen 1975) in small and finite
strains: a learned free energy ``psi(eps, alpha)`` of a few internal variables ``alpha``,
convex, and a learned dissipation potential ``phi(alpha')``, convex, nonnegative and zero
at zero; the state of a step is the minimizer of the incremental potential (Ortiz &
Stainier 1999),

    alpha_{n+1} = argmin_alpha  psi(eps_{n+1}, alpha) + dt phi((alpha - alpha_n) / dt),

a convex problem solved by Newton's method on its gradient (a backtracking line search on
the potential), and ``sigma = d psi / d eps``. By construction the dissipation of a step,
``D dt = -d psi/d alpha . (alpha_{n+1} - alpha_n)``, is nonnegative (``phi`` convex and
least at zero), the consistent tangent is symmetric and positive definite (the Schur
complement of the Hessian of the incremental potential, by the implicit function theorem)
and the law is stable whatever its parameters.

In the normalized variables of the models (:class:`~morphogen.learning.data.Scales`; ``x
= eps / s_eps``, ``alpha`` dimensionless, ``z = (x, alpha)``):

* the free energy ``psi = s_sigma s_eps (W(z) - W(0) - z . dW/dz(0))``, ``W(z) = z . A z / 2
  + sum_i w_i g(u_i . z + b_i)``: an input-convex network of one hidden layer (Amos, Xu &
  Kolter 2017; ``g`` the softplus, ``w = softplus(.) >= 0``) and a quadratic term ``A = L
  L^T``, convex in ``(eps, alpha)`` jointly, zero with its gradient at the natural state
  (no stress, no thermodynamic force);
* the dissipation of a step ``Phi(dalpha) = phi_1(dalpha) + dalpha . V dalpha / (2 tau)``
  (``tau = dt / s_t``): with ``plastic``, a rate-independent part ``phi_1``, positively
  homogeneous of degree 1 beyond the smoothing ``delta`` and taken on the increment (the
  time step drops out), ``phi_1(v) = sum_k S(t_k(v)) - S(0) - c_k . v / 2``, ``t_k = sum_j
  P_kj n_j(v) + c_k . v``, ``n_j = sqrt(|B_j v|^2 + delta^2) - delta`` (smoothed norms),
  ``S(t) = (t + sqrt(t^2 + delta^2)) / 2`` (a smoothed ramp), ``P = softplus(.) >= 0``:
  convex, zero with its gradient at zero, hence nonnegative, its elastic domain the
  support set of ``phi_1`` sharpened at the scale ``delta`` of the increments; with
  ``rate``, a viscous part ``V = M M^T`` (a quadratic potential of the rate).

The training fits the stresses along whole paths from the natural state: the gradient of
the objective goes through the minimization of each step by implicit differentiation (the
adjoint of a step solves ``K mu = lambda`` with ``K`` the Hessian of the incremental
potential in ``alpha``) and back through the steps (``lambda_n = d2 Phi mu_n``, the next
state depending on the previous one through the dissipation only); the gradients of the
parameters are those of the derivatives of ``W`` and ``Phi`` along the adjoint directions,
written out and compiled by Numba (no automatic differentiation).

In finite strains (data of the Green-Lagrange strain ``E`` and the second Piola-Kirchhoff
stress ``S``) the free energy is of an objective Lagrangian strain, ``psi(H, alpha)``:

* ``strain="logarithmic"`` (the default): ``H = 1/2 ln C`` (``C = I + 2 E``), the space of
  the Hencky points of morphoGen (Miehe, Apel & Lambrecht 2002): ``T = d psi/dH`` and ``S =
  T : dH/dE = J^T T`` (``J = dH/dE``, 6 x 6, from the divided differences of ``1/2 ln``
  in the eigenbasis of ``C``, Daleckii-Krein; :mod:`morphogen.learning.logarithmic`, the
  kernels of every learned law of ``ln C``), the tangent ``dS/dE = J^T (dT/dH) J + T :
  d2H/dE2``; internal variables additive in ``H`` (an isochoric flow of the logarithmic
  strain) represent isotropic multiplicative plasticity;
* ``strain="green-lagrange"``: ``H = E``, ``S = d psi/dE``, the tangent the Schur
  complement itself (positive definite in ``E``).

Either way ``S : dE = T : dH``, so that the dissipation of a step is the same
``-d psi/d alpha . (alpha_{n+1} - alpha_n) >= 0`` and ``S`` derives from ``psi(E)``
(``J`` does not depend on the parameters: the training sees the gaps of ``S`` through
``J``); a superposed rotation leaves ``C``, hence ``S`` and the state, unchanged.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from morphogen.learning.data import Dataset
from morphogen.learning.logarithmic import (
    _conjugate,
    _geometric,
    _logarithmic,
    _measures,
    _symmetrize,
)
from morphogen.learning.networks import (
    STRAINS,
    Model,
    _inverse_softplus,
    _linear_fit,
    _sigmoid,
    _softplus,
)

#: Newton's method of a step: the tolerance of the gradient of the incremental potential
#: (normalized forces), the iterations, the factor of Armijo's condition
NEWTON_TOL = 1e-10
NEWTON_MAX = 60
_ARMIJO = 1e-4
#: The parameters of a model: of the free energy, of the rate-independent and of the
#: viscous dissipation
_PSI = ("U", "b", "w", "L")
_PHI = ("B", "P", "C")
_VISCOUS = ("M",)


@njit(cache=True)
def _soft(t):
    """The softplus of a scalar, without overflow."""
    if t > 30.0:
        return t
    if t < -30.0:
        return math.exp(t)
    return math.log1p(math.exp(t))


@njit(cache=True)
def _sig(t):
    return 0.5 * (1.0 + math.tanh(0.5 * t))


@njit(cache=True)
def _energy(z, U, b, w, A, g0, grad, hess, order):
    """``W(z) - z . g0`` (``g0 = dW/dz(0)``) and, with ``order`` 1 or 2, its gradient and
    Hessian (written into ``grad`` and ``hess``)."""
    d, h = z.size, b.size
    value = 0.0
    for i in range(d):
        s = 0.0
        for j in range(d):
            s += A[i, j] * z[j]
        value += (0.5 * s - g0[i]) * z[i]
        if order > 0:
            grad[i] = s - g0[i]
    if order > 1:
        for i in range(d):
            for j in range(d):
                hess[i, j] = A[i, j]
    for k in range(h):
        a = b[k]
        for j in range(d):
            a += U[k, j] * z[j]
        value += w[k] * _soft(a)
        if order > 0:
            sg = _sig(a)
            c = w[k] * sg
            for j in range(d):
                grad[j] += c * U[k, j]
            if order > 1:
                c2 = c * (1.0 - sg)
                for i in range(d):
                    ci = c2 * U[k, i]
                    for j in range(d):
                        hess[i, j] += ci * U[k, j]
    return value


@njit(cache=True)
def _dissipation(v, B, G, P, C, V, tau, delta, plastic, viscous, grad, hess, order):
    """The dissipation of a step ``Phi(v)`` of the increment ``v`` and, with ``order`` 1
    or 2, its gradient and Hessian (``G_j = B_j^T B_j``)."""
    m = v.size
    value = 0.0
    if order > 0:
        grad[:] = 0.0
    if order > 1:
        hess[:, :] = 0.0
    if viscous:
        for i in range(m):
            s = 0.0
            for j in range(m):
                s += V[i, j] * v[j]
            value += 0.5 * s * v[i] / tau
            if order > 0:
                grad[i] += s / tau
            if order > 1:
                for j in range(m):
                    hess[i, j] += V[i, j] / tau
    if not plastic:
        return value
    n_norms, n_ridges = B.shape[0], C.shape[0]
    norm = np.empty(n_norms)
    radius = np.empty(n_norms)
    dn = np.empty((n_norms, m))  # the gradients of the norms
    for j in range(n_norms):
        ss = 0.0
        for i in range(m):
            s = 0.0
            for l in range(m):
                s += B[j, i, l] * v[l]
            ss += s * s
        radius[j] = math.sqrt(ss + delta * delta)
        norm[j] = ss / (radius[j] + delta)  # sqrt(ss + delta^2) - delta, without cancellation
        if order > 0:
            for l in range(m):
                s = 0.0
                for i in range(m):
                    s += G[j, l, i] * v[i]
                dn[j, l] = s / radius[j]
    f = np.zeros(n_norms)
    dt = np.empty(m)
    for k in range(n_ridges):
        t = 0.0
        for j in range(n_norms):
            t += P[k, j] * norm[j]
        lin = 0.0
        for l in range(m):
            t += C[k, l] * v[l]
            lin += C[k, l] * v[l]
        q = math.sqrt(t * t + delta * delta)
        value += 0.5 * (t + q) - 0.5 * delta - 0.5 * lin
        if order == 0:
            continue
        s1 = 0.5 * (1.0 + t / q)
        for l in range(m):
            s = C[k, l]
            for j in range(n_norms):
                s += P[k, j] * dn[j, l]
            dt[l] = s
            grad[l] += s1 * s - 0.5 * C[k, l]
        if order > 1:
            s2 = 0.5 * delta * delta / (q * q * q)
            for l in range(m):
                for i in range(m):
                    hess[l, i] += s2 * dt[l] * dt[i]
            for j in range(n_norms):
                f[j] += s1 * P[k, j]
    if order > 1:  # sum_j f_j d2 n_j, d2 n_j = (B_j^T B_j - dn_j dn_j^T) / r_j
        for j in range(n_norms):
            for l in range(m):
                for i in range(m):
                    hess[l, i] += f[j] * (G[j, l, i] - dn[j, l] * dn[j, i]) / radius[j]
    return value


@njit(cache=True)
def _solve(K, rhs, out):
    """``K^-1 rhs`` of a symmetric positive definite matrix by Cholesky's factorization
    (a small shift of the diagonal if round-off spoils it)."""
    m = rhs.size
    low = np.zeros((m, m))
    shift = 0.0
    for _ in range(8):
        ok = True
        for j in range(m):
            s = K[j, j] + shift
            for k in range(j):
                s -= low[j, k] * low[j, k]
            if not s > 0.0:
                ok = False
                break
            low[j, j] = math.sqrt(s)
            for i in range(j + 1, m):
                s = K[i, j]
                for k in range(j):
                    s -= low[i, k] * low[j, k]
                low[i, j] = s / low[j, j]
        if ok:
            break
        trace = 0.0
        for j in range(m):
            trace += abs(K[j, j])
        shift = max(10.0 * shift, 1e-14 * trace / m + 1e-300)
    for i in range(m):
        s = rhs[i]
        for k in range(i):
            s -= low[i, k] * out[k]
        out[i] = s / low[i, i]
    for i in range(m - 1, -1, -1):
        s = out[i]
        for k in range(i + 1, m):
            s -= low[k, i] * out[k]
        out[i] = s / low[i, i]


@njit(cache=True)
def _update(x, a0, tau, U, b, w, A, g0, B, G, P, C, V, delta, plastic, viscous,
            z, gz, hz, gp, hp, kmat):  # fmt: skip
    """The state of a step: the minimizer ``alpha`` of the incremental potential at the
    strain ``x`` from the state ``a0`` (Newton's method from ``a0``, a backtracking line
    search). On return ``z = (x, alpha)``, the gradient ``gz`` and Hessian ``hz`` of the
    free energy at ``z``, those of the dissipation at ``alpha - a0`` (``gp``, ``hp``) and
    ``kmat``, the Hessian of the potential in ``alpha``; returns the iterations, or minus
    them without convergence."""
    m = a0.size
    d = 6 + m
    for i in range(6):
        z[i] = x[i]
    for i in range(m):
        z[6 + i] = a0[i]
    inc = np.zeros(m)
    res = np.empty(m)
    step = np.empty(m)
    trial = np.empty(d)
    for it in range(NEWTON_MAX + 1):
        value = _energy(z, U, b, w, A, g0, gz, hz, 2)
        for i in range(m):
            inc[i] = z[6 + i] - a0[i]
        value += _dissipation(inc, B, G, P, C, V, tau, delta, plastic, viscous, gp, hp, 2)
        size = 0.0
        for i in range(m):
            res[i] = gz[6 + i] + gp[i]
            size = max(size, abs(res[i]))
            for j in range(m):
                kmat[i, j] = hz[6 + i, 6 + j] + hp[i, j]
        if size <= NEWTON_TOL:
            return it
        if it == NEWTON_MAX:
            break
        _solve(kmat, res, step)
        slope = 0.0
        for i in range(m):
            step[i] = -step[i]
            slope += res[i] * step[i]
        t = 1.0
        # far from the solution: Armijo's backtracking on the potential, while the decrease
        # it predicts is above the round-off of the potential (below, the test would fail
        # whatever the step, and the full Newton step converges)
        if size > 1e-6 and -slope > 1e-12 * (1.0 + abs(value)):
            for i in range(6):
                trial[i] = x[i]
            for _ in range(40):
                for i in range(m):
                    trial[6 + i] = z[6 + i] + t * step[i]
                    inc[i] = trial[6 + i] - a0[i]
                got = _energy(trial, U, b, w, A, g0, gz, hz, 0)
                got += _dissipation(inc, B, G, P, C, V, tau, delta, plastic, viscous, gp, hp, 0)
                if got <= value + _ARMIJO * t * slope:
                    break
                t *= 0.5
        for i in range(m):
            z[6 + i] += t * step[i]
    return -NEWTON_MAX


@njit(cache=True)
def _schur(hz, kmat, out):
    """The consistent tangent ``d2W/dx2 - d2W/dx dalpha K^-1 d2W/dalpha dx`` (6 x 6,
    symmetrized: symmetric to round-off)."""
    m = kmat.shape[0]
    col = np.empty(m)
    sol = np.empty(m)
    for c in range(6):
        for i in range(m):
            col[i] = hz[6 + i, c]
        _solve(kmat, col, sol)
        for r in range(6):
            s = hz[r, c]
            for i in range(m):
                s -= hz[r, 6 + i] * sol[i]
            out[r, c] = s
    for r in range(6):
        for c in range(r):
            s = 0.5 * (out[r, c] + out[c, r])
            out[r, c] = s
            out[c, r] = s


@njit(cache=True)
def _stress(gz, hz, kmat, log, jac, q, mu, dq, scale, y, tangent):
    """The normalized stress ``y`` and tangent of a step from the derivatives of the free
    energy at its state: the stress and the Schur complement in the strain of the free
    energy, or (``log``) those conjugate to the Green-Lagrange strain of the stress ``T``
    conjugate to the logarithmic strain, ``J^T T`` and ``J^T K J + T : d2H/dE2``."""
    if not log:
        for i in range(6):
            y[i] = gz[i]
        _schur(hz, kmat, tangent)
        return
    k6 = np.empty((6, 6))
    _schur(hz, kmat, k6)
    _conjugate(jac, gz[:6], k6, y, tangent)
    _geometric(gz[:6], q, mu, dq, scale, tangent)
    _symmetrize(tangent)


@njit(cache=True)
def _points(x, a0, tau, U, b, w, A, g0, B, G, P, C, V, delta, plastic, viscous, log, scale):
    """A step of points (n x 6 strains, n x m states, n time steps): their normalized
    stresses, tangents and states at its end, and the iterations of each (``log``: the
    strains normalized Green-Lagrange strains, the free energy of their logarithmic
    strains; NaN stresses where ``C`` is not positive definite)."""
    n, m = a0.shape
    d = 6 + m
    y, tangent, alpha = np.empty((n, 6)), np.empty((n, 6, 6)), np.empty((n, m))
    iterations = np.empty(n, dtype=np.int64)
    z, gz, hz = np.empty(d), np.empty(d), np.empty((d, d))
    gp, hp, kmat = np.empty(m), np.empty((m, m)), np.empty((m, m))
    h, jac, q, mu = np.empty(6), np.empty((6, 6)), np.empty((3, 3)), np.empty(3)
    dq = np.empty((6, 3, 3))
    for p in range(n):
        if log:
            if _logarithmic(x[p], scale, h, jac, q, mu, dq):
                y[p] = np.nan
                tangent[p] = np.nan
                alpha[p] = a0[p]
                iterations[p] = -1
                continue
        else:
            h[:] = x[p]
        iterations[p] = _update(h, a0[p], tau[p], U, b, w, A, g0, B, G, P, C, V, delta,
                                plastic, viscous, z, gz, hz, gp, hp, kmat)  # fmt: skip
        for i in range(m):
            alpha[p, i] = z[6 + i]
        _stress(gz, hz, kmat, log, jac, q, mu, dq, scale, y[p], tangent[p])
    return y, tangent, alpha, iterations


@njit(cache=True)
def _paths(x, tau, U, b, w, A, g0, w0, B, G, P, C, V, delta, plastic, viscous, log, scale):
    """Paths from the natural state (paths x steps x 6 strains, paths x steps time steps):
    the normalized stresses, states, free energies (``W(z) - W(0) - z . g0``,
    ``w0 = W(0)``), dissipations of the steps (``-dpsi/dalpha . dalpha``), tangents and
    the iterations of each step (``log``: see :func:`_points`)."""
    n, steps = tau.shape
    m = V.shape[0]
    d = 6 + m
    y, alpha = np.empty((n, steps, 6)), np.empty((n, steps, m))
    energy, dissipation = np.empty((n, steps)), np.empty((n, steps))
    tangent = np.empty((n, steps, 6, 6))
    iterations = np.empty((n, steps), dtype=np.int64)
    z, gz, hz = np.empty(d), np.empty(d), np.empty((d, d))
    gp, hp, kmat = np.empty(m), np.empty((m, m)), np.empty((m, m))
    h, jac, q, mu = np.empty(6), np.empty((6, 6)), np.empty((3, 3)), np.empty(3)
    dq = np.empty((6, 3, 3))
    for p in range(n):
        state = np.zeros(m)
        for t in range(steps):
            if log:
                if _logarithmic(x[p, t], scale, h, jac, q, mu, dq):  # the rest of the path
                    y[p, t:] = np.nan
                    alpha[p, t:] = np.nan
                    energy[p, t:] = np.nan
                    dissipation[p, t:] = np.nan
                    tangent[p, t:] = np.nan
                    iterations[p, t:] = -1
                    break
            else:
                h[:] = x[p, t]
            iterations[p, t] = _update(h, state, tau[p, t], U, b, w, A, g0, B, G, P, C, V,
                                       delta, plastic, viscous, z, gz, hz, gp, hp,
                                       kmat)  # fmt: skip
            spent = 0.0
            for i in range(m):
                spent -= gz[6 + i] * (z[6 + i] - state[i])
                state[i] = z[6 + i]
                alpha[p, t, i] = z[6 + i]
            dissipation[p, t] = spent
            _stress(gz, hz, kmat, log, jac, q, mu, dq, scale, y[p, t], tangent[p, t])
            energy[p, t] = _energy(z, U, b, w, A, g0, gz, hz, 0) - w0
    return y, alpha, energy, dissipation, tangent, iterations


@njit(cache=True)
def _energy_params(z, u, U, b, w, gU, gb, gw, gA):
    """Adds the gradients with respect to ``U``, ``b``, ``w`` and ``A`` of ``u . dW/dz
    (z)``, the derivative of the free energy along ``u``."""
    d, h = z.size, b.size
    for i in range(d):
        for j in range(d):
            gA[i, j] += u[i] * z[j]
    for k in range(h):
        a = b[k]
        uu = 0.0
        for j in range(d):
            a += U[k, j] * z[j]
            uu += U[k, j] * u[j]
        sg = _sig(a)
        ds = sg * (1.0 - sg)
        gw[k] += sg * uu
        gb[k] += w[k] * ds * uu
        for j in range(d):
            gU[k, j] += w[k] * (ds * uu * z[j] + sg * u[j])


@njit(cache=True)
def _dissipation_params(v, mu, B, P, C, tau, delta, plastic, viscous, gB, gP, gC, gV):
    """Adds the gradients with respect to ``B``, ``P``, ``C`` and ``V`` of ``mu . dPhi/dv
    (v)``, the derivative of the dissipation of the increment ``v`` along ``mu``."""
    m = v.size
    if viscous:
        for i in range(m):
            for j in range(m):
                gV[i, j] += mu[i] * v[j] / tau
    if not plastic:
        return
    n_norms, n_ridges = B.shape[0], C.shape[0]
    bv, bm = np.empty((n_norms, m)), np.empty((n_norms, m))
    norm, radius, along = np.empty(n_norms), np.empty(n_norms), np.empty(n_norms)
    for j in range(n_norms):
        ss = 0.0
        cross = 0.0
        for i in range(m):
            s = 0.0
            sm = 0.0
            for l in range(m):
                s += B[j, i, l] * v[l]
                sm += B[j, i, l] * mu[l]
            bv[j, i] = s
            bm[j, i] = sm
            ss += s * s
            cross += s * sm
        radius[j] = math.sqrt(ss + delta * delta)
        norm[j] = ss / (radius[j] + delta)
        along[j] = cross / radius[j]  # mu . dn_j
    e, f = np.zeros(n_norms), np.zeros(n_norms)
    for k in range(n_ridges):
        t = 0.0
        g = 0.0
        for j in range(n_norms):
            t += P[k, j] * norm[j]
            g += P[k, j] * along[j]
        for l in range(m):
            t += C[k, l] * v[l]
            g += C[k, l] * mu[l]
        q = math.sqrt(t * t + delta * delta)
        s1 = 0.5 * (1.0 + t / q)
        s2 = 0.5 * delta * delta / (q * q * q)
        for l in range(m):
            gC[k, l] += s2 * g * v[l] + (s1 - 0.5) * mu[l]
        for j in range(n_norms):
            gP[k, j] += s2 * g * norm[j] + s1 * along[j]
            e[j] += s2 * g * P[k, j]
            f[j] += s1 * P[k, j]
    for j in range(n_norms):
        r = radius[j]
        cross = along[j] * r
        for i in range(m):
            for l in range(m):
                gB[j, i, l] += (e[j] * bv[j, i] * v[l] / r
                                + f[j] * ((bv[j, i] * mu[l] + bm[j, i] * v[l]) / r
                                          - cross * bv[j, i] * v[l] / (r * r * r)))  # fmt: skip


@njit(cache=True)
def _gradients(x, tau, y, U, b, w, A, g0, B, G, P, C, V, delta, plastic, viscous, log, scale,
               grad):  # fmt: skip
    """The mean square of the gaps of the normalized stresses of paths from the natural
    state and, with ``grad``, its gradients with respect to ``U``, ``b``, ``w``, ``A``,
    ``B``, ``P``, ``C`` and ``V`` (the values of the parameters, before their
    reparametrizations): the minimization of each step differentiated implicitly, the
    adjoints carried back through the steps. With ``log`` the strains are normalized
    Green-Lagrange strains and the stresses ``S = J^T T``: the gaps of ``S`` reach ``T`` as
    ``J r`` (``J`` of the data). Returns the loss, the gradients and the iterations of
    Newton's method that did not converge."""
    n, steps = tau.shape
    m = V.shape[0]
    d = 6 + m
    total = steps * n * 6
    gU, gb, gw, gA = np.zeros_like(U), np.zeros_like(b), np.zeros_like(w), np.zeros_like(A)
    gB, gP, gC, gV = np.zeros_like(B), np.zeros_like(P), np.zeros_like(C), np.zeros_like(V)
    zs, hs = np.empty((steps, d)), np.empty((steps, d, d))
    ks, hps = np.empty((steps, m, m)), np.empty((steps, m, m))
    incs, gaps = np.empty((steps, m)), np.empty((steps, 6))
    z, gz, hz = np.empty(d), np.empty(d), np.empty((d, d))
    gp, hp, kmat = np.empty(m), np.empty((m, m)), np.empty((m, m))
    lam, mu, nxt, u = np.empty(m), np.empty(m), np.zeros(m), np.empty(d)
    h, jac, q, eig = np.empty(6), np.empty((6, 6)), np.empty((3, 3)), np.empty(3)
    dq, r = np.empty((6, 3, 3)), np.empty(6)
    summed = np.zeros(d)
    loss = 0.0
    failed = 0
    for p in range(n):
        state = np.zeros(m)
        for t in range(steps):
            if log:
                if _logarithmic(x[p, t], scale, h, jac, q, eig, dq):
                    return math.nan, gU, gb, gw, gA, gB, gP, gC, gV, failed + 1
            else:
                h[:] = x[p, t]
            it = _update(h, state, tau[p, t], U, b, w, A, g0, B, G, P, C, V, delta,
                         plastic, viscous, z, gz, hz, gp, hp, kmat)  # fmt: skip
            if it < 0:
                failed += 1
            for i in range(6):
                s = gz[i]
                if log:  # S = J^T T
                    s = 0.0
                    for k in range(6):
                        s += jac[k, i] * gz[k]
                r[i] = s - y[p, t, i]
                loss += r[i] * r[i]
            for i in range(6):
                if log:  # d loss / dT = J d loss / dS
                    s = 0.0
                    for k in range(6):
                        s += jac[i, k] * r[k]
                    gaps[t, i] = 2.0 * s / total
                else:
                    gaps[t, i] = 2.0 * r[i] / total
            for i in range(m):
                incs[t, i] = z[6 + i] - state[i]
                state[i] = z[6 + i]
            if grad:
                zs[t] = z
                hs[t] = hz
                ks[t] = kmat
                hps[t] = hp
        if not grad:
            continue
        nxt[:] = 0.0
        for t in range(steps - 1, -1, -1):
            # the adjoint of the state of the step: the next steps and the stress of this one
            for i in range(m):
                s = nxt[i]
                for c in range(6):
                    s += hs[t, 6 + i, c] * gaps[t, c]
                lam[i] = s
            _solve(ks[t], lam, mu)
            for c in range(6):
                u[c] = gaps[t, c]
            for i in range(m):
                u[6 + i] = -mu[i]
                mu[i] = -mu[i]
            for i in range(d):
                summed[i] += u[i]
            _energy_params(zs[t], u, U, b, w, gU, gb, gw, gA)
            _dissipation_params(incs[t], mu, B, P, C, tau[p, t], delta, plastic, viscous,
                                gB, gP, gC, gV)  # fmt: skip
            for i in range(m):  # the previous state: through the dissipation only
                s = 0.0
                for j in range(m):
                    s -= hps[t, i, j] * mu[j]
                nxt[i] = s
    if grad:  # the gradient at the natural state g0 = dW/dz(0), subtracted from every one
        for i in range(d):
            summed[i] = -summed[i]
        _energy_params(np.zeros(d), summed, U, b, w, gU, gb, gw, gA)
    return loss / total, gU, gb, gw, gA, gB, gP, gC, gV, failed


class Standard(Model):
    """A learned generalized standard material (see the module): ``internal``, the number
    of internal variables; ``hidden``, the softplus ridges of its free energy; ``rate``, a
    viscous dissipation (the time step matters); ``plastic``, a rate-independent one (by
    default; ``rate`` alone gives viscoelasticity); ``smoothing``, the scale ``delta`` of
    the normalized increments below which the rate-independent dissipation is smoothed;
    ``seed``, that of its initial parameters. Its dissipation has ``internal`` smoothed
    norms and ``internal`` ridges. ``strain``: that of its free energy in finite strains
    (:data:`~morphogen.learning.networks.STRAINS`), the logarithmic strain (the default)
    or the Green-Lagrange strain; in small strains the free energy is of the small strain,
    which both linearize to."""

    kind = "standard"
    recurrent = True

    def __init__(self, hidden: int = 32, *, internal: int = 8, rate: bool = False,
                 plastic: bool = True, smoothing: float = 0.01, strain: str = "logarithmic",
                 seed: int = 0) -> None:  # fmt: skip
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1
               for v in (hidden, internal)):  # fmt: skip
            raise ValueError(
                "hidden, internal: the ridges of the energy and the internal variables, "
                f"positive integers, got {hidden!r}, {internal!r}"
            )
        if not (rate or plastic):
            raise ValueError("standard model: a dissipation, rate-independent (plastic) or "
                             "viscous (rate), or both")  # fmt: skip
        if not 0 < smoothing < math.inf:
            raise ValueError("smoothing: positive and finite")
        if strain not in STRAINS:
            raise ValueError(f"strain: that of the free energy, one of {STRAINS}, got {strain!r}")
        super().__init__(hidden=int(hidden), internal=int(internal), rate=bool(rate),
                         plastic=bool(plastic), smoothing=float(smoothing), strain=str(strain),
                         seed=int(seed))  # fmt: skip

    @property
    def n_state(self) -> int:
        return self.options["internal"]

    def prepare(self, data: Dataset) -> tuple:
        """The normalized strains at the ends of the steps (paths x steps x 6), the
        normalized time steps (paths x steps) and the normalized stresses."""
        s = self._scales()
        return (data.strain[:, 1:] / s.strain, np.diff(data.times, axis=1) / s.time,
                data.stress[:, 1:] / s.stress)  # fmt: skip

    def _initial(self, rng, x, tau, y) -> dict:
        h, m = self.options["hidden"], self.options["internal"]
        d = 6 + m
        # in ln C, the fit of T = J^-T S to the logarithmic strain
        strain, stress = self._conjugates(x.reshape(-1, 6), y.reshape(-1, 6))
        # the quadratic term: the symmetric positive part of the linear fit for the strain,
        # the internal variables coupled to it at random
        fit = _linear_fit(strain, stress)
        values, vectors = np.linalg.eigh(0.5 * (fit + fit.T))
        values = np.maximum(values, 1e-3 * max(values.max(), 1e-12))
        low = np.zeros((d, d))
        low[:6, :6] = vectors * np.sqrt(values)
        low[6:, :6] = rng.normal(0.0, 0.5 * math.sqrt(values.mean() / 6.0), (m, 6))
        low[6:, 6:] = np.eye(m)
        p = {"U": rng.normal(0.0, 1.0 / math.sqrt(d), (h, d)), "b": np.zeros(h),
             "w": _inverse_softplus(0.1 / h) + rng.normal(0.0, 0.3, h), "L": low}  # fmt: skip
        if self.options["plastic"]:
            p["B"] = rng.normal(0.0, 1.0 / math.sqrt(m), (m, m, m))
            p["P"] = _inverse_softplus(0.3 / m) + rng.normal(0.0, 0.3, (m, m))
            p["C"] = rng.normal(0.0, 0.3 / math.sqrt(m), (m, m))
        if self.options["rate"]:  # relaxation times from one mean step to a whole path
            times = np.geomspace(1.0, max(float(x.shape[1]), 2.0), m)
            stiff = np.diag(low[6:] @ low[6:].T)
            p["M"] = np.diag(np.sqrt(times * stiff)) + rng.normal(0.0, 0.03, (m, m))
        return p

    def _values(self) -> tuple:
        """The arrays of the kernels: the values of the parameters (``w = softplus``, ``A =
        L L^T``, ...), ``dW/dz(0)`` and ``W(0)``; then whether the free energy is of the
        logarithmic strain and the scale of the strains."""
        p, m = self.params, self.options["internal"]
        w, low = _softplus(p["w"]), p["L"]
        amat = low @ low.T
        g0 = (w * _sigmoid(p["b"])) @ p["U"]
        w0 = float(np.sum(w * _softplus(p["b"])))
        if self.options["plastic"]:
            b, c, pm = p["B"], p["C"], _softplus(p["P"])
            g = np.einsum("jil,jik->jlk", b, b)
        else:  # placeholders the kernels skip
            b, g, c, pm = (np.zeros((1, m, m)), np.zeros((1, m, m)), np.zeros((1, m)),
                           np.zeros((1, 1)))  # fmt: skip
        v = p["M"] @ p["M"].T if self.options["rate"] else np.zeros((m, m))
        delta = self.options["smoothing"]
        return (p["U"], p["b"], w, amat, g0, w0, b, g, pm, c, v, delta,
                self.options["plastic"], self.options["rate"], self.logarithmic,
                self._scales().strain)  # fmt: skip

    def _taus(self, times, shape) -> np.ndarray:
        """The normalized time steps of paths (paths x steps), ones without a viscous
        dissipation (the time step then drops out)."""
        if not self.options["rate"]:
            return np.ones(shape)
        if times is None:
            raise ValueError("standard model of a rate: give the times of the paths")
        dt = np.diff(np.atleast_2d(np.asarray(times, dtype=float)), axis=1)
        return np.ascontiguousarray(np.broadcast_to(dt, shape) / self._scales().time)

    def objective(self, arrays, grad=True):
        x, tau, y = (np.ascontiguousarray(a, dtype=float) for a in arrays)
        u, b, w, amat, g0, _, bmat, g, pm, c, v, delta, plastic, viscous, log, scale = (
            self._values())  # fmt: skip
        if not self.options["rate"]:
            tau = np.ones_like(tau)
        loss, *grads, _ = _gradients(x, tau, y, u, b, w, amat, g0, bmat, g, pm, c, v, delta,
                                     plastic, viscous, log, scale, grad)  # fmt: skip
        if not grad:
            return float(loss), None
        gu, gb, gw, ga, gbm, gpm, gc, gv = grads
        p = self.params
        out = {"U": gu, "b": gb, "w": gw * _sigmoid(p["w"]), "L": (ga + ga.T) @ p["L"]}
        if plastic:
            out.update(B=gbm, P=gpm * _sigmoid(p["P"]), C=gc)
        if viscous:
            out["M"] = (gv + gv.T) @ p["M"]
        return float(loss), out

    def history(self, strain, times=None) -> dict:
        """The evolution of paths from the natural state (paths x points x 6, or one path):
        their ``stress``, internal variables (``internal``), free energy (``energy``, per
        unit volume) and the energy dissipated by each step (``dissipation``, ``-dpsi/
        dalpha . dalpha``, nonnegative to the round-off of Newton's method), the first
        point the natural state; the consistent ``tangent`` of each step (its first
        point that of the natural state, not used); the ``iterations`` of Newton's method
        of each step (negative where it stopped without converging). In finite strains:
        the Green-Lagrange strains, the second Piola-Kirchhoff stresses and their tangents
        ``dS/dE``."""
        s = self._scales()
        strain = np.asarray(strain, dtype=float)
        single = strain.ndim == 2
        strain = strain[None] if single else strain
        if times is not None and single:
            times = np.asarray(times, dtype=float)[None]
        x = np.ascontiguousarray(strain[:, 1:] / s.strain)
        tau = self._taus(times, x.shape[:2])
        u, b, w, amat, g0, w0, bmat, g, pm, c, v, delta, plastic, viscous, log, scale = (
            self._values())  # fmt: skip
        y, alpha, energy, spent, tangent, its = _paths(x, tau, u, b, w, amat, g0, w0, bmat, g,
                                                      pm, c, v, delta, plastic, viscous, log,
                                                      scale)  # fmt: skip
        n = strain.shape[0]

        def first(a, value=0.0):  # the natural state before the steps
            return np.concatenate([np.full((n, 1, *a.shape[2:]), value), a], axis=1)

        work = s.stress * s.strain
        out = {"stress": first(s.stress * y), "internal": first(alpha),
               "energy": first(work * energy), "dissipation": first(work * spent),
               "tangent": first(s.stress / s.strain * tangent, np.nan),
               "iterations": first(its, 0)}  # fmt: skip
        return {k: a[0] for k, a in out.items()} if single else out

    def predict(self, strain, times=None) -> np.ndarray:
        return self.history(strain, times)["stress"]

    def tangent(self, strain):
        raise ValueError("a standard model has the tangents of its steps (step)")

    def energy(self, strain, internal) -> np.ndarray:
        """The free energy (per unit volume) at strains (..., 6; Green-Lagrange strains in
        finite strains) and internal variables (..., internal)."""
        s = self._scales()
        x = np.asarray(strain, dtype=float) / s.strain
        if self.logarithmic:
            x = _measures(np.ascontiguousarray(x.reshape(-1, 6)), s.strain)[0].reshape(x.shape)
        alpha = np.asarray(internal, dtype=float)
        lead = np.broadcast_shapes(x.shape[:-1], alpha.shape[:-1])
        z = np.concatenate([np.broadcast_to(x, (*lead, 6)),
                            np.broadcast_to(alpha, (*lead, alpha.shape[-1]))], axis=-1)  # fmt: skip
        u, b, w, amat, g0, w0, *_ = self._values()
        flat = z.reshape(-1, z.shape[-1])
        values = (0.5 * np.einsum("ni,ij,nj->n", flat, amat, flat) - flat @ g0
                  + _softplus(flat @ u.T + b) @ w - w0)  # fmt: skip
        return s.stress * s.strain * values.reshape(z.shape[:-1])

    def step(self, e_new, e_old, h_old, dt):
        s = self._scales()
        e_new = np.asarray(e_new, dtype=float)
        n = e_new.shape[0]
        x = np.ascontiguousarray(e_new / s.strain)
        a0 = np.ascontiguousarray(np.asarray(h_old, dtype=float).reshape(n, self.n_state))
        # a step of no time: the viscous variables frozen (an infinite viscous resistance)
        tau = np.full(n, max(float(dt) / s.time, 1e-12) if self.options["rate"] else 1.0)
        u, b, w, amat, g0, _, bmat, g, pm, c, v, delta, plastic, viscous, log, scale = (
            self._values())  # fmt: skip
        y, tangent, alpha, _ = _points(x, a0, tau, u, b, w, amat, g0, bmat, g, pm, c, v, delta,
                                       plastic, viscous, log, scale)  # fmt: skip
        return s.stress * y, s.stress / s.strain * tangent, alpha
