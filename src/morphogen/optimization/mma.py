"""Method of moving asymptotes (Svanberg 1987, 2002).

Each iteration replaces the problem

    minimize f0(x)  subject to  fi(x) <= 0 (i = 1..m),  xmin <= x <= xmax

by a convex separable approximation built on moving asymptotes ``low < x < upp`` and
solves it by a primal-dual interior-point method, as in Svanberg's notes "MMA and GCMMA,
two methods for nonlinear optimization" (2007), with artificial variables ``y_i`` of
cost ``c_i y_i + d_i y_i^2 / 2`` that make every subproblem feasible.
"""

from __future__ import annotations

import numpy as np


class MMA:
    """Optimizer state: call :meth:`update` with the current point, objective and
    constraint values and gradients; it returns the next point."""

    def __init__(
        self,
        n: int,
        m: int,
        xmin,
        xmax,
        *,
        move: float = 0.5,
        asyinit: float = 0.5,
        asyincr: float = 1.2,
        asydecr: float = 0.7,
        c: float = 1000.0,
    ):
        self.n, self.m = int(n), int(m)
        if self.m < 1 or self.m >= self.n:
            raise ValueError("MMA here needs 1 <= m < n constraints")
        self.xmin = np.broadcast_to(np.asarray(xmin, dtype=float), (self.n,)).copy()
        self.xmax = np.broadcast_to(np.asarray(xmax, dtype=float), (self.n,)).copy()
        self.move = float(move)
        self.asyinit, self.asyincr, self.asydecr = asyinit, asyincr, asydecr
        self.a0 = 1.0
        self.a = np.zeros(self.m)
        self.c = np.full(self.m, float(c))
        self.d = np.ones(self.m)
        self.iteration = 0
        self.xold1 = None
        self.xold2 = None
        self.low = None
        self.upp = None

    def reset(self) -> None:
        """Forget the asymptotes after a change of the problem (a continuation step):
        asymptotes tightened by past oscillations would hold the design in place."""
        self.iteration = 0
        self.xold1 = None
        self.xold2 = None

    def update(self, x, f0, df0, fval, dfdx) -> np.ndarray:
        x = np.asarray(x, dtype=float).copy()
        df0 = np.asarray(df0, dtype=float).reshape(self.n)
        fval = np.atleast_1d(np.asarray(fval, dtype=float))
        dfdx = np.asarray(dfdx, dtype=float).reshape(self.m, self.n)
        self.iteration += 1
        span = self.xmax - self.xmin
        if self.iteration <= 2 or self.xold2 is None:
            self.low = x - self.asyinit * span
            self.upp = x + self.asyinit * span
        else:
            sign = (x - self.xold1) * (self.xold1 - self.xold2)
            factor = np.ones(self.n)
            factor[sign > 0] = self.asyincr
            factor[sign < 0] = self.asydecr
            self.low = x - factor * (self.xold1 - self.low)
            self.upp = x + factor * (self.upp - self.xold1)
            self.low = np.clip(self.low, x - 10.0 * span, x - 0.01 * span)
            self.upp = np.clip(self.upp, x + 0.01 * span, x + 10.0 * span)
        albefa = 0.1
        alpha = np.maximum.reduce(
            [self.xmin, self.low + albefa * (x - self.low), x - self.move * span]
        )
        beta = np.minimum.reduce(
            [self.xmax, self.upp - albefa * (self.upp - x), x + self.move * span]
        )
        xmami = np.maximum(span, 1e-5)
        ux1 = self.upp - x
        xl1 = x - self.low
        ux2, xl2 = ux1 * ux1, xl1 * xl1
        raa0 = 1e-5
        p0 = np.maximum(df0, 0.0)
        q0 = np.maximum(-df0, 0.0)
        pq0 = 0.001 * (p0 + q0) + raa0 / xmami
        p0 = (p0 + pq0) * ux2
        q0 = (q0 + pq0) * xl2
        p = np.maximum(dfdx, 0.0)
        q = np.maximum(-dfdx, 0.0)
        pq = 0.001 * (p + q) + raa0 / xmami
        p = (p + pq) * ux2
        q = (q + pq) * xl2
        b = p @ (1.0 / ux1) + q @ (1.0 / xl1) - fval
        xnew = _subsolve(
            self.m,
            self.n,
            self.low,
            self.upp,
            alpha,
            beta,
            p0,
            q0,
            p,
            q,
            self.a0,
            self.a,
            b,
            self.c,
            self.d,
        )
        self.xold2 = self.xold1
        self.xold1 = x
        return xnew


def _subsolve(
    m, n, low, upp, alpha, beta, p0, q0, p, q, a0, a, b, c, d, epsimin: float = 1e-7
) -> np.ndarray:
    """Primal-dual Newton solution of the MMA subproblem (Svanberg's subsolv)."""
    een = np.ones(n)
    eem = np.ones(m)
    epsi = 1.0
    x = 0.5 * (alpha + beta)
    y = eem.copy()
    z = 1.0
    lam = eem.copy()
    xsi = np.maximum(een / (x - alpha), een)
    eta = np.maximum(een / (beta - x), een)
    mu = np.maximum(eem, 0.5 * c)
    zet = 1.0
    s = eem.copy()

    def residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi):
        ux1 = upp - x
        xl1 = x - low
        plam = p0 + p.T @ lam
        qlam = q0 + q.T @ lam
        gvec = p @ (1.0 / ux1) + q @ (1.0 / xl1)
        dpsidx = plam / ux1**2 - qlam / xl1**2
        return np.concatenate(
            [
                dpsidx - xsi + eta,
                c + d * y - mu - lam,
                [a0 - zet - a @ lam],
                gvec - a * z - y + s - b,
                xsi * (x - alpha) - epsi,
                eta * (beta - x) - epsi,
                mu * y - epsi,
                [zet * z - epsi],
                lam * s - epsi,
            ]
        )

    while epsi > epsimin:
        res = residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi)
        resnorm = np.linalg.norm(res)
        resmax = np.max(np.abs(res))
        inner = 0
        while resmax > 0.9 * epsi and inner < 200:
            inner += 1
            ux1 = upp - x
            xl1 = x - low
            ux2, xl2 = ux1**2, xl1**2
            ux3, xl3 = ux1 * ux2, xl1 * xl2
            plam = p0 + p.T @ lam
            qlam = q0 + q.T @ lam
            gvec = p @ (1.0 / ux1) + q @ (1.0 / xl1)
            gg = p / ux2 - q / xl2  # (m, n)
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsi / (x - alpha) + epsi / (beta - x)
            dely = c + d * y - lam - epsi / y
            delz = a0 - a @ lam - epsi / z
            dellam = gvec - a * z - y - b + epsi / lam
            diagx = 2.0 * (plam / ux3 + qlam / xl3) + xsi / (x - alpha) + eta / (beta - x)
            diagy = d + mu / y
            diaglamyi = s / lam + 1.0 / diagy
            blam = dellam + dely / diagy - gg @ (delx / diagx)
            alam = np.diag(diaglamyi) + (gg / diagx) @ gg.T
            aa = np.zeros((m + 1, m + 1))
            aa[:m, :m] = alam
            aa[:m, m] = a
            aa[m, :m] = a
            aa[m, m] = -zet / z
            sol = np.linalg.solve(aa, np.concatenate([blam, [delz]]))
            dlam, dz = sol[:m], sol[m]
            dx = -delx / diagx - (gg.T @ dlam) / diagx
            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsi / (x - alpha) - xsi * dx / (x - alpha)
            deta = -eta + epsi / (beta - x) + eta * dx / (beta - x)
            dmu = -mu + epsi / y - mu * dy / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsi / lam - s * dlam / lam
            xx = np.concatenate([y, [z], lam, xsi, eta, mu, [zet], s])
            dxx = np.concatenate([dy, [dz], dlam, dxsi, deta, dmu, [dzet], ds])
            stmxx = np.max(-1.01 * dxx / xx)
            stmalfa = np.max(-1.01 * dx / (x - alpha))
            stmbeta = np.max(1.01 * dx / (beta - x))
            step = 1.0 / max(stmxx, stmalfa, stmbeta, 1.0)
            old = (x, y, z, lam, xsi, eta, mu, zet, s)
            newnorm = 2.0 * resnorm
            tries = 0
            while newnorm > resnorm and tries < 50:
                tries += 1
                x = old[0] + step * dx
                y = old[1] + step * dy
                z = old[2] + step * dz
                lam = old[3] + step * dlam
                xsi = old[4] + step * dxsi
                eta = old[5] + step * deta
                mu = old[6] + step * dmu
                zet = old[7] + step * dzet
                s = old[8] + step * ds
                res = residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi)
                newnorm = np.linalg.norm(res)
                step *= 0.5
            resnorm = newnorm
            resmax = np.max(np.abs(res))
        epsi *= 0.1
    return x
