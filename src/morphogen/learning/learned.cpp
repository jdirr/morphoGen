// The learned laws of morphoGen (morphogen.learning) in the MFront sources of
// morphogen.learning.mfront: ports, line by line, of the forward passes of the models and of
// their consistent tangents (networks.py: FeedForward, Potential, Recurrent; standard.py:
// Standard, its Newton solve of the incremental potential), in the normalized variables of
// the models. NumPy and Numba are the reference: a change of a model is made in both, and
// tests/test_learning_mfront.py compares them through MFront and MGIS.
//
// The source prepends the sizes and the trained parameters of its model (constexpr in the
// namespace of the behaviour: SCALE_STRAIN, SCALE_STRESS, SCALE_INCREMENT, SCALE_TIME, then
// those of its kind) and takes the common part of this file and the section of its kind
// (from "// ---- <kind>" to the next).
//
// Components: morphoGen's Voigt order 11, 22, 33, 23, 31, 12, engineering shears of the
// strains, in every array of this file (the trained parameters are those of the models);
// the source turns TFEL's components (11, 22, 33, then sqrt(2) times 12, 13, 23) to them
// and back.
//
// Every law is
//   int law(const double* e1, const double* e0, const double* s0, double dt,
//           double* sig, double* tan, double* s1)
// e1 and e0: the strains that the model sees at the end and at the start of the step (6:
// the small strain, the logarithmic strain 1/2 ln C or the Green-Lagrange strain, without
// the thermal strain, in the axes of the material); s0: the state of the model at the start
// of the step (its internal variables); dt: the time step. On return: sig, the stress (6:
// that conjugate to the strain the model sees), tan, its derivative with respect to e1 (6 x
// 6, row by row; engineering shears), s1, the state at the end; 0, or 1 where the stress or
// its tangent is not finite (a failed point, as in morphoGen).

// ---- common

inline double sigmoid(double x) { return 0.5 * (1.0 + std::tanh(0.5 * x)); }

inline int finite_values(const double* sig, const double* tan) {
  for (int i = 0; i != 6; ++i) {
    if (!std::isfinite(sig[i])) {
      return 1;
    }
  }
  for (int i = 0; i != 36; ++i) {
    if (!std::isfinite(tan[i])) {
      return 1;
    }
  }
  return 0;
}

// ---- feedforward
// y = L x + Wout z_K + bout, z_{k+1} = tanh(W_k z_k + b_k), z_0 = x (FeedForward._output),
// and its Jacobian forward through the layers (FeedForward._jacobian). Constants: N_LAYERS,
// SIZES, WIDEST, W (the hidden weights W_k, out x in, layer after layer), B (their biases),
// WOUT (6 x SIZES[N_LAYERS - 1]), BOUT (6), LIN (6 x 6).

inline int law(const double* e1, const double*, const double*, double, double* sig,
               double* tan, double*) {
  double x[6], z[WIDEST], zn[WIDEST], t[6 * WIDEST], tn[6 * WIDEST];
  for (int i = 0; i != 6; ++i) {
    x[i] = e1[i] / SCALE_STRAIN;
    z[i] = x[i];
    for (int d = 0; d != 6; ++d) {
      t[6 * i + d] = i == d ? 1.0 : 0.0;
    }
  }
  int n = 6, w = 0, b = 0;
  for (int k = 0; k != N_LAYERS; ++k) {
    const int m = SIZES[k];
    for (int i = 0; i != m; ++i) {
      double a = 0.0;
      for (int j = 0; j != n; ++j) {
        a += z[j] * W[w + n * i + j];
      }
      zn[i] = std::tanh(a + B[b + i]);
      const double g = 1.0 - zn[i] * zn[i];
      for (int d = 0; d != 6; ++d) {
        double s = 0.0;
        for (int j = 0; j != n; ++j) {
          s += W[w + n * i + j] * t[6 * j + d];
        }
        tn[6 * i + d] = g * s;
      }
    }
    for (int i = 0; i != m; ++i) {
      z[i] = zn[i];
      for (int d = 0; d != 6; ++d) {
        t[6 * i + d] = tn[6 * i + d];
      }
    }
    w += m * n;
    b += m;
    n = m;
  }
  for (int r = 0; r != 6; ++r) {
    double y = 0.0;
    for (int j = 0; j != n; ++j) {
      y += z[j] * WOUT[n * r + j];
    }
    y += BOUT[r];
    double lin = 0.0;
    for (int c = 0; c != 6; ++c) {
      lin += x[c] * LIN[6 * r + c];
    }
    sig[r] = SCALE_STRESS * (y + lin);
    for (int d = 0; d != 6; ++d) {
      double s = 0.0;
      for (int j = 0; j != n; ++j) {
        s += WOUT[n * r + j] * t[6 * j + d];
      }
      tan[6 * r + d] = SCALE_STRESS / SCALE_STRAIN * (s + LIN[6 * r + d]);
    }
  }
  return finite_values(sig, tan);
}

// ---- potential
// y = dW/dx (x) - dW/dx (0), W(x) = x . A x / 2 + w . z_K, z_k = g(U_k x + b_k + W_k z_{k-1})
// an input-convex network (Potential._gradient: the gradient backward through the layers,
// the Hessian forward derivatives of it, symmetrized). Constants: N_LAYERS, SIZES, OFFSETS
// (of the units of each layer), TOTAL, WIDEST, U (TOTAL x 6), BIAS (TOTAL), WPOS (the
// nonnegative W_k = softplus(V_k) of the layers k > 0, SIZES[k] x SIZES[k - 1], layer after
// layer; WOFFSETS their offsets), WOUT (w = softplus(v), SIZES[N_LAYERS - 1]), AMAT (A = L
// L^T, 6 x 6), G0 (dW/dx (0)).

inline double softplus(double a) {  // NumPy's logaddexp(0, a)
  return (a > 0.0 ? a : 0.0) + std::log1p(std::exp(-std::fabs(a)));
}

inline int law(const double* e1, const double*, const double*, double, double* sig,
               double* tan, double*) {
  double x[6], z[TOTAL], s[TOTAL], ds[TOTAL], delta[TOTAL], tang[6 * TOTAL];
  for (int d = 0; d != 6; ++d) {
    x[d] = e1[d] / SCALE_STRAIN;
  }
  for (int k = 0; k != N_LAYERS; ++k) {  // the layers and their derivatives da_k / dx
    const int o = OFFSETS[k];
    for (int i = 0; i != SIZES[k]; ++i) {
      double a = 0.0;
      for (int d = 0; d != 6; ++d) {
        a += x[d] * U[6 * (o + i) + d];
      }
      a += BIAS[o + i];
      if (k > 0) {
        const int p = OFFSETS[k - 1], hp = SIZES[k - 1];
        double v = 0.0;
        for (int j = 0; j != hp; ++j) {
          v += z[p + j] * WPOS[WOFFSETS[k] + hp * i + j];
        }
        a += v;
      }
      z[o + i] = softplus(a);
      s[o + i] = sigmoid(a);
      ds[o + i] = s[o + i] * (1.0 - s[o + i]);
      for (int d = 0; d != 6; ++d) {
        double t = U[6 * (o + i) + d];
        if (k > 0) {
          const int p = OFFSETS[k - 1], hp = SIZES[k - 1];
          double v = 0.0;
          for (int j = 0; j != hp; ++j) {
            v += WPOS[WOFFSETS[k] + hp * i + j] * (s[p + j] * tang[6 * (p + j) + d]);
          }
          t += v;
        }
        tang[6 * (o + i) + d] = t;
      }
    }
  }
  const int last = N_LAYERS - 1, ol = OFFSETS[last];
  for (int i = 0; i != SIZES[last]; ++i) {  // dW / da_k, backward
    delta[ol + i] = WOUT[i] * s[ol + i];
  }
  for (int k = last; k > 0; --k) {
    const int o = OFFSETS[k], h = SIZES[k], p = OFFSETS[k - 1], hp = SIZES[k - 1];
    for (int j = 0; j != hp; ++j) {
      double v = 0.0;
      for (int i = 0; i != h; ++i) {
        v += delta[o + i] * WPOS[WOFFSETS[k] + hp * i + j];
      }
      delta[p + j] = v * s[p + j];
    }
  }
  for (int r = 0; r != 6; ++r) {
    double g = 0.0;
    for (int k = 0; k != N_LAYERS; ++k) {
      double v = 0.0;
      for (int i = 0; i != SIZES[k]; ++i) {
        v += delta[OFFSETS[k] + i] * U[6 * (OFFSETS[k] + i) + r];
      }
      g += v;
    }
    double q = 0.0;
    for (int c = 0; c != 6; ++c) {
      q += x[c] * AMAT[6 * c + r];
    }
    sig[r] = SCALE_STRESS * ((g + q) - G0[r]);
  }
  // the Hessian: dd = d(dW/da_k)/dx, from the last layer down
  double hess[36], dd[6 * WIDEST], nd[6 * WIDEST];
  for (int i = 0; i != SIZES[last]; ++i) {
    for (int d = 0; d != 6; ++d) {
      dd[6 * i + d] = (WOUT[i] * ds[ol + i]) * tang[6 * (ol + i) + d];
    }
  }
  for (int j = 0; j != 6; ++j) {
    for (int d = 0; d != 6; ++d) {
      double v = 0.0;
      for (int i = 0; i != SIZES[last]; ++i) {
        v += dd[6 * i + d] * U[6 * (ol + i) + j];
      }
      hess[6 * j + d] = v;
    }
  }
  for (int k = last; k > 0; --k) {
    const int o = OFFSETS[k], h = SIZES[k], p = OFFSETS[k - 1], hp = SIZES[k - 1];
    for (int j = 0; j != hp; ++j) {
      double q = 0.0;
      for (int i = 0; i != h; ++i) {
        q += delta[o + i] * WPOS[WOFFSETS[k] + hp * i + j];
      }
      q *= ds[p + j];
      for (int d = 0; d != 6; ++d) {
        double back = 0.0;
        for (int i = 0; i != h; ++i) {
          back += dd[6 * i + d] * WPOS[WOFFSETS[k] + hp * i + j];
        }
        nd[6 * j + d] = back * s[p + j] + q * tang[6 * (p + j) + d];
      }
    }
    for (int j = 0; j != hp; ++j) {
      for (int d = 0; d != 6; ++d) {
        dd[6 * j + d] = nd[6 * j + d];
      }
    }
    for (int jj = 0; jj != 6; ++jj) {
      for (int d = 0; d != 6; ++d) {
        double v = 0.0;
        for (int j = 0; j != hp; ++j) {
          v += dd[6 * j + d] * U[6 * (p + j) + jj];
        }
        hess[6 * jj + d] += v;
      }
    }
  }
  for (int i = 0; i != 36; ++i) {
    hess[i] += AMAT[i];
  }
  for (int r = 0; r != 6; ++r) {
    for (int c = 0; c != 6; ++c) {
      tan[6 * r + c] = SCALE_STRESS / SCALE_STRAIN * (0.5 * (hess[6 * r + c] + hess[6 * c + r]));
    }
  }
  return finite_values(sig, tan);
}

// ---- recurrent
// A gated recurrent unit over the steps (Recurrent.step): its input the strain at the end
// of the step and the increment of the step (and the time step, with RATE), its state the
// internal variables of the law, y = V h + Q x + c; the tangent the derivative with respect
// to the strain at the end of the step, the state at its start held. Constants: HIDDEN,
// N_INPUTS, RATE, LOGARITHMIC (the increment of the normalized logarithmic strain, as
// Recurrent.step writes it), WI (N_INPUTS x 3 HIDDEN), WH (HIDDEN x 3 HIDDEN), BG (3 HIDDEN),
// BN (HIDDEN), V (HIDDEN x 6), Q (6 x 6), C (6).

inline int law(const double* e1, const double* e0, const double* s0, double dt, double* sig,
               double* tan, double* s1) {
  constexpr int H = HIDDEN;
  double u[N_INPUTS], seen[6], gi[3 * H], gh[3 * H], r[H], zg[H], hn[H], cand[H];
  double dstate[6 * H];
  for (int i = 0; i != 6; ++i) {
    seen[i] = e1[i] / SCALE_STRAIN;
    u[i] = seen[i];
    if (LOGARITHMIC) {
      u[6 + i] = (seen[i] - e0[i] / SCALE_STRAIN) * (SCALE_STRAIN / SCALE_INCREMENT);
    } else {
      u[6 + i] = (e1[i] - e0[i]) / SCALE_INCREMENT;
    }
  }
  if (RATE) {  // the last input, the thirteenth
    u[N_INPUTS - 1] = dt / SCALE_TIME;
  }
  for (int j = 0; j != 3 * H; ++j) {
    double a = 0.0;
    for (int c = 0; c != N_INPUTS; ++c) {
      a += u[c] * WI[3 * H * c + j];
    }
    gi[j] = a + BG[j];
    double v = 0.0;
    for (int c = 0; c != H; ++c) {
      v += s0[c] * WH[3 * H * c + j];
    }
    gh[j] = v;
  }
  for (int j = 0; j != H; ++j) {
    r[j] = sigmoid(gi[j] + gh[j]);
    zg[j] = sigmoid(gi[H + j] + gh[H + j]);
    hn[j] = gh[2 * H + j] + BN[j];
    cand[j] = std::tanh(gi[2 * H + j] + r[j] * hn[j]);
    s1[j] = cand[j] + zg[j] * (s0[j] - cand[j]);
    for (int d = 0; d != 6; ++d) {  // the inputs of the strain at the end, of the increment
      const double dr = WI[3 * H * d + j] / SCALE_STRAIN
                        + WI[3 * H * (6 + d) + j] / SCALE_INCREMENT;
      const double dz = WI[3 * H * d + H + j] / SCALE_STRAIN
                        + WI[3 * H * (6 + d) + H + j] / SCALE_INCREMENT;
      const double dn = WI[3 * H * d + 2 * H + j] / SCALE_STRAIN
                        + WI[3 * H * (6 + d) + 2 * H + j] / SCALE_INCREMENT;
      const double ddr = r[j] * (1.0 - r[j]) * dr;
      const double ddz = zg[j] * (1.0 - zg[j]) * dz;
      const double dcand = (1.0 - cand[j] * cand[j]) * (dn + ddr * hn[j]);
      dstate[6 * j + d] = (1.0 - zg[j]) * dcand + ddz * (s0[j] - cand[j]);
    }
  }
  for (int k = 0; k != 6; ++k) {
    double y = 0.0;
    for (int j = 0; j != H; ++j) {
      y += s1[j] * V[6 * j + k];
    }
    double q = 0.0;
    for (int c = 0; c != 6; ++c) {
      q += seen[c] * Q[6 * k + c];
    }
    sig[k] = SCALE_STRESS * ((y + q) + C[k]);
    for (int d = 0; d != 6; ++d) {
      double v = 0.0;
      for (int j = 0; j != H; ++j) {
        v += dstate[6 * j + d] * V[6 * j + k];
      }
      tan[6 * k + d] = SCALE_STRESS * (v + Q[6 * k + d] / SCALE_STRAIN);
    }
  }
  return finite_values(sig, tan);
}

// ---- standard
// A learned generalized standard material (standard.py): the state of a step the minimizer
// of the incremental potential W(z) + Phi(alpha - alpha_n), z = (x, alpha), by Newton's
// method with Armijo's backtracking (_update), the stress dW/dx and the tangent the Schur
// complement of its Hessian (_schur). Constants: M (internal variables), D = 6 + M, NH (the
// ridges of the free energy), U (NH x D), BIAS (NH), WR (w = softplus, NH), A (L L^T, D x
// D), G0 (dW/dz (0)), N_NORMS, N_RIDGES, BM (B, N_NORMS x M x M), GM (B_j^T B_j), P
// (softplus, N_RIDGES x N_NORMS), CM (N_RIDGES x M), VM (M M^T, M x M), DELTA, PLASTIC,
// VISCOUS, NEWTON_TOL, NEWTON_MAX, ARMIJO.

inline double soft(double t) {  // the softplus of a scalar, without overflow (_soft)
  if (t > 30.0) {
    return t;
  }
  if (t < -30.0) {
    return std::exp(t);
  }
  return std::log1p(std::exp(t));
}

// W(z) - z . g0 and, with order 1 or 2, its gradient and Hessian (_energy)
inline double energy(const double* z, double* grad, double* hess, int order) {
  double value = 0.0;
  for (int i = 0; i != D; ++i) {
    double s = 0.0;
    for (int j = 0; j != D; ++j) {
      s += A[D * i + j] * z[j];
    }
    value += (0.5 * s - G0[i]) * z[i];
    if (order > 0) {
      grad[i] = s - G0[i];
    }
  }
  if (order > 1) {
    for (int i = 0; i != D * D; ++i) {
      hess[i] = A[i];
    }
  }
  for (int k = 0; k != NH; ++k) {
    double a = BIAS[k];
    for (int j = 0; j != D; ++j) {
      a += U[D * k + j] * z[j];
    }
    value += WR[k] * soft(a);
    if (order > 0) {
      const double sg = sigmoid(a);
      const double c = WR[k] * sg;
      for (int j = 0; j != D; ++j) {
        grad[j] += c * U[D * k + j];
      }
      if (order > 1) {
        const double c2 = c * (1.0 - sg);
        for (int i = 0; i != D; ++i) {
          const double ci = c2 * U[D * k + i];
          for (int j = 0; j != D; ++j) {
            hess[D * i + j] += ci * U[D * k + j];
          }
        }
      }
    }
  }
  return value;
}

// the dissipation of a step Phi(v) of the increment v and, with order 1 or 2, its gradient
// and Hessian (_dissipation)
inline double dissipation(const double* v, double tau, double* grad, double* hess, int order) {
  double value = 0.0;
  if (order > 0) {
    for (int i = 0; i != M; ++i) {
      grad[i] = 0.0;
    }
  }
  if (order > 1) {
    for (int i = 0; i != M * M; ++i) {
      hess[i] = 0.0;
    }
  }
  if (VISCOUS) {
    for (int i = 0; i != M; ++i) {
      double s = 0.0;
      for (int j = 0; j != M; ++j) {
        s += VM[M * i + j] * v[j];
      }
      value += 0.5 * s * v[i] / tau;
      if (order > 0) {
        grad[i] += s / tau;
      }
      if (order > 1) {
        for (int j = 0; j != M; ++j) {
          hess[M * i + j] += VM[M * i + j] / tau;
        }
      }
    }
  }
  if (!PLASTIC) {
    return value;
  }
  double norm[N_NORMS], radius[N_NORMS], dn[N_NORMS * M], f[N_NORMS], dt[M];
  for (int j = 0; j != N_NORMS; ++j) {
    double ss = 0.0;
    for (int i = 0; i != M; ++i) {
      double s = 0.0;
      for (int l = 0; l != M; ++l) {
        s += BM[M * M * j + M * i + l] * v[l];
      }
      ss += s * s;
    }
    radius[j] = std::sqrt(ss + DELTA * DELTA);
    norm[j] = ss / (radius[j] + DELTA);  // sqrt(ss + delta^2) - delta, without cancellation
    if (order > 0) {
      for (int l = 0; l != M; ++l) {
        double s = 0.0;
        for (int i = 0; i != M; ++i) {
          s += GM[M * M * j + M * l + i] * v[i];
        }
        dn[M * j + l] = s / radius[j];
      }
    }
    f[j] = 0.0;
  }
  for (int k = 0; k != N_RIDGES; ++k) {
    double t = 0.0;
    for (int j = 0; j != N_NORMS; ++j) {
      t += P[N_NORMS * k + j] * norm[j];
    }
    double lin = 0.0;
    for (int l = 0; l != M; ++l) {
      t += CM[M * k + l] * v[l];
      lin += CM[M * k + l] * v[l];
    }
    const double q = std::sqrt(t * t + DELTA * DELTA);
    value += 0.5 * (t + q) - 0.5 * DELTA - 0.5 * lin;
    if (order == 0) {
      continue;
    }
    const double s1 = 0.5 * (1.0 + t / q);
    for (int l = 0; l != M; ++l) {
      double s = CM[M * k + l];
      for (int j = 0; j != N_NORMS; ++j) {
        s += P[N_NORMS * k + j] * dn[M * j + l];
      }
      dt[l] = s;
      grad[l] += s1 * s - 0.5 * CM[M * k + l];
    }
    if (order > 1) {
      const double s2 = 0.5 * DELTA * DELTA / (q * q * q);
      for (int l = 0; l != M; ++l) {
        for (int i = 0; i != M; ++i) {
          hess[M * l + i] += s2 * dt[l] * dt[i];
        }
      }
      for (int j = 0; j != N_NORMS; ++j) {
        f[j] += s1 * P[N_NORMS * k + j];
      }
    }
  }
  if (order > 1) {  // sum_j f_j d2 n_j, d2 n_j = (B_j^T B_j - dn_j dn_j^T) / r_j
    for (int j = 0; j != N_NORMS; ++j) {
      for (int l = 0; l != M; ++l) {
        for (int i = 0; i != M; ++i) {
          hess[M * l + i] += f[j] * (GM[M * M * j + M * l + i] - dn[M * j + l] * dn[M * j + i])
                             / radius[j];
        }
      }
    }
  }
  return value;
}

// K^-1 rhs of a symmetric positive definite matrix by Cholesky's factorization, a small
// shift of the diagonal if round-off spoils it (_solve)
inline void solve(const double* K, const double* rhs, double* out) {
  double low[M * M];
  for (int i = 0; i != M * M; ++i) {
    low[i] = 0.0;
  }
  double shift = 0.0;
  for (int attempt = 0; attempt != 8; ++attempt) {
    bool ok = true;
    for (int j = 0; j != M && ok; ++j) {
      double s = K[M * j + j] + shift;
      for (int k = 0; k != j; ++k) {
        s -= low[M * j + k] * low[M * j + k];
      }
      if (!(s > 0.0)) {
        ok = false;
        break;
      }
      low[M * j + j] = std::sqrt(s);
      for (int i = j + 1; i != M; ++i) {
        double v = K[M * i + j];
        for (int k = 0; k != j; ++k) {
          v -= low[M * i + k] * low[M * j + k];
        }
        low[M * i + j] = v / low[M * j + j];
      }
    }
    if (ok) {
      break;
    }
    double trace = 0.0;
    for (int j = 0; j != M; ++j) {
      trace += std::fabs(K[M * j + j]);
    }
    const double floor = 1e-14 * trace / M + 1e-300;
    shift = 10.0 * shift > floor ? 10.0 * shift : floor;
  }
  for (int i = 0; i != M; ++i) {
    double s = rhs[i];
    for (int k = 0; k != i; ++k) {
      s -= low[M * i + k] * out[k];
    }
    out[i] = s / low[M * i + i];
  }
  for (int i = M - 1; i >= 0; --i) {
    double s = out[i];
    for (int k = i + 1; k != M; ++k) {
      s -= low[M * k + i] * out[k];
    }
    out[i] = s / low[M * i + i];
  }
}

// the state of a step: the minimizer alpha of the incremental potential at the strain x
// from the state a0 (Newton's method from a0, a backtracking line search); on return z =
// (x, alpha), the gradient gz and Hessian hz of the free energy at z and kmat, the Hessian
// of the potential in alpha; returns the iterations, or minus them without convergence
// (_update)
inline int update(const double* x, const double* a0, double tau, double* z, double* gz,
                  double* hz, double* kmat) {
  double inc[M], res[M], step[M], trial[D], gp[M], hp[M * M];
  for (int i = 0; i != 6; ++i) {
    z[i] = x[i];
  }
  for (int i = 0; i != M; ++i) {
    z[6 + i] = a0[i];
    inc[i] = 0.0;
  }
  for (int it = 0; it != NEWTON_MAX + 1; ++it) {
    double value = energy(z, gz, hz, 2);
    for (int i = 0; i != M; ++i) {
      inc[i] = z[6 + i] - a0[i];
    }
    value += dissipation(inc, tau, gp, hp, 2);
    double size = 0.0;
    for (int i = 0; i != M; ++i) {
      res[i] = gz[6 + i] + gp[i];
      size = std::fabs(res[i]) > size ? std::fabs(res[i]) : size;
      for (int j = 0; j != M; ++j) {
        kmat[M * i + j] = hz[D * (6 + i) + 6 + j] + hp[M * i + j];
      }
    }
    if (size <= NEWTON_TOL) {
      return it;
    }
    if (it == NEWTON_MAX) {
      break;
    }
    solve(kmat, res, step);
    double slope = 0.0;
    for (int i = 0; i != M; ++i) {
      step[i] = -step[i];
      slope += res[i] * step[i];
    }
    double t = 1.0;
    // far from the solution: Armijo's backtracking on the potential, while the decrease it
    // predicts is above the round-off of the potential
    if (size > 1e-6 && -slope > 1e-12 * (1.0 + std::fabs(value))) {
      for (int i = 0; i != 6; ++i) {
        trial[i] = x[i];
      }
      for (int ls = 0; ls != 40; ++ls) {
        for (int i = 0; i != M; ++i) {
          trial[6 + i] = z[6 + i] + t * step[i];
          inc[i] = trial[6 + i] - a0[i];
        }
        double got = energy(trial, gz, hz, 0);
        got += dissipation(inc, tau, gp, hp, 0);
        if (got <= value + ARMIJO * t * slope) {
          break;
        }
        t *= 0.5;
      }
    }
    for (int i = 0; i != M; ++i) {
      z[6 + i] += t * step[i];
    }
  }
  return -NEWTON_MAX;
}

// the consistent tangent d2W/dx2 - d2W/dx dalpha K^-1 d2W/dalpha dx (6 x 6, symmetrized:
// _schur)
inline void schur(const double* hz, const double* kmat, double* out) {
  double col[M], sol[M];
  for (int c = 0; c != 6; ++c) {
    for (int i = 0; i != M; ++i) {
      col[i] = hz[D * (6 + i) + c];
    }
    solve(kmat, col, sol);
    for (int r = 0; r != 6; ++r) {
      double s = hz[D * r + c];
      for (int i = 0; i != M; ++i) {
        s -= hz[D * r + 6 + i] * sol[i];
      }
      out[6 * r + c] = s;
    }
  }
  for (int r = 0; r != 6; ++r) {
    for (int c = 0; c != r; ++c) {
      const double s = 0.5 * (out[6 * r + c] + out[6 * c + r]);
      out[6 * r + c] = s;
      out[6 * c + r] = s;
    }
  }
}

inline int law(const double* e1, const double*, const double* s0, double dt, double* sig,
               double* tan, double* s1) {
  double x[6], z[D], gz[D], hz[D * D], kmat[M * M], k6[36];
  for (int i = 0; i != 6; ++i) {
    x[i] = e1[i] / SCALE_STRAIN;
  }
  // a step of no time: the viscous variables frozen (an infinite viscous resistance)
  const double tau = VISCOUS ? (dt / SCALE_TIME > 1e-12 ? dt / SCALE_TIME : 1e-12) : 1.0;
  update(x, s0, tau, z, gz, hz, kmat);
  schur(hz, kmat, k6);
  for (int i = 0; i != 6; ++i) {
    sig[i] = SCALE_STRESS * gz[i];
  }
  for (int i = 0; i != 36; ++i) {
    tan[i] = SCALE_STRESS / SCALE_STRAIN * k6[i];
  }
  for (int i = 0; i != M; ++i) {
    s1[i] = z[6 + i];
  }
  return finite_values(sig, tan);
}
