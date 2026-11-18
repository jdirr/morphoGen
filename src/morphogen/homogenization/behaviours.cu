// Behaviours of the integration points on the GPU (morphogen.homogenization.gpu): ports,
// line by line, of the Numba kernels of morphogen.homogenization.tensors (small dense
// algebra), .behaviours (von Mises elasto-(visco)plasticity, Hill's criterion), .crystal
// (Méric-Cailletaud crystal plasticity), .porous (porous plasticity: GTN, Rousselier, the
// elliptic criterion) and .finite (the space of the logarithmic strain);
// one thread per point. The Numba kernels are the reference: a change of a behaviour is made
// in both, and tests/test_gpu.py compares them point by point.
//
// Prepended by morphogen.homogenization.gpu: NS (slip systems of the arrays of the
// crystals), WIDTH (internal variables of a point) and the constants of the Python modules
// (CP_ALPHA, STATE_P, STATE_X, KIN, N_PARAMS, PARAMS_WIDTH, ROW_WIDTH, N_CP, MAX_ISOTROPIC,
// ELASTIC, PLASTIC, CRYSTAL, MFRONT, HILL, POROUS, HILL_IT, SLIP_FLOOR, LOCAL_TOL,
// MAX_LOCAL, MAX_ROUNDS, SHRINK, SUBSTEP_LEVELS, and those of the porous rows: P_..., the
// criteria GURSON, ROUSSELIER, ELLIPTIC, POROUS_ISOTROPIC, POROUS_KINEMATIC, STATE_FG,
// STATE_FN, STATE_SSTAR, STATE_BACK). The rows of parameters are ROW_WIDTH wide: those of
// von Mises (N_PARAMS), then the coefficients of Hill (PARAMS_WIDTH), or those of a porous
// point.
// The points of MFront behaviours (MFRONT) are skipped: MGIS integrates them on the CPU.
//
// Arrays of the points are stored point by point, as on the CPU: strains (n x 6,
// engineering) or displacement gradients (n x 9, row by row), stresses, tangents (n x 6 x 6
// or n x 9 x 9), internal variables (n x WIDTH).

#define MG_INF __longlong_as_double(0x7ff0000000000000LL)

// max and min with the semantics of Python's (and Numba's): the first argument unless the
// second is strictly greater (smaller)
__device__ __forceinline__ double pmax(double a, double b) { return b > a ? b : a; }
__device__ __forceinline__ double pmin(double a, double b) { return b < a ? b : a; }

// ---- algebra: morphogen.homogenization.tensors -------------------------------------------

__device__ __forceinline__ void matmul3(double a[3][3], double b[3][3], double out[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
}

__device__ __forceinline__ void tmatmul3(double a[3][3], double b[3][3], double out[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out[i][j] = a[0][i] * b[0][j] + a[1][i] * b[1][j] + a[2][i] * b[2][j];
}

__device__ __forceinline__ void matmult3(double a[3][3], double b[3][3], double out[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out[i][j] = a[i][0] * b[j][0] + a[i][1] * b[j][1] + a[i][2] * b[j][2];
}

__device__ __forceinline__ void strain_voigt(double t[3][3], double* out) {
    out[0] = t[0][0];
    out[1] = t[1][1];
    out[2] = t[2][2];
    out[3] = t[1][2] + t[2][1];
    out[4] = t[2][0] + t[0][2];
    out[5] = t[0][1] + t[1][0];
}

__device__ __forceinline__ void stress_tensor(const double* v, double out[3][3]) {
    out[0][0] = v[0];
    out[1][1] = v[1];
    out[2][2] = v[2];
    out[1][2] = v[3];
    out[2][1] = v[3];
    out[2][0] = v[4];
    out[0][2] = v[4];
    out[0][1] = v[5];
    out[1][0] = v[5];
}

__device__ __forceinline__ void elastic_stress(const double* d, double strain[3][3],
                                               double out[3][3], double* e6, double* s6) {
    strain_voigt(strain, e6);
    for (int r = 0; r < 6; ++r) {
        double acc = 0.0;
        for (int c = 0; c < 6; ++c) acc += d[6 * r + c] * e6[c];
        s6[r] = acc;
    }
    stress_tensor(s6, out);
}

__device__ __forceinline__ double project(const double* lv, const double* nv,
                                          double t[3][3]) {
    double acc = 0.0;
    for (int i = 0; i < 3; ++i)
        acc += lv[i] * (t[i][0] * nv[0] + t[i][1] * nv[1] + t[i][2] * nv[2]);
    return acc;
}

// a (n x n, row by row) replaced by its symmetric part (symmetrize)
__device__ __forceinline__ void symmetrize(double* a, int n) {
    for (int r = 0; r < n; ++r)
        for (int c = r + 1; c < n; ++c) {
            double m = 0.5 * (a[n * r + c] + a[n * c + r]);
            a[n * r + c] = m;
            a[n * c + r] = m;
        }
}

__device__ double inverse3(double a[3][3], double out[3][3]) {
    double c00 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    double c01 = a[1][2] * a[2][0] - a[1][0] * a[2][2];
    double c02 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
    double det = a[0][0] * c00 + a[0][1] * c01 + a[0][2] * c02;
    double inv = 1.0 / det;
    out[0][0] = c00 * inv;
    out[1][0] = c01 * inv;
    out[2][0] = c02 * inv;
    out[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv;
    out[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv;
    out[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv;
    out[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv;
    out[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv;
    out[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv;
    return det;
}

// a (n x n, rows of lda) x = b (n x m, rows of ldb) by Gaussian elimination with partial
// pivoting, in place; returns 1 if a is singular
__device__ __noinline__ int gauss_solve(double* a, int lda, double* b, int ldb, int n, int m) {
    for (int k = 0; k < n; ++k) {
        int piv = k;
        double big = fabs(a[k * lda + k]);
        for (int i = k + 1; i < n; ++i) {
            if (fabs(a[i * lda + k]) > big) {
                big = fabs(a[i * lda + k]);
                piv = i;
            }
        }
        if (!(big > 1e-300) || !isfinite(big)) return 1;
        if (piv != k) {
            for (int j = 0; j < n; ++j) {
                double tmp = a[k * lda + j];
                a[k * lda + j] = a[piv * lda + j];
                a[piv * lda + j] = tmp;
            }
            for (int j = 0; j < m; ++j) {
                double tmp = b[k * ldb + j];
                b[k * ldb + j] = b[piv * ldb + j];
                b[piv * ldb + j] = tmp;
            }
        }
        double inv = 1.0 / a[k * lda + k];
        for (int i = k + 1; i < n; ++i) {
            double f = a[i * lda + k] * inv;
            if (f != 0.0) {
                for (int j = k + 1; j < n; ++j) a[i * lda + j] -= f * a[k * lda + j];
                for (int j = 0; j < m; ++j) b[i * ldb + j] -= f * b[k * ldb + j];
            }
        }
    }
    for (int k = n - 1; k >= 0; --k) {
        double inv = 1.0 / a[k * lda + k];
        for (int j = 0; j < m; ++j) {
            double acc = b[k * ldb + j];
            for (int i = k + 1; i < n; ++i) acc -= a[k * lda + i] * b[i * ldb + j];
            b[k * ldb + j] = acc * inv;
        }
    }
    return 0;
}

// eigenvalues w and eigenvectors (columns of v) of a symmetric matrix by Jacobi's method;
// a is overwritten
__device__ __noinline__ void eigh3(double a[3][3], double* w, double v[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) v[i][j] = i == j ? 1.0 : 0.0;
    for (int it = 0; it < 60; ++it) {
        double off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        double scale = a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2];
        if (off <= 1e-36 * scale || off == 0.0) break;
        for (int p = 0; p < 2; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                double apq = a[p][q];
                if (apq == 0.0) continue;
                double theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                double t;
                if (fabs(theta) > 1e150) {
                    t = 0.5 / theta;
                } else {
                    t = 1.0 / (fabs(theta) + sqrt(theta * theta + 1.0));
                    if (theta < 0.0) t = -t;
                }
                double c = 1.0 / sqrt(t * t + 1.0);
                double s = t * c;
                for (int k = 0; k < 3; ++k) {
                    double akp = a[k][p];
                    double akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    double apk = a[p][k];
                    double aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; ++k) {
                    double vkp = v[k][p];
                    double vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    for (int i = 0; i < 3; ++i) w[i] = a[i][i];
}

__device__ __forceinline__ double half_log(double mu) { return 0.5 * log1p(mu); }

__device__ double log_divided(double x, double y) {
    double z = (x - y) / (1.0 + y);
    if (fabs(z) < 1e-4)
        return 0.5 / (1.0 + y) * (1.0 - z * (0.5 - z * (1.0 / 3.0 - 0.25 * z)));
    return 0.5 * log1p(z) / (x - y);
}

__device__ double log_divided2(double x, double y, double z) {
    double lo = pmin(x, pmin(y, z));
    double hi = pmax(x, pmax(y, z));
    double mid = x + y + z - lo - hi;
    double m = 1.0 + (x + y + z) / 3.0;
    if (hi - lo <= 1e-4 * m) {
        double d0 = 1.0 + x - m;
        double d1 = 1.0 + y - m;
        double d2 = 1.0 + z - m;
        double h2 = 0.5 * (d0 * d0 + d1 * d1 + d2 * d2);
        double h3 = d0 * d1 * d2;
        double m2 = m * m;
        return -0.25 / m2 - h2 / (8.0 * m2 * m2) + h3 / (10.0 * m2 * m2 * m);
    }
    return (log_divided(hi, mid) - log_divided(mid, lo)) / (hi - lo);
}

// ---- von Mises: morphogen.homogenization.behaviours ----------------------------------------

__device__ __forceinline__ double ddot6(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
         + 2.0 * (a[3] * b[3] + a[4] * b[4] + a[5] * b[5]);
}

__device__ __forceinline__ void j2_hardening(const double* par, double p, double* r,
                                             double* dr) {
    double rr = par[3] * p;
    double d = par[3];
    for (int i = 0; i < MAX_ISOTROPIC; ++i) {
        double q = par[6 + 2 * i];
        if (q != 0.0) {
            double b = par[7 + 2 * i];
            double e = exp(-b * p);
            rr += q * (1.0 - e);
            d += q * b * e;
        }
    }
    *r = rr;
    *dr = d;
}

__device__ double j2_phi(const double* par, int nk, const double* s_tr, const double* old,
                         double dp, double* xi, double* nrm, double* w, double* dphi,
                         double* jm_out) {
    double mu = par[1];
    double csum = 0.0;
    double c2sum = 0.0;
    for (int r = 0; r < 6; ++r) {
        xi[r] = s_tr[r];
        w[r] = 0.0;
    }
    for (int k = 0; k < nk; ++k) {
        double c = par[KIN + 2 * k];
        double d = par[KIN + 2 * k + 1];
        double th = 1.0 / (1.0 + d * dp);
        csum += th * c;
        c2sum += th * th * c;
        int base = STATE_X + 6 * k;
        for (int r = 0; r < 6; ++r) {
            double xk = old[base + r];
            xi[r] -= th * xk;
            w[r] += d * th * th * xk;
        }
    }
    double jm = sqrt(pmax(1.5 * ddot6(xi, xi), 0.0));
    double inv = jm > 0.0 ? 1.5 / jm : 0.0;
    for (int r = 0; r < 6; ++r) nrm[r] = inv * xi[r];
    double rr, dr;
    j2_hardening(par, old[STATE_P] + dp, &rr, &dr);
    double phi = jm - (3.0 * mu + csum) * dp - rr - par[2];
    *dphi = ddot6(nrm, w) - 3.0 * mu - c2sum - dr;
    *jm_out = jm;
    return phi;
}

__device__ void elastic_tangent(double lam, double mu, double* tan) {
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c) tan[6 * r + c] = 0.0;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) tan[6 * r + c] = lam;
        tan[6 * r + r] += 2.0 * mu;
        tan[6 * (3 + r) + 3 + r] = mu;
    }
}

// implicit update of a von Mises point (_j2_update), its tangent symmetrized with sym;
// returns 1 on failure
__device__ __noinline__ int j2_update(const double* par, int nk, const double* eps,
                                      const double* eth, double dt, const double* old,
                                      double* nw, double* sig, double* tan, int sym) {
    double lam = par[0];
    double mu = par[1];
    double kv = par[4];
    double nv = par[5];
    double ee[6], s_tr[6], xi[6], nrm[6], w[6], d[6], dxi[6];
    for (int r = 0; r < 6; ++r) {
        double e = r < 3 ? eps[r] : 0.5 * eps[r];
        ee[r] = e - eth[r] - old[r];
    }
    double tr = ee[0] + ee[1] + ee[2];
    for (int r = 0; r < 6; ++r) sig[r] = 2.0 * mu * ee[r];
    for (int r = 0; r < 3; ++r) sig[r] += lam * tr;
    double mean = (sig[0] + sig[1] + sig[2]) / 3.0;
    for (int r = 0; r < 6; ++r) s_tr[r] = r < 3 ? sig[r] - mean : sig[r];
    for (int r = 0; r < WIDTH; ++r) nw[r] = old[r];
    double dphi, jm;
    double f_tr = j2_phi(par, nk, s_tr, old, 0.0, xi, nrm, w, &dphi, &jm);
    double rr0, dr0;
    j2_hardening(par, old[STATE_P], &rr0, &dr0);
    double scale = jm + par[2] + fabs(rr0) + 1e-300;
    if (f_tr <= 1e-13 * scale) {
        elastic_tangent(lam, mu, tan);
        return 0;
    }
    double tol = 1e-12 * scale;
    double dp = 0.0;
    double y = 0.0;
    bool converged = false;
    if (kv > 0.0) {
        // viscoplastic: unknown y = (dp / dt)^(1/N), the viscous stress over K
        double lo = 0.0, hi = f_tr / kv;
        y = 0.5 * hi;
        for (int it = 0; it < 200; ++it) {
            dp = dt * pow(y, nv);
            double phi = j2_phi(par, nk, s_tr, old, dp, xi, nrm, w, &dphi, &jm);
            double res = phi - kv * y;
            if (fabs(res) <= tol) {
                converged = true;
                break;
            }
            if (res > 0.0)
                lo = y;
            else
                hi = y;
            double der = dphi * nv * dt * pow(y, nv - 1.0) - kv;
            double step = der < 0.0 ? y - res / der : 0.5 * (lo + hi);
            if (!(lo < step && step < hi)) step = 0.5 * (lo + hi);
            if (fabs(step - y) <= 1e-15 * hi) {
                y = step;
                converged = true;
                break;
            }
            y = step;
        }
        dp = dt * pow(y, nv);
    } else {
        // rate-independent: unknown dp, bracketed by doubling
        double lo = 0.0, hi = f_tr / (3.0 * mu);
        for (int it = 0; it < 200; ++it) {
            double phi = j2_phi(par, nk, s_tr, old, hi, xi, nrm, w, &dphi, &jm);
            if (phi < 0.0) break;
            lo = hi;
            hi *= 2.0;
        }
        dp = lo + 0.5 * (hi - lo);
        for (int it = 0; it < 200; ++it) {
            double phi = j2_phi(par, nk, s_tr, old, dp, xi, nrm, w, &dphi, &jm);
            if (fabs(phi) <= tol) {
                converged = true;
                break;
            }
            if (phi > 0.0)
                lo = dp;
            else
                hi = dp;
            double step = dphi < 0.0 ? dp - phi / dphi : 0.5 * (lo + hi);
            if (!(lo < step && step < hi)) step = 0.5 * (lo + hi);
            if (fabs(step - dp) <= 1e-15 * hi) {
                dp = step;
                converged = true;
                break;
            }
            dp = step;
        }
    }
    if (!converged || !isfinite(dp)) return 1;
    j2_phi(par, nk, s_tr, old, dp, xi, nrm, w, &dphi, &jm);
    if (jm <= 0.0) return 1;
    // new state and stress
    for (int r = 0; r < 6; ++r) {
        nw[r] = old[r] + dp * nrm[r];
        sig[r] -= 2.0 * mu * dp * nrm[r];
    }
    nw[STATE_P] = old[STATE_P] + dp;
    for (int k = 0; k < nk; ++k) {
        double c = par[KIN + 2 * k];
        double th = 1.0 / (1.0 + par[KIN + 2 * k + 1] * dp);
        int base = STATE_X + 6 * k;
        for (int r = 0; r < 6; ++r)
            nw[base + r] = th * (old[base + r] + 2.0 / 3.0 * c * dp * nrm[r]);
    }
    // consistent tangent: d dp = 2 mu / h n : d eps
    double h = -dphi;
    if (kv > 0.0 && y > 0.0) {
        double lg = (1.0 - nv) * log(y);
        h = h + (lg < 700.0 ? kv / (nv * dt) * exp(lg) : MG_INF);
    }
    double inv_h = (h > 0.0 && isfinite(h)) ? 1.0 / h : 0.0;
    for (int j = 0; j < 6; ++j) {
        for (int r = 0; r < 6; ++r) d[r] = 0.0;
        d[j] = j < 3 ? 1.0 : 0.5;
        double trd = j < 3 ? 1.0 : 0.0;
        double ddp = 2.0 * mu * ddot6(nrm, d) * inv_h;
        for (int r = 0; r < 6; ++r) {
            double dev = r < 3 ? d[r] - trd / 3.0 : d[r];
            dxi[r] = 2.0 * mu * dev + w[r] * ddp;
        }
        double ndxi = ddot6(nrm, dxi);
        for (int r = 0; r < 6; ++r) {
            double dn = 1.5 / jm * (dxi[r] - 2.0 / 3.0 * nrm[r] * ndxi);
            double ds = 2.0 * mu * d[r] - 2.0 * mu * (ddp * nrm[r] + dp * dn);
            if (r < 3) ds += lam * trd;
            tan[6 * r + j] = ds;
        }
    }
    if (sym) symmetrize(tan, 6);
    return 0;
}

// ---- Hill's criterion: morphogen.homogenization.behaviours (_hill_update) ----------------

// a = 3/2 P M P in Mandel's notation (_hill_matrix)
__device__ void hill_matrix(const double* h, double* a) {
    for (int r = 0; r < 36; ++r) a[r] = 0.0;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            double acc = 0.0;
            for (int k = 0; k < 3; ++k) {
                double pr = (r == k ? 1.0 : 0.0) - 1.0 / 3.0;
                double pc = (c == k ? 1.0 : 0.0) - 1.0 / 3.0;
                acc += pr * h[k] * pc;
            }
            a[6 * r + c] = 1.5 * acc;
        }
    for (int r = 3; r < 6; ++r) a[6 * r + r] = 1.5 * h[r];
}

// residuals of a Hill update and their Jacobian (_hill_system); returns J_H (0 where the
// normal is not defined), with dp and d dp / d q
__device__ double hill_system(const double* par, int nk, const double* a, const double* s_tr,
                              const double* xn, double p0, const double* de, double q,
                              double dt, double* res, double* jac, double* sr, double* nrm,
                              double* w, double* nn, double* dp_out, double* ddp_out) {
    double lam = par[0];
    double mu = par[1];
    double kv = par[4];
    double nv = par[5];
    double dp, ddp;
    if (kv > 0.0) {
        dp = dt * pow(q, nv);
        ddp = q > 0.0 ? nv * dt * pow(q, nv - 1.0) : (nv == 1.0 ? dt : 0.0);
    } else {
        dp = q;
        ddp = 1.0;
    }
    *dp_out = dp;
    *ddp_out = ddp;
    double cb = 0.0;
    double cdb = 0.0;
    for (int r = 0; r < 6; ++r) {
        sr[r] = s_tr[r];
        w[r] = 0.0;
    }
    for (int k = 0; k < nk; ++k) {
        double c = par[KIN + 2 * k];
        double d = par[KIN + 2 * k + 1];
        double th = 1.0 / (1.0 + d * dp);
        cb += 2.0 / 3.0 * th * c;
        cdb += 2.0 / 3.0 * c * d * th * th;
        for (int r = 0; r < 6; ++r) {
            sr[r] -= th * xn[6 * k + r];
            w[r] += d * th * th * xn[6 * k + r];
        }
    }
    double tr = de[0] + de[1] + de[2];
    for (int r = 0; r < 6; ++r) {
        double b = (2.0 * mu + cb) * de[r] + (r < 3 ? lam * tr : 0.0);
        sr[r] -= b;
        w[r] += cdb * de[r];
    }
    double jj = 0.0;
    for (int r = 0; r < 6; ++r) {
        double acc = 0.0;
        for (int c = 0; c < 6; ++c) acc += a[6 * r + c] * sr[c];
        nrm[r] = acc;
        jj += acc * sr[r];
    }
    if (!(jj > 0.0)) return 0.0;
    double jh = sqrt(jj);
    for (int r = 0; r < 6; ++r) nrm[r] /= jh;
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c) nn[6 * r + c] = (a[6 * r + c] - nrm[r] * nrm[c]) / jh;
    double rr, dr;
    j2_hardening(par, p0 + dp, &rr, &dr);
    for (int r = 0; r < 6; ++r) res[r] = de[r] - dp * nrm[r];
    res[6] = jh - rr - par[2] - (kv > 0.0 ? kv * q : 0.0);
    for (int r = 0; r < 6; ++r) {
        double rows = 0.0;
        for (int c = 0; c < 3; ++c) rows += nn[6 * r + c];
        for (int c = 0; c < 6; ++c)
            jac[7 * r + c] = dp * ((2.0 * mu + cb) * nn[6 * r + c] + (c < 3 ? lam * rows : 0.0));
        jac[7 * r + r] += 1.0;
        double acc = 0.0;
        for (int c = 0; c < 6; ++c) acc += nn[6 * r + c] * w[c];
        jac[7 * r + 6] = (-nrm[r] - dp * acc) * ddp;
    }
    double ntr = nrm[0] + nrm[1] + nrm[2];
    double nw = 0.0;
    for (int c = 0; c < 6; ++c) {
        jac[7 * 6 + c] = -((2.0 * mu + cb) * nrm[c] + (c < 3 ? lam * ntr : 0.0));
        nw += nrm[c] * w[c];
    }
    jac[7 * 6 + 6] = (nw - dr) * ddp - (kv > 0.0 ? kv : 0.0);
    return jh;
}

// a x = b for the m columns of b (n x m, row by row), Gauss's elimination with partial
// pivoting (_solve_in_place); 0 if a is singular
__device__ int hill_solve(double* a, double* b, int n, int m) {
    for (int k = 0; k < n; ++k) {
        int piv = k;
        for (int r = k + 1; r < n; ++r)
            if (fabs(a[n * r + k]) > fabs(a[n * piv + k])) piv = r;
        if (!(fabs(a[n * piv + k]) > 0.0)) return 0;
        if (piv != k) {
            for (int c = 0; c < n; ++c) {
                double t = a[n * k + c];
                a[n * k + c] = a[n * piv + c];
                a[n * piv + c] = t;
            }
            for (int c = 0; c < m; ++c) {
                double t = b[m * k + c];
                b[m * k + c] = b[m * piv + c];
                b[m * piv + c] = t;
            }
        }
        for (int r = k + 1; r < n; ++r) {
            double f = a[n * r + k] / a[n * k + k];
            if (f != 0.0) {
                for (int c = k; c < n; ++c) a[n * r + c] -= f * a[n * k + c];
                for (int c = 0; c < m; ++c) b[m * r + c] -= f * b[m * k + c];
            }
        }
    }
    for (int k = n - 1; k >= 0; --k)
        for (int c = 0; c < m; ++c) {
            double acc = b[m * k + c];
            for (int j = k + 1; j < n; ++j) acc -= a[n * k + j] * b[m * j + c];
            b[m * k + c] = acc / a[n * k + k];
        }
    return 1;
}

// implicit update of a point of Hill's criterion (_hill_update), its tangent symmetrized
// with sym; returns 1 on failure
__device__ __noinline__ int hill_update(const double* par, int nk, const double* eps,
                                        const double* eth, double dt, const double* old,
                                        double* nw, double* sig, double* tan, int sym) {
    double lam = par[0];
    double mu = par[1];
    double kv = par[4];
    double nv = par[5];
    const double sq2 = sqrt(2.0);
    double f[6], h[6], a[36], s_tr[6], ee[6], xn[6 * 4];
    for (int r = 0; r < 6; ++r) {
        f[r] = r < 3 ? 1.0 : sq2;
        h[r] = par[N_PARAMS + r];
    }
    hill_matrix(h, a);
    for (int r = 0; r < 6; ++r) {
        double e = r < 3 ? eps[r] : 0.5 * eps[r];
        ee[r] = e - eth[r] - old[r];
    }
    double tr = ee[0] + ee[1] + ee[2];
    for (int r = 0; r < 6; ++r) s_tr[r] = f[r] * (2.0 * mu * ee[r] + (r < 3 ? lam * tr : 0.0));
    for (int k = 0; k < nk; ++k)
        for (int r = 0; r < 6; ++r) xn[6 * k + r] = f[r] * old[STATE_X + 6 * k + r];
    for (int r = 0; r < WIDTH; ++r) nw[r] = old[r];
    double p0 = old[STATE_P];
    double res[7], jac[49], work[49], sr[6], nrm[6], w[6], nn[36], de[6], trial[6], step[7];
    for (int r = 0; r < 6; ++r) de[r] = 0.0;
    double dp, ddp;
    double jh = hill_system(par, nk, a, s_tr, xn, p0, de, 0.0, dt, res, jac, sr, nrm, w, nn,
                            &dp, &ddp);
    double f_tr = jh > 0.0 ? res[6] : -1.0;
    double rr0, dr0;
    j2_hardening(par, p0, &rr0, &dr0);
    double scale = jh + par[2] + fabs(rr0) + 1e-300;
    if (f_tr <= 1e-13 * scale) {
        elastic_tangent(lam, mu, tan);
        for (int r = 0; r < 6; ++r) sig[r] = s_tr[r] / f[r];
        return 0;
    }
    // first guess: the plastic increment of the elastic predictor along its normal
    double stiff = 0.0;
    double ntr = nrm[0] + nrm[1] + nrm[2];
    for (int r = 0; r < 6; ++r) stiff += nrm[r] * (2.0 * mu * nrm[r] + (r < 3 ? lam * ntr : 0.0));
    double nn2 = 0.0;
    for (int r = 0; r < 6; ++r) nn2 += nrm[r] * nrm[r];
    for (int k = 0; k < nk; ++k) stiff += 2.0 / 3.0 * par[KIN + 2 * k] * nn2;
    dp = f_tr / (stiff + pmax(dr0, 0.0));
    double q = kv > 0.0 ? pmin(pow(dp / dt, 1.0 / nv), f_tr / kv) : dp;
    if (kv > 0.0) dp = dt * pow(q, nv);
    for (int r = 0; r < 6; ++r) de[r] = dp * nrm[r];
    double e_scale = f_tr / (3.0 * mu) + 1e-300;
    jh = hill_system(par, nk, a, s_tr, xn, p0, de, q, dt, res, jac, sr, nrm, w, nn, &dp, &ddp);
    if (jh <= 0.0) return 1;
    int converged = 0;
    for (int it = 0; it < HILL_IT; ++it) {
        double merit = 0.0;
        for (int r = 0; r < 6; ++r) merit += (res[r] / e_scale) * (res[r] / e_scale);
        merit += (res[6] / scale) * (res[6] / scale);
        double big = 0.0;
        for (int r = 0; r < 6; ++r) big = pmax(big, fabs(res[r]));
        if (big <= 1e-12 * e_scale && fabs(res[6]) <= 1e-12 * scale) {
            converged = 1;
            break;
        }
        for (int r = 0; r < 7; ++r) {
            step[r] = -res[r];
            for (int c = 0; c < 7; ++c) work[7 * r + c] = jac[7 * r + c];
        }
        if (!hill_solve(work, step, 7, 1)) return 1;
        // backtracking on the scaled residuals: the system of the accepted point is that of
        // the next iteration
        double alpha = 1.0;
        int accepted = 0;
        double qt = q;
        double dpt = dp;
        for (int ls = 0; ls < 16; ++ls) {
            qt = q + alpha * step[6];
            if (qt < 0.0) {
                alpha *= 0.5;
                continue;
            }
            for (int r = 0; r < 6; ++r) trial[r] = de[r] + alpha * step[r];
            double ddpt;
            double jt = hill_system(par, nk, a, s_tr, xn, p0, trial, qt, dt, res, jac, sr, nrm, w,
                                    nn, &dpt, &ddpt);
            if (jt > 0.0) {
                double m = 0.0;
                for (int r = 0; r < 6; ++r) m += (res[r] / e_scale) * (res[r] / e_scale);
                m += (res[6] / scale) * (res[6] / scale);
                if (m <= (1.0 - 1e-4 * alpha) * merit) {
                    accepted = 1;
                    break;
                }
            }
            alpha *= 0.5;
        }
        if (!accepted) return 1;
        q = qt;
        dp = dpt;
        for (int r = 0; r < 6; ++r) de[r] = trial[r];
    }
    if (!converged || !isfinite(dp)) return 1;
    // new state and stress (tensor components)
    tr = de[0] + de[1] + de[2];
    for (int r = 0; r < 6; ++r) {
        nw[r] = old[r] + de[r] / f[r];
        sig[r] = (s_tr[r] - (2.0 * mu * de[r] + (r < 3 ? lam * tr : 0.0))) / f[r];
    }
    nw[STATE_P] = p0 + dp;
    for (int k = 0; k < nk; ++k) {
        double th = 1.0 / (1.0 + par[KIN + 2 * k + 1] * dp);
        double c = par[KIN + 2 * k];
        for (int r = 0; r < 6; ++r)
            nw[STATE_X + 6 * k + r] = th * (xn[6 * k + r] + 2.0 / 3.0 * c * de[r]) / f[r];
    }
    // consistent tangent: jac d(de, q) = -d res / d eps_m, d sigma = C (d eps_m - d de)
    double cm[36], rhs[42];
    for (int r = 0; r < 36; ++r) cm[r] = 0.0;
    for (int r = 0; r < 6; ++r) cm[6 * r + r] = 2.0 * mu;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) cm[6 * r + c] += lam;
    for (int c = 0; c < 6; ++c) {
        for (int r = 0; r < 6; ++r) {
            double acc = 0.0;
            for (int j = 0; j < 6; ++j) acc += nn[6 * r + j] * cm[6 * j + c];
            rhs[6 * r + c] = dp * acc;
        }
        double acc = 0.0;
        for (int j = 0; j < 6; ++j) acc += nrm[j] * cm[6 * j + c];
        rhs[6 * 6 + c] = -acc;
    }
    for (int r = 0; r < 49; ++r) work[r] = jac[r];
    if (!hill_solve(work, rhs, 7, 6)) return 1;
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c) {
            double acc = cm[6 * r + c];
            for (int j = 0; j < 6; ++j) acc -= cm[6 * r + j] * rhs[6 * j + c];
            tan[6 * r + c] = acc / (f[r] * f[c]);
        }
    if (sym) symmetrize(tan, 6);
    return 0;
}

// ---- crystal plasticity: morphogen.homogenization.crystal ---------------------------------

// The work arrays of crystal.work_arrays: wv (res, tau, fv, sgn, tau_tr, trial, res_t,
// dg, act, fv_t, the exponentials of the hardening), wm (jac, lu, dtau, amat), wr, w3, wg,
// w6 (m, C : m), s6, idx and the internal variables of two substeps.
struct Work {
    double wv[11][NS];
    double wm[4][NS][NS];
    double wr[NS][9];
    double w3[24][3][3];
    double wg[3][NS][3][3];
    double w6[2][NS][6];
    double s6[6][9];
    int idx[NS];
    double bufs[2][WIDTH];
};

// Parameters of a slip system s: par[N_CP * s + (tau0, Q, b, C, D, K, n)]; interaction
// matrix h[NS * s + r]; slip systems sys[6 * s + (l, n)]

__device__ void tau_small(int ns, const double* dg, const double* tau_tr, double amat[NS][NS],
                          double* tau, double dtau[NS][NS], bool with_jac) {
    for (int s = 0; s < ns; ++s) {
        double t = tau_tr[s];
        for (int r = 0; r < ns; ++r) t -= amat[s][r] * dg[r];
        tau[s] = t;
    }
    if (with_jac)
        for (int s = 0; s < ns; ++s)
            for (int r = 0; r < ns; ++r) dtau[s][r] = -amat[s][r];
}

__device__ __noinline__ void tau_finite(int ns, const double* dg, const double* sys,
                                        double fe_tr[3][3], const double* dm,
                                        double eth[3][3], double* tau, double dtau[NS][NS],
                                        bool with_jac, double w3[24][3][3],
                                        double wg[3][NS][3][3], double s6[6][9]) {
    double(*gam)[3] = w3[0];
    double(*g)[3] = w3[1];
    double(*fe)[3] = w3[2];
    double(*ce)[3] = w3[3];
    double(*se)[3] = w3[4];
    double(*mm)[3] = w3[5];
    double(*t1)[3] = w3[6];
    double(*t2)[3] = w3[7];
    double(*ee)[3] = w3[8];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) gam[i][j] = 0.0;
    for (int r = 0; r < ns; ++r) {
        double d = dg[r];
        if (d != 0.0)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) gam[i][j] += d * sys[6 * r + i] * sys[6 * r + 3 + j];
    }
    matmul3(gam, gam, t1);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) g[i][j] = (i == j ? 1.0 : 0.0) - gam[i][j] + 0.5 * t1[i][j];
    matmul3(fe_tr, g, fe);
    tmatmul3(fe, fe, ce);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            ee[i][j] = 0.5 * (ce[i][j] - (i == j ? 1.0 : 0.0)) - eth[i][j];
    elastic_stress(dm, ee, se, s6[0], s6[1]);
    matmul3(ce, se, mm);
    for (int s = 0; s < ns; ++s) tau[s] = project(sys + 6 * s, sys + 6 * s + 3, mm);
    if (!with_jac) return;
    double* av = s6[2];
    for (int r = 0; r < ns; ++r) {
        double(*dgr)[3] = wg[0][r];
        double(*dce)[3] = wg[1][r];
        double(*dsr)[3] = wg[2][r];
        const double* lr = sys + 6 * r;
        const double* nr = sys + 6 * r + 3;
        for (int i = 0; i < 3; ++i) {
            av[i] = gam[0][i] * nr[0] + gam[1][i] * nr[1] + gam[2][i] * nr[2];
            av[3 + i] = gam[i][0] * lr[0] + gam[i][1] * lr[1] + gam[i][2] * lr[2];
        }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                dgr[i][j] = -lr[i] * nr[j] + 0.5 * (lr[i] * av[j] + av[3 + i] * nr[j]);
        matmul3(fe_tr, dgr, t1);
        tmatmul3(t1, fe, t2);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                dce[i][j] = t2[i][j] + t2[j][i];
                ee[i][j] = 0.5 * dce[i][j];
            }
        elastic_stress(dm, ee, dsr, s6[0], s6[1]);
        matmul3(dce, se, t1);
        matmul3(ce, dsr, t2);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) t1[i][j] += t2[i][j];
        for (int s = 0; s < ns; ++s) dtau[s][r] = project(sys + 6 * s, sys + 6 * s + 3, t1);
    }
}

// resolved shear stresses, overstresses fv, residuals of the active systems and their
// Jacobian (_cp_eval); returns the largest |R_s|
__device__ __noinline__ double cp_eval(bool finite, const double* par, const double* h,
                                       int ns, double dt, const double* alpha0,
                                       const double* v0, const double* dg, const double* act,
                                       const double* sgn, const double* sys, const double* dm,
                                       double fe_tr[3][3], double eth[3][3], Work* wk,
                                       double* res, double* fv, bool with_jac) {
    double* tau = wk->wv[1];
    double(*dtau)[NS] = wk->wm[2];
    double* ex = wk->wv[10];  // exp(-b v_r) of the systems r for the b of the last family
    double b_ex = 0.0;
    int have = 0;
    if (finite)
        tau_finite(ns, dg, sys, fe_tr, dm, eth, tau, dtau, with_jac, wk->w3, wk->wg, wk->s6);
    else
        tau_small(ns, dg, wk->wv[4], wk->wm[3], tau, dtau, with_jac);
    double worst = 0.0;
    for (int s = 0; s < ns; ++s) {
        const double* ps = par + N_CP * s;
        double tau0 = ps[0], q = ps[1], b = ps[2], c = ps[3], d = ps[4], k = ps[5], n = ps[6];
        double mag = fabs(dg[s]);
        double over = tau[s] - c * (alpha0[s] + dg[s]) / (1.0 + d * mag);
        double iso = tau0;
        if (q != 0.0) {
            if (!have || b != b_ex) {  // the systems of a family share their b
                for (int r = 0; r < ns; ++r) ex[r] = exp(-b * (v0[r] + fabs(dg[r])));
                b_ex = b;
                have = 1;
            }
            double acc = 0.0;
            for (int r = 0; r < ns; ++r) acc += h[NS * s + r] * (1.0 - ex[r]);
            iso += q * acc;
        }
        fv[s] = fabs(over) - iso;
        if (act[s] > 0.5) {
            res[s] = sgn[s] * over - iso - k * pow(mag / dt, 1.0 / n);
            worst = pmax(worst, fabs(res[s]));
        } else {
            res[s] = 0.0;
        }
    }
    if (!isfinite(worst)) return MG_INF;
    if (with_jac) {
        double(*jac)[NS] = wk->wm[0];
        for (int s = 0; s < ns; ++s) {
            if (act[s] < 0.5) continue;
            const double* ps = par + N_CP * s;
            double q = ps[1], b = ps[2], c = ps[3], d = ps[4], k = ps[5], n = ps[6];
            double mag = fabs(dg[s]);
            if (q != 0.0 && (!have || b != b_ex)) {
                for (int r = 0; r < ns; ++r) ex[r] = exp(-b * (v0[r] + fabs(dg[r])));
                b_ex = b;
                have = 1;
            }
            for (int r = 0; r < ns; ++r) {
                if (act[r] < 0.5) continue;
                double v = sgn[s] * sgn[r] * dtau[s][r];
                if (q != 0.0) v -= q * h[NS * s + r] * b * ex[r];
                if (r == s) {
                    double den = 1.0 + d * mag;
                    v -= c * (1.0 - d * sgn[s] * alpha0[s]) / (den * den);
                    v -= k / (n * dt) * pow(mag / dt, 1.0 / n - 1.0);
                }
                jac[s][r] = v;
            }
        }
    }
    return worst;
}

__device__ int active_indices(const double* act, int ns, int* idx) {
    int na = 0;
    for (int s = 0; s < ns; ++s)
        if (act[s] > 0.5) idx[na++] = s;
    return na;
}

// Newton's method on the magnitudes of the slips of the active systems (_active_newton)
__device__ __noinline__ int active_newton(bool finite, const double* par, const double* h,
                                          int ns, double dt, const double* alpha0,
                                          const double* v0, const double* sys,
                                          const double* dm, double fe_tr[3][3],
                                          double eth[3][3], Work* wk, double tol) {
    double* res = wk->wv[0];
    double* fv = wk->wv[2];
    double* sgn = wk->wv[3];
    double* trial = wk->wv[5];
    double* res_t = wk->wv[6];
    double* dg = wk->wv[7];
    double* act = wk->wv[8];
    double* fv_t = wk->wv[9];
    double(*jac)[NS] = wk->wm[0];
    double(*lu)[NS] = wk->wm[1];
    double(*wr)[9] = wk->wr;
    int* idx = wk->idx;
    for (int it = 0; it < MAX_LOCAL; ++it) {
        double worst = cp_eval(finite, par, h, ns, dt, alpha0, v0, dg, act, sgn, sys, dm, fe_tr,
                               eth, wk, res, fv, true);
        if (!isfinite(worst)) return 1;
        if (worst <= tol) return 0;
        bool dropped = false;
        for (int s = 0; s < ns; ++s) {
            if (act[s] > 0.5 && fabs(dg[s]) <= SLIP_FLOOR && res[s] < 0.0) {
                act[s] = 0.0;
                dg[s] = 0.0;
                dropped = true;
            }
        }
        if (dropped) continue;
        int na = active_indices(act, ns, idx);
        for (int a = 0; a < na; ++a) {
            for (int c = 0; c < na; ++c) lu[a][c] = jac[idx[a]][idx[c]];
            wr[a][0] = -res[idx[a]];
        }
        if (gauss_solve(&lu[0][0], NS, &wr[0][0], 9, na, 1) != 0) return 1;
        double norm0 = 0.0;
        for (int a = 0; a < na; ++a) norm0 += res[idx[a]] * res[idx[a]];
        double t = 1.0;
        bool accepted = false;
        for (int ls = 0; ls < 30; ++ls) {
            for (int s = 0; s < ns; ++s) trial[s] = dg[s];
            for (int a = 0; a < na; ++a) {
                int s = idx[a];
                double mag = fabs(dg[s]) + t * wr[a][0];
                trial[s] = sgn[s] * pmax(pmax(mag, SHRINK * fabs(dg[s])), SLIP_FLOOR);
            }
            double w = cp_eval(finite, par, h, ns, dt, alpha0, v0, trial, act, sgn, sys, dm,
                               fe_tr, eth, wk, res_t, fv_t, false);
            double norm1 = 0.0;
            for (int a = 0; a < na; ++a) norm1 += res_t[idx[a]] * res_t[idx[a]];
            if (isfinite(w) && norm1 <= (1.0 - 1e-4 * t) * norm0) {
                accepted = true;
                break;
            }
            t *= 0.5;
        }
        if (!accepted) {
            // stagnation at the level of rounding errors
            cp_eval(finite, par, h, ns, dt, alpha0, v0, dg, act, sgn, sys, dm, fe_tr, eth, wk,
                    res, fv, true);
            return worst <= 1e3 * tol ? 0 : 1;
        }
        for (int s = 0; s < ns; ++s) dg[s] = trial[s];
    }
    return 1;
}

// slips of the step by the active set strategy (_cp_solve)
__device__ __noinline__ int cp_solve(bool finite, const double* par, const double* h, int ns,
                                     double dt, const double* alpha0, const double* v0,
                                     const double* sys, const double* dm, double fe_tr[3][3],
                                     double eth[3][3], Work* wk) {
    double* res = wk->wv[0];
    double* tau = wk->wv[1];
    double* fv = wk->wv[2];
    double* sgn = wk->wv[3];
    double* dg = wk->wv[7];
    double* act = wk->wv[8];
    for (int s = 0; s < ns; ++s) {
        if (fabs(dg[s]) > SLIP_FLOOR) {
            act[s] = 1.0;
            sgn[s] = dg[s] > 0.0 ? 1.0 : -1.0;
        } else {
            act[s] = 0.0;
            dg[s] = 0.0;
        }
    }
    double worst = cp_eval(finite, par, h, ns, dt, alpha0, v0, dg, act, sgn, sys, dm, fe_tr,
                           eth, wk, res, fv, true);
    if (!isfinite(worst)) return 1;
    double scale = 0.0;
    for (int s = 0; s < ns; ++s)
        scale = pmax(scale, fabs(tau[s]) + par[N_CP * s] + par[N_CP * s + 5]);
    double tol = LOCAL_TOL * scale;
    for (int round = 0; round < MAX_ROUNDS; ++round) {
        if (active_newton(finite, par, h, ns, dt, alpha0, v0, sys, dm, fe_tr, eth, wk, tol))
            return 1;
        bool changed = false;
        for (int s = 0; s < ns; ++s) {
            if (act[s] > 0.5) continue;
            double c = par[N_CP * s + 3];
            double k = par[N_CP * s + 5];
            double n = par[N_CP * s + 6];
            if (fv[s] <= k * pow(SLIP_FLOOR / dt, 1.0 / n) + tol) continue;
            // a new active system, from below: a slip that relaxes a tenth of its
            // overstress at most, with a viscous stress half of it at most
            act[s] = 1.0;
            sgn[s] = tau[s] - c * alpha0[s] >= 0.0 ? 1.0 : -1.0;
            double relax = fabs(wk->wm[2][s][s]) + c;
            double d0 = pmin(dt * pow(0.5 * fv[s] / k, n), 0.1 * fv[s] / pmax(relax, 1e-300));
            dg[s] = sgn[s] * pmax(d0, SLIP_FLOOR);
            changed = true;
        }
        if (!changed) return 0;
    }
    return 1;
}

__device__ void initial_slips(const double* old, const double* nw, const double* par, int ns,
                              double* dg, int guess, double ratio) {
    for (int s = 0; s < ns; ++s) dg[s] = 0.0;
    if (guess == 0) return;
    const int a = CP_ALPHA;
    for (int s = 0; s < ns; ++s) {
        double d = par[N_CP * s + 4];
        double val;
        if (guess == 1) {
            double mag = nw[a + ns + s] - old[a + ns + s];
            val = nw[a + s] * (1.0 + d * mag) - old[a + s];
        } else {
            double mag = old[a + ns + s] - nw[a + ns + s];
            val = ratio * (old[a + s] * (1.0 + d * mag) - nw[a + s]);
        }
        if (isfinite(val)) dg[s] = val;
    }
}

__device__ int solve_slips(bool finite, const double* par, const double* h, int ns, double dt,
                           const double* old, const double* nw, const double* sys,
                           const double* dm, double fe_tr[3][3], double eth[3][3], int guess,
                           double ratio, Work* wk) {
    double* dg = wk->wv[7];
    const double* alpha0 = old + CP_ALPHA;
    const double* v0 = old + CP_ALPHA + ns;
    initial_slips(old, nw, par, ns, dg, guess, ratio);
    if (cp_solve(finite, par, h, ns, dt, alpha0, v0, sys, dm, fe_tr, eth, wk) == 0) return 0;
    if (guess == 0) return 1;
    initial_slips(old, nw, par, ns, dg, 0, 1.0);
    return cp_solve(finite, par, h, ns, dt, alpha0, v0, sys, dm, fe_tr, eth, wk);
}

__device__ void update_hardening(const double* par, int ns, const double* dg,
                                 const double* old, double* nw) {
    for (int s = 0; s < ns; ++s) {
        double d = par[N_CP * s + 4];
        nw[CP_ALPHA + s] = (old[CP_ALPHA + s] + dg[s]) / (1.0 + d * fabs(dg[s]));
        nw[CP_ALPHA + ns + s] = old[CP_ALPHA + ns + s] + fabs(dg[s]);
    }
}

__device__ void schmid_small(const double* sys, int ns, const double* dm, double m[NS][6],
                             double cm[NS][6], double* me) {
    for (int s = 0; s < ns; ++s) {
        double l0 = sys[6 * s], l1 = sys[6 * s + 1], l2 = sys[6 * s + 2];
        double n0 = sys[6 * s + 3], n1 = sys[6 * s + 4], n2 = sys[6 * s + 5];
        m[s][0] = l0 * n0;
        m[s][1] = l1 * n1;
        m[s][2] = l2 * n2;
        m[s][3] = 0.5 * (l1 * n2 + l2 * n1);
        m[s][4] = 0.5 * (l2 * n0 + l0 * n2);
        m[s][5] = 0.5 * (l0 * n1 + l1 * n0);
        for (int r = 0; r < 6; ++r) me[r] = r < 3 ? m[s][r] : 2.0 * m[s][r];
        for (int r = 0; r < 6; ++r) {
            double acc = 0.0;
            for (int c = 0; c < 6; ++c) acc += dm[6 * r + c] * me[c];
            cm[s][r] = acc;
        }
    }
}

// backward Euler step of a small-strain point (_small_step)
__device__ __noinline__ int small_step(const double* sys, const double* par, const double* h,
                                       int ns, const double* dm, const double* emech, double dt,
                                       const double* old, double* nw, double* sig, int guess,
                                       double ratio, Work* wk) {
    double(*m)[6] = wk->w6[0];
    double(*cm)[6] = wk->w6[1];
    double* tau_tr = wk->wv[4];
    double(*amat)[NS] = wk->wm[3];
    double* ee = wk->s6[0];
    for (int r = 0; r < 6; ++r) ee[r] = emech[r] - old[r] * (r < 3 ? 1.0 : 2.0);
    for (int r = 0; r < 6; ++r) {
        double acc = 0.0;
        for (int c = 0; c < 6; ++c) acc += dm[6 * r + c] * ee[c];
        sig[r] = acc;
    }
    for (int s = 0; s < ns; ++s) {
        tau_tr[s] = ddot6(m[s], sig);
        for (int r = 0; r < ns; ++r) amat[s][r] = ddot6(m[s], cm[r]);
    }
    if (solve_slips(false, par, h, ns, dt, old, nw, sys, dm, wk->w3[0], wk->w3[0], guess, ratio,
                    wk))
        return 1;
    for (int r = 0; r < WIDTH; ++r) nw[r] = old[r];
    double* dg = wk->wv[7];
    double* dep = wk->s6[1];
    for (int r = 0; r < 6; ++r) dep[r] = 0.0;
    for (int s = 0; s < ns; ++s)
        for (int r = 0; r < 6; ++r) {
            dep[r] += dg[s] * m[s][r];
            sig[r] -= dg[s] * cm[s][r];
        }
    for (int r = 0; r < 6; ++r) nw[r] = old[r] + dep[r];
    nw[6] = old[6] + sqrt(pmax(2.0 / 3.0 * ddot6(dep, dep), 0.0));
    update_hardening(par, ns, dg, old, nw);
    return 0;
}

// implicit update of a crystal point in small strains (cp_small), its tangent symmetrized
// with sym
__device__ __noinline__ int cp_small(const double* sys, const double* par, const double* h,
                                     int ns, const double* dm, const double* eps,
                                     const double* eth, double dt, const double* old,
                                     double* nw, double* sig, double* tan, int guess,
                                     double ratio, Work* wk, int sym) {
    double* sgn = wk->wv[3];
    double(*jac)[NS] = wk->wm[0];
    double(*lu)[NS] = wk->wm[1];
    double(*wr)[9] = wk->wr;
    double(*m)[6] = wk->w6[0];
    double(*cm)[6] = wk->w6[1];
    schmid_small(sys, ns, dm, m, cm, wk->s6[3]);
    double* emech = wk->s6[4];
    for (int r = 0; r < 6; ++r) emech[r] = eps[r] - eth[r] * (r < 3 ? 1.0 : 2.0);
    const int slot = CP_ALPHA + 2 * ns;
    if (small_step(sys, par, h, ns, dm, emech, dt, old, nw, sig, guess, ratio, wk) != 0) {
        double* ek = wk->s6[5];
        bool done = false;
        for (int level = 0; level < SUBSTEP_LEVELS; ++level) {
            int count = 1 << (2 * (level + 1));
            double* cur = wk->bufs[0];
            double* nxt = wk->bufs[1];
            for (int r = 0; r < WIDTH; ++r) cur[r] = old[r];
            bool ok = true;
            for (int k = 1; k <= count; ++k) {
                for (int r = 0; r < 6; ++r)
                    ek[r] = old[slot + r] + (double)k / count * (emech[r] - old[slot + r]);
                if (small_step(sys, par, h, ns, dm, ek, dt / count, cur, nxt, sig,
                               k > 1 ? 2 : 0, 1.0, wk)) {
                    ok = false;
                    break;
                }
                double* tmp = cur;
                cur = nxt;
                nxt = tmp;
            }
            if (ok) {
                for (int r = 0; r < WIDTH; ++r) nw[r] = cur[r];
                done = true;
                break;
            }
        }
        if (!done) return 1;
    }
    for (int r = 0; r < 6; ++r) nw[slot + r] = emech[r];
    // consistent tangent (of the last substep)
    int* idx = wk->idx;
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c) tan[6 * r + c] = dm[6 * r + c];
    int na = active_indices(wk->wv[8], ns, idx);
    if (na == 0) return 0;
    for (int a = 0; a < na; ++a) {
        int s = idx[a];
        for (int b = 0; b < na; ++b) lu[a][b] = jac[s][idx[b]];
        for (int j = 0; j < 6; ++j) wr[a][j] = -sgn[s] * cm[s][j];
    }
    if (gauss_solve(&lu[0][0], NS, &wr[0][0], 9, na, 6) != 0) return 1;
    for (int j = 0; j < 6; ++j)
        for (int r = 0; r < 6; ++r) {
            double acc = 0.0;
            for (int a = 0; a < na; ++a) {
                int s = idx[a];
                acc += sgn[s] * cm[s][r] * wr[a][j];
            }
            tan[6 * r + j] -= acc;
        }
    if (sym) symmetrize(tan, 6);
    return 0;
}

// backward Euler step of a finite-strain point (_finite_step)
__device__ __noinline__ int finite_step(const double* sys, const double* par, const double* h,
                                        int ns, const double* dm, const double* hg,
                                        double eth[3][3], double dt, const double* old,
                                        double* nw, int guess, double ratio, Work* wk) {
    double(*f)[3] = wk->w3[9];
    double(*fpn)[3] = wk->w3[10];
    double(*fe_tr)[3] = wk->w3[11];
    double(*fpi)[3] = wk->w3[13];
    double(*t1)[3] = wk->w3[15];
    double(*t2)[3] = wk->w3[16];
    const int off = CP_ALPHA + 2 * ns;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            f[i][j] = (i == j ? 1.0 : 0.0) + hg[3 * i + j];
            fpn[i][j] = (i == j ? 1.0 : 0.0) + old[off + 3 * i + j];
        }
    matmul3(f, fpn, fe_tr);
    if (solve_slips(true, par, h, ns, dt, old, nw, sys, dm, fe_tr, eth, guess, ratio, wk))
        return 1;
    for (int r = 0; r < WIDTH; ++r) nw[r] = old[r];
    double(*gam)[3] = wk->w3[0];
    matmul3(fpn, wk->w3[1], fpi);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) nw[off + 3 * i + j] = fpi[i][j] - (i == j ? 1.0 : 0.0);
    inverse3(fpi, t1);
    tmatmul3(t1, t1, t2);
    nw[0] = 0.5 * (t2[0][0] - 1.0);
    nw[1] = 0.5 * (t2[1][1] - 1.0);
    nw[2] = 0.5 * (t2[2][2] - 1.0);
    nw[3] = 0.5 * t2[1][2];
    nw[4] = 0.5 * t2[2][0];
    nw[5] = 0.5 * t2[0][1];
    double acc = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double sym = 0.5 * (gam[i][j] + gam[j][i]);
            acc += sym * sym;
        }
    nw[6] = old[6] + sqrt(2.0 / 3.0 * acc);
    update_hardening(par, ns, wk->wv[7], old, nw);
    return 0;
}

// implicit update of a crystal point in finite strains (cp_finite): first Piola-Kirchhoff
// stress and tangent dP/dF
__device__ __noinline__ int cp_finite(const double* sys, const double* par, const double* h,
                                      int ns, const double* dm, const double* hg,
                                      const double* eth6, double dt, const double* old,
                                      double* nw, double* pk1, double* tan, int guess,
                                      double ratio, Work* wk) {
    double* sgn = wk->wv[3];
    double(*wr)[9] = wk->wr;
    double(*eth)[3] = wk->w3[12];
    stress_tensor(eth6, eth);
    const int hslot = CP_ALPHA + 2 * ns + 9;
    if (finite_step(sys, par, h, ns, dm, hg, eth, dt, old, nw, guess, ratio, wk) != 0) {
        double* hk = wk->s6[5];
        bool done = false;
        for (int level = 0; level < SUBSTEP_LEVELS; ++level) {
            int count = 1 << (2 * (level + 1));
            double* cur = wk->bufs[0];
            double* nxt = wk->bufs[1];
            for (int r = 0; r < WIDTH; ++r) cur[r] = old[r];
            bool ok = true;
            for (int k = 1; k <= count; ++k) {
                for (int r = 0; r < 9; ++r)
                    hk[r] = old[hslot + r] + (double)k / count * (hg[r] - old[hslot + r]);
                if (finite_step(sys, par, h, ns, dm, hk, eth, dt / count, cur, nxt,
                                k > 1 ? 2 : 0, 1.0, wk)) {
                    ok = false;
                    break;
                }
                double* tmp = cur;
                cur = nxt;
                nxt = tmp;
            }
            if (ok) {
                for (int r = 0; r < WIDTH; ++r) nw[r] = cur[r];
                done = true;
                break;
            }
        }
        if (!done) return 1;
    }
    for (int r = 0; r < 9; ++r) nw[hslot + r] = hg[r];
    // stress and tangent of the (last sub)step
    double(*f)[3] = wk->w3[9];
    double(*fpn)[3] = wk->w3[10];
    double(*fpi)[3] = wk->w3[13];
    double(*s2)[3] = wk->w3[14];
    double(*t1)[3] = wk->w3[15];
    double(*t2)[3] = wk->w3[16];
    double(*t3)[3] = wk->w3[17];
    double(*t4)[3] = wk->w3[18];
    double(*dpr)[3] = wk->w3[19];
    double(*fe)[3] = wk->w3[2];
    double(*ce)[3] = wk->w3[3];
    double(*se)[3] = wk->w3[4];
    double(*ee)[3] = wk->w3[8];
    int* idx = wk->idx;
    int na = active_indices(wk->wv[8], ns, idx);
    matmul3(fpi, se, t1);
    matmult3(t1, fpi, s2);
    matmul3(f, s2, t1);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) pk1[3 * i + j] = t1[i][j];
    // tangent at fixed slips, and d tau / d F
    for (int kl = 0; kl < 9; ++kl) {
        int k = kl / 3;
        int ll = kl % 3;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) t1[i][j] = fpi[ll][i] * fe[k][j];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                t2[i][j] = t1[i][j] + t1[j][i];
                ee[i][j] = 0.5 * t2[i][j];
            }
        elastic_stress(dm, ee, t3, wk->s6[0], wk->s6[1]);
        matmul3(t2, se, t1);
        matmul3(ce, t3, t4);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) t1[i][j] += t4[i][j];
        for (int a = 0; a < na; ++a) {
            int s = idx[a];
            wr[a][kl] = -sgn[s] * project(sys + 6 * s, sys + 6 * s + 3, t1);
        }
        matmul3(fe, t3, t1);
        matmult3(t1, fpi, t2);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double v = t2[i][j];
                if (i == k) v += s2[ll][j];
                tan[9 * (3 * i + j) + kl] = v;
            }
    }
    if (na > 0) {
        double(*jac)[NS] = wk->wm[0];
        double(*lu)[NS] = wk->wm[1];
        for (int a = 0; a < na; ++a)
            for (int b = 0; b < na; ++b) lu[a][b] = jac[idx[a]][idx[b]];
        if (gauss_solve(&lu[0][0], NS, &wr[0][0], 9, na, 9) != 0) return 1;
        matmul3(fe, se, t4);
        for (int a = 0; a < na; ++a) {
            int r = idx[a];
            // d P / d dgamma_r, with d F_p^-1 = F_p^-1(n) dG_r
            matmul3(fpn, wk->wg[0][r], t1);
            matmul3(f, t1, t2);
            matmul3(t2, se, t3);
            matmult3(t3, fpi, dpr);
            matmul3(fe, wk->wg[2][r], t2);
            matmult3(t2, fpi, t3);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) dpr[i][j] += t3[i][j];
            matmult3(t4, t1, t3);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) dpr[i][j] += t3[i][j];
            for (int kl = 0; kl < 9; ++kl) {
                double x = sgn[r] * wr[a][kl];
                if (x != 0.0)
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j) tan[9 * (3 * i + j) + kl] += dpr[i][j] * x;
            }
        }
    }
    return 0;
}

// ---- porous plasticity: morphogen.homogenization.porous -----------------------------------

// R(p) = R0 + H p + sum Q (1 - exp(-b p)) (_hardening)
__device__ __forceinline__ double porous_hardening(const double* par, double p) {
    double r = par[2] + par[3] * p;
    for (int i = 0; i < POROUS_ISOTROPIC; ++i) {
        double q = par[6 + 2 * i];
        if (q != 0.0) r += q * (1.0 - exp(-par[7 + 2 * i] * p));
    }
    return r;
}

// R'(p) (_hardening_slope)
__device__ __forceinline__ double porous_hardening_slope(const double* par, double p) {
    double r = par[3];
    for (int i = 0; i < POROUS_ISOTROPIC; ++i) {
        double q = par[6 + 2 * i];
        if (q != 0.0) {
            double b = par[7 + 2 * i];
            r += q * b * exp(-b * p);
        }
    }
    return r;
}

// the piecewise linear function of the table at par[at], dflt without points (_table_value)
__device__ double table_value(const double* par, int at, double x, double dflt) {
    int n = (int)par[at];
    if (n == 0) return dflt;
    if (n == 1) return par[at + 2];
    int k = 0;
    while (k < n - 2 && x > par[at + 1 + 2 * (k + 1)]) ++k;
    double x0 = par[at + 1 + 2 * k];
    double y0 = par[at + 2 + 2 * k];
    double x1 = par[at + 3 + 2 * k];
    double y1 = par[at + 4 + 2 * k];
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
}

// its slope, dflt without points (_table_slope)
__device__ double table_slope(const double* par, int at, double x, double dflt) {
    int n = (int)par[at];
    if (n == 0) return dflt;
    if (n == 1) return 0.0;
    int k = 0;
    while (k < n - 2 && x > par[at + 1 + 2 * (k + 1)]) ++k;
    double dy = par[at + 4 + 2 * k] - par[at + 2 + 2 * k];
    return dy / (par[at + 3 + 2 * k] - par[at + 1 + 2 * k]);
}

// the matrix stress of the GTN potential and its derivatives by seq and sm, the root
// bracketed from guess if it is positive, else from an estimate (_gurson_stress)
__device__ double gurson_stress(double q1, double q2, double q3, double fs, double seq,
                                double sm, double guess, double* s_q, double* s_m) {
    if (fs <= 0.0) {  // von Mises
        *s_q = 1.0;
        *s_m = 0.0;
        return seq;
    }
    if (seq == 0.0 && sm == 0.0) {
        *s_q = 1.0;
        *s_m = 0.0;
        return 0.0;
    }
    double c = 1.0 + q3 * fs * fs;
    if (c - 2.0 * q1 * fs <= 0.0) {  // the surface has vanished
        *s_q = 0.0;
        *s_m = 0.0;
        return MG_INF;
    }
    double s, grow;
    if (guess > 0.0 && isfinite(guess)) {
        s = guess;
        grow = 1.0 + 1e-6;
    } else {
        double k = 1.5 * q2 * sm;
        s = sqrt(seq * seq + k * k) / pmax(1.0 - q1 * fs, 0.05);
        grow = 2.0;
    }
    double lo = 0.0, hi = 0.0;
    for (int it = 0; it < 200; ++it) {
        double t = 1.5 * q2 * sm / s;
        double a = seq / s;
        double phi = a * a + 2.0 * q1 * fs * cosh(t) - c;
        if (phi > 0.0) {
            lo = s;
            if (hi == 0.0) {
                s *= grow;
                grow = 1.0 + pmin(8.0 * (grow - 1.0), 1.0);
                continue;
            }
        } else {
            hi = s;
            if (lo == 0.0) {
                s /= grow;
                grow = 1.0 + pmin(8.0 * (grow - 1.0), 1.0);
                continue;
            }
        }
        break;
    }
    for (int it = 0; it < 100; ++it) {
        double t = 1.5 * q2 * sm / s;
        double a = seq / s;
        double phi = a * a + 2.0 * q1 * fs * cosh(t) - c;
        if (phi > 0.0)
            lo = s;
        else
            hi = s;
        double dphi = -2.0 / s * (a * a + q1 * fs * t * sinh(t));  // dPhi/ds < 0
        double step = s - phi / dphi;
        if (!(lo < step && step < hi)) step = 0.5 * (lo + hi);
        if (fabs(step - s) <= 1e-15 * s) {
            s = step;
            break;
        }
        s = step;
    }
    double t = 1.5 * q2 * sm / s;
    double a = seq / s;
    double den = a * a + q1 * fs * t * sinh(t);
    *s_q = a / den;
    *s_m = 1.5 * q1 * q2 * fs * sinh(t) / den;
    return s;
}

// the matrix stress of the potential of a point and its derivatives by seq and sm at the
// porosity f, GTN's iteration started from guess if it is positive (_matrix_stress)
__device__ double matrix_stress(const double* par, double seq, double sm, double f,
                                double guess, double* s_q, double* s_m) {
    int kind = (int)par[P_CRITERION];
    if (kind == GURSON) {
        double fs = table_value(par, P_FSTAR, f, f);
        fs = pmin(pmax(fs, 0.0), par[P_CAP]);
        return gurson_stress(par[P_Q1], par[P_Q2], par[P_Q3], fs, seq, sm, guess, s_q, s_m);
    }
    if (kind == ROUSSELIER) {
        double fe = pmin(pmax(f, 0.0), par[P_CAP]);
        double s1 = par[P_SIGMA1];
        double e = par[P_D] * s1 * fe * exp(pmin(par[P_QR] * sm / ((1.0 - fe) * s1), 700.0));
        *s_q = 1.0 / (1.0 - fe);
        *s_m = e * par[P_QR] / ((1.0 - fe) * s1);
        return seq / (1.0 - fe) + e;
    }
    double fe = pmax(f, 0.0);
    double cc = pmax(table_value(par, P_C, fe, 1.0), 0.0);
    double ff = pmax(table_value(par, P_F, fe, 0.0), 0.0);
    double s2 = cc * seq * seq + 9.0 * ff * sm * sm;
    if (s2 <= 0.0) {
        *s_q = 1.0;
        *s_m = 0.0;
        return 0.0;
    }
    double s = sqrt(s2);
    *s_q = cc * seq / s;
    *s_m = 9.0 * ff * sm / s;
    return s;
}

// the rate A(p) of the nucleation of porosity per unit of dp (_nucleation)
__device__ double nucleation(const double* par, double p) {
    int kind = (int)par[P_NUCLEATION];
    if (kind == 0 || p < par[P_NUCLEATION + 4]) return 0.0;
    double a = par[P_NUCLEATION + 1], b = par[P_NUCLEATION + 2], c = par[P_NUCLEATION + 3];
    if (kind == 1) return a;
    if (kind == 2) {
        double z = (p - b) / c;
        return a / (c * sqrt(2.0 * 3.141592653589793)) * exp(-0.5 * z * z);
    }
    return a * exp(-p / b);
}

// A'(p), zero below the threshold (_nucleation_slope)
__device__ double nucleation_slope(const double* par, double p) {
    int kind = (int)par[P_NUCLEATION];
    if (kind <= 1 || p < par[P_NUCLEATION + 4]) return 0.0;
    double a = par[P_NUCLEATION + 1], b = par[P_NUCLEATION + 2], c = par[P_NUCLEATION + 3];
    if (kind == 2) {
        double z = (p - b) / c;
        return -a / (c * sqrt(2.0 * 3.141592653589793)) * exp(-0.5 * z * z) * z / c;
    }
    return -a / b * exp(-p / b);
}

// the derivatives of the matrix stress s of (seq, sm, f) in d: by seq, sm and f, then the
// second ones by (seq, seq), (seq, sm), (sm, sm), (seq, f) and (sm, f); those of GTN by the
// implicit function theorem on its potential (_matrix_derivatives)
__device__ void matrix_derivatives(const double* par, double seq, double sm, double f,
                                   double s, double* d) {
    for (int i = 0; i < 8; ++i) d[i] = 0.0;
    int kind = (int)par[P_CRITERION];
    if (kind == GURSON) {
        double raw = table_value(par, P_FSTAR, f, f);
        double cap = par[P_CAP];
        double fs = pmin(pmax(raw, 0.0), cap);
        double dfs = (0.0 <= raw && raw < cap) ? table_slope(par, P_FSTAR, f, 1.0) : 0.0;
        if (!(s > 0.0)) {
            d[0] = 1.0;
            return;
        }
        double q1 = par[P_Q1];
        double q3 = par[P_Q3];
        double k = 1.5 * par[P_Q2];
        double u = k * sm / s;
        double ch = cosh(u);
        double sh = sinh(u);
        double a = seq / s;
        double p_s = -2.0 * (a * a + q1 * fs * u * sh) / s;
        double p_q = 2.0 * a / s;
        double p_m = 2.0 * q1 * fs * k * sh / s;
        double p_f = 2.0 * (q1 * ch - q3 * fs);
        double s2 = s * s;
        double p_qq = 2.0 / s2;
        double p_qs = -4.0 * a / s2;
        double p_mm = 2.0 * q1 * fs * k * k * ch / s2;
        double p_ms = -2.0 * q1 * fs * k * (u * ch + sh) / s2;
        double p_mf = 2.0 * q1 * k * sh / s;
        double p_fs = -2.0 * q1 * u * sh / s;
        double p_ss = (6.0 * a * a + 2.0 * q1 * fs * (2.0 * u * sh + u * u * ch)) / s2;
        double s_q = -p_q / p_s;
        double s_m = -p_m / p_s;
        double s_f = -p_f / p_s;
        d[0] = s_q;
        d[1] = s_m;
        d[2] = s_f * dfs;
        d[3] = -(p_qq + 2.0 * p_qs * s_q + p_ss * s_q * s_q) / p_s;
        d[4] = -(p_qs * s_m + p_ms * s_q + p_ss * s_q * s_m) / p_s;
        d[5] = -(p_mm + 2.0 * p_ms * s_m + p_ss * s_m * s_m) / p_s;
        d[6] = -(p_qs * s_f + p_fs * s_q + p_ss * s_q * s_f) / p_s * dfs;
        d[7] = -(p_mf + p_ms * s_f + p_fs * s_m + p_ss * s_m * s_f) / p_s * dfs;
        return;
    }
    if (kind == ROUSSELIER) {
        double cap = par[P_CAP];
        double fe = pmin(pmax(f, 0.0), cap);
        double dfe = (0.0 <= f && f < cap) ? 1.0 : 0.0;
        double s1 = par[P_SIGMA1];
        double qr = par[P_QR];
        double a = 1.0 / (1.0 - fe);
        double arg = qr * sm * a / s1;
        int unclamped = arg < 700.0;
        double ex = exp(unclamped ? arg : 700.0);
        double g = par[P_D] * s1 * fe * ex;
        double dg = par[P_D] * s1 * ex + (unclamped ? g * qr * sm * a * a / s1 : 0.0);
        d[0] = a;
        d[1] = g * qr * a / s1;
        d[2] = (seq * a * a + dg) * dfe;
        d[5] = unclamped ? d[1] * qr * a / s1 : 0.0;
        d[6] = a * a * dfe;
        d[7] = qr / s1 * (dg * a + g * a * a) * dfe;
        return;
    }
    double fe = pmax(f, 0.0);
    double dfe = f >= 0.0 ? 1.0 : 0.0;
    double c_raw = table_value(par, P_C, fe, 1.0);
    double f_raw = table_value(par, P_F, fe, 0.0);
    double cc = pmax(c_raw, 0.0);
    double ff = pmax(f_raw, 0.0);
    double dc = c_raw > 0.0 ? table_slope(par, P_C, fe, 0.0) * dfe : 0.0;
    double dff = f_raw > 0.0 ? table_slope(par, P_F, fe, 0.0) * dfe : 0.0;
    if (!(s > 0.0)) {
        d[0] = 1.0;
        return;
    }
    double s3 = s * s * s;
    d[0] = cc * seq / s;
    d[1] = 9.0 * ff * sm / s;
    d[2] = (seq * seq * dc + 9.0 * sm * sm * dff) / (2.0 * s);
    d[3] = 9.0 * cc * ff * sm * sm / s3;
    d[4] = -9.0 * cc * ff * seq * sm / s3;
    d[5] = 9.0 * ff * cc * seq * seq / s3;
    d[6] = dc * seq / s - cc * seq * d[2] / (s * s);
    d[7] = 9.0 * dff * sm / s - 9.0 * ff * sm * d[2] / (s * s);
}

// the four equations of the update at x = (dv, dq, z, f), the matrix stress sought from
// guess: the matrix stress, or -1 when x is out of bounds (_residual)
// the invariants of the relative trial stress after a plastic increment dp: J*, m*, c and
// their derivatives by dp in rel[0..5]; kin: the trial deviator (row 0) and the deviators
// and mean stresses of the backstresses (rows 1..., 7 wide) (_relative)
__device__ void porous_relative(const double* par, const double* kin, double sm_tr,
                                double seq_tr, double dp, double* rel) {
    int nk = (int)par[P_NKIN];
    if (nk == 0) {
        rel[0] = seq_tr;
        rel[1] = sm_tr;
        for (int i = 2; i < 6; ++i) rel[i] = 0.0;
        return;
    }
    double m = sm_tr, dm = 0.0, c = 0.0, dc = 0.0;
    for (int k = 0; k < nk; ++k) {
        double d = par[P_KIN + 2 * k + 1];
        double th = 1.0 / (1.0 + d * dp);
        c += th * par[P_KIN + 2 * k];
        dc -= d * th * th * par[P_KIN + 2 * k];
        m -= th * kin[7 * (1 + k) + 6];
        dm += d * th * th * kin[7 * (1 + k) + 6];
    }
    double xx = 0.0, xw = 0.0;
    for (int r = 0; r < 6; ++r) {
        double xi = kin[r], w = 0.0;
        for (int k = 0; k < nk; ++k) {
            double d = par[P_KIN + 2 * k + 1];
            double th = 1.0 / (1.0 + d * dp);
            xi -= th * kin[7 * (1 + k) + r];
            w += d * th * th * kin[7 * (1 + k) + r];
        }
        double g = r < 3 ? 1.0 : 2.0;
        xx += g * xi * xi;
        xw += g * xi * w;
    }
    double j = sqrt(pmax(1.5 * xx, 0.0));
    rel[0] = j;
    rel[1] = m;
    rel[2] = c;
    rel[3] = j > 0.0 ? 1.5 * xw / j : 0.0;
    rel[4] = dm;
    rel[5] = dc;
}

// the deviator xi of the relative trial stress after dp and its derivative w by dp
// (_relative_deviator)
__device__ void porous_relative_deviator(const double* par, const double* kin, double dp,
                                         double* xi, double* w) {
    for (int r = 0; r < 6; ++r) {
        xi[r] = kin[r];
        w[r] = 0.0;
    }
    for (int k = 0; k < (int)par[P_NKIN]; ++k) {
        double d = par[P_KIN + 2 * k + 1];
        double th = 1.0 / (1.0 + d * dp);
        for (int r = 0; r < 6; ++r) {
            xi[r] -= th * kin[7 * (1 + k) + r];
            w[r] += d * th * th * kin[7 * (1 + k) + r];
        }
    }
}

__device__ double porous_residual(const double* par, const double* x, double sm_tr,
                                  double seq_tr, double kb, double mu, double f_old,
                                  double p_old, double dt, int vertex, double* out,
                                  double guess, const double* kin) {
    double dv = x[0], dq = x[1], z = x[2], f = x[3];
    if (z < 0.0 || !(0.0 <= f && f < 0.9999)) return -1.0;
    int viscous = par[4] > 0.0;
    double dp = viscous ? dt * pow(z, par[5]) : z;
    double rel[6];
    porous_relative(par, kin, sm_tr, seq_tr, dp, rel);
    double c = rel[2];
    double sm = rel[1] - (kb + 2.0 * c / 9.0) * dv;
    double seq = rel[0] - (3.0 * mu + c) * dq;
    double s_q, s_m;
    double s = matrix_stress(par, seq, sm, f, guess, &s_q, &s_m);
    if (!isfinite(s)) return -1.0;
    double p = p_old + dp;
    out[0] = vertex ? dq - rel[0] / (3.0 * mu + c) : dv * s_q - dq * s_m;
    out[1] = s - porous_hardening(par, p) - (viscous ? par[4] * z : 0.0);
    out[2] = (1.0 - f) * s * dp - sm * dv - seq * dq;
    out[3] = f - f_old - (1.0 - f) * dv - nucleation(par, p) * dp;
    return s;
}

// a x = b (4 x 4, n_rhs columns of b, 4 x 2 row-major) by Gaussian elimination with
// partial pivoting, in place in b (m: a work matrix); 0 if singular (_solve4)
__device__ int solve4(const double* a, double* b, int n_rhs, double* m) {
    for (int r = 0; r < 16; ++r) m[r] = a[r];
    for (int col = 0; col < 4; ++col) {
        int piv = col;
        for (int r = col + 1; r < 4; ++r)
            if (fabs(m[4 * r + col]) > fabs(m[4 * piv + col])) piv = r;
        if (fabs(m[4 * piv + col]) < 1e-300) return 0;
        if (piv != col) {
            for (int c = 0; c < 4; ++c) {
                double t = m[4 * col + c];
                m[4 * col + c] = m[4 * piv + c];
                m[4 * piv + c] = t;
            }
            for (int c = 0; c < n_rhs; ++c) {
                double t = b[2 * col + c];
                b[2 * col + c] = b[2 * piv + c];
                b[2 * piv + c] = t;
            }
        }
        for (int r = col + 1; r < 4; ++r) {
            double g = m[4 * r + col] / m[4 * col + col];
            if (g != 0.0) {
                for (int c = col; c < 4; ++c) m[4 * r + c] -= g * m[4 * col + c];
                for (int c = 0; c < n_rhs; ++c) b[2 * r + c] -= g * b[2 * col + c];
            }
        }
    }
    for (int col = 3; col >= 0; --col)
        for (int c = 0; c < n_rhs; ++c) {
            double acc = b[2 * col + c];
            for (int k = col + 1; k < 4; ++k) acc -= m[4 * col + k] * b[2 * k + c];
            b[2 * col + c] = acc / m[4 * col + col];
        }
    return 1;
}

__device__ __forceinline__ double scaled_norm(const double* r, double e_scale,
                                              double s_scale) {
    return pmax(pmax(pmax(fabs(r[0]) / e_scale, fabs(r[1]) / s_scale),
                     fabs(r[2]) / (s_scale * e_scale)),
                fabs(r[3]));
}

// the Jacobian jac (4 x 4) of the residual at x, whose matrix stress is s, and in dr
// (4 x 2) its derivatives by the trial invariants sm_tr and seq_tr; d: a work vector of 8
// (_jacobian)
__device__ void porous_jacobian(const double* par, const double* x, double sm_tr,
                                double seq_tr, double kb, double mu, double f_old,
                                double p_old, double dt, int vertex, double s, double* d,
                                double* jac, double* dr, const double* kin) {
    double dv = x[0], dq = x[1], z = x[2], f = x[3];
    double kv = par[4];
    int viscous = kv > 0.0;
    double nv = par[5];
    double dp, dpz;
    if (viscous) {
        dp = dt * pow(z, nv);
        dpz = z > 0.0 ? nv * dt * pow(z, nv - 1.0) : (nv == 1.0 ? dt : 0.0);
    } else {
        dp = z;
        dpz = 1.0;
    }
    double rel[6];
    porous_relative(par, kin, sm_tr, seq_tr, dp, rel);
    double jstar = rel[0], c = rel[2], dj = rel[3], dm = rel[4], dc = rel[5];
    double kbe = kb + 2.0 * c / 9.0;
    double mue = 3.0 * mu + c;
    double sm = rel[1] - kbe * dv;
    double seq = jstar - mue * dq;
    // the derivatives of the relative invariants by dp (the recovery of the backstresses)
    double seq_p = dj - dc * dq;
    double sm_p = dm - 2.0 * dc / 9.0 * dv;
    matrix_derivatives(par, seq, sm, f, s, d);
    double s_q = d[0], s_m = d[1], s_f = d[2];
    double s_qq = d[3], s_qm = d[4], s_mm = d[5], s_qf = d[6], s_mf = d[7];
    double s_p = s_q * seq_p + s_m * sm_p;
    double p = p_old + dp;
    if (vertex) {  // dq - J* / (3 mu + c)
        jac[0] = 0.0;
        jac[1] = 1.0;
        jac[2] = -dpz * (dj - jstar * dc / mue) / mue;
        jac[3] = 0.0;
        dr[0] = 0.0;
        dr[1] = -1.0 / mue;
    } else {  // dv S_q - dq S_m
        double nq = dv * s_qq - dq * s_qm;
        double nm = dv * s_qm - dq * s_mm;
        jac[0] = s_q - kbe * nm;
        jac[1] = -s_m - mue * nq;
        jac[2] = dpz * (nq * seq_p + nm * sm_p);
        jac[3] = dv * s_qf - dq * s_mf;
        dr[0] = nm;
        dr[1] = nq;
    }
    // s - R(p) - K z
    jac[4] = -kbe * s_m;
    jac[5] = -mue * s_q;
    jac[6] = (s_p - porous_hardening_slope(par, p)) * dpz - (viscous ? kv : 0.0);
    jac[7] = s_f;
    dr[2] = s_m;
    dr[3] = s_q;
    // (1 - f) s dp - sm dv - seq dq
    double g = 1.0 - f;
    jac[8] = -g * dp * kbe * s_m + kbe * dv - sm;
    jac[9] = -mue * g * dp * s_q + mue * dq - seq;
    jac[10] = (g * s + g * dp * s_p - dv * sm_p - dq * seq_p) * dpz;
    jac[11] = (g * s_f - s) * dp;
    dr[4] = g * dp * s_m - dv;
    dr[5] = g * dp * s_q - dq;
    // f - f_old - (1 - f) dv - A(p) dp
    jac[12] = -g;
    jac[13] = 0.0;
    jac[14] = -dpz * (nucleation(par, p) + nucleation_slope(par, p) * dp);
    jac[15] = 1.0 + dv;
    dr[6] = 0.0;
    dr[7] = 0.0;
}

// implicit update of a porous point (_porous_update), its tangent symmetrized with sym;
// returns 1 on failure
__device__ __noinline__ int porous_update(const double* par, const double* eps,
                                          const double* eth, double dt, const double* old,
                                          double* nw, double* sig, double* tan, int sym) {
    double lam = par[0];
    double mu = par[1];
    double kb = lam + 2.0 * mu / 3.0;
    int viscous = par[4] > 0.0;
    double ee[6], s_tr[6], nrm[6], d[6], xi[6], w[6], x[4], res[4], trial[4], rt[4];
    double jac[16], m4[16], rhs[8], dr[8], ds[8], rel[6];
    double kin[7 * (1 + POROUS_KINEMATIC)];
    for (int r = 0; r < 6; ++r) {
        double e = r < 3 ? eps[r] : 0.5 * eps[r];
        ee[r] = e - eth[r] - old[r];
    }
    double tr = ee[0] + ee[1] + ee[2];
    double sm_tr = kb * tr;
    for (int r = 0; r < 6; ++r) s_tr[r] = 2.0 * mu * (r < 3 ? ee[r] - tr / 3.0 : ee[r]);
    double seq_tr = sqrt(pmax(1.5 * ddot6(s_tr, s_tr), 0.0));
    for (int r = 0; r < WIDTH; ++r) nw[r] = old[r];
    // the trial deviator and the deviators and mean stresses of the backstresses
    int nk = (int)par[P_NKIN];
    for (int r = 0; r < 6; ++r) kin[r] = s_tr[r];
    for (int k = 0; k < nk; ++k) {
        int base = STATE_BACK + 6 * k;
        double xm = (old[base] + old[base + 1] + old[base + 2]) / 3.0;
        for (int r = 0; r < 6; ++r) kin[7 * (1 + k) + r] = old[base + r] - (r < 3 ? xm : 0.0);
        kin[7 * (1 + k) + 6] = xm;
    }
    double f_old = pmax(par[P_F0] + old[STATE_FG] + old[STATE_FN], 0.0);
    double p_old = old[STATE_P];
    // the relative trial stress (the trial one without backstresses)
    porous_relative(par, kin, sm_tr, seq_tr, 0.0, rel);
    double jstar0 = rel[0], mstar0 = rel[1], c0 = rel[2];
    double sq0, smd0;
    double s0 = matrix_stress(par, jstar0, mstar0, f_old, 0.0, &sq0, &smd0);
    double r_old = porous_hardening(par, p_old);
    // the stresses of the trial state (not s*, whose exponential term of Rousselier may be
    // out of all proportion with them)
    double s_scale =
        pmax(pmax(pmax(pmax(pmax(seq_tr, fabs(sm_tr)), jstar0), fabs(mstar0)), r_old), 1e-300);
    double tiny = 1e-14 * s_scale;
    double inv = seq_tr > tiny ? 1.5 / seq_tr : 0.0;
    for (int r = 0; r < 6; ++r) nrm[r] = inv * s_tr[r];
    if (!isfinite(s0)) return 1;
    if (s0 - r_old <= 1e-13 * s_scale) {
        for (int r = 0; r < 6; ++r) sig[r] = s_tr[r] + (r < 3 ? sm_tr : 0.0);
        elastic_tangent(lam, mu, tan);
        nw[STATE_SSTAR] = s0;
        return 0;
    }
    // plastic flow: initial guess, the cutting-plane step along the normal at the trial
    // stress
    double e_scale = s_scale / (3.0 * mu);
    double work = jstar0 * sq0 + mstar0 * smd0;
    double dp_dl = work > 0.0 ? work / ((1.0 - f_old) * s0) : 1.0 / (1.0 - f_old);
    double below = pmax(p_old - 1e-6, 0.0);
    double slope = (porous_hardening(par, p_old + 1e-6) - porous_hardening(par, below)) /
                   (p_old + 1e-6 - below);
    double kbe = kb + 2.0 * c0 / 9.0;
    double mue = 3.0 * mu + c0;
    double lam0 = (s0 - r_old) / (kbe * smd0 * smd0 + mue * sq0 * sq0 + slope * dp_dl);
    double dp0 = lam0 * dp_dl;
    x[0] = lam0 * smd0;
    x[1] = pmin(lam0 * sq0, 0.5 * jstar0 / mue);
    x[3] = f_old;
    if (viscous) {  // the viscous stress of the rate-independent increment, at most the
                    // overstress
        double bound = (s0 - r_old) / par[4];
        x[2] = dt > 0.0 ? pmin(pow(dp0 / dt, 1.0 / par[5]), bound) : bound;
    } else {
        x[2] = dp0;
    }
    int vertex = jstar0 <= tiny && (int)par[P_CRITERION] == ROUSSELIER;
    int converged = 0;
    double sx = 0.0;
    for (int attempt = 0; attempt < 2; ++attempt) {
        // sx: the matrix stress of x, the guess of the points of the line search
        sx = porous_residual(par, x, sm_tr, seq_tr, kb, mu, f_old, p_old, dt, vertex, res, 0.0,
                             kin);
        if (sx < 0.0) {
            x[0] = 0.0;
            x[1] = 0.0;
            x[2] = 0.0;
            x[3] = f_old;
            sx = porous_residual(par, x, sm_tr, seq_tr, kb, mu, f_old, p_old, dt, vertex, res,
                                 0.0, kin);
            if (sx < 0.0) return 1;
        }
        double norm = scaled_norm(res, e_scale, s_scale);
        for (int it = 0; it < 80; ++it) {
            if (norm <= 1e-12) {
                converged = 1;
                break;
            }
            porous_jacobian(par, x, sm_tr, seq_tr, kb, mu, f_old, p_old, dt, vertex, sx, ds,
                            jac, dr, kin);
            for (int i = 0; i < 4; ++i) rhs[2 * i] = -res[i];
            if (!solve4(jac, rhs, 1, m4)) return 1;
            // the porosity falls at most by 90 % per iteration
            double alpha = 1.0;
            if (rhs[6] < 0.0 && x[3] > 0.0) alpha = pmin(1.0, 0.9 * x[3] / -rhs[6]);
            int accepted = 0;
            double n_t = norm;
            double inside = -1.0;
            for (int ls = 0; ls < 30; ++ls) {
                for (int i = 0; i < 4; ++i) trial[i] = x[i] + alpha * rhs[2 * i];
                trial[3] = pmax(trial[3], 0.0);  // a dense point stays dense
                inside = porous_residual(par, trial, sm_tr, seq_tr, kb, mu, f_old, p_old, dt,
                                         vertex, rt, sx, kin);
                if (inside >= 0.0) {
                    n_t = scaled_norm(rt, e_scale, s_scale);
                    if (n_t < (1.0 - 1e-4 * alpha) * norm || alpha < 1e-6) {
                        accepted = 1;
                        break;
                    }
                }
                alpha *= 0.5;
            }
            if (!accepted) return 1;
            for (int i = 0; i < 4; ++i) {
                x[i] = trial[i];
                res[i] = rt[i];
            }
            sx = inside;
            norm = n_t;
        }
        if (!converged) return 1;
        double dp1 = viscous ? dt * pow(x[2], par[5]) : x[2];
        porous_relative(par, kin, sm_tr, seq_tr, dp1, rel);
        double seq = rel[0] - (3.0 * mu + rel[2]) * x[1];
        if (seq >= -1e-10 * s_scale || vertex) break;
        // Rousselier's vertex: the hydrostatic return
        vertex = 1;
        converged = 0;
        x[1] = rel[0] / (3.0 * mu + rel[2]);
    }
    if (!converged) return 1;
    double dv = x[0], dq = x[1], z = x[2], f = x[3];
    double dp, dpz;
    if (viscous) {
        dp = dt * pow(z, par[5]);
        dpz = z > 0.0 ? par[5] * dt * pow(z, par[5] - 1.0) : (par[5] == 1.0 ? dt : 0.0);
    } else {
        dp = z;
        dpz = 1.0;
    }
    porous_relative(par, kin, sm_tr, seq_tr, dp, rel);
    double jstar = rel[0], c = rel[2];
    double sm = rel[1] - (kb + 2.0 * c / 9.0) * dv;
    double seq = pmax(jstar - (3.0 * mu + c) * dq, 0.0);
    double sq_, smd_;
    double s = matrix_stress(par, seq, sm, f, 0.0, &sq_, &smd_);
    // the deviator of the relative trial stress, whose direction the plastic strain takes
    porous_relative_deviator(par, kin, dp, xi, w);
    if (nk > 0) {
        double inv1 = jstar > tiny ? 1.5 / jstar : 0.0;
        for (int r = 0; r < 6; ++r) nrm[r] = inv1 * xi[r];
    }
    for (int r = 0; r < 6; ++r) nw[r] = old[r] + (r < 3 ? dv / 3.0 : 0.0) + dq * nrm[r];
    nw[STATE_P] = p_old + dp;
    nw[STATE_FN] = old[STATE_FN] + nucleation(par, p_old + dp) * dp;
    // no porosity below zero (a dense point whose volume change is a roundoff)
    nw[STATE_FG] = pmax(old[STATE_FG] + (1.0 - f) * dv, -par[P_F0] - nw[STATE_FN]);
    nw[STATE_SSTAR] = s;
    for (int k = 0; k < nk; ++k) {  // X = th (X_n + 2/3 C deps_p), its hydrostatic part included
        double th = 1.0 / (1.0 + par[P_KIN + 2 * k + 1] * dp);
        double cc = 2.0 / 3.0 * par[P_KIN + 2 * k];
        int base = STATE_BACK + 6 * k;
        for (int r = 0; r < 6; ++r) {
            double dep = (r < 3 ? dv / 3.0 : 0.0) + dq * nrm[r];
            nw[base + r] = th * (old[base + r] + cc * dep);
        }
    }
    // sig = sig_tr - K dv I - 2 mu dq n: the deviator (s_tr - xi) + (1 - 3 mu dq / J*) xi
    double ratio_s = jstar > tiny ? pmax(jstar - 3.0 * mu * dq, 0.0) / jstar : 1.0;
    double sm_abs = sm_tr - kb * dv;
    for (int r = 0; r < 6; ++r)
        sig[r] = (s_tr[r] - xi[r]) + ratio_s * xi[r] + (r < 3 ? sm_abs : 0.0);
    // consistent tangent: the sensitivity of the solution to the trial invariants (with
    // backstresses: m* and J*, dJ* = n : ds_tr)
    porous_jacobian(par, x, sm_tr, seq_tr, kb, mu, f_old, p_old, dt, vertex, sx, ds, jac, dr,
                    kin);
    for (int i = 0; i < 4; ++i) {
        rhs[2 * i] = -dr[2 * i];
        rhs[2 * i + 1] = -dr[2 * i + 1];
    }
    if (!solve4(jac, rhs, 2, m4)) return 1;
    double gq = rhs[3];
    double ratio = jstar > tiny ? ratio_s : 1.0 - 3.0 * mu * gq;
    // the turn of the direction with dp: dn = 3 / (2 J*) (w - 2/3 (n : w) n) ddp
    double turn = nk > 0 && jstar > tiny ? 3.0 * mu * dq / jstar : 0.0;
    double nwd = ddot6(nrm, w);
    for (int j = 0; j < 6; ++j) {
        for (int r = 0; r < 6; ++r) d[r] = 0.0;
        d[j] = j < 3 ? 1.0 : 0.5;
        double trd = j < 3 ? 1.0 : 0.0;
        double nd = ddot6(nrm, d);
        double dsm_tr = kb * trd;
        double dseq_tr = 2.0 * mu * nd;
        double ddv = rhs[0] * dsm_tr + rhs[1] * dseq_tr;
        double ddq = rhs[2] * dsm_tr + rhs[3] * dseq_tr;
        double ddp = turn * dpz * (rhs[4] * dsm_tr + rhs[5] * dseq_tr);
        double dsm = dsm_tr - kb * ddv;
        double dseq = dseq_tr - 3.0 * mu * ddq;
        for (int r = 0; r < 6; ++r) {
            double dev = r < 3 ? d[r] - trd / 3.0 : d[r];
            double value =
                2.0 / 3.0 * dseq * nrm[r] + 2.0 * mu * ratio * (dev - 2.0 / 3.0 * nd * nrm[r]);
            if (turn != 0.0) value -= ddp * (w[r] - 2.0 / 3.0 * nwd * nrm[r]);
            if (r < 3) value += dsm;
            tan[6 * r + j] = value;
        }
    }
    if (sym) symmetrize(tan, 6);
    return 0;
}

// ---- finite strains: morphogen.homogenization.finite --------------------------------------

// elastic, von Mises, Hill or porous point in the space of the logarithmic strain
// (_log_point), the tangent of the logarithmic strain symmetrized with sym
__device__ __noinline__ int log_point(int kind, const double* dm, const double* par, int nk,
                                      const double* hg, const double* eth, double dt,
                                      const double* old, double* nw, double* pk1,
                                      double* tan, int sym) {
    double f[3][3], e2[3][3], q[3][3], tq[3][3], s[3][3], t1[3][3], t2[3][3], dcq[3][3];
    double dtq[3][3], f1[3][3], f2[3][3][3];
    double mu[3], eps6[6], sig6[6], d6[6], ds6[6], tan6[36];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) f[i][j] = (i == j ? 1.0 : 0.0) + hg[3 * i + j];
    // C - I = H + H^T + H^T H, without cancellation for small strains
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double acc = hg[3 * i + j] + hg[3 * j + i];
            for (int k = 0; k < 3; ++k) acc += hg[3 * k + i] * hg[3 * k + j];
            e2[i][j] = acc;
        }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) t1[i][j] = e2[i][j];
    eigh3(t1, mu, q);
    for (int a = 0; a < 3; ++a)
        if (!(mu[a] > -1.0 && isfinite(mu[a]))) return 1;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double acc = 0.0;
            for (int a = 0; a < 3; ++a) acc += q[i][a] * q[j][a] * half_log(mu[a]);
            t1[i][j] = acc;
        }
    strain_voigt(t1, eps6);
    if (kind == PLASTIC) {
        if (j2_update(par, nk, eps6, eth, dt, old, nw, sig6, tan6, sym) != 0) return 1;
    } else if (kind == HILL) {
        if (hill_update(par, nk, eps6, eth, dt, old, nw, sig6, tan6, sym) != 0) return 1;
    } else if (kind == POROUS) {
        if (porous_update(par, eps6, eth, dt, old, nw, sig6, tan6, sym) != 0) return 1;
    } else {
        for (int r = 0; r < 6; ++r) d6[r] = eps6[r] - eth[r] * (r < 3 ? 1.0 : 2.0);
        for (int r = 0; r < 6; ++r) {
            double acc = 0.0;
            for (int c = 0; c < 6; ++c) {
                acc += dm[6 * r + c] * d6[c];
                tan6[6 * r + c] = dm[6 * r + c];
            }
            sig6[r] = acc;
        }
    }
    stress_tensor(sig6, t1);
    tmatmul3(q, t1, t2);
    matmul3(t2, q, tq);
    // the divided differences are symmetric in their arguments: their 6 and 10 values
    for (int a = 0; a < 3; ++a)
        for (int b = a; b < 3; ++b) {
            double v = log_divided(mu[a], mu[b]);
            f1[a][b] = v;
            f1[b][a] = v;
            for (int c = b; c < 3; ++c) {
                v = log_divided2(mu[a], mu[b], mu[c]);
                f2[a][b][c] = v;
                f2[a][c][b] = v;
                f2[b][a][c] = v;
                f2[b][c][a] = v;
                f2[c][a][b] = v;
                f2[c][b][a] = v;
            }
        }
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) t1[a][b] = 2.0 * f1[a][b] * tq[a][b];
    matmul3(q, t1, t2);
    matmult3(t2, q, s);
    matmul3(f, s, t1);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) pk1[3 * i + j] = t1[i][j];
    // dS is linear in dC: F dS for the six directions of dC, B = e_m (x) e_n + e_n (x) e_m
    // (e_m (x) e_m when m = n), then for each dF = e_k (x) e_l the sum of the dC_mn F dS(B)
    const int pm[6] = {0, 1, 2, 1, 0, 0};
    const int pn[6] = {0, 1, 2, 2, 2, 1};
    double gb[6][3][3];
    for (int bb = 0; bb < 6; ++bb) {
        int m = pm[bb], n = pn[bb];
        for (int a = 0; a < 3; ++a)  // Q^T B Q
            for (int b = 0; b < 3; ++b) {
                double v = q[m][a] * q[n][b];
                if (m != n) v += q[n][a] * q[m][b];
                dcq[a][b] = v;
            }
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) t1[a][b] = f1[a][b] * dcq[a][b];
        matmul3(q, t1, t2);
        matmult3(t2, q, t1);
        strain_voigt(t1, d6);
        for (int r = 0; r < 6; ++r) {
            double acc = 0.0;
            for (int c = 0; c < 6; ++c) acc += tan6[6 * r + c] * d6[c];
            ds6[r] = acc;
        }
        stress_tensor(ds6, t1);
        tmatmul3(q, t1, t2);
        matmul3(t2, q, dtq);
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                double acc = 2.0 * f1[a][b] * dtq[a][b];
                for (int c = 0; c < 3; ++c)
                    acc += 2.0 * f2[a][b][c] * (dcq[a][c] * tq[c][b] + tq[a][c] * dcq[c][b]);
                t1[a][b] = acc;
            }
        matmul3(q, t1, t2);
        matmult3(t2, q, t1);
        matmul3(f, t1, gb[bb]);
    }
    for (int kl = 0; kl < 9; ++kl) {
        int k = kl / 3;
        int ll = kl % 3;
        // dC = dF^T F + F^T dF for dF = e_k (x) e_l: dC_mn = d_ml F_kn + d_nl F_km
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) t2[i][j] = 0.0;
        for (int bb = 0; bb < 6; ++bb) {
            int m = pm[bb], n = pn[bb];
            double c = (m == ll ? f[k][n] : 0.0) + (n == ll ? f[k][m] : 0.0);
            if (c != 0.0)
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) t2[i][j] += c * gb[bb][i][j];
        }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double v = t2[i][j];
                if (i == k) v += s[ll][j];
                tan[9 * (3 * i + j) + kl] = v;
            }
    }
    return 0;
}

// ---- the integration of the points: update_points and update_points_finite --------------

// Stress, internal variables and tangent (symmetrized with sym) of the points first, ...,
// n - 1 for the strains eps (engineering, n x 6) at the end of a step dt
// (behaviours.update_points); failed counts the points whose integration failed.
extern "C" __global__ void integrate_small(
    const long long first, const long long n, const long long* kind, const long long* mat,
    const long long* pmat, const double* matrices, const double* eigen, const double* params,
    const int nk, const double* systems, const double* cp_par, const double* cp_h,
    const long long* cp_n, const double* eps, const double dtemp, const double dt,
    const double* old, double* nw, double* sigma, double* tangent, const int guess,
    const double ratio, const int sym, unsigned int* failed) {
    long long i = first + (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int m = (int)mat[i];
    const int k = (int)kind[i];
    if (k == MFRONT) return;
    const double* dm = matrices + 36LL * m;
    const double* e = eps + 6 * i;
    double* sg = sigma + 6 * i;
    double* tg = tangent + 36 * i;
    if (k == ELASTIC) {
        double mech[6];
        for (int r = 0; r < 6; ++r) mech[r] = e[r] - dtemp * eigen[6 * m + r];
        for (int r = 0; r < 6; ++r) {
            double s = 0.0;
            for (int c = 0; c < 6; ++c) {
                s += dm[6 * r + c] * mech[c];
                tg[6 * r + c] = dm[6 * r + c];
            }
            sg[r] = s;
        }
        return;
    }
    double eth[6];
    for (int r = 0; r < 6; ++r) eth[r] = dtemp * eigen[6 * m + r] * (r < 3 ? 1.0 : 0.5);
    const double* o = old + (long long)WIDTH * i;
    double* w = nw + (long long)WIDTH * i;
    int fail;
    if (k == PLASTIC) {
        fail = j2_update(params + (long long)ROW_WIDTH * pmat[i], nk, e, eth, dt, o, w, sg,
                         tg, sym);
    } else if (k == HILL) {
        fail = hill_update(params + (long long)ROW_WIDTH * pmat[i], nk, e, eth, dt, o, w, sg,
                           tg, sym);
    } else if (k == POROUS) {
        fail = porous_update(params + (long long)ROW_WIDTH * pmat[i], e, eth, dt, o, w, sg, tg,
                             sym);
    } else {
        const long long c = pmat[i];
        Work wk;
        fail = cp_small(systems + (long long)m * NS * 6, cp_par + c * NS * N_CP,
                        cp_h + c * NS * NS, (int)cp_n[c], dm, e, eth, dt, o, w, sg, tg, guess,
                        ratio, &wk, sym);
    }
    if (fail) atomicAdd(failed, 1u);
}

// First Piola-Kirchhoff stress, internal variables and tangent dP/dF (symmetrized with
// sym) of the points first, ..., n - 1 for the displacement gradients hg (n x 9) at the end
// of a step (finite.update_points_finite).
extern "C" __global__ void integrate_finite(
    const long long first, const long long n, const long long* kind, const long long* mat,
    const long long* pmat, const double* matrices, const double* eigen, const double* params,
    const int nk, const double* systems, const double* cp_par, const double* cp_h,
    const long long* cp_n, const double* hg, const double dtemp, const double dt,
    const double* old, double* nw, double* pk1, double* tangent, const int guess,
    const double ratio, const int sym, unsigned int* failed) {
    long long i = first + (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int m = (int)mat[i];
    const int k = (int)kind[i];
    if (k == MFRONT) return;
    double eth[6];
    for (int r = 0; r < 6; ++r) eth[r] = dtemp * eigen[6 * m + r] * (r < 3 ? 1.0 : 0.5);
    const double* dm = matrices + 36LL * m;
    const double* o = old + (long long)WIDTH * i;
    double* w = nw + (long long)WIDTH * i;
    double* p = pk1 + 9 * i;
    double* a = tangent + 81 * i;
    int fail;
    if (k == CRYSTAL) {
        const long long c = pmat[i];
        Work wk;
        fail = cp_finite(systems + (long long)m * NS * 6, cp_par + c * NS * N_CP,
                         cp_h + c * NS * NS, (int)cp_n[c], dm, hg + 9 * i, eth, dt, o, w, p, a,
                         guess, ratio, &wk);
    } else {
        const long long row = pmat[i] > 0 ? pmat[i] : 0;
        fail = log_point(k, dm, params + (long long)ROW_WIDTH * row, nk, hg + 9 * i,
                         eth, dt, o, w, p, a, sym);
    }
    if (fail == 0) {
        if (sym) symmetrize(a, 9);
    } else {
        atomicAdd(failed, 1u);
    }
}
