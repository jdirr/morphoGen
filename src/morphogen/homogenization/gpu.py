"""The FFT solvers of voxel images on the GPU (CuPy; see :mod:`morphogen.gpu`).

The conjugate gradients of the linear solver (:mod:`morphogen.homogenization.fft`) and
the iterations of the loading paths (:mod:`morphogen.homogenization.fft_nonlinear`) run
on the GPU, in double precision:

* the stiffness operators of the voxels are applied matrix-free by CUDA kernels, ports
  of the Numba kernels of :mod:`morphogen.homogenization.kernels` (one thread per node,
  so no atomics; the point stresses one thread per voxel), with the element matrices of
  the materials or the tangents of the points;
* the Fourier preconditioner is cuFFT transforms and a kernel multiplying the blocks of
  its inverse symbol, computed on the GPU;
* the dot products are reductions of CuPy, without cuBLAS; GMRES, for the full tangents
  that are not symmetric, keeps its basis on the GPU (the dot products of a Gram-Schmidt
  pass in one reduction over its rows, the vectors of its deflated restarts combined in
  place) and its small Hessenberg matrix and harmonic Ritz vectors on the CPU, as the
  space it recycles from a solve to the next along a path (GCRO-DR).

Along loading paths (:class:`VoxelPaths`) the arrays of the integration points stay on
the GPU: their measures of strain, the integration of their behaviours (the CUDA kernels
of :file:`behaviours.cu`, ports of the Numba kernels of von Mises plasticity, crystal
plasticity and the logarithmic strain space, one thread per point), the stresses
conjugate to B-bar and F-bar, the nodal forces and the tangent operator. The Newton
iterations, on the CPU, exchange with the GPU the nodal unknowns, the residuals and the
integrals of the stresses; the outputs of each step take the stresses, the measures and
the cumulated plastic strain of the points back.
"""

from __future__ import annotations

import logging
import math
import time
from functools import lru_cache, partial

import numpy as np

from morphogen.homogenization import behaviours as _vm
from morphogen.homogenization import finite as fin
from morphogen.homogenization.fft_nonlinear import _VoxelPaths
from morphogen.homogenization.kernels import CORNER_INDEX, CORNERS
from morphogen.homogenization.nonlinear import _asymmetry_of, _mfront_points
from morphogen.homogenization.solvers import (
    _REORTHOGONALIZE,
    GMRES_DEFLATE,
    GMRES_RESTART,
    GMRES_STAGNATION,
    NEGLIGIBLE,
    Recycling,
    SolveInfo,
    _deflation,
    _gcro_dr,
    _keep,
    _stagnates,
)
from morphogen.stopping import checkpoint

log = logging.getLogger("morphogen")

_THREADS = 256
#: Threads of a block of the kernels of the behaviours. The points of a launch adapt to
#: its duration (about _LAUNCH_S seconds, between the bounds), so that each launch stays far
#: below the 2 s of the watchdog of the display drivers of Windows even in the steps whose
#: points take long (large increments, substeps)
_POINT_THREADS = 64
_CHUNK_MIN, _CHUNK_START, _CHUNK_MAX = 1 << 10, 1 << 14, 1 << 20
_LAUNCH_S = 0.2

_SOURCE = r"""
#include <cupy/complex.cuh>

__constant__ int CORNERS[8][3] = {CORNERS};
__constant__ int CORNER_INDEX[2][2][2] = {CORNER_INDEX};

__device__ __forceinline__ void element_nodes(long long e, int nx, int ny, int nz,
                                              long long* nodes) {
    int k = (int)(e % nz);
    int j = (int)((e / nz) % ny);
    int i = (int)(e / ((long long)ny * nz));
    for (int q = 0; q < 8; ++q) {
        int ii = (i + CORNERS[q][0]) % nx;
        int jj = (j + CORNERS[q][1]) % ny;
        int kk = (k + CORNERS[q][2]) % nz;
        nodes[q] = ((long long)ii * ny + jj) * nz + kk;
    }
}

// out = K w on the periodic grid: mats[g] is the element matrix (8 nd x 8 nd) of the
// voxels of group g (-1: no stiffness); one thread per node.
extern "C" __global__ void apply_operator(const int nx, const int ny, const int nz,
                                          const int nd, const long long* group,
                                          const double* mats, const double* w,
                                          double* out) {
    long long n = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (long long)nx * ny * nz) return;
    int k = (int)(n % nz);
    int j = (int)((n / nz) % ny);
    int i = (int)(n / ((long long)ny * nz));
    const int size = 8 * nd;
    double acc[3] = {0.0, 0.0, 0.0};
    for (int a = 0; a < 2; ++a) {
        int ei = i - a >= 0 ? i - a : nx - 1;
        for (int b = 0; b < 2; ++b) {
            int ej = j - b >= 0 ? j - b : ny - 1;
            for (int c = 0; c < 2; ++c) {
                int ek = k - c >= 0 ? k - c : nz - 1;
                long long g = group[((long long)ei * ny + ej) * nz + ek];
                if (g < 0) continue;
                int ln = CORNER_INDEX[a][b][c];
                const double* mat = mats + g * size * size;
                for (int q = 0; q < 8; ++q) {
                    int qi = ei + CORNERS[q][0]; if (qi == nx) qi = 0;
                    int qj = ej + CORNERS[q][1]; if (qj == ny) qj = 0;
                    int qk = ek + CORNERS[q][2]; if (qk == nz) qk = 0;
                    long long m = ((long long)qi * ny + qj) * nz + qk;
                    for (int c2 = 0; c2 < nd; ++c2) {
                        double wv = w[m * nd + c2];
                        for (int r = 0; r < nd; ++r)
                            acc[r] += mat[(ln * nd + r) * size + q * nd + c2] * wv;
                    }
                }
            }
        }
    }
    for (int r = 0; r < nd; ++r) out[n * nd + r] = acc[r];
}

// Inverse of the symbol of a homogeneous operator (stencil blocks: 3 x 3 x 3 x nd x nd)
// at the frequencies of the real FFT of the grid (frequencies x nd x nd), zero at the
// zero frequency: the port of morphogen.homogenization.kernels.fourier_inverse.
extern "C" __global__ void fourier_inverse(const int nx, const int ny, const int nz,
                                           const int nd, const double* blocks,
                                           complex<double>* out) {
    const int nzh = nz / 2 + 1;
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= (long long)nx * ny * nzh) return;
    int fz = (int)(f % nzh);
    int fy = (int)((f / nzh) % ny);
    int fx = (int)(f / ((long long)ny * nzh));
    complex<double>* o = out + f * nd * nd;
    if (fx == 0 && fy == 0 && fz == 0) {
        for (int r = 0; r < nd * nd; ++r) o[r] = complex<double>(0.0, 0.0);
        return;
    }
    const double pi2 = 6.283185307179586476925286766559;
    double tx = pi2 * fx / nx, ty = pi2 * fy / ny, tz = pi2 * fz / nz;
    complex<double> px[3], py[3], pz[3];
    for (int a = 0; a < 3; ++a) {
        px[a] = complex<double>(cos((a - 1) * tx), sin((a - 1) * tx));
        py[a] = complex<double>(cos((a - 1) * ty), sin((a - 1) * ty));
        pz[a] = complex<double>(cos((a - 1) * tz), sin((a - 1) * tz));
    }
    complex<double> s[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) s[i][j] = complex<double>(0.0, 0.0);
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            for (int c = 0; c < 3; ++c) {
                complex<double> phase = px[a] * py[b] * pz[c];
                const double* blk = blocks + (((a * 3 + b) * 3 + c) * nd) * nd;
                for (int i = 0; i < nd; ++i)
                    for (int j = 0; j < nd; ++j) s[i][j] += phase * blk[i * nd + j];
            }
    if (nd == 1) {
        o[0] = complex<double>(1.0, 0.0) / s[0][0];
        return;
    }
    complex<double> c00 = s[1][1] * s[2][2] - s[1][2] * s[2][1];
    complex<double> c01 = s[1][2] * s[2][0] - s[1][0] * s[2][2];
    complex<double> c02 = s[1][0] * s[2][1] - s[1][1] * s[2][0];
    complex<double> inv = complex<double>(1.0, 0.0)
                        / (s[0][0] * c00 + s[0][1] * c01 + s[0][2] * c02);
    o[0] = c00 * inv;
    o[1] = (s[0][2] * s[2][1] - s[0][1] * s[2][2]) * inv;
    o[2] = (s[0][1] * s[1][2] - s[0][2] * s[1][1]) * inv;
    o[3] = c01 * inv;
    o[4] = (s[0][0] * s[2][2] - s[0][2] * s[2][0]) * inv;
    o[5] = (s[0][2] * s[1][0] - s[0][0] * s[1][2]) * inv;
    o[6] = c02 * inv;
    o[7] = (s[0][1] * s[2][0] - s[0][0] * s[2][1]) * inv;
    o[8] = (s[0][0] * s[1][1] - s[0][1] * s[1][0]) * inv;
}

// out[i, f] = sum_j inv[f, i, j] hat[j, f] for complex numbers stored as pairs of
// doubles: the inverse symbol (frequencies x nd x nd), the fields component by component.
extern "C" __global__ void block_multiply(const long long n_freq, const int nd,
                                          const double* inv, const double* hat,
                                          double* out) {
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_freq) return;
    for (int i = 0; i < nd; ++i) {
        double re = 0.0, im = 0.0;
        for (int j = 0; j < nd; ++j) {
            const double* m = inv + 2 * ((f * nd + i) * nd + j);
            const double* h = hat + 2 * ((long long)j * n_freq + f);
            re += m[0] * h[0] - m[1] * h[1];
            im += m[0] * h[1] + m[1] * h[0];
        }
        out[2 * ((long long)i * n_freq + f)] = re;
        out[2 * ((long long)i * n_freq + f) + 1] = im;
    }
}

__device__ __forceinline__ void gradient(const double ue[8][3], const double* gg, int g,
                                         double* h) {
    for (int r = 0; r < 9; ++r) h[r] = 0.0;
    for (int a = 0; a < 8; ++a) {
        double gx = gg[(g * 3 + 0) * 8 + a];
        double gy = gg[(g * 3 + 1) * 8 + a];
        double gz = gg[(g * 3 + 2) * 8 + a];
        for (int i = 0; i < 3; ++i) {
            double u = ue[a][i];
            h[3 * i] += u * gx;
            h[3 * i + 1] += u * gy;
            h[3 * i + 2] += u * gz;
        }
    }
}

__device__ __forceinline__ double det3(const double* f) {
    return f[0] * (f[4] * f[8] - f[5] * f[7]) - f[1] * (f[3] * f[8] - f[5] * f[6])
         + f[2] * (f[3] * f[7] - f[4] * f[6]);
}

__device__ __forceinline__ void inverse_transpose(const double* f, double det, double* o) {
    double inv = 1.0 / det;
    o[0] = (f[4] * f[8] - f[5] * f[7]) * inv;
    o[1] = (f[5] * f[6] - f[3] * f[8]) * inv;
    o[2] = (f[3] * f[7] - f[4] * f[6]) * inv;
    o[3] = (f[2] * f[7] - f[1] * f[8]) * inv;
    o[4] = (f[0] * f[8] - f[2] * f[6]) * inv;
    o[5] = (f[1] * f[6] - f[0] * f[7]) * inv;
    o[6] = (f[1] * f[5] - f[2] * f[4]) * inv;
    o[7] = (f[2] * f[3] - f[0] * f[5]) * inv;
    o[8] = (f[0] * f[4] - f[1] * f[3]) * inv;
}

// Stress increments of the Gauss points (their conjugates with bbar) for the nodal
// displacements v plus the uniform gradient direction (9, row by row): the port of
// morphogen.homogenization.kernels.voxel_tangent_stress; one thread per voxel.
extern "C" __global__ void voxel_increments(const int nx, const int ny, const int nz,
                                            const long long n_el, const long long* elem,
                                            const double* v, const double* direction,
                                            const double* tangent, const double* s,
                                            const double* base, const double* gg,
                                            const int finite, const int bbar,
                                            double* out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_el) return;
    long long nodes[8];
    element_nodes(elem[t], nx, ny, nz, nodes);
    double ue[8][3];
    for (int a = 0; a < 8; ++a)
        for (int c = 0; c < 3; ++c) ue[a][c] = v[3 * nodes[a] + c];
    double dh[8][9];
    for (int g = 0; g < 8; ++g) {
        gradient(ue, gg, g, dh[g]);
        for (int r = 0; r < 9; ++r) dh[g][r] += direction[r];
    }
    double ds[8][9];
    if (!finite) {
        double mean = 0.0;
        for (int g = 0; g < 8; ++g) mean += (dh[g][0] + dh[g][4] + dh[g][8]) / 8.0;
        double mean_p = 0.0;
        for (int g = 0; g < 8; ++g) {
            const double* h = dh[g];
            double de[6] = {h[0], h[4], h[8], h[5] + h[7], h[6] + h[2], h[1] + h[3]};
            if (bbar) {
                double shift = (mean - de[0] - de[1] - de[2]) / 3.0;
                for (int r = 0; r < 3; ++r) de[r] += shift;
            }
            long long p = 8 * t + g;
            for (int r = 0; r < 6; ++r) {
                double acc = 0.0;
                for (int c = 0; c < 6; ++c) acc += tangent[(p * 6 + r) * 6 + c] * de[c];
                ds[g][r] = acc;
            }
            mean_p += (ds[g][0] + ds[g][1] + ds[g][2]) / 24.0;
        }
        for (int g = 0; g < 8; ++g) {
            long long p = 8 * t + g;
            double shift = bbar ? mean_p - (ds[g][0] + ds[g][1] + ds[g][2]) / 3.0 : 0.0;
            for (int r = 0; r < 6; ++r) out[p * 6 + r] = ds[g][r] + (r < 3 ? shift : 0.0);
        }
        return;
    }
    if (!bbar) {
        for (int g = 0; g < 8; ++g) {
            long long p = 8 * t + g;
            for (int r = 0; r < 9; ++r) {
                double acc = 0.0;
                for (int c = 0; c < 9; ++c) acc += tangent[(p * 9 + r) * 9 + c] * dh[g][c];
                out[p * 9 + r] = acc;
            }
        }
        return;
    }
    // F-bar: F_bar = alpha F, alpha = (J_bar / J)^(1/3)
    double alpha[8], dalpha[8], tau[8], dtau[8];
    double f[9], fit[9];
    double mean_log = 0.0, mean_c = 0.0;
    for (int g = 0; g < 8; ++g) {
        long long p = 8 * t + g;
        for (int r = 0; r < 9; ++r) f[r] = base[p * 9 + r] + (r % 4 == 0 ? 1.0 : 0.0);
        double det = det3(f);
        inverse_transpose(f, det, fit);
        alpha[g] = log(det);
        mean_log += alpha[g] / 8.0;
        double c = 0.0;
        for (int r = 0; r < 9; ++r) c += fit[r] * dh[g][r];
        dalpha[g] = c;
        mean_c += c / 8.0;
    }
    double mean_tau = 0.0, mean_dtau = 0.0;
    for (int g = 0; g < 8; ++g) {
        long long p = 8 * t + g;
        double a = exp((mean_log - alpha[g]) / 3.0);
        alpha[g] = a;
        double da = a * (mean_c - dalpha[g]) / 3.0;
        dalpha[g] = da;
        double de[9];
        for (int r = 0; r < 9; ++r) {
            f[r] = base[p * 9 + r] + (r % 4 == 0 ? 1.0 : 0.0);
            de[r] = a * dh[g][r] + da * f[r];
        }
        double pf = 0.0, dpf = 0.0, pdf = 0.0;
        for (int r = 0; r < 9; ++r) {
            double acc = 0.0;
            for (int c = 0; c < 9; ++c) acc += tangent[(p * 9 + r) * 9 + c] * de[c];
            ds[g][r] = acc;
            pf += s[p * 9 + r] * f[r];
            dpf += acc * f[r];
            pdf += s[p * 9 + r] * dh[g][r];
        }
        tau[g] = a * pf / 3.0;
        dtau[g] = (da * pf + a * (dpf + pdf)) / 3.0;
        mean_tau += tau[g] / 8.0;
        mean_dtau += dtau[g] / 8.0;
    }
    for (int g = 0; g < 8; ++g) {
        long long p = 8 * t + g;
        for (int r = 0; r < 9; ++r) f[r] = base[p * 9 + r] + (r % 4 == 0 ? 1.0 : 0.0);
        inverse_transpose(f, det3(f), fit);
        double a = alpha[g], da = dalpha[g];
        double ct = mean_tau - tau[g], cdt = mean_dtau - dtau[g];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                // d(F^-T)_ij = -(F^-T dF^T F^-T)_ij
                double dfit = 0.0;
                for (int k = 0; k < 3; ++k)
                    for (int m = 0; m < 3; ++m)
                        dfit -= fit[3 * i + k] * dh[g][3 * m + k] * fit[3 * m + j];
                int r = 3 * i + j;
                out[p * 9 + r] = da * s[p * 9 + r] + a * ds[g][r] + cdt * fit[r] + ct * dfit;
            }
        }
    }
}

// Nodal forces of the point stresses s (int B^T s; in finite strains int P . grad N),
// and the squared norm of the contributions of the points to each node: the port of
// morphogen.homogenization.kernels.voxel_forces; one thread per node.
extern "C" __global__ void voxel_forces(const int nx, const int ny, const int nz,
                                        const long long* slot, const double* s,
                                        const double weight, const double* gg,
                                        const int finite, double* out, double* sq) {
    long long node = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= (long long)nx * ny * nz) return;
    int k = (int)(node % nz);
    int j = (int)((node / nz) % ny);
    int i = (int)(node / ((long long)ny * nz));
    double fx = 0.0, fy = 0.0, fz = 0.0, q = 0.0;
    for (int a = 0; a < 2; ++a) {
        int ei = i - a >= 0 ? i - a : nx - 1;
        for (int b = 0; b < 2; ++b) {
            int ej = j - b >= 0 ? j - b : ny - 1;
            for (int c = 0; c < 2; ++c) {
                int ek = k - c >= 0 ? k - c : nz - 1;
                long long t = slot[((long long)ei * ny + ej) * nz + ek];
                if (t < 0) continue;
                int ln = CORNER_INDEX[a][b][c];
                for (int g = 0; g < 8; ++g) {
                    long long p = 8 * t + g;
                    double gx = weight * gg[(g * 3 + 0) * 8 + ln];
                    double gy = weight * gg[(g * 3 + 1) * 8 + ln];
                    double gz = weight * gg[(g * 3 + 2) * 8 + ln];
                    double cx, cy, cz;
                    if (finite) {
                        const double* t9 = s + p * 9;
                        cx = t9[0] * gx + t9[1] * gy + t9[2] * gz;
                        cy = t9[3] * gx + t9[4] * gy + t9[5] * gz;
                        cz = t9[6] * gx + t9[7] * gy + t9[8] * gz;
                    } else {
                        const double* t6 = s + p * 6;
                        cx = gx * t6[0] + gz * t6[4] + gy * t6[5];
                        cy = gy * t6[1] + gz * t6[3] + gx * t6[5];
                        cz = gz * t6[2] + gy * t6[3] + gx * t6[4];
                    }
                    fx += cx; fy += cy; fz += cz;
                    q += cx * cx + cy * cy + cz * cz;
                }
            }
        }
    }
    out[3 * node] = fx;
    out[3 * node + 1] = fy;
    out[3 * node + 2] = fz;
    sq[node] = q;
}

// Measures of strain of the Gauss points of the voxels elem for the nodal fluctuation w and
// the macroscopic gradient macro (9, row by row): the port of
// morphogen.homogenization.kernels.voxel_measures (B-bar or F-bar with bbar, the gradients
// themselves into base in finite strains); one thread per voxel.
extern "C" __global__ void voxel_measures(const int nx, const int ny, const int nz,
                                          const long long n_el, const long long* elem,
                                          const double* w, const double* macro,
                                          const double* gg, const int finite, const int bbar,
                                          double* out, double* base) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_el) return;
    long long nodes[8];
    element_nodes(elem[t], nx, ny, nz, nodes);
    double ue[8][3];
    for (int a = 0; a < 8; ++a)
        for (int c = 0; c < 3; ++c) ue[a][c] = w[3 * nodes[a] + c];
    double hs[8][9];
    for (int g = 0; g < 8; ++g) {
        gradient(ue, gg, g, hs[g]);
        for (int r = 0; r < 9; ++r) hs[g][r] += macro[r];
    }
    if (!finite) {
        double mean = 0.0;
        for (int g = 0; g < 8; ++g) mean += hs[g][0] + hs[g][4] + hs[g][8];
        mean /= 8.0;
        for (int g = 0; g < 8; ++g) {
            const double* h = hs[g];
            double m[6] = {h[0], h[4], h[8], h[5] + h[7], h[6] + h[2], h[1] + h[3]};
            if (bbar) {
                double shift = (mean - m[0] - m[1] - m[2]) / 3.0;
                for (int r = 0; r < 3; ++r) m[r] += shift;
            }
            long long p = 8 * t + g;
            for (int r = 0; r < 6; ++r) out[p * 6 + r] = m[r];
        }
        return;
    }
    bool valid = true;
    double mean = 0.0;
    double log_j[8];
    for (int g = 0; g < 8; ++g) {
        hs[g][0] += 1.0;
        hs[g][4] += 1.0;
        hs[g][8] += 1.0;
        double det = det3(hs[g]);
        if (!(det > 0.0)) {
            valid = false;
            break;
        }
        log_j[g] = log(det);
        mean += log_j[g] / 8.0;
    }
    const double nan = __longlong_as_double(0x7ff8000000000000LL);
    for (int g = 0; g < 8; ++g) {
        long long p = 8 * t + g;
        if (!valid) {  // a voxel turned inside out: its behaviours fail
            for (int r = 0; r < 9; ++r) {
                out[p * 9 + r] = nan;
                if (bbar) base[p * 9 + r] = nan;
            }
            continue;
        }
        double alpha = bbar ? exp((mean - log_j[g]) / 3.0) : 1.0;
        for (int r = 0; r < 9; ++r) {
            double f = hs[g][r];
            double unit = r % 4 == 0 ? 1.0 : 0.0;
            out[p * 9 + r] = alpha * f - unit;
            if (bbar) base[p * 9 + r] = f - unit;
        }
    }
}

// The stresses of the points of n_el voxels replaced, in place, by those conjugate to the
// measures of B-bar or F-bar: the port of morphogen.homogenization.kernels.voxel_effective;
// one thread per voxel.
extern "C" __global__ void voxel_effective(const long long n_el, const int finite, double* s,
                                           const double* base) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_el) return;
    if (!finite) {
        double mean = 0.0;
        for (int g = 0; g < 8; ++g) {
            long long p = 8 * t + g;
            mean += (s[p * 6] + s[p * 6 + 1] + s[p * 6 + 2]) / 24.0;
        }
        for (int g = 0; g < 8; ++g) {
            long long p = 8 * t + g;
            double shift = mean - (s[p * 6] + s[p * 6 + 1] + s[p * 6 + 2]) / 3.0;
            for (int r = 0; r < 3; ++r) s[p * 6 + r] += shift;
        }
        return;
    }
    double alpha[8], tau[8], f[9], fit[9];
    double mean_log = 0.0;
    for (int g = 0; g < 8; ++g) {
        long long p = 8 * t + g;
        for (int r = 0; r < 9; ++r) f[r] = base[p * 9 + r] + (r % 4 == 0 ? 1.0 : 0.0);
        alpha[g] = log(det3(f));
        mean_log += alpha[g] / 8.0;
    }
    double mean_tau = 0.0;
    for (int g = 0; g < 8; ++g) {
        long long p = 8 * t + g;
        alpha[g] = exp((mean_log - alpha[g]) / 3.0);
        double pf = 0.0;
        for (int r = 0; r < 9; ++r)
            pf += s[p * 9 + r] * (base[p * 9 + r] + (r % 4 == 0 ? 1.0 : 0.0));
        tau[g] = alpha[g] * pf / 3.0;
        mean_tau += tau[g] / 8.0;
    }
    for (int g = 0; g < 8; ++g) {
        long long p = 8 * t + g;
        for (int r = 0; r < 9; ++r) f[r] = base[p * 9 + r] + (r % 4 == 0 ? 1.0 : 0.0);
        inverse_transpose(f, det3(f), fit);
        for (int r = 0; r < 9; ++r)
            s[p * 9 + r] = alpha[g] * s[p * 9 + r] + (mean_tau - tau[g]) * fit[r];
    }
}
"""


def _cupy():
    import cupy

    return cupy


def _c_array(a) -> str:
    """C initializer of a nested integer array."""
    a = np.asarray(a)
    if a.ndim == 0:
        return str(int(a))
    return "{" + ", ".join(_c_array(x) for x in a) + "}"


@lru_cache(maxsize=1)
def _module():
    cp = _cupy()
    code = _SOURCE.replace("{CORNERS}", _c_array(CORNERS)).replace(
        "{CORNER_INDEX}", _c_array(CORNER_INDEX)
    )
    return cp.RawModule(code=code)


def _launch(name: str, n: int, args) -> None:
    if n > 0:
        _module().get_function(name)(((n + _THREADS - 1) // _THREADS,), (_THREADS,), args)


@lru_cache(maxsize=8)
def _behaviours(ns: int, width: int):
    """The CUDA module of the behaviours (:file:`behaviours.cu`) for ``ns`` slip systems
    in the arrays of the crystals and ``width`` internal variables per point, with the
    constants of the Python modules."""
    from importlib import resources

    cp = _cupy()
    constants = _vm.kernel_constants(ns, width)
    defines = "".join(f"#define {name} {value!r}\n" for name, value in constants.items())
    source = resources.files("morphogen.homogenization") / "behaviours.cu"
    return cp.RawModule(code=defines + source.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _dot_kernel():
    cp = _cupy()
    return cp.ReductionKernel("float64 x, float64 y", "float64 z", "x * y", "a + b", "z = a",
                              "0", "morphogen_dot")  # fmt: skip


def dot(a, b) -> float:
    """Dot product of two vectors on the GPU (a CuPy reduction, without cuBLAS)."""
    return float(_dot_kernel()(a, b))


@lru_cache(maxsize=1)
def _updates():
    """The updates of the conjugate gradients, in place and fused."""
    cp = _cupy()
    step = cp.ElementwiseKernel("float64 alpha, float64 p, float64 ap", "float64 x, float64 r",
                                "x += alpha * p; r -= alpha * ap", "morphogen_cg_step")  # fmt: skip
    direction = cp.ElementwiseKernel("float64 beta, float64 z", "float64 p", "p = z + beta * p",
                                     "morphogen_cg_direction")  # fmt: skip
    return step, direction


def conjugate_gradient(apply, precondition, rhs, tol, maxiter):
    """Preconditioned conjugate gradient from ``x = 0`` on the GPU, the algorithm of
    :func:`morphogen.homogenization.solvers.conjugate_gradient` (:func:`minres` after a
    direction of negative curvature)."""
    cp = _cupy()
    step, direction = _updates()
    x = cp.zeros_like(rhs)
    norm_b = math.sqrt(dot(rhs, rhs))
    if norm_b == 0.0:
        return x, 0, 0.0
    r = rhs.copy()
    z = precondition(r)
    p = z.copy()
    rz = dot(r, z)
    res = 1.0
    it = 0
    for it in range(1, maxiter + 1):
        checkpoint()
        ap = apply(p)
        pap = dot(p, ap)
        if pap <= 0.0:
            norm_r = math.sqrt(dot(r, r))
            if pap < 0.0 and norm_r > 0.0:
                if norm_r > norm_b:  # an iterate worse than zero
                    x[:] = 0.0
                    r = rhs.copy()
                    norm_r = norm_b
                dx, more, rel = minres(apply, precondition, r, tol * norm_b / norm_r,
                                       maxiter - it)  # fmt: skip
                return x + dx, it + more, rel * norm_r / norm_b
            break
        alpha = rz / pap
        step(alpha, p, ap, x, r)
        res = math.sqrt(dot(r, r)) / norm_b
        if res <= tol:
            break
        z = precondition(r)
        rz_new = dot(r, z)
        if rz_new <= 0.0:  # the residual is round-off the preconditioner cancels
            break
        direction(rz_new / rz, z, p)
        rz = rz_new
    return x, it, res


@lru_cache(maxsize=1)
def _subtract_kernel():
    """``w -= basis[:k]^T coef[:k]`` on the GPU (the rows of the basis in their order), in
    place: the second half of a Gram-Schmidt pass of :func:`gmres`."""
    cp = _cupy()
    return cp.ElementwiseKernel(
        "raw float64 basis, raw float64 coef, int64 k, int64 n", "float64 w",
        "double acc = w; for (long long j = 0; j < k; ++j) acc -= coef[j] * basis[j * n + i]; "
        "w = acc;", "morphogen_gmres_subtract")  # fmt: skip


@lru_cache(maxsize=1)
def _tangent_kernels():
    """The squared norms of ``C - C^T`` and of ``C`` for the tangent ``C`` of each point, and
    its symmetrization in place as :func:`morphogen.homogenization.tensors.symmetrize` does
    it (one thread per point)."""
    cp = _cupy()
    norms = cp.ElementwiseKernel(
        "raw float64 t, int64 nc", "float64 skew, float64 full",
        "long long o = i * nc * nc; double s = 0.0, c = 0.0; "
        "for (long long r = 0; r < nc; ++r) for (long long q = 0; q < nc; ++q) { "
        "double a = t[o + r * nc + q]; double d = a - t[o + q * nc + r]; c += a * a; "
        "s += d * d; } skew = s; full = c;", "morphogen_tangent_norms")  # fmt: skip
    symmetrize = cp.ElementwiseKernel(
        "int64 nc", "raw float64 t",
        "long long o = i * nc * nc; for (long long r = 0; r < nc; ++r) "
        "for (long long q = r + 1; q < nc; ++q) { "
        "double m = 0.5 * (t[o + r * nc + q] + t[o + q * nc + r]); t[o + r * nc + q] = m; "
        "t[o + q * nc + r] = m; }", "morphogen_symmetrize_points")  # fmt: skip
    return norms, symmetrize


@lru_cache(maxsize=64)
def _combine_kernel(cols: int):
    """``basis[:cols] = p^T basis[:rows]`` on the GPU, in place (``p``: rows x cols): one
    thread per component of the vectors, its ``cols`` new values summed in the order of the
    basis; the vectors that a deflated restart of :func:`gmres` keeps (and those that its
    recycling keeps: up to ``2k + 3`` columns)."""
    cp = _cupy()
    return cp.ElementwiseKernel(
        "raw float64 p, int64 rows, int64 n", "raw float64 basis",
        f"double acc[{cols}]; for (int c = 0; c < {cols}; ++c) acc[c] = 0.0; "
        "for (long long j = 0; j < rows; ++j) { double b = basis[j * n + i]; "
        f"for (int c = 0; c < {cols}; ++c) acc[c] += p[j * {cols} + c] * b; }} "
        f"for (int c = 0; c < {cols}; ++c) basis[c * n + i] = acc[c];",
        f"morphogen_gmres_combine_{cols}")  # fmt: skip


class _Device:
    """The operations on the vectors of the recycling of :func:`gmres` (GCRO-DR,
    :func:`morphogen.homogenization.solvers._gcro_dr`) on the GPU: the reductions and the
    kernels of its Gram-Schmidt passes and of its deflated restarts."""

    @staticmethod
    def norm(w) -> float:
        return math.sqrt(dot(w, w))

    @staticmethod
    def zeros_like(a):
        return _cupy().zeros_like(a)

    @staticmethod
    def empty(rows: int, n: int):
        return _cupy().empty((rows, n))

    @staticmethod
    def dots(basis, k: int, w) -> np.ndarray:
        return _cupy().asnumpy(_dot_kernel()(basis[:k], w, axis=1))

    @staticmethod
    def subtract(basis, k: int, coef, w) -> None:
        cp = _cupy()
        _subtract_kernel()(basis, cp.asarray(np.ascontiguousarray(coef, dtype=np.float64)),
                           np.int64(k), np.int64(w.size), w)  # fmt: skip

    @staticmethod
    def combine(basis, rows: int, p) -> None:
        cp = _cupy()
        p = np.ascontiguousarray(p, dtype=np.float64)
        n = int(basis.shape[1])
        _combine_kernel(int(p.shape[1]))(cp.asarray(p), np.int64(rows), np.int64(n), basis,
                                         size=n)  # fmt: skip


_DEVICE = _Device()


def gmres(apply, precondition, rhs, tol, maxiter, restart: int = GMRES_RESTART,
          deflate: int = GMRES_DEFLATE, recycle: Recycling | None = None,
          stagnation: int = 0):  # fmt: skip
    """GMRES with deflated restarts, preconditioned on the right on the GPU, the algorithm
    of :func:`morphogen.homogenization.solvers.gmres`: the basis on the GPU (the dot
    products of a Gram-Schmidt pass in one reduction over its rows, the vectors kept at a
    deflated restart combined in place), the Hessenberg matrix, its factors and the harmonic
    Ritz vectors on the CPU; ``recycle`` and ``stagnation`` as there, the recycled vectors
    on the GPU."""
    cp = _cupy()
    subtract = _subtract_kernel()
    x = cp.zeros_like(rhs)
    norm_b = math.sqrt(dot(rhs, rhs))
    if norm_b == 0.0:
        return x, 0, 0.0
    n = int(rhs.size)
    m = max(1, min(int(restart), int(maxiter)))
    k_max = min(int(deflate), m - 2)
    if recycle is not None and recycle.u is not None and k_max > 0:
        out = _gcro_dr(_DEVICE, apply, precondition, rhs, tol, maxiter, m, k_max, recycle,
                       stagnation)  # fmt: skip
        if out is not None:
            return out
    ends = [1.0]
    basis = cp.empty((m + 1, n))
    h = np.zeros((m + 1, m))
    tri = np.zeros((m + 1, m))
    cs, sn, g, c = np.zeros(m), np.zeros(m), np.zeros(m + 1), np.zeros(m + 1)
    q0 = np.ones((1, 1))
    beta, total, res = norm_b, 0, 1.0
    s, deflated = 0, False
    cp.divide(rhs, beta, out=basis[0])
    c[0] = beta
    while total < maxiter:
        g[:] = 0.0
        g[: s + 1] = q0.T @ c[: s + 1]
        k = s
        for j in range(s, min(m, s + maxiter - total)):
            checkpoint()
            w = apply(precondition(basis[j]))
            h[:, j] = 0.0
            before = math.sqrt(dot(w, w))
            for _ in range(2):
                coef = _dot_kernel()(basis[: j + 1], w, axis=1)
                subtract(basis, coef, np.int64(j + 1), np.int64(n), w)
                h[: j + 1, j] += cp.asnumpy(coef)
                after = math.sqrt(dot(w, w))
                if after > _REORTHOGONALIZE * before:
                    break
                before = after
            h[j + 1, j] = after
            col = tri[:, j]
            col[:] = h[:, j]
            if s:
                col[: s + 1] = q0.T @ col[: s + 1]
            for i in range(s, j):
                a, b = col[i], col[i + 1]
                col[i] = cs[i] * a + sn[i] * b
                col[i + 1] = cs[i] * b - sn[i] * a
            gamma = math.hypot(col[j], after)
            if not gamma > 0.0:
                break
            cs[j], sn[j] = col[j] / gamma, after / gamma
            col[j], col[j + 1] = gamma, 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] *= cs[j]
            k = j + 1
            total += 1
            res = abs(g[k]) / norm_b
            if res <= tol or after == 0.0:
                break
            cp.divide(w, after, out=basis[j + 1])
        if k == 0:
            break
        y = np.zeros(m + 1)
        for i in range(k - 1, -1, -1):
            y[i] = (g[i] - tri[i, i + 1 : k] @ y[i + 1 : k]) / tri[i, i]
        combined = cp.zeros_like(rhs)
        subtract(basis, cp.asarray(-y[:k]), np.int64(k), np.int64(n), combined)
        x += precondition(combined)
        if total >= maxiter or (res <= tol and not deflated):
            break
        previous = beta
        full = res > tol and k == m and h[m, m - 1] > 0.0
        if full and stagnation:
            ends.append(res)
            if _stagnates(ends, stagnation):
                break
        start = _deflation(h, c, y, m, k_max) if full and k_max > 0 else None
        if start is not None:
            p, hk, ck = start
            beta = res * norm_b
            if not beta < previous:
                break
            s = p.shape[1] - 1
            _combine_kernel(s + 1)(cp.asarray(np.ascontiguousarray(p)), np.int64(m + 1),
                                   np.int64(n), basis, size=n)  # fmt: skip
            h[:] = 0.0
            h[: s + 1, :s] = hk
            c[:] = 0.0
            c[: s + 1] = ck
            q0, r0 = np.linalg.qr(hk, mode="complete")
            tri[:] = 0.0
            tri[: s + 1, :s] = r0
            deflated = True
            continue
        r = rhs - apply(x)
        beta = math.sqrt(dot(r, r))
        res = beta / norm_b
        if res <= tol or not beta < previous:
            break
        if deflated and recycle is not None:  # before the plain cycle overwrites them
            _keep(_DEVICE, recycle, basis, h[: s + 1, :s])
        cp.divide(r, beta, out=basis[0])
        c[:] = 0.0
        c[0] = beta
        s, q0, deflated = 0, np.ones((1, 1)), False
    if deflated and recycle is not None:
        _keep(_DEVICE, recycle, basis, h[: s + 1, :s])
    return x, total, res


def minres(apply, precondition, rhs, tol, maxiter):
    """Preconditioned MINRES on the GPU, the algorithm of
    :func:`morphogen.homogenization.solvers.minres`."""
    cp = _cupy()
    x = cp.zeros_like(rhs)
    norm_b = math.sqrt(dot(rhs, rhs))
    if norm_b == 0.0:
        return x, 0, 0.0
    r = rhs.copy()
    res = 1.0
    total = 0
    while total < maxiter:
        dx, its = _minres_cycle(apply, precondition, r, tol / res, maxiter - total)
        total += its
        x += dx
        r = rhs - apply(x)
        res = math.sqrt(dot(r, r)) / norm_b
        if res <= tol or its == 0:
            break
    return x, total, res


def _minres_cycle(apply, precondition, rhs, tol, maxiter):
    """One cycle of :func:`minres` from ``x = 0``: (x, iterations)."""
    cp = _cupy()
    x = cp.zeros_like(rhs)
    r1 = rhs.copy()
    y = precondition(r1)
    beta1 = dot(r1, y)
    if not beta1 > 0.0:
        return x, 0
    beta1 = math.sqrt(beta1)
    oldb, beta, dbar, epsln, phibar, cs, sn = 0.0, beta1, 0.0, 0.0, beta1, -1.0, 0.0
    w = cp.zeros_like(rhs)
    w2 = cp.zeros_like(rhs)
    r2 = r1
    it = 0
    for it in range(1, maxiter + 1):
        checkpoint()
        v = y / beta
        y = apply(v)
        if it >= 2:
            y -= (beta / oldb) * r1
        alfa = dot(v, y)
        y -= (alfa / beta) * r2
        r1 = r2
        r2 = y
        y = precondition(r2)
        oldb = beta
        beta = dot(r2, y)
        if beta < 0.0:  # a preconditioner that is not positive definite
            break
        beta = math.sqrt(beta)
        oldeps = epsln
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        epsln = sn * beta
        dbar = -cs * beta
        gamma = max(math.hypot(gbar, beta), 1e-300)
        cs = gbar / gamma
        sn = beta / gamma
        phi = cs * phibar
        phibar = sn * phibar
        w1 = w2
        w2 = w
        w = (v - oldeps * w1 - delta * w2) / gamma
        x += phi * w
        if abs(phibar) <= tol * beta1 or beta == 0.0:
            break
    return x, it


class FourierPreconditioner:
    """The Fourier preconditioner of :class:`morphogen.homogenization.fft.
    FourierPreconditioner` on the GPU, for the element matrix ``k_ref`` of the reference
    medium: its inverse symbol computed on the GPU, the fields component by component,
    cuFFT transforms of the grid."""

    def __init__(self, shape, k_ref, nd: int):
        from morphogen.homogenization.fft import _stencil

        cp = _cupy()
        self.shape = tuple(int(s) for s in shape)
        self.nd = int(nd)
        nx, ny, nz = self.shape
        self.n_freq = nx * ny * (nz // 2 + 1)
        stencil = _stencil(np.asarray(k_ref, dtype=float), self.nd)
        blocks = cp.asarray(np.ascontiguousarray(stencil))
        self.inverse = cp.empty((self.n_freq, self.nd, self.nd), dtype=np.complex128)
        _launch("fourier_inverse", self.n_freq, (np.int32(nx), np.int32(ny), np.int32(nz),
                np.int32(self.nd), blocks, self.inverse))  # fmt: skip

    def __call__(self, r):
        cp = _cupy()
        nd, shape = self.nd, self.shape
        field = cp.ascontiguousarray(r.reshape(-1, nd).T).reshape(nd, *shape)
        hat = cp.ascontiguousarray(cp.fft.rfftn(field, axes=(1, 2, 3)))
        out = cp.empty_like(hat)
        _launch("block_multiply", self.n_freq, (np.int64(self.n_freq), np.int32(nd),
                self.inverse, hat, out))  # fmt: skip
        back = cp.fft.irfftn(out, s=shape, axes=(1, 2, 3))
        return cp.ascontiguousarray(back.reshape(nd, -1).T).reshape(-1)


class LinearOperator:
    """The stiffness operator of the voxels of a linear FFT problem on the GPU (the
    element matrices of its materials)."""

    def __init__(self, problem):
        cp = _cupy()
        self.shape = problem.shape
        self.nd = problem.nd
        self.group = cp.asarray(np.ascontiguousarray(problem.mat, dtype=np.int64))
        self.mats = cp.asarray(np.ascontiguousarray(problem.element_matrices, dtype=np.float64))

    def __call__(self, w):
        cp = _cupy()
        nx, ny, nz = self.shape
        out = cp.empty_like(w)
        _launch("apply_operator", nx * ny * nz, (np.int32(nx), np.int32(ny), np.int32(nz),
                np.int32(self.nd), self.group, self.mats, w, out))  # fmt: skip
        return out


def solve_linear(problem, rhs, tol: float, maxiter: int):
    """Solution of the linear system of an FFT problem on the GPU: (x, iterations,
    relative residual), ``x`` on the CPU. The operator and the preconditioner are kept
    by the problem."""
    cp = _cupy()
    state = problem.__dict__.setdefault("_gpu", {})
    if "operator" not in state:
        state["operator"] = LinearOperator(problem)
        k_ref = problem._reference_element()
        state["precondition"] = FourierPreconditioner(problem.shape, k_ref, problem.nd)
    x, it, res = conjugate_gradient(state["operator"], state["precondition"], cp.asarray(rhs),
                                    tol, maxiter)  # fmt: skip
    return cp.asnumpy(x), it, res


class VoxelPaths(_VoxelPaths):
    """The discretization of the loading paths on the voxels
    (:class:`~morphogen.homogenization.fft_nonlinear._VoxelPaths`) on the GPU: the arrays of
    the points (measures of strain, stresses, tangents, internal variables) stay in its
    memory, where the measures, the behaviours, the stresses conjugate to B-bar and F-bar,
    the nodal forces and the products by the tangent operator are computed. The weights of
    the points stay on the CPU (they are uniform)."""

    def _buffers(self, n_pts: int, nc: int) -> None:
        cp = _cupy()
        p = self.problem
        beh = p.behaviours
        self.elem_gpu = cp.asarray(np.ascontiguousarray(self.elem, dtype=np.int64))
        self.slot_gpu = cp.asarray(np.ascontiguousarray(self.slot, dtype=np.int64))
        self.g_gpu = cp.asarray(np.ascontiguousarray(self.g_gauss, dtype=np.float64))
        self.precondition = FourierPreconditioner(self.shape, self.reference, 3)
        self.work = cp.empty((n_pts, nc))
        # F-bar: the gradients of the points and the stresses of their behaviours
        keep = self.finite and self.bbar
        self.base = cp.zeros((n_pts if keep else 1, 9))
        self.stress = cp.zeros((n_pts if keep else 1, nc))
        self.zero = cp.zeros(9)
        self.still = cp.zeros(self.n_q)
        self.squares = cp.empty(int(np.prod(self.shape)))
        self.tangent = None
        self.phase_points = None  # the points of each phase, for the asymmetry
        # the behaviours of the points
        self.points = tuple(cp.asarray(np.ascontiguousarray(a, dtype=np.int64))
                            for a in (p.kind, p.point_mat, p.pmat))  # fmt: skip
        matrices, eigen, params, nk, systems, cp_par, cp_h, cp_n = beh.data
        if any(crystal.extended for crystal in beh.crystals):
            raise ValueError(
                "GPU: the twins and the cleavage planes of the crystals are integrated on the "
                "CPU only (device: cpu)"
            )
        # the rows of the kernels: von Mises's parameters and Hill's coefficients, or the
        # parameters of a porous point
        width = _vm.ROW_WIDTH
        params = np.pad(params, ((0, 0), (0, max(0, width - params.shape[1]))))[:, :width]
        dev = [cp.asarray(np.ascontiguousarray(a, dtype=np.float64))
               for a in (matrices, eigen, params, systems, cp_par, cp_h)]  # fmt: skip
        self.data = (*dev[:3], np.int32(nk), *dev[3:],
                     cp.asarray(np.ascontiguousarray(cp_n, dtype=np.int64)))  # fmt: skip
        self.module = _behaviours(max(int(cp_par.shape[1]), 1), beh.width)
        self.failed = cp.zeros(1, dtype=np.uint32)
        self.chunk = _CHUNK_START

    def arrays(self, n_pts: int, nc: int, width: int) -> tuple:
        cp = _cupy()
        return (cp.zeros((n_pts, nc)), np.full(n_pts, self.weight), cp.zeros((n_pts, nc)),
                cp.zeros((n_pts, nc, nc)), cp.zeros((n_pts, width)),
                cp.zeros((n_pts, width)))  # fmt: skip

    def host(self, *arrays) -> tuple:
        cp = _cupy()
        return tuple(cp.asnumpy(a) if isinstance(a, cp.ndarray) else a for a in arrays)

    def measures(self, q, e, grads, wts) -> None:
        cp = _cupy()
        nx, ny, nz = self.shape
        n_el = int(self.elem.size)
        q_gpu = cp.asarray(np.ascontiguousarray(q, dtype=np.float64))
        macro = cp.asarray(np.ascontiguousarray(self._macro(e), dtype=np.float64))
        _launch("voxel_measures", n_el, (np.int32(nx), np.int32(ny), np.int32(nz),
                np.int64(n_el), self.elem_gpu, q_gpu, macro, self.g_gpu, np.int32(self.finite),
                np.int32(self.bbar), grads, self.base))  # fmt: skip
        wts.fill(self.weight)

    def integrate(self, grads, dtemp, dt, old, new, stress, tangent, guess, ratio) -> int:
        """The behaviours of the points on the GPU (:file:`behaviours.cu`), in launches of
        ``chunk`` points that follow their durations, their tangents symmetrized as the
        problem says."""
        cp = _cupy()
        kernel = self.module.get_function("integrate_finite" if self.finite else "integrate_small")
        symmetric = bool(self.problem.symmetrize)
        self.failed.fill(0)
        n = int(grads.shape[0])
        first = 0
        while first < n:
            end = min(first + self.chunk, n)
            blocks = (end - first + _POINT_THREADS - 1) // _POINT_THREADS
            t0 = time.perf_counter()
            kernel((blocks,), (_POINT_THREADS,), (np.int64(first), np.int64(end), *self.points,
                   *self.data, grads, np.float64(dtemp), np.float64(dt), old, new, stress,
                   tangent, np.int32(guess), np.float64(ratio), np.int32(symmetric),
                   self.failed))  # fmt: skip
            cp.cuda.get_current_stream().synchronize()
            elapsed = time.perf_counter() - t0
            if elapsed > 2.0 * _LAUNCH_S:
                self.chunk = max(self.chunk // 4, _CHUNK_MIN)
            elif elapsed < 0.5 * _LAUNCH_S and end - first == self.chunk:
                self.chunk = min(2 * self.chunk, _CHUNK_MAX)
            first = end
        failed = int(self.failed.get()[0])
        points = _mfront_points(self.problem)
        if points is not None and points.points.size:
            # MGIS integrates the points of MFront behaviours on the CPU
            idx = cp.asarray(points.points)
            nc = grads.shape[1]
            out = [cp.asnumpy(new[idx]), np.empty((idx.size, nc)), np.empty((idx.size, nc, nc))]
            failed += points.integrate_compact(cp.asnumpy(grads[idx]), dtemp, dt,
                                               cp.asnumpy(old[idx]), *out)  # fmt: skip
            new[idx], stress[idx], tangent[idx] = (cp.asarray(a) for a in out)
        return failed

    def stress_scale(self, sigma, wts) -> float:
        cp = _cupy()
        return self.weight * float(cp.sqrt((sigma * sigma).sum(axis=1)).sum())

    def _nodal(self, s):
        """Nodal forces of the point stresses ``s`` (on the GPU) and the norm of the
        contributions of the points before they cancel."""
        cp = _cupy()
        nx, ny, nz = self.shape
        out = cp.empty(self.n_q)
        _launch("voxel_forces", nx * ny * nz, (np.int32(nx), np.int32(ny), np.int32(nz),
                self.slot_gpu, s, np.float64(self.weight), self.g_gpu, np.int32(self.finite),
                out, self.squares))  # fmt: skip
        return out, math.sqrt(float(self.squares.sum()))

    def _host_forces(self, s) -> np.ndarray:
        """Nodal forces on the CPU, zero when they are round-off of uniform stresses."""
        cp = _cupy()
        out, scale = self._nodal(s)
        r = cp.asnumpy(out)
        if np.linalg.norm(r) <= NEGLIGIBLE * scale:
            r[:] = 0.0
        return r, scale

    def forces(self, sigma, wts):
        cp = _cupy()
        if self.bbar:
            if self.finite:
                cp.copyto(self.stress, sigma)
            n_el = int(self.elem.size)
            _launch("voxel_effective", n_el, (np.int64(n_el), np.int32(self.finite), sigma,
                    self.base))  # fmt: skip
        r, scale = self._host_forces(sigma)
        return r, self.weight * cp.asnumpy(sigma.sum(axis=0)), scale

    def _increments(self, v, direction, tangent):
        nx, ny, nz = self.shape
        n_el = int(self.elem.size)
        _launch("voxel_increments", n_el, (np.int32(nx), np.int32(ny), np.int32(nz),
                np.int64(n_el), self.elem_gpu, v, direction, tangent, self.stress, self.base,
                self.g_gpu, np.int32(self.finite), np.int32(self.bbar), self.work))  # fmt: skip
        return self.work

    def __call__(self, v):
        """The tangent operator of the last linearization applied to the nodal
        displacements ``v`` (on the GPU)."""
        return self._nodal(self._increments(v, self.zero, self.tangent))[0]

    def linearize(self, tangent, wts, first: bool, symmetric: bool | None = None):
        self.tangent = tangent
        symmetric = self.problem.symmetrize if symmetric is None else symmetric
        return KrylovSolver(self, self.tol, self.maxiter, symmetric, self.recycling)

    def asymmetry(self, tangent, wts) -> float:
        """The largest asymmetry of the tangents of the points of the phases
        (:func:`morphogen.homogenization.nonlinear.asymmetry`), their norms on the GPU."""
        cp = _cupy()
        n, nc = int(tangent.shape[0]), int(tangent.shape[1])
        if n == 0:
            return 0.0
        skew, full = cp.empty(n), cp.empty(n)
        _tangent_kernels()[0](tangent, np.int64(nc), skew, full)
        if self.phase_points is None:
            groups = np.asarray(self.problem.point_group)
            self.phase_points = [cp.asarray(np.flatnonzero(groups == g))
                                 for g in np.unique(groups)]  # fmt: skip
        return max(_asymmetry_of(float(skew[p].sum()), float(full[p].sum()))
                   for p in self.phase_points)  # uniform weights  # fmt: skip

    def symmetrize(self, tangent) -> None:
        """The tangents of the points replaced by their symmetric parts on the GPU, in place
        (to the bit those of the kernels that symmetrize them)."""
        n, nc = int(tangent.shape[0]), int(tangent.shape[1])
        _tangent_kernels()[1](np.int64(nc), tangent, size=n)

    def macro_response(self, tangent, wts, v) -> np.ndarray:
        cp = _cupy()
        ds = self._increments(cp.asarray(np.ascontiguousarray(v, dtype=np.float64)), self.zero,
                              tangent)  # fmt: skip
        total = self.weight * cp.asnumpy(ds.sum(axis=0))
        return fin.macro_part(total) if self.finite else total

    def coupling(self, tangent, wts, unknown):
        cp = _cupy()
        g = np.empty((self.n_q, unknown.size))
        kee = np.zeros((6, 6))
        for k, j in enumerate(unknown):
            unit = np.zeros(6)
            unit[j] = 1.0
            ds = self._increments(self.still, cp.asarray(self._macro(unit)), tangent)
            g[:, k] = self._host_forces(ds)[0]
            total = self.weight * cp.asnumpy(ds.sum(axis=0))
            kee[:, j] = fin.macro_part(total) if self.finite else total
        return g, kee


class KrylovSolver:
    """Conjugate gradients of the tangent operator of :class:`VoxelPaths` on the GPU (GMRES
    if it is not ``symmetric``, recycling the space of ``recycle`` and stopped where it
    stagnates), with the interface of
    :class:`morphogen.homogenization.fft_nonlinear._KrylovSolver` (vectors on the CPU)."""

    method = "fft-gpu"

    def __init__(self, operator: VoxelPaths, tol: float, maxiter: int, symmetric: bool = True,
                 recycle: Recycling | None = None):  # fmt: skip
        self.operator = operator
        self.tol = float(tol)
        self.maxiter = int(maxiter)
        self.krylov = conjugate_gradient
        if not symmetric:
            self.krylov = partial(gmres, recycle=recycle, stagnation=GMRES_STAGNATION)
        if recycle is not None:
            recycle.renew()

    def solve(self, rhs, x0=None, tol=None) -> tuple[np.ndarray, SolveInfo]:
        """Solution of ``K x = rhs`` to the relative residual ``tol``, from ``x0`` if
        given; the mean of ``rhs`` (round-off) is removed."""
        cp = _cupy()
        t0 = time.perf_counter()
        b = np.asarray(rhs, dtype=float).reshape(-1, 3)
        b = (b - b.mean(axis=0)).ravel()
        n = b.size
        tol = self.tol if tol is None else float(tol)
        norm_b = float(np.linalg.norm(b))
        if norm_b == 0.0:
            return np.zeros(n), SolveInfo(self.method, n, 0, 0.0, 0.0, 0.0)
        b_gpu = cp.asarray(b)
        start = None if x0 is None or not np.any(x0) else cp.asarray(x0, dtype=np.float64)
        r = b_gpu if start is None else b_gpu - self.operator(start)
        norm_r = math.sqrt(dot(r, r))
        iterations, residual = 0, norm_r / norm_b
        if norm_r <= tol * norm_b:
            x = cp.zeros(n) if start is None else start
        else:
            op = self.operator
            x, iterations, rel = self.krylov(op, op.precondition, r, tol * norm_b / norm_r,
                                             self.maxiter)  # fmt: skip
            residual = rel * norm_r / norm_b
            if start is not None:
                x += start
        if residual > 10 * tol:
            log.warning(
                "FFT solver stopped at relative residual %.2e (tolerance %.1e)", residual, tol
            )
        info = SolveInfo(self.method, n, iterations, residual, 0.0, time.perf_counter() - t0)
        return cp.asnumpy(x), info
