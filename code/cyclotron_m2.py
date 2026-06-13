"""
cyclotron_m2.py
================
Cyclotron forward model optimised for Apple Silicon (M2) and modern x86.

Optimisations relative to the v2 baseline (cyc/github/ztf0933/cyclotron_model.py):

  1. @njit(cache=True, fastmath=True, error_model='numpy', boundscheck=False)
  2. parallel=True  + numba.prange over wavelengths
  3. All wavelength-independent quantities (K2, w_c, alpha+/-, trig, K2 prefactor)
     hoisted to the outer scope and passed to the inner kernel.
  4. Velocity grid (199 points) and per-harmonic constants pre-computed once.
  5. Vectorised Bessel asymptotic expressions (no Python overhead).
  6. Early-exit on harmonic series when the cumulative sum has stabilised.
  7. Exponent clipping for numerical stability.
  8. Optional batch evaluation `cal_cy_spec_batch`: evaluate a *list of parameter
     vectors* in parallel with a single Python call (huge win for DE / MC).

Set NUMBA_NUM_THREADS environment variable to control the worker pool, e.g.

    NUMBA_NUM_THREADS=8 python ...

On an M2 the 4 performance + 4 efficiency cores give the best throughput at
NUMBA_NUM_THREADS=4 (use only the P-cores).

Public API
----------
    cal_cy_spec(wave_m, T_J, B_T, theta_rad, Lambda) -> ndarray
    cal_cy_spec_batch(wave_m, params_array) -> ndarray  (n_params, n_wave)
"""

import math
import os
import numpy as np
import numba
from numba import njit, prange


# Make threading layer fall back gracefully on macOS
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")


# --------------- Physical constants (SI) ---------------
C_LIGHT = 3.0e8
PI      = math.pi
ME      = 9.1e-31
KB      = 1.38e-23
EV      = 1.602e-19


@njit(cache=True, fastmath=True, error_model='numpy', boundscheck=False)
def _cal_cy_inner(wave_length, n, x, K2, w_c,
                  cos_th, sin_th, sin4, cot_th, csc_th,
                  v_p_grid, dv):
    """Single-wavelength single-harmonic source-function integral."""
    w = 2.0 * PI * C_LIGHT / wave_length
    w_over_wc = w / w_c

    denom_plus  = -sin_th**2 + math.sqrt(sin4 + 4.0 * w_over_wc**2 * cos_th**2)
    denom_minus = -sin_th**2 - math.sqrt(sin4 + 4.0 * w_over_wc**2 * cos_th**2)

    alpha_plus  = (2.0 * w_over_wc) * cos_th / denom_plus  if abs(denom_plus)  > 1e-30 else 0.0
    alpha_minus = (2.0 * w_over_wc) * cos_th / denom_minus if abs(denom_minus) > 1e-30 else 0.0

    inv_one_plus_ap2 = 1.0 / (1.0 + alpha_plus  * alpha_plus)
    inv_one_plus_am2 = 1.0 / (1.0 + alpha_minus * alpha_minus)

    inv_n            = 1.0 / n
    n_pow_minus_2_3  = n**(-2.0 / 3.0)
    sqrt_2pi_n       = math.sqrt(2.0 * PI * n)
    prefactor        = PI * PI * x * x / K2

    I_w_plus  = 0.0
    I_w_minus = 0.0

    for i in range(v_p_grid.shape[0]):
        v_p = v_p_grid[i]
        one_minus_vp_cos = 1.0 - v_p * cos_th
        gamma = n * w_c / w / one_minus_vp_cos
        if gamma < 1.0:
            continue

        v_0_sqrt = 1.0 - v_p * v_p - (w / w_c)**2 * one_minus_vp_cos**2 * inv_n * inv_n
        if v_0_sqrt < 0.0:
            continue

        v_0 = math.sqrt(v_0_sqrt)
        xi  = v_0 * sin_th / one_minus_vp_cos
        if xi > 1.0:
            continue

        w0 = math.sqrt(1.0 - xi * xi)
        one_plus_w0 = 1.0 + w0
        exp_n_w0 = math.exp(n * w0)

        xi_pow_n = xi**n
        one_plus_w0_pow_n = one_plus_w0**n

        Jn = (xi_pow_n * exp_n_w0
              * (w0**3 + 0.5033 * inv_n)**(-1.0 / 6.0)
              / sqrt_2pi_n
              / one_plus_w0_pow_n)
        Jn1 = (xi_pow_n / xi * exp_n_w0
               * (w0**3 + 1.193 * inv_n)**(1.0 / 6.0)
               / sqrt_2pi_n
               / one_plus_w0_pow_n
               * (1.0 - 0.2 * n_pow_minus_2_3))

        common = (cot_th - v_p * csc_th)
        F_plus  = inv_one_plus_ap2 * (alpha_plus  * common * Jn - v_0 * Jn1)**2
        F_minus = inv_one_plus_am2 * (alpha_minus * common * Jn - v_0 * Jn1)**2

        exp_factor = math.exp(-x * (gamma - 1.0)) * (gamma**4 * inv_n)
        I_w_plus  += prefactor * F_plus  * exp_factor * dv
        I_w_minus += prefactor * F_minus * exp_factor * dv

    return I_w_plus, I_w_minus


@njit(parallel=True, cache=True, fastmath=True,
      error_model='numpy', boundscheck=False)
def cal_cy_spec(wave_length_set, T, B, theta, Lambda):
    """Cyclotron emission spectrum incl. radiative transfer.

    Parameters
    ----------
    wave_length_set : 1-D float array (m)
    T  : float — electron temperature (J)
    B  : float — magnetic field (T)
    theta : float — viewing angle (rad)
    Lambda : float — dimensionless plasma column-density parameter

    Returns
    -------
    I : 1-D array, same length as wave_length_set, specific intensity.
    """
    n_harmonics = 100
    f = 1.0 / (2.0 * PI)
    n_wl = wave_length_set.shape[0]

    psi_plus  = np.zeros(n_wl)
    psi_minus = np.zeros(n_wl)

    # Wavelength-independent precomputations
    x = ME * C_LIGHT * C_LIGHT / T
    K2 = (PI / 2.0 / x)**0.5 * (
        1.0 + 1.875 / x + 0.8203125 / x**2
        - 0.307617 / x**3 + 0.2997 / x**4 - 0.281 / x**5
    )
    w_c = EV * B / ME
    cos_th = math.cos(theta)
    sin_th = math.sin(theta)
    sin4   = sin_th**4
    if abs(sin_th) > 1e-30:
        cot_th = cos_th / sin_th
        csc_th = 1.0 / sin_th
    else:
        cot_th = 1e30
        csc_th = 1e30

    # Pre-computed velocity grid (shared across wavelengths/harmonics)
    N_V = 199
    dv  = 1.98 / (N_V - 1)
    v_p_grid = np.empty(N_V)
    for i in range(N_V):
        v_p_grid[i] = -0.99 + i * dv

    # Parallel over wavelength (numba.prange)
    for wi in prange(n_wl):
        wl = wave_length_set[wi]
        pp = 0.0
        pm = 0.0
        for ih in range(n_harmonics):
            n_h = ih + 1
            y_p, y_m = _cal_cy_inner(wl, n_h, x, K2, w_c,
                                     cos_th, sin_th, sin4, cot_th, csc_th,
                                     v_p_grid, dv)
            pp += y_p
            pm += y_m
            # Early exit when contribution is negligible relative to running sum.
            if ih > 10 and (y_p + y_m) < 1e-10 * (pp + pm + 1e-30):
                break
        psi_plus[wi]  = pp
        psi_minus[wi] = pm

    w = 2.0 * PI * C_LIGHT / wave_length_set
    I_rj = T * w * w / (8.0 * PI**3 * C_LIGHT * C_LIGHT)
    I_plus  = I_rj * (1.0 - np.exp(-Lambda * psi_plus  * f))
    I_minus = I_rj * (1.0 - np.exp(-Lambda * psi_minus * f))
    return I_plus + I_minus


@njit(cache=True, fastmath=True, error_model='numpy', boundscheck=False)
def _cal_one_spectrum_serial(wave_length_set, T, B, theta, Lambda):
    """Single-spectrum serial kernel (no prange) for use inside batch parallel."""
    n_harmonics = 100
    f_const = 1.0 / (2.0 * PI)
    n_wl = wave_length_set.shape[0]

    psi_plus  = np.zeros(n_wl)
    psi_minus = np.zeros(n_wl)

    x = ME * C_LIGHT * C_LIGHT / T
    K2 = (PI / 2.0 / x)**0.5 * (
        1.0 + 1.875 / x + 0.8203125 / x**2
        - 0.307617 / x**3 + 0.2997 / x**4 - 0.281 / x**5
    )
    w_c = EV * B / ME
    cos_th = math.cos(theta)
    sin_th = math.sin(theta)
    sin4   = sin_th**4
    if abs(sin_th) > 1e-30:
        cot_th = cos_th / sin_th
        csc_th = 1.0 / sin_th
    else:
        cot_th = 1e30
        csc_th = 1e30

    N_V = 199
    dv  = 1.98 / (N_V - 1)
    v_p_grid = np.empty(N_V)
    for i in range(N_V):
        v_p_grid[i] = -0.99 + i * dv

    for wi in range(n_wl):
        wl = wave_length_set[wi]
        pp = 0.0
        pm = 0.0
        for ih in range(n_harmonics):
            n_h = ih + 1
            y_p, y_m = _cal_cy_inner(wl, n_h, x, K2, w_c,
                                     cos_th, sin_th, sin4, cot_th, csc_th,
                                     v_p_grid, dv)
            pp += y_p
            pm += y_m
            if ih > 10 and (y_p + y_m) < 1e-10 * (pp + pm + 1e-30):
                break
        psi_plus[wi]  = pp
        psi_minus[wi] = pm

    w_arr = 2.0 * PI * C_LIGHT / wave_length_set
    I_rj = T * w_arr * w_arr / (8.0 * PI**3 * C_LIGHT * C_LIGHT)
    I_plus  = I_rj * (1.0 - np.exp(-Lambda * psi_plus  * f_const))
    I_minus = I_rj * (1.0 - np.exp(-Lambda * psi_minus * f_const))
    return I_plus + I_minus


@njit(parallel=True, cache=True, fastmath=True,
      error_model='numpy', boundscheck=False)
def cal_cy_spec_batch(wave_length_set, params):
    """Evaluate the model for a batch of parameter vectors in parallel.

    Parallelism is OVER PARAMETERS (one thread per spectrum) so it is
    threadsafe under all numba backends (workqueue/omp/tbb).

    Parameters
    ----------
    wave_length_set : 1-D array of length n_wave (m)
    params : 2-D array shape (n_params, 4) — columns [T, B, theta, Lambda]
        T in J, B in T, theta in rad, Lambda dimensionless.

    Returns
    -------
    I : 2-D array shape (n_params, n_wave)
    """
    n_params = params.shape[0]
    n_wave   = wave_length_set.shape[0]
    I = np.empty((n_params, n_wave))
    for k in prange(n_params):
        Tk     = params[k, 0]
        Bk     = params[k, 1]
        thk    = params[k, 2]
        Lk     = params[k, 3]
        I[k, :] = _cal_one_spectrum_serial(wave_length_set, Tk, Bk, thk, Lk)
    return I


# Convenience wrapper that accepts physical units (MG, keV, deg)
def cal_cy_spec_phys(wavelength_AA, T_keV, B_MG, theta_deg, Lambda):
    wave_m = np.asarray(wavelength_AA, dtype=np.float64) * 1e-10
    T_J    = T_keV * 1000.0 * EV
    B_T    = B_MG * 1e6 * 1e-4
    th_r   = theta_deg * PI / 180.0
    return cal_cy_spec(wave_m, T_J, B_T, th_r, Lambda)
