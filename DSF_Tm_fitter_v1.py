#!/usr/bin/env python3
"""
This function takes two arrays, T (temperature, Celsius) and F (fluorescence signal)
> F array may contain any number of replicates, as long as the number, n, is specified in the function call

returns fitted values for Tm and the standard error of fit from the covariance matrix (floats)
"""

from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

## finds local minimum (lower baseline) and maximum (upper baseline) to trim dataset for fitting

import numpy as np
from scipy.signal import argrelextrema

def trim_dsf_for_fitting(temperature, signal, min_points=5, T_cutoff=30):
    """
    Trim DSF data for sigmoid fitting (batch-safe).

    Rules:
    - Lower baseline: first local minimum
    - Upper baseline: max signal at T > T_cutoff
    """
    T = np.asarray(temperature, dtype=float)
    F = np.asarray(signal, dtype=float)

    # remove NaNs/Infs
    mask = np.isfinite(T) & np.isfinite(F)
    T = T[mask]
    F = F[mask]

    if len(F) == 0:
        return T, F

    # --- first local minimum ---
    minima_idx = argrelextrema(F, np.less, order=2)[0]
    idx_min = int(minima_idx[0]) if len(minima_idx) > 0 else int(np.argmin(F))

    # --- maximum at T > T_cutoff ---
    mask_upper = T > T_cutoff
    if np.any(mask_upper):
        idx_candidates = np.where(mask_upper)[0]
        idx_max = int(idx_candidates[np.argmax(F[idx_candidates])])
    else:
        idx_max = int(np.argmax(F))

    # --- ensure idx_max > idx_min ---
    if idx_max <= idx_min:
        # fallback: return full curve
        return T, F

    # --- trim ---
    T_trim = T[idx_min:idx_max + 1]
    F_trim = F[idx_min:idx_max + 1]

    # fallback if trimming leaves too few points
    if len(T_trim) < min_points:
        return T, F

    return T_trim, F_trim


def DSF_sigmoid(T, Tm, slope, uB, lB):
    # Clip slope to avoid divide by zero
    slope = max(abs(slope), 0.01)
    x = (Tm - T) / slope
    x = np.clip(x, -500, 500)  # avoid overflow
    return lB + (uB - lB) / (1 + np.exp(x))

## actual data processing begins here ##
def fit_dsf(T, F):

    T_trim, F_trim = trim_dsf_for_fitting(T, F)

    guess = [np.median(T_trim), 1, max(F_trim), min(F_trim)]
    popt, pcov = optimize.curve_fit(DSF_sigmoid, T_trim, F_trim, guess, maxfev = 20000)

    Tfit = np.arange(min(T_trim), max(T_trim), 0.1)
    fit = np.array(DSF_sigmoid(Tfit, *popt))

    Tm = popt[0]
    err = np.sqrt(np.diag(pcov))[0]
    TmR = np.round(Tm, 2)
    errR = np.round(err, 2)

    return TmR, errR
