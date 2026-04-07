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
from scipy.signal import find_peaks

def trim_dsf_derivative(T, F, window=10, smooth=True):
    T = np.asarray(T, dtype=float)
    F = np.asarray(F, dtype=float)

    # Remove NaNs
    mask = np.isfinite(T) & np.isfinite(F)
    T = T[mask]
    F = F[mask]

    if len(F) < 5:
        return T, F

    # --- optional smoothing (helps a lot) ---
    F_smooth = np.convolve(F, np.ones(5)/5, mode='same')

    # --- compute derivative ---
    dF_dT = np.gradient(F_smooth, T)

    edge=5
    # --- find transition center ---
    idx_peak = int(np.argmax(np.abs(dF_dT[edge:-edge]))) + edge  # for normal DSF (increasing signal)

    # If curves can be reversed, use this instead:
    # idx_peak = int(np.argmax(np.abs(dF_dT)))

    # --- define trimming window ---
    idx_min = max(0, idx_peak - window)
    idx_max = min(len(F) - 1, idx_peak + window)

    return T[idx_min:idx_max+1], F[idx_min:idx_max+1]

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
