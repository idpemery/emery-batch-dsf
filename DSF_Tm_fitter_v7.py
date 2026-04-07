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

from scipy.signal import argrelextrema
import numpy as np

from scipy.signal import argrelextrema
import numpy as np

import numpy as np
from scipy.signal import argrelextrema

import numpy as np
from scipy.signal import argrelextrema

def trim_dsf_first_local_min(T, F, min_points=5, min_peak_height=0.05):
    """
    Trim DSF data from first local minimum → first maximum after it,
    strictly within 30–70 °C.
    
    Does NOT fall back to the global minimum; if no local min exists, returns empty arrays.
    """
    T = np.asarray(T, dtype=float)
    F = np.asarray(F, dtype=float)
    
    # Remove NaNs/Infs
    mask = np.isfinite(T) & np.isfinite(F)
    T = T[mask]
    F = F[mask]
    
    if len(F) == 0:
        return np.array([]), np.array([])
    
    # Restrict to 30–70°C window
    mask_window = (T >= 30) & (T <= 70)
    if not np.any(mask_window):
        return np.array([]), np.array([])
    
    T_win = T[mask_window]
    F_win = F[mask_window]
    indices_win = np.where(mask_window)[0]
    
    # Find all local minima in the window
    minima_idx = argrelextrema(F_win, np.less, order=2)[0]
    
    if len(minima_idx) == 0:
        # STRICT: no local minimum → fail
        return np.array([]), np.array([])
    
    # Pick the first local minimum
    idx_min = minima_idx[0]
    
    # Find local maxima after the minimum
    maxima_idx = argrelextrema(F_win, np.greater, order=1)[0]
    maxima_after_min = maxima_idx[maxima_idx > idx_min]
    
    # Apply minimum peak height threshold
    peak_threshold = F_win[idx_min] + min_peak_height * (F_win.max() - F_win.min())
    maxima_after_min = maxima_after_min[F_win[maxima_after_min] > peak_threshold]
    
    if len(maxima_after_min) == 0:
        return np.array([]), np.array([])

    idx_max = maxima_after_min[0]
    
    # Ensure enough points
    if (idx_max - idx_min + 1) < min_points:
        return np.array([]), np.array([])
    
    # Map back to original indices
    idx_min_orig = indices_win[idx_min]
    idx_max_orig = indices_win[idx_max]
    
    return T[idx_min_orig:idx_max_orig+1], F[idx_min_orig:idx_max_orig+1]
    
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
