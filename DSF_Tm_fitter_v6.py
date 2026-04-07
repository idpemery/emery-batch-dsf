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

def trim_dsf_for_fitting(T, F, min_points=5, min_peak_height=0.05, smooth_window=3):
    """
    Trim DSF data using smoothed derivative to find first local minimum 
    and first local maximum after it within 30–70 °C.
    
    Parameters
    ----------
    T : array-like
        Temperature values (°C)
    F : array-like
        Fluorescence values
    min_points : int
        Minimum number of points required in trimmed slice
    min_peak_height : float
        Minimum fraction of full F range that the max must exceed the min
    smooth_window : int
        Window size for moving average smoothing (must be >=1)
    
    Returns
    -------
    T_trim, F_trim : np.ndarray
        Trimmed arrays; empty if no valid min→max found
    """
    T = np.asarray(T, dtype=float)
    F = np.asarray(F, dtype=float)
    
    # Remove NaNs/Infs
    mask = np.isfinite(T) & np.isfinite(F)
    T = T[mask]
    F = F[mask]
    
    if len(F) == 0:
        return np.array([]), np.array([])
    
    # Restrict to 30–70°C
    mask_window = (T >= 30) & (T <= 70)
    if not np.any(mask_window):
        return np.array([]), np.array([])
    
    T_win = T[mask_window]
    F_win = F[mask_window]
    indices_win = np.where(mask_window)[0]
    
    # --- Smooth fluorescence using moving average ---
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        F_smooth = np.convolve(F_win, kernel, mode='same')
    else:
        F_smooth = F_win.copy()
    
    # Compute derivative
    dF = np.diff(F_smooth)
    
    # Detect zero crossings
    sign_change = np.diff(np.sign(dF))
    
    # Local minima: derivative goes from negative → positive
    minima_idx = np.where(sign_change > 0)[0] + 1
    
    if len(minima_idx) == 0:
        return np.array([]), np.array([])
    
    # First local minimum
    idx_min = minima_idx[0]
    
    # Local maxima: derivative goes from positive → negative
    maxima_idx = np.where(sign_change < 0)[0] + 1
    maxima_after_min = maxima_idx[maxima_idx > idx_min]
    
    # Apply minimum peak height
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
