# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Vesicle tracks to displacement, speed and MSD statistics with power-law fits.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 5E–G; Fig. S16A–C (per-track tables read by plot_comparative_data.py
                and plot_compare_data_sig.py)
What it does  : For each pooled folder: one-pixel kymograph tracks -> position vs time
                (um, s), Gaussian smoothing (3 samples) and 10x spline resampling; speeds;
                time-averaged MSD; fit of log MSD vs log tau for tau > T_GAP (single-regime
                alpha and K_alpha = 0.5*10^intercept), a two-regime fit and MSD = a*tau + b*tau^2.
Inputs        : <base_dir>/<folder>/tracks/*.png (outputs of pool_tracks_v2.py); SETTING picks the dataset
Outputs       : <base_dir>/<folder>/plots3/: PNG plots, track_analysis.xlsx/.csv and data/*.csv.
                NOTE: everything already inside plots3 is deleted at the start of each run.
Environment   : environment-analysis.yml (Python 3.7)
Run           : python tracks_processing_fit_v3.py   (set SETTING)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import warnings
from scipy.stats import linregress  # Import linregress for linear fitting
import pandas as pd
from scipy.interpolate import UnivariateSpline
# --- USER PATHS -------------------------------------------------------------
# Point OSSICLE_DATA / OSSICLE_OUT at your copies (see data/README.md).
import os
from pathlib import Path
try:
    _HERE = Path(__file__).resolve()
except NameError:  # interactive (e.g. Spyder cell) execution
    _HERE = Path.cwd().resolve() / "_"
REPO_ROOT = next((p for p in _HERE.parents if (p / "CITATION.cff").exists()), _HERE.parent)
DATA_ROOT = Path(os.environ.get("OSSICLE_DATA", REPO_ROOT / "data"))
OUT_ROOT = Path(os.environ.get("OSSICLE_OUT", REPO_ROOT / "outputs"))
# ----------------------------------------------------------------------------


# Set font as Helvetica
mult_factor = 1.5
plt.rcParams['font.family'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 12 * mult_factor  # Increase font size for readability
plt.rcParams['axes.titlesize'] = 12 * mult_factor
plt.rcParams['axes.labelsize'] = 11 * mult_factor
plt.rcParams['xtick.labelsize'] = 10 * mult_factor
plt.rcParams['ytick.labelsize'] = 10 * mult_factor
plt.rcParams['legend.fontsize'] = 7 * mult_factor

# Suppress the DLL warning if desired
warnings.filterwarnings("ignore", message="loaded more than 1 DLL from .libs")

# Spatial / temporal scales and data folders per dataset. The original file kept the
# alternatives as commented-out lines; choose one with SETTING. The default reproduces the
# values that were active in the original file. All Fig. 5 / Fig. S16 datasets (5, 6 and
# 7 Sep 2025) were analysed with 0.0598188 um/px and 1.42 s/frame.
SETTING = "fig5_S16_6Sep"
SETTINGS = {
    # 6 Sep 2025: Ctrl 1 (Image 5), DMSO (Image 12), 1 uM Noc after 15/25/60 min (Image 13_*)
    "fig5_S16_6Sep": {
        "PIXEL_TO_MICRON": 0.0598188, "PIXEL_TO_SECOND": 1.42,
        "base_dir": str(DATA_ROOT / "vesicles" / "confocal_2025" / "6 Sep 2025 membrane test" / "good"),
        "folders": ["Image 5/pooled_MAX", "Image 5/pooled_AVRG", "Image 12/pooled_MAX", "Image 12/pooled_AVRG", "Image 13_after_15_min/pooled_MAX", "Image 13_after_15_min/pooled_AVRG", "Image 13_after_25_min/pooled_MAX", "Image 13_after_25_min/pooled_AVRG", "Image 13_after_60_min/pooled_MAX", "Image 13_after_60_min/pooled_AVRG"],
    },
    # 7 Sep 2025, 4th animal: Ctrl 2 (Image 20), 5 uM Noc after 10-75 min (Image 21_*)
    "fig5_7Sep": {
        "PIXEL_TO_MICRON": 0.0598188, "PIXEL_TO_SECOND": 1.42,
        "base_dir": str(DATA_ROOT / "vesicles" / "confocal_2025" / "7 Sep 2025 membrane test" / "4th animal" / "good"),
        "folders": ["Image 20/pooled_MAX", "Image 20/pooled_AVRG", "Image 21_after_10min/pooled_MAX", "Image 21_after_10min/pooled_AVRG", "Image 21_after_25min/pooled_MAX", "Image 21_after_25min/pooled_AVRG", "Image 21_after_50min/pooled_MAX", "Image 21_after_50min/pooled_AVRG", "Image 21_after_75min/pooled_MAX", "Image 21_after_75min/pooled_AVRG"],
    },
    # 5 Sep 2025, session 2: Ctrl 3 (Image 10), 10 uM Noc (Image 15); Images 3, 4, 17
    "fig5_5Sep": {
        "PIXEL_TO_MICRON": 0.0598188, "PIXEL_TO_SECOND": 1.42,
        "base_dir": str(DATA_ROOT / "vesicles" / "confocal_2025" / "5 Sep 2025 membrane test" / "session2" / "good"),
        "folders": ["Image 3/pooled_MAX", "Image 3/pooled_AVRG", "Image 4/pooled_MAX", "Image 4/pooled_AVRG", "Image 10/pooled_MAX", "Image 10/pooled_AVRG", "Image 15/pooled_MAX", "Image 15/pooled_AVRG", "Image 17/pooled_MAX", "Image 17/pooled_AVRG"],
    },
    # Fig. S15 front-plate data (2024): 0.1 um/px, 0.5901 s/frame. With this file's
    # T_GAP = 3.0 and spline resampling the S15I exponents are NOT reproduced; S15I was
    # made with s15_front_plate_2024/tracks_processing_fit.py (T_GAP = 1.0, no spline).
    "S15_front_plate_2024": {
        "PIXEL_TO_MICRON": 0.1, "PIXEL_TO_SECOND": 0.5901,
        "base_dir": str(DATA_ROOT / "vesicles" / "front_plate_2024"),
        "folders": ["pooled"],
    },
}
# Two further calibrations were present (commented out) in the original file and are
# not used by any dataset above:
# PIXEL_TO_MICRON = 0.0897282; PIXEL_TO_SECOND = 1.42
# PIXEL_TO_MICRON = 0.0897282; PIXEL_TO_SECOND = 1.80

PIXEL_TO_MICRON = SETTINGS[SETTING]["PIXEL_TO_MICRON"]
PIXEL_TO_SECOND = SETTINGS[SETTING]["PIXEL_TO_SECOND"]

# Time gap threshold for fitting (in seconds)
T_GAP = 3.0  # Only fit data where time_lags > T_GAP

# Option to save plots
SAVE_PLOTS = True

# Option to perform spline fitting
SPLINE_FIT_DATA = True  # Set to True to perform spline fitting


# Function to extract tracks from binary images
def tracks_processing(images):
    tracks = []
    for image in images:
        track = []
        for i in range(image.shape[0]):
            row = image[i]
            if 255 in row:
                positions = np.where(row == 255)[0]
                for pos in positions:
                    track.append([i, pos])
        if len(track) > 0:
            tracks.append(np.array(track))
        else:
            print("Warning: No track found in one of the images.")
    return tracks  # Return as a list

# Function to process tracks: shift origin, scale data, and optionally spline fit
def process_tracks(tracks):
    processed_tracks = []
    for track in tracks:

        #print track id as the position in the list
        print(f"Processing track {len(processed_tracks)+1} with {len(track)} points.")
        
        shifted_track = track - track[0]  # Shift origin
        shifted_track = shifted_track.astype(float)
        # Scale x and y positions
        shifted_track[:, 1] *= PIXEL_TO_MICRON  # x positions to μm
        shifted_track[:, 0] *= PIXEL_TO_SECOND  # y positions to s

        if SPLINE_FIT_DATA:
            # Perform spline fitting to get higher-resolution data
            # Extract time and x positions
            time = shifted_track[:, 0]
            x_positions = shifted_track[:, 1]

            ### NEW: Gaussian smoothing (kernel size 3; if 2 is desired, Gaussian requires odd -> use 3)
            try:
                k = 3  # 2 is not valid for Gaussian (must be odd); 3 is the smallest applicable
                x_positions_sm = cv2.GaussianBlur(x_positions.reshape(-1, 1), (k, 1), 0).ravel()
            except Exception:
                # Fallback: if OpenCV complains, use original
                x_positions_sm = x_positions

            # Define new time array with smaller intervals
            time_interval = np.mean(np.diff(time))  # Original time interval
            new_time_interval = time_interval / 10  # Increase resolution by factor of 10
            new_time = np.arange(time[0], time[-1], new_time_interval)
            # Fit spline to x_positions as a function of time
            spline = UnivariateSpline(time, x_positions_sm, s=0)  # s=0 for interpolation
            # Evaluate spline at new time points
            new_x_positions = spline(new_time)
            # Create new shifted_track with interpolated data
            shifted_track = np.column_stack((new_time, new_x_positions))
        # Append processed track
        processed_tracks.append(shifted_track)
    return processed_tracks


# Function to calculate displacement
def calculate_displacement(track):
    displacements = np.abs(track[:, 1])  # Absolute x position
    return displacements  # Units: μm

# Function to calculate speed over specified time step gap
def calculate_speed(track, step=1):
    """
    Calculates speeds over a specified time step gap.

    Parameters:
        track (ndarray): The track data.
        step (int): The number of time steps to skip for calculating speeds.

    Returns:
        mid_times (ndarray): The mid-point times for each speed calculation.
        speeds (ndarray): The speeds calculated.
    """
    time = track[:, 0]
    x_positions = track[:, 1]
    if len(time) <= step:
        return np.array([]), np.array([])
    time_intervals = time[step:] - time[:-step]
    velocities = (x_positions[step:] - x_positions[:-step]) / time_intervals
    speeds = np.abs(velocities)
    mid_times = (time[:-step] + time[step:]) / 2
    return mid_times, speeds  # Units: μm/s


# Function to calculate mean squared displacement
def calculate_msd(track, max_lag=None):
    time = track[:, 0]
    positions = track[:, 1]
    N = len(positions)
    if max_lag is None:
        max_lag = N // 2
    msd = []
    time_lags = []
    for lag in range(1, max_lag):
        diffs = positions[lag:] - positions[:-lag]
        sq_distances = diffs**2
        msd.append(np.mean(sq_distances))
        # Time lag for this lag
        time_diffs = time[lag:] - time[:-lag]
        time_lag = np.mean(time_diffs)
        time_lags.append(time_lag)
    return np.array(time_lags), np.array(msd)  # Units: s, μm²


# Modified function to fit MSD curves and collect statistics
def fit_msd(time_lags, msd, t_gap):
    """
    Perform linear fitting on the log-log MSD data beyond the specified time gap.

    Parameters:
        time_lags (array): Array of time lags (in seconds).
        msd (array): Array of MSD values (in μm²).
        t_gap (float): Minimum time lag (in seconds) to include in the fit.

    Returns:
        slope (float): Scaling exponent from the fit.
        intercept (float): Intercept from the fit.
        r_value (float): Correlation coefficient.
        p_value (float): Two-sided p-value for a hypothesis test.
        std_err (float): Standard error of the estimated gradient.
    """
    # Select data beyond t_gap
    mask = time_lags > t_gap
    selected_time_lags = time_lags[mask]
    selected_msd = msd[mask]

    # Check if there are enough points to perform the fit
    if len(selected_time_lags) < 2:
        # Return NaN values if not enough data
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Take logarithm of selected time lags and MSD values
    log_time = np.log10(selected_time_lags)
    log_msd = np.log10(selected_msd)

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(log_time, log_msd)
    return slope, intercept, r_value, p_value, std_err

### NEW: helpers for two-regime power-law fit and diffusion+drift mixture
def _linfit_log10(x, y):
    """Linear regression in log10 space; returns dict with slope, intercept, r2, se_slope, se_intercept."""
    res = linregress(x, y)
    # approximate SE for intercept from slope SE and variance of x
    # Using analytical from linregress: res.stderr (slope SE). For intercept SE, recompute:
    n = len(x)
    if n < 3:
        se_intercept = np.nan
    else:
        xbar = np.mean(x)
        sxx = np.sum((x - xbar)**2)
        # residual standard error
        yhat = res.intercept + res.slope * x
        rss = np.sum((y - yhat)**2)
        s2 = rss / (n - 2) if n > 2 else np.nan
        se_intercept = np.sqrt(s2 * (1/n + xbar**2 / sxx)) if sxx > 0 and n > 2 else np.nan
    return {
        "slope": res.slope,
        "intercept": res.intercept,
        "r2": res.rvalue**2,
        "se_slope": res.stderr,
        "se_intercept": se_intercept
    }

def fit_two_regime_powerlaw(time_lags, msd, t_gap, min_pts=5):
    """
    Split log10(MSD)=a+b*log10(tau) into two regimes by choosing a breakpoint that minimizes SSE.
    Returns per-regime (alpha, Kalpha) with 95% CIs and the chosen breakpoint.
    Kalpha is defined by MSD = 2*Kalpha * tau^alpha.
    """
    mask = (time_lags > t_gap) & (msd > 0)
    tau = time_lags[mask]
    M = msd[mask]
    if len(tau) < 2*min_pts:
        return None  # not enough data

    logt = np.log10(tau)
    logm = np.log10(M)

    best = None
    # candidate split indices ensuring min_pts on each side
    for k in range(min_pts, len(tau) - min_pts):
        x1, y1 = logt[:k], logm[:k]
        x2, y2 = logt[k:], logm[k:]
        f1 = _linfit_log10(x1, y1)
        f2 = _linfit_log10(x2, y2)
        y1hat = f1["intercept"] + f1["slope"]*x1
        y2hat = f2["intercept"] + f2["slope"]*x2
        sse = np.sum((y1 - y1hat)**2) + np.sum((y2 - y2hat)**2)
        if (best is None) or (sse < best["sse"]):
            best = {"idx": k, "f1": f1, "f2": f2, "sse": sse}

    if best is None:
        return None

    # Convert intercepts to Kalpha using MSD = 2 Kalpha tau^alpha => intercept = log10(2 Kalpha)
    z = 1.96  # ~95% CI
    def pack(fit):
        alpha = fit["slope"]
        alpha_ci = (alpha - z*fit["se_slope"], alpha + z*fit["se_slope"]) if np.isfinite(fit["se_slope"]) else (np.nan, np.nan)
        # intercept CI in log10 units
        b = fit["intercept"]
        if np.isfinite(fit["se_intercept"]):
            b_lo, b_hi = b - z*fit["se_intercept"], b + z*fit["se_intercept"]
        else:
            b_lo, b_hi = np.nan, np.nan
        K = 0.5 * (10**b)
        K_lo = 0.5 * (10**b_lo) if np.isfinite(b_lo) else np.nan
        K_hi = 0.5 * (10**b_hi) if np.isfinite(b_hi) else np.nan
        return {
            "alpha": alpha, "alpha_lo": alpha_ci[0], "alpha_hi": alpha_ci[1],
            "Kalpha": K, "Kalpha_lo": K_lo, "Kalpha_hi": K_hi,
            "r2": fit["r2"]
        }

    bp_tau = tau[best["idx"]]
    reg1 = pack(best["f1"])
    reg2 = pack(best["f2"])
    return {"break_tau": bp_tau, "regime1": reg1, "regime2": reg2}

def fit_diffusion_plus_drift(time_lags, msd, t_gap, min_pts=6):
    """
    Fit MSD = a*tau + b*tau^2 (linear domain) for tau>t_gap.
    Return D=a/2 and v=sqrt(b) with ~95% CIs using OLS covariance.
    """
    mask = (time_lags > t_gap) & np.isfinite(msd)
    tau = time_lags[mask]
    y = msd[mask]
    if len(tau) < min_pts:
        return None

    X = np.column_stack([tau, tau**2])
    # OLS via normal equations
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    a, b = beta[0], beta[1]
    yhat = X @ beta
    n, p = len(y), 2
    rss = np.sum((y - yhat)**2)
    if n <= p:
        return {"D": np.nan, "D_lo": np.nan, "D_hi": np.nan, "v": np.nan, "v_lo": np.nan, "v_hi": np.nan, "r2": np.nan}
    s2 = rss / (n - p)
    cov = s2 * np.linalg.inv(XtX)
    se_a = np.sqrt(cov[0, 0]) if cov[0, 0] >= 0 else np.nan
    se_b = np.sqrt(cov[1, 1]) if cov[1, 1] >= 0 else np.nan
    z = 1.96

    D = a / 2.0
    D_lo = (a - z*se_a) / 2.0 if np.isfinite(se_a) else np.nan
    D_hi = (a + z*se_a) / 2.0 if np.isfinite(se_a) else np.nan

    if b > 0 and np.isfinite(se_b):
        v = np.sqrt(b)
        # delta method: var(v) ≈ (1/(4 b)) var(b)
        se_v = se_b / (2.0 * v)
        v_lo, v_hi = v - z*se_v, v + z*se_v
    else:
        v, v_lo, v_hi = np.nan, np.nan, np.nan

    tss = np.sum((y - np.mean(y))**2)
    r2 = 1 - rss / tss if tss > 0 else np.nan
    return {"D": D, "D_lo": D_lo, "D_hi": D_hi, "v": v, "v_lo": v_lo, "v_hi": v_hi, "r2": r2}

# Function to plot tracks
def plot_tracks(tracks, colors):
    plt.figure(figsize=(6, 6))
    for i, track in enumerate(tracks):
        time = track[:, 0]
        x_position = track[:, 1]
        plt.plot(time, x_position, marker='o', color=colors[i], alpha=0.7,
                 label=f'Track {i+1}', markeredgewidth=0, markersize=3)
    plt.xlabel('Time (s)')
    plt.ylabel('X Position ($\mu$m)')
    plt.title('Tracks Over Time')
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.legend()
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(os.path.join(PLOT_FOLDER, 'tracks_over_time.png'), dpi=600)
    # plt.show()

# Function to plot displacement over time
def plot_displacement(tracks, colors):
    plt.figure(figsize=(6, 6))
    for i, track in enumerate(tracks):
        time = track[:, 0]
        displacement = calculate_displacement(track)
        plt.plot(time, displacement, color=colors[i], alpha=0.7, label=f'Track {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement ($\mu$m)')
    plt.title('Displacement Over Time')
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.legend()
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(os.path.join(PLOT_FOLDER, 'displacement_over_time.png'), dpi=600)
    # plt.show()

# Function to plot speed over time AND save per-track time-series speeds
def plot_speed(tracks, colors):
    plt.figure(figsize=(6, 6))

    # --- collect data for CSV ---
    rows = []

    for i, track in enumerate(tracks):
        mid_times, speeds = calculate_speed(track)  # step=1 by default
        if mid_times.size and speeds.size:
            # plot
            plt.plot(mid_times, speeds, color=colors[i], alpha=0.7, label=f'Track {i+1}')
            # collect rows
            rows.append(pd.DataFrame({
                "track_id": i + 1,
                "step": 1,
                "mid_time_s": mid_times,
                "speed_um_per_s": speeds
            }))

    plt.xlabel('Time (s)')
    plt.ylabel('Speed ($\mu$m/s)')
    plt.title('Speed Over Time')
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.legend()
    plt.tight_layout()

    if SAVE_PLOTS:
        # save plot
        plt.savefig(os.path.join(PLOT_FOLDER, 'speed_over_time.png'), dpi=600)
        # save CSV
        if rows:
            data_dir = os.path.join(PLOT_FOLDER, "data")
            os.makedirs(data_dir, exist_ok=True)
            df_speed_ts = pd.concat(rows, ignore_index=True)
            df_speed_ts.to_csv(os.path.join(data_dir, "speeds_time_series.csv"), index=False)
    # plt.show()

# Function to plot a histogram of speeds AND save speeds + histogram data
def plot_speed_histogram(tracks):
    """
    Plots a histogram of the speeds from all tracks combined (broken Y-axis),
    and saves (i) all speeds with track IDs, and (ii) histogram bins+counts to CSV.
    """
    import matplotlib.patches as mpatches  # avoid name clash with hist return

    # Collect all speeds (and track IDs)
    all_speeds = []
    all_track_ids = []
    for i, track in enumerate(tracks):
        _, speeds = calculate_speed(track)
        if len(speeds):
            all_speeds.extend(speeds)
            all_track_ids.extend([i + 1] * len(speeds))
    all_speeds = np.array(all_speeds)

    # Set up the figure with two subplots (axes) stacked vertically
    f, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(6, 6),
                                          gridspec_kw={'height_ratios': [1, 2]})

    # Define the number of bins
    bins = 30

    # Choose the middle color from the 'Blues' colormap
    cmap = plt.get_cmap('Blues')
    bar_color = cmap(0.75)

    # Plot the histogram on both axes (and capture counts/edges)
    counts, bin_edges, hist_patches = ax_top.hist(
        all_speeds, bins=bins, color=bar_color, edgecolor='black', alpha=0.7
    )
    ax_bottom.hist(all_speeds, bins=bins, color=bar_color, edgecolor='black', alpha=0.7)

    # Determine the y-axis limits for both plots
    y_break = max(counts) * 0.01 if counts.size else 1.0

    # Adjust y-axis limits
    ax_top.set_ylim(y_break, max(counts) * 1.05 if counts.size else 1.05)
    ax_bottom.set_ylim(0, y_break)

    # Hide the spines between ax_top and ax_bottom
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(labeltop=False)
    ax_bottom.xaxis.tick_bottom()

    # Add diagonal lines to indicate the break
    d = .015
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # spacing
    plt.subplots_adjust(hspace=0.06)

    # labels
    ax_top.set_ylabel('')
    ax_bottom.set_xlabel('Speed ($\mu$m/s)')
    ax_bottom.set_ylabel('Frequency')
    ax_top.set_title('Histogram of Speeds')

    ax_top.grid(True, linestyle='--', alpha=0.5)
    ax_bottom.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if SAVE_PLOTS:
        # save figure
        plt.savefig(os.path.join(PLOT_FOLDER, 'speed_histogram_broken_axis.png'), dpi=600)

        # save CSVs
        data_dir = os.path.join(PLOT_FOLDER, "data")
        os.makedirs(data_dir, exist_ok=True)

        # (i) all speeds with track IDs
        if all_speeds.size:
            df_all = pd.DataFrame({"track_id": all_track_ids, "speed_um_per_s": all_speeds})
            df_all.to_csv(os.path.join(data_dir, "speeds_all_tracks.csv"), index=False)

        # (ii) histogram bins + counts
        if counts.size:
            df_hist = pd.DataFrame({
                "bin_left": bin_edges[:-1],
                "bin_right": bin_edges[1:],
                "count": counts
            })
            df_hist.to_csv(os.path.join(data_dir, "speed_histogram_counts.csv"), index=False)
    # plt.show()

# New function to plot normalized speed histogram AND save speeds + histogram data
# Normalized histogram with broken y-axis (area = 1), matching the look/feel of plot_speed_histogram
def plot_speed_histogram_normalized(tracks, step=1, bins=30, hist_range=None):
    """
    Plots a normalized histogram (area = 1) of the speeds from all tracks combined,
    using a broken Y-axis to reveal small contributions. Saves:
      (i) all speeds with track IDs, and
      (ii) histogram bins + density to CSV.
    """
    # Collect all speeds (and track IDs)
    all_speeds = []
    all_track_ids = []
    for i, track in enumerate(tracks):
        _, speeds = calculate_speed(track, step=step)
        if len(speeds):
            all_speeds.extend(speeds)
            all_track_ids.extend([i + 1] * len(speeds))

    all_speeds = np.asarray(all_speeds, dtype=float)

    # Early exit + write empty tables (keep pipeline predictable)
    if all_speeds.size == 0 or not np.any(np.isfinite(all_speeds)):
        if SAVE_PLOTS:
            data_dir = os.path.join(PLOT_FOLDER, "data")
            os.makedirs(data_dir, exist_ok=True)
            pd.DataFrame(columns=["track_id", "speed_um_per_s"]).to_csv(
                os.path.join(data_dir, "speeds_all_tracks.csv"), index=False
            )
            pd.DataFrame(columns=["bin_left", "bin_right", "density"]).to_csv(
                os.path.join(data_dir, "speed_histogram_density.csv"), index=False
            )
        return

    # Clean NaN/Inf (defensive)
    finite_mask = np.isfinite(all_speeds)
    all_speeds = all_speeds[finite_mask]
    all_track_ids = np.asarray(all_track_ids, dtype=int)[finite_mask]

    # Figure with two vertically stacked axes (broken y-axis), same formatting
    f, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(6, 6),
        gridspec_kw={'height_ratios': [1, 2]}
    )

    # Choose color consistent with your style
    cmap = plt.get_cmap('Blues')
    bar_color = cmap(0.75)

    # Plot normalized histogram on both axes
    # (Matplotlib returns "n" as density when density=True)
    n_top, bin_edges, _ = ax_top.hist(
        all_speeds, bins=bins, range=hist_range,
        density=True, color=bar_color, edgecolor='black', alpha=0.7
    )
    ax_bottom.hist(
        all_speeds, bins=bin_edges, density=True,
        color=bar_color, edgecolor='black', alpha=0.7
    )

    # Determine break
    ymax = float(np.max(n_top)) if n_top.size else 0.0
    # Use a visible break (5% of peak), clamp to >= small positive
    y_break = max(1e-3, 0.05 * ymax) if ymax > 0 else 1.0

    # Set y-lims (avoid identical bounds at the seam)
    if ymax > 0:
        ax_top.set_ylim(y_break * 1.001, ymax * 1.05)
        ax_bottom.set_ylim(0.0, y_break * 0.999)
    else:
        # Degenerate case (unlikely if data exists)
        ax_top.set_ylim(0.001, 0.002)
        ax_bottom.set_ylim(0.0, 0.001)

    # Hide spines between plots; add diagonal "break" marks
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(labeltop=False)
    ax_bottom.xaxis.tick_bottom()

    d = .015  # diagonal mark size
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Spacing and labels (match your style)
    plt.subplots_adjust(hspace=0.06)
    ax_top.set_ylabel('')
    ax_bottom.set_xlabel('Speed ($\\mu$m/s)')
    ax_bottom.set_ylabel('Probability density')
    ax_top.set_title('Normalized Histogram of Speeds (area = 1)')

    ax_top.grid(True, linestyle='--', alpha=0.5)
    ax_bottom.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save figure + CSVs consistent with your pipeline
    if SAVE_PLOTS:
        # Figure
        plt.savefig(os.path.join(PLOT_FOLDER, 'speed_histogram_normalized_broken_axis.png'), dpi=600)

        # Data
        data_dir = os.path.join(PLOT_FOLDER, "data")
        os.makedirs(data_dir, exist_ok=True)

        # (i) all speeds with track IDs
        pd.DataFrame({"track_id": all_track_ids, "speed_um_per_s": all_speeds}).to_csv(
            os.path.join(data_dir, "speeds_all_tracks.csv"), index=False
        )

        # (ii) histogram bins + density
        # Compute density once more with np.histogram to ensure exact values match saved table
        density, edges = np.histogram(all_speeds, bins=bin_edges, density=True)
        pd.DataFrame({
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "density": density
        }).to_csv(os.path.join(data_dir, "speed_histogram_density.csv"), index=False)
    # plt.show()


# New function to plot speed histograms for different time steps AND save CSVs
def plot_speed_histograms_multiple_steps(tracks, steps=[1, 2, 4, 8, 16], normalize=False):
    """
    Plots speed histograms for different time steps and saves:
      (1) Long-format speeds for each step/track
      (2) Histogram bins+counts (or densities if normalize=True) per step
    """
    plt.figure(figsize=(6, 6))
    all_speeds_per_step = {}
    max_speed = 0

    # Colors for different histograms
    cmap = plt.get_cmap('Blues')
    colors = [cmap(i) for i in np.linspace(0.4, 1, len(steps))]

    # --- Compute a global max speed and shared bins ---
    global_max_speed = 0.0
    for step in steps:
        for track in tracks:
            _, speeds = calculate_speed(track, step=step)
            if len(speeds) > 0:
                m = float(np.max(speeds))
                if m > global_max_speed:
                    global_max_speed = m

    if global_max_speed <= 0:
        print("No speeds to plot for any step.")
        plt.close()
        return

    bins = np.linspace(0.0, global_max_speed, 30)  # fixed shared bin edges

    any_plotted = False

    # --- collectors for CSVs ---
    rows_long = []      # per-speed long format
    rows_hist = []      # per-step histogram (bin_left, bin_right, value)

    for idx, step in enumerate(steps):
        all_speeds = []
        # collect per-track with IDs for CSV
        for i, track in enumerate(tracks):
            _, speeds = calculate_speed(track, step=step)
            if len(speeds):
                all_speeds.extend(speeds)
                # long-format rows
                rows_long.append(pd.DataFrame({
                    "step": step,
                    "track_id": i + 1,
                    "speed_um_per_s": speeds
                }))

        all_speeds = np.array(all_speeds)
        all_speeds_per_step[step] = all_speeds
        if len(all_speeds) > 0:
            any_plotted = True
            # Plot histogram with shared bins; normalize controlled by flag
            plt.hist(all_speeds, bins=bins, alpha=0.5, color=colors[idx],
                     label=f'Time Step = {step}', edgecolor='black', density=normalize)

            # store histogram table for this step
            counts, edges = np.histogram(all_speeds, bins=bins, density=normalize)
            rows_hist.append(pd.DataFrame({
                "step": step,
                "bin_left": edges[:-1],
                "bin_right": edges[1:],
                "value": counts
            }))

    if not any_plotted:
        print("No speeds for any time step.")
        plt.close()
        return

    plt.xlabel('Speed ($\\mu$m/s)')
    plt.ylabel('Probability Density' if normalize else 'Count')
    plt.title('Speed Histograms for Different Time Steps')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if SAVE_PLOTS:
        # save plot
        plt.savefig(os.path.join(PLOT_FOLDER, 'speed_histograms_multiple_steps.png'), dpi=600)

        # save CSVs
        data_dir = os.path.join(PLOT_FOLDER, "data")
        os.makedirs(data_dir, exist_ok=True)

        if rows_long:
            df_long = pd.concat(rows_long, ignore_index=True)
            df_long.to_csv(os.path.join(data_dir, "speeds_by_step_long.csv"), index=False)

        if rows_hist:
            df_hist = pd.concat(rows_hist, ignore_index=True)
            df_hist.to_csv(os.path.join(data_dir, "speed_histograms_by_step.csv"), index=False)
    # plt.show()


# Modified function to plot MSD with linear fits and statistics + SAVE CSVs
def plot_msd(tracks, colors, t_gap):
    plt.figure(figsize=(6, 6))
    fit_results = []
    analysis_rows = []          # combined (single + two-regime + mix) for Excel/CSV
    single_rows = []            # single-regime only (clean CSV)
    two_regime_rows = []        # two-regime only
    mix_rows = []               # diffusion+drift only
    msd_rows_all = []           # all MSD points per track
    msd_rows_masked = []        # MSD points used in plotting (tau > t_gap)

    masked_max_lags = []  # for reference lines

    for i, track in enumerate(tracks):
        track_id = i + 1
        time_lags, msd = calculate_msd(track)

        # --- save raw MSD points per track ---
        if len(time_lags):
            msd_rows_all.append(pd.DataFrame({
                "track_id": track_id,
                "tau_s": time_lags,
                "msd_um2": msd
            }))

        # single-regime fit beyond t_gap
        slope, intercept, r_value, p_value, std_err = fit_msd(time_lags, msd, t_gap)

        # Plot data used (tau > t_gap)
        mask_plot = (time_lags > t_gap)
        if np.any(mask_plot):
            plt.loglog(time_lags[mask_plot], msd[mask_plot], marker='o', color=colors[i],
                       alpha=0.7, markeredgewidth=0, markersize=3)
            masked_max_lags.append(np.max(time_lags[mask_plot]))

            # save masked MSD points
            msd_rows_masked.append(pd.DataFrame({
                "track_id": track_id,
                "tau_s": time_lags[mask_plot],
                "msd_um2": msd[mask_plot]
            }))

        # Collect printout list (keep your console output behavior)
        if not np.isnan(slope):
            fit_results.append({
                'Track': track_id,
                'Slope': slope,
                'Intercept': intercept,
                'R_squared': r_value**2,
                'P_value': p_value,
                'Std_err': std_err
            })

        # --- two-regime and diffusion+drift fits ---
        two_reg = fit_two_regime_powerlaw(time_lags, msd, t_gap, min_pts=5)
        mix = fit_diffusion_plus_drift(time_lags, msd, t_gap, min_pts=6)

        # --- assemble a single analysis row per track (with NaNs if fits failed) ---
        row = {
            "Track": track_id,
            "alpha_single": slope if slope == slope else np.nan,
            "Kalpha_single": 0.5*(10**intercept) if intercept == intercept else np.nan,
            "R2_single": (r_value**2) if r_value == r_value else np.nan,
            "intercept_single": intercept if intercept == intercept else np.nan,
            "pval_single": p_value if p_value == p_value else np.nan,
            "stderr_alpha_single": std_err if std_err == std_err else np.nan,
        }

        # keep a clean single-fit table too
        single_rows.append({
            "track_id": track_id,
            "alpha_single": row["alpha_single"],
            "Kalpha_single": row["Kalpha_single"],
            "R2_single": row["R2_single"],
            "intercept_single": row["intercept_single"],
            "pval_single": row["pval_single"],
            "stderr_alpha_single": row["stderr_alpha_single"],
            "t_min_fit_s": float(np.min(time_lags[mask_plot])) if np.any(mask_plot) else np.nan,
            "t_max_fit_s": float(np.max(time_lags[mask_plot])) if np.any(mask_plot) else np.nan,
            "n_points_fit": int(np.count_nonzero(mask_plot)) if np.any(mask_plot) else 0
        })

        if two_reg is not None:
            r1 = two_reg["regime1"]; r2 = two_reg["regime2"]
            row.update({
                "break_tau": two_reg["break_tau"],
                "alpha_reg1": r1["alpha"], "alpha_reg1_lo": r1["alpha_lo"], "alpha_reg1_hi": r1["alpha_hi"],
                "Kalpha_reg1": r1["Kalpha"], "Kalpha_reg1_lo": r1["Kalpha_lo"], "Kalpha_reg1_hi": r1["Kalpha_hi"],
                "R2_reg1": r1["r2"],
                "alpha_reg2": r2["alpha"], "alpha_reg2_lo": r2["alpha_lo"], "alpha_reg2_hi": r2["alpha_hi"],
                "Kalpha_reg2": r2["Kalpha"], "Kalpha_reg2_lo": r2["Kalpha_lo"], "Kalpha_reg2_hi": r2["Kalpha_hi"],
                "R2_reg2": r2["r2"]
            })
            two_regime_rows.append({
                "track_id": track_id,
                "break_tau": two_reg["break_tau"],
                "alpha_reg1": r1["alpha"], "alpha_reg1_lo": r1["alpha_lo"], "alpha_reg1_hi": r1["alpha_hi"],
                "Kalpha_reg1": r1["Kalpha"], "Kalpha_reg1_lo": r1["Kalpha_lo"], "Kalpha_reg1_hi": r1["Kalpha_hi"],
                "R2_reg1": r1["r2"],
                "alpha_reg2": r2["alpha"], "alpha_reg2_lo": r2["alpha_lo"], "alpha_reg2_hi": r2["alpha_hi"],
                "Kalpha_reg2": r2["Kalpha"], "Kalpha_reg2_lo": r2["Kalpha_lo"], "Kalpha_reg2_hi": r2["Kalpha_hi"],
                "R2_reg2": r2["r2"]
            })
        else:
            row.update({
                "break_tau": np.nan,
                "alpha_reg1": np.nan, "alpha_reg1_lo": np.nan, "alpha_reg1_hi": np.nan,
                "Kalpha_reg1": np.nan, "Kalpha_reg1_lo": np.nan, "Kalpha_reg1_hi": np.nan,
                "R2_reg1": np.nan,
                "alpha_reg2": np.nan, "alpha_reg2_lo": np.nan, "alpha_reg2_hi": np.nan,
                "Kalpha_reg2": np.nan, "Kalpha_reg2_lo": np.nan, "Kalpha_reg2_hi": np.nan,
                "R2_reg2": np.nan
            })

        if mix is not None:
            row.update({
                "D": mix["D"], "D_lo": mix["D_lo"], "D_hi": mix["D_hi"],
                "v": mix["v"], "v_lo": mix["v_lo"], "v_hi": mix["v_hi"],
                "R2_mix": mix["r2"]
            })
            mix_rows.append({
                "track_id": track_id,
                "D": mix["D"], "D_lo": mix["D_lo"], "D_hi": mix["D_hi"],
                "v": mix["v"], "v_lo": mix["v_lo"], "v_hi": mix["v_hi"],
                "R2_mix": mix["r2"]
            })
        else:
            row.update({
                "D": np.nan, "D_lo": np.nan, "D_hi": np.nan,
                "v": np.nan, "v_lo": np.nan, "v_hi": np.nan,
                "R2_mix": np.nan
            })

        analysis_rows.append(row)

    # Reference lines
    if masked_max_lags:
        ref_time = np.logspace(np.log10(t_gap), np.log10(max(masked_max_lags)), num=100)
        D_eff = 1   # guide
        v_eff = 1   # guide
        plt.loglog(ref_time, 2 * D_eff * ref_time, 'k--', linewidth=2, label='Passive Diffusion (slope=1)')
        plt.loglog(ref_time, (v_eff * ref_time)**2, 'k:', linewidth=2, label='Active Transport (slope=2)')

    plt.xlabel('Time Lag (s)')
    plt.ylabel('Mean Squared Displacement ($\mu$m$^2$)')
    plt.title('MSD Over Time Lag (Log-Log Scale)')
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    # show only reference lines in legend (last two handles)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles and labels:
        plt.legend(handles[-2:], labels[-2:], loc='upper left')

    # Stats box
    if fit_results:
        slopes = [r['Slope'] for r in fit_results if r['Slope'] == r['Slope']]
        if len(slopes):
            textstr = (
                'Scaling Exponent (Slope):\n'
                f'Min: {np.min(slopes):.3f}\n'
                f'Max: {np.max(slopes):.3f}\n'
                f'Mean: {np.mean(slopes):.3f}'
            )
            plt.gca().text(
                0.62, 0.03, textstr,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
            )

    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(os.path.join(PLOT_FOLDER, 'msd_log_log.png'), dpi=600)
    # plt.show()

    # Console printout (unchanged)
    print("MSD Linear Fit Results (log-log scale):")
    for result in fit_results:
        print(f"Track {result['Track']}:")
        print(f"  Slope (Scaling Exponent): {result['Slope']:.3f}")
        print(f"  Intercept: {result['Intercept']:.3f}")
        print(f"  R-squared: {result['R_squared']:.3f}")
        print(f"  P-value: {result['P_value']:.3e}")
        print(f"  Standard Error: {result['Std_err']:.3f}")
        print()

    # --- SAVE TABLES ---
    if analysis_rows:
        df = pd.DataFrame(analysis_rows)
        out_base = (PLOT_FOLDER if SAVE_PLOTS else BASE_FOLDER)
        out_xlsx = os.path.join(out_base, "track_analysis.xlsx")
        out_csv  = os.path.join(out_base, "track_analysis.csv")
        try:
            df.to_excel(out_xlsx, index=False)
            df.to_csv(out_csv, index=False)
            print(f"Saved analysis to: {out_xlsx}")
            print(f"Saved analysis to: {out_csv}")
        except Exception as e:
            print(f"Failed to save analysis: {e}")

    # Save the clean per-fit tables + MSD points to PLOT_FOLDER/data/
    if SAVE_PLOTS:
        data_dir = os.path.join(PLOT_FOLDER, "data")
        os.makedirs(data_dir, exist_ok=True)

        # MSD points
        if msd_rows_all:
            pd.concat(msd_rows_all, ignore_index=True).to_csv(
                os.path.join(data_dir, "msd_points_all_tracks.csv"), index=False
            )
        if msd_rows_masked:
            pd.concat(msd_rows_masked, ignore_index=True).to_csv(
                os.path.join(data_dir, "msd_points_masked_tau_gt_TGAP.csv"), index=False
            )

        # Single-regime fit table
        if single_rows:
            pd.DataFrame(single_rows).to_csv(
                os.path.join(data_dir, "msd_fit_single.csv"), index=False
            )

        # Two-regime fit table
        if two_regime_rows:
            pd.DataFrame(two_regime_rows).to_csv(
                os.path.join(data_dir, "msd_fit_two_regime.csv"), index=False
            )

        # Diffusion + drift fit table
        if mix_rows:
            pd.DataFrame(mix_rows).to_csv(
                os.path.join(data_dir, "msd_fit_diffusion_plus_drift.csv"), index=False
            )



def execute(base_folder):
    # Track images folder
    track_image_folder = os.path.join(base_folder, "tracks")  # Change this to your folder containing track images
    # Read only the png images in the folder
    image_list = [image for image in os.listdir(track_image_folder) if image.endswith('.png')]
    images = [cv2.imread(os.path.join(track_image_folder, image), cv2.IMREAD_GRAYSCALE) for image in image_list]
    tracks = tracks_processing(images)
    proc_tracks = process_tracks(tracks)

    # Generate a fixed list of colors for all tracks
    cmap = plt.get_cmap('Blues')
    colors = [cmap(i) for i in np.linspace(0, 1, len(proc_tracks))]

    # Plotting functions with colors
    plot_tracks(proc_tracks, colors) 
    plot_displacement(proc_tracks, colors) 
    plot_speed(proc_tracks, colors) 
    plot_speed_histogram(proc_tracks)
    plot_speed_histogram_normalized(proc_tracks, step=1, bins=30, hist_range=None)
    plot_speed_histograms_multiple_steps(proc_tracks, steps=[1, 2, 4, 8, 16])
    plot_msd(proc_tracks, colors, T_GAP)


if __name__ == '__main__':
    base_dir = SETTINGS[SETTING]["base_dir"]
    folders = SETTINGS[SETTING]["folders"]
               
    for folder in folders:
        #make both folder names global for use in functions
        global BASE_FOLDER, PLOT_FOLDER

        BASE_FOLDER = os.path.join(base_dir, folder)
        PLOT_FOLDER = os.path.join(BASE_FOLDER, "plots3")
        os.makedirs(PLOT_FOLDER, exist_ok=True)

        print(f"Processing folder: {BASE_FOLDER}")
        print(f"Plots will be saved to: {PLOT_FOLDER}")

        #remove any existing contents in the plots folder
        for f in os.listdir(PLOT_FOLDER):
            f_path = os.path.join(PLOT_FOLDER, f)
            try:
                if os.path.isfile(f_path) or os.path.islink(f_path):
                    os.unlink(f_path)
                elif os.path.isdir(f_path):
                    shutil.rmtree(f_path)
            except Exception as e:
                print(f'Failed to delete {f_path}. Reason: {e}')

        execute(BASE_FOLDER)

        print("Processing complete for this folder.")
