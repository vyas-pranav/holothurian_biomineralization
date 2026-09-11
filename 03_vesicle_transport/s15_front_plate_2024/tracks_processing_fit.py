# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""MSD analysis of the 2024 front-plate vesicle tracks (Fig. S15).

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. S15 (MSD exponents of Fig. S15I; also the track and speed plots)
What it does  : One-pixel kymograph tracks -> position vs time (um, s); plots of tracks,
                displacement, speed and speed histograms; time-averaged MSD with a line fit
                of log MSD vs log tau for tau > T_GAP = 1 s. The MSD plot prints min / max /
                mean of the exponents (Fig. S15I: 93 tracks, 0.044 / 1.813 / 0.733).
Inputs        : <BASE_FOLDER>/tracks/*.png (S15: DATA_ROOT/vesicles/front_plate_2024/pooled/tracks)
Outputs       : <BASE_FOLDER>/plots/*.png (tracks_over_time, displacement_over_time,
                velocity_over_time, speed_histogram_broken_axis, speed_histograms_multiple_steps, msd_log_log)
Environment   : environment-analysis.yml (Python 3.7)
Run           : python tracks_processing_fit.py   (set SETTING = "S15" for Fig. S15)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import warnings
from scipy.stats import linregress  # Import linregress for linear fitting
import pandas as pd
from scipy.interpolate import UnivariateSpline  # For spline fitting
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

# Spatial / temporal scales and track folder. The original file kept the Fig. S15 values
# as a commented-out alternative; choose one with SETTING. The default reproduces the
# values that were active in the original file (a 2025 confocal data set).
SETTING = "confocal_2025_Image13_25min"
SETTINGS = {
    # values active in the original file (6 Sep 2025, Image 13 after 25 min, older pooling)
    "confocal_2025_Image13_25min": {
        "PIXEL_TO_MICRON": 0.0897282, "PIXEL_TO_SECOND": 1.42,
        "BASE_FOLDER": str(DATA_ROOT / "vesicles" / "confocal_2025" / "6 Sep 2025 membrane test" / "good" / "Image 13_after_25_min" / "pooled_tracks"),
    },
    # Fig. S15 (front plate, 2024): 0.1 um per x pixel, 0.5901 s per y pixel (frame)
    "S15": {
        "PIXEL_TO_MICRON": 0.1, "PIXEL_TO_SECOND": 0.5901,
        "BASE_FOLDER": str(DATA_ROOT / "vesicles" / "front_plate_2024" / "pooled"),
    },
}
PIXEL_TO_MICRON = SETTINGS[SETTING]["PIXEL_TO_MICRON"]
PIXEL_TO_SECOND = SETTINGS[SETTING]["PIXEL_TO_SECOND"]

# Time gap threshold for fitting (in seconds)
T_GAP = 1.0  # Only fit data where time_lags > T_GAP

# Option to save plots
BASE_FOLDER = SETTINGS[SETTING]["BASE_FOLDER"]
SAVE_PLOTS = True
PLOT_FOLDER = os.path.join(BASE_FOLDER, "plots")

# Option to perform spline fitting
SPLINE_FIT_DATA = False  # Set to True to perform spline fitting

# Create plot folder if it doesn't exist
if SAVE_PLOTS and not os.path.exists(PLOT_FOLDER):
    os.makedirs(PLOT_FOLDER)

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
            # Define new time array with smaller intervals
            time_interval = np.mean(np.diff(time))  # Original time interval
            new_time_interval = time_interval / 10  # Increase resolution by factor of 10
            new_time = np.arange(time[0], time[-1], new_time_interval)
            # Fit spline to x_positions as a function of time
            spline = UnivariateSpline(time, x_positions, s=0)  # s=0 for interpolation
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

# Function to calculate velocity over specified time step gap
def calculate_velocity(track, step=1):
    """
    Calculates velocities and speeds over a specified time step gap.

    Parameters:
        track (ndarray): The track data.
        step (int): The number of time steps to skip for calculating velocities.

    Returns:
        mid_times (ndarray): The mid-point times for each velocity calculation.
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
    plt.show()

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
    plt.show()

# Function to plot velocity over time
def plot_velocity(tracks, colors):
    plt.figure(figsize=(6, 6))
    for i, track in enumerate(tracks):
        mid_times, speeds = calculate_velocity(track)
        plt.plot(mid_times, speeds, color=colors[i], alpha=0.7, label=f'Track {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed ($\mu$m/s)')
    plt.title('Velocity Over Time')
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.legend()
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(os.path.join(PLOT_FOLDER, 'velocity_over_time.png'), dpi=600)
    plt.show()

# Function to plot a histogram of speeds with a broken Y-axis
def plot_speed_histogram(tracks):
    """
    Plots a histogram of the speeds from all tracks combined, with a broken Y-axis
    to better visualize lower frequency bins. The histogram bars are colored using
    the middle color from the 'Blues' colormap.

    Parameters:
        tracks (list of ndarray): List of processed tracks.
    """
    import matplotlib.patches as patches

    # Collect all speeds from all tracks
    all_speeds = []
    for track in tracks:
        _, speeds = calculate_velocity(track)
        all_speeds.extend(speeds)
    all_speeds = np.array(all_speeds)

    # Set up the figure with two subplots (axes) stacked vertically
    f, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(6, 6), 
                                          gridspec_kw={'height_ratios': [1, 2]})

    # Define the number of bins
    bins = 30

    # Choose the middle color from the 'Blues' colormap
    cmap = plt.get_cmap('Blues')
    bar_color = cmap(0.75)  # Middle color

    # Plot the histogram on both axes
    counts, bins_edges, patches = ax_top.hist(all_speeds, bins=bins, color=bar_color, edgecolor='black', alpha=0.7)
    ax_bottom.hist(all_speeds, bins=bins, color=bar_color, edgecolor='black', alpha=0.7)

    # Determine the y-axis limits for both plots
    # Set a break point where the y-axis will be split
    y_break = max(counts) * 0.01  # Adjust as needed

    # Adjust y-axis limits
    ax_top.set_ylim(y_break, max(counts) * 1.05)  # Slightly above the max count
    ax_bottom.set_ylim(0, y_break)

    # Hide the spines between ax_top and ax_bottom
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(labeltop=False)  # Don't put tick labels at the top
    ax_bottom.xaxis.tick_bottom()

    # Add diagonal lines to indicate the break
    d = .015  # Size of diagonal lines in axes coordinates
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)        # Top-left diagonal
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right diagonal

    kwargs.update(transform=ax_bottom.transAxes)  # Switch to the bottom axes
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom-left diagonal
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Bottom-right diagonal

    #set gap between subplots
    plt.subplots_adjust(hspace=0.06)    

    #turn off y axis axes label 'frequency' for top plot
    # ax_top.set_yticklabels([])
    ax_top.set_ylabel('')

    # Set labels and title
    ax_bottom.set_xlabel('Speed ($\mu$m/s)')
    ax_bottom.set_ylabel('Frequency')
    ax_top.set_ylabel('')
    ax_top.set_title('Histogram of Speeds')

    # Add gridlines
    ax_top.grid(True, linestyle='--', alpha=0.5)
    ax_bottom.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save and show the plot
    if SAVE_PLOTS:
        plt.savefig(os.path.join(PLOT_FOLDER, 'speed_histogram_broken_axis.png'), dpi=600)
    plt.show()


# New function to plot speed histograms for different time steps
def plot_speed_histograms_multiple_steps(tracks, steps=[1, 2, 4, 8, 16]):
    """
    Plots speed histograms for different time steps over which velocities are calculated.

    Parameters:
        tracks (list of ndarray): List of processed tracks.
        steps (list of int): List of time step gaps to use for velocity calculations.
    """
    plt.figure(figsize=(6, 6))
    all_speeds_per_step = {}
    max_speed = 0

    # Colors for different histograms
    cmap = plt.get_cmap('Blues')
    colors = [cmap(i) for i in np.linspace(0.4, 1, len(steps))]

    for idx, step in enumerate(steps):
        all_speeds = []
        for track in tracks:
            _, speeds = calculate_velocity(track, step=step)
            all_speeds.extend(speeds)
        all_speeds = np.array(all_speeds)
        all_speeds_per_step[step] = all_speeds
        if len(all_speeds) > 0:
            max_speed = max(max_speed, np.max(all_speeds))

            # Plot histogram
            plt.hist(all_speeds, bins=30, alpha=0.5, color=colors[idx],
                     label=f'Time Step = {step}', edgecolor='black', density=False)

    plt.xlabel('Speed ($\mu$m/s)')
    plt.ylabel('Probability Density')
    plt.title('Speed Histograms for Different Time Steps')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(os.path.join(PLOT_FOLDER, 'speed_histograms_multiple_steps.png'), dpi=600)
    plt.show()


# Modified function to plot MSD with linear fits and statistics
def plot_msd(tracks, colors, t_gap):
    plt.figure(figsize=(6, 6))
    fit_results = []
    for i, track in enumerate(tracks):
        time_lags, msd = calculate_msd(track)
        # Perform linear fitting on data where time_lags > t_gap
        slope, intercept, r_value, p_value, std_err = fit_msd(time_lags, msd, t_gap)

        # Handle cases where fitting was not possible
        if np.isnan(slope):
            print(f"Not enough data beyond t_gap = {t_gap} s for Track {i+1}")
            continue  # Skip to the next track

        fit_results.append({
            'Track': i+1,
            'Slope': slope,
            'Intercept': intercept,
            'R_squared': r_value**2,
            'P_value': p_value,
            'Std_err': std_err
        })
        # Plot MSD data
        plt.loglog(time_lags, msd, marker='o', color=colors[i], alpha=0.7,
                   markeredgewidth=0, markersize=3)
        # Plot fitted line over the fitting range
        fit_time_lags = np.linspace(min(time_lags[time_lags > t_gap]), max(time_lags), 100)
        fit_msd_values = 10**(intercept) * fit_time_lags**slope
        # plt.loglog(fit_time_lags, fit_msd_values, color=colors[i], linestyle='--', linewidth=1)
    
    # Add reference lines
    ref_time = np.logspace(np.log10(t_gap), np.log10(max(time_lags)), num=100)
    D_eff = 1  # Adjust as needed
    msd_passive = 2 * D_eff * ref_time
    plt.loglog(ref_time, msd_passive, 'k--', linewidth=2, label='Passive Diffusion (slope=1)')
    v_eff = 1  # Adjust as needed
    msd_active = (v_eff * ref_time)**2
    plt.loglog(ref_time, msd_active, 'k:', linewidth=2, label='Active Transport (slope=2)')
    
    plt.xlabel('Time Lag (s)')
    plt.ylabel('Mean Squared Displacement ($\mu$m$^2$)')
    plt.title('MSD Over Time Lag (Log-Log Scale)')
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    
    # Legend for reference lines
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[-2:], labels[-2:], loc='upper left')
    
    # Display slope statistics if any fits were successful
    if fit_results:
        slopes = [result['Slope'] for result in fit_results]
        min_slope = min(slopes)
        max_slope = max(slopes)
        mean_slope = np.mean(slopes)
        textstr = (
            'Scaling Exponent (Slope):\n'
            f'Min: {min_slope:.3f}\n'
            f'Max: {max_slope:.3f}\n'
            f'Mean: {mean_slope:.3f}'
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
    plt.show()
    
    # Print fit results
    print("MSD Linear Fit Results (log-log scale):")
    for result in fit_results:
        print(f"Track {result['Track']}:")
        print(f"  Slope (Scaling Exponent): {result['Slope']:.3f}")
        print(f"  Intercept: {result['Intercept']:.3f}")
        print(f"  R-squared: {result['R_squared']:.3f}")
        print(f"  P-value: {result['P_value']:.3e}")
        print(f"  Standard Error: {result['Std_err']:.3f}")
        print()

def main():
    # Track images folder
    track_image_folder = os.path.join(BASE_FOLDER, "tracks")  # Change this to your folder containing track images
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
    plot_velocity(proc_tracks, colors) 
    plot_speed_histogram(proc_tracks)
    plot_speed_histograms_multiple_steps(proc_tracks, steps=[1, 2, 4, 8, 16])
    plot_msd(proc_tracks, colors, T_GAP)

if __name__ == '__main__':
    main()








