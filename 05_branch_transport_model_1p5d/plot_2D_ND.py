# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2025 Pranav Vyas
"""Plot 1.5D branch geometry profiles in the (Pe_f^-1, Da_d) plane.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 6B; Fig. S19
What it does  : Reads the final snapshot of every run of the non-dimensional
                sweep (para_sweep_non_dim.py), rebuilds the branch width
                profile w(x) from the deposited field c_d, and draws each
                profile (true aspect ratio, 10 um scale bar) at its
                (Pe_f^-1, Da_d) position on log-log axes. To reduce clutter, runs
                with 0.01 < Da_d < 0.1 are thinned to every 2nd and runs with
                Da_d <= 0.01 to every 4th (within each Da_d group, in folder order).
                The file also contains the simulator class (needed to unpickle
                the snapshots) and earlier plotting variants that main does not call.
Inputs        : OUT_ROOT/05_branch_transport_model_1p5d/sweep_study_ND/
                phase_space_study_ND_pairwise_<timestamp>/ (set my_root_folder)
Outputs       : <that folder>/plots/geometry_profiles_Da_d_invPe_loglog_small.eps
Environment   : environment-models.yml (Python 3.13)
Run           : python plot_2D_ND.py

Symbols (code -> SI): D2 -> D_f, k4 -> k_d, k5 -> k_t, epsilon -> eps,
vm -> v_m, m_dot_0 -> mdot_0, c2/c4/cg -> c_f/c_d/c_g (dimensional classes),
cf_bar/cd_bar/cg_bar -> non-dimensional c_f/c_d/c_g, Pe2_inv/Pe_inv -> Pe_f^-1,
Da4/Da_d -> Da_d, Da5/Da_t -> Da_t, Da_epsilon/Da_eps -> Da_eps (SI Eqs. 3-8).
"""
import os
import re
import pickle
import numpy as np
import pyvista as pv

import matplotlib.pyplot as plt
import datetime
import shutil
import cv2
from itertools import product
import traceback
from matplotlib.ticker import FuncFormatter

# --- USER PATHS -------------------------------------------------------------
# Point OSSICLE_DATA / OSSICLE_OUT at your copies (see data/README.md).
from pathlib import Path
try:
    _HERE = Path(__file__).resolve()
except NameError:  # interactive (e.g. Spyder cell) execution
    _HERE = Path.cwd().resolve() / "_"
REPO_ROOT = next((p for p in _HERE.parents if (p / "CITATION.cff").exists()), _HERE.parent)
DATA_ROOT = Path(os.environ.get("OSSICLE_DATA", REPO_ROOT / "data"))
OUT_ROOT = Path(os.environ.get("OSSICLE_OUT", REPO_ROOT / "outputs"))
# ----------------------------------------------------------------------------

#choose font helvetica
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.family'] = "sans-serif"  


class ReactionDiffusionSimulator:
    """
    A 1D Reaction-Diffusion simulator that models:
      - Diffusion/advection (primarily for c2 in this version)
      - Precipitation reaction updating c4
      - A growth zone concentration 'cg'
      - Dynamic domain growth
      - Running, storing, and plotting results
    """
    def __init__(self, parameters_file, folder_dir=None, save_pkl=False, save_plot=False):
        """
        Constructor that reads parameters from file and initializes simulation variables.
        :param parameters_file: Path to a text file containing parameters (key = value pairs).
        :param folder_dir: Directory path where pkl/plots can be saved (optional).
        :param save_pkl: Boolean controlling whether to save .pkl snapshots.
        :param save_plot: Boolean controlling whether to save .png plots.
        """
        self.folder_dir = folder_dir
        self.save_pkl = save_pkl
        self.save_plot = save_plot

        self.read_parameters(parameters_file)
        self.initialize_variables_dim()

    def read_parameters(self, parameters_file):
        """
        Read parameter values from a text file in the format:
           key = value
        and set them as attributes of the simulator instance.
        """
        with open(parameters_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if not line or '=' not in line:
                    continue
                key, value = line.split('=')
                key, value = key.strip(), float(value.strip())
                if key == 'N':
                    setattr(self, key, int(value))
                else:
                    setattr(self, key, value)

        # Speedups by storing some "inverse" or "constant" values once:
        self.inv_delta_x = 1.0 / self.delta_x
        self.inv_l_g = 1.0 / self.l_g
        self.delta_x_sq = self.delta_x * self.delta_x

        # Compute 'mu' once (since mu = 4 / (rho*pi*w0^2))
        self.mu = 4.0 / (self.rho * np.pi * self.w0 * self.w0)
        
        # Some derived dimensionless parameters
        self.Pe2_inv = self.D2 / (self.vm * self.w0)
        self.Da4 = self.k4 * self.w0 / self.vm
        self.Da5 = self.k5 * self.w0 / self.vm
        self.Da_epsilon = self.epsilon / self.vm

        self.delta_t_bar = self.delta_t * self.vm / self.w0
        self.delta_x_bar = self.delta_x / self.w0
        self.l_g_bar = self.l_g / self.w0
        self.N_bar = int(self.N)
        self.time = 0

        # Additional attributes for tracking
        self.calc_tip_range = int(3 * self.l_g / self.delta_x)
        self.clim = self.delta_x / (self.mu * self.l_g)
        self.mu_int = self.mu
        self.max_conc = 0

        # Quick parameter printouts for verification
        print(
            'D2=', self.D2,
            'k4=', self.k4,
            'k5=', self.k5,
            'epsilon=', self.epsilon,
            'csat=', self.csat,
            'vm=', self.vm,
            'mu=', self.mu,
            'l_g=', self.l_g,
            'm_dot_0=', self.m_dot_0,
            'delta_t=', self.delta_t,
            'delta_x=', self.delta_x,
            'N=', self.N
        )
        print(
            'w0=', self.w0,
            'rho=', self.rho,
            'Pe2_inv=', self.Pe2_inv,
            'Da4=', self.Da4,
            'Da5=', self.Da5,
            'Da_epsilon=', self.Da_epsilon
        )
        print(
            'delta_t_bar=', self.delta_t_bar,
            'delta_x_bar=', self.delta_x_bar,
            'l_g_bar=', self.l_g_bar,
            'N_bar=', self.N_bar
        )

    def initialize_variables_dim(self):
        """
        Initialize the concentration arrays (c2, c4) with preallocation,
        set the domain length, etc.
        """
        # 1) Preallocate big arrays
        self.max_N = 2000  # Chosen large enough for entire simulation
        self.c2 = np.zeros(self.max_N)
        self.c4 = np.zeros(self.max_N)

        # 2) Fill only the first self.N entries (the 'active' domain):
        self.c2[:self.N] = 0.0
        self.c4[:self.N] = 1.0 / self.mu

        # Growth zone concentration
        self.cg = 0.0

        # Domain length in micrometers, based on initial N
        self.l = self.delta_x * self.N

        # Calculate domain width
        local_mass = self.c4[:self.N] * self.delta_x
        local_volume = local_mass / self.rho
        self.w = np.sqrt(4.0 * local_volume / self.delta_x / np.pi)

        # Total initial mass
        c2_mass = np.sum(self.c2[:self.N]) * self.delta_x
        c4_mass = np.sum(self.c4[:self.N]) * self.delta_x
        self.total_mass = c2_mass + c4_mass + self.cg * self.l_g

        # Keep track of total mass injected
        self.total_mass_injected = self.total_mass

        # For c2 updates near the domain tip
        self.calc_start_index = max(self.N - self.calc_tip_range, 0)

        # Conditionally save a .pkl snapshot
        if self.save_pkl and self.folder_dir is not None:
            pkl_path = os.path.join(self.folder_dir, f"{int(round(self.time))}.pkl")
            with open(pkl_path, 'wb') as f:
                pickle.dump(self, f)

    def run_simulation_dim(self, num_steps):
        """
        Advance the simulation by 'num_steps' steps using preallocated arrays.
        """
        # Local references to avoid repeated lookups
        c2 = self.c2
        c4 = self.c4
        delta_x = self.delta_x
        delta_t = self.delta_t
        vm = self.vm
        D2 = self.D2
        k4 = self.k4
        epsilon = self.epsilon
        csat = self.csat
        rho = self.rho

        for _ in range(num_steps):
            # 1. Precipitation (vectorized)
            c2_active = c2[:self.N]  # slice for current domain
            c4_active = c4[:self.N]

            precipitate = k4 * (c2_active - csat)
            precipitate[precipitate < 0] = 0

            # 2. Diffusion & advection in the interior (calc_start_index to N-2)
            start = self.calc_start_index + 1
            end   = self.N - 1
            c2_slice  = c2_active[start:end]
            c2_left   = c2_active[start-1 : end-1]
            c2_right  = c2_active[start+1 : end+1]
            prec_slice = precipitate[start:end]

            c2_active[start:end] += delta_t * (
                (D2 / (delta_x**2)) * (c2_left - 2.0*c2_slice + c2_right)
                - vm * (c2_slice - c2_left) / delta_x
                - prec_slice
            )

            # 3. c4 update
            c4_active += delta_t * precipitate

            # 4. Growth zone accumulation
            tip_val = c2_active[self.N - 1]
            self.cg += (delta_t * epsilon * tip_val) / self.l_g

            # 5. Boundary conditions
            # Left boundary
            c2_active[0] = (
                c2_active[1]*D2 + self.m_dot_0*delta_x
            ) / (D2 + vm*delta_x)

            # Right boundary
            c2_active[self.N - 1] = (
                c2_active[self.N - 2]*(D2 + vm*delta_x)
            ) / (D2 + epsilon*delta_x)

            # 6. Domain growth
            c2_tip = c2_active[self.N - 1] 
            self.mu_int = 1.0 / (1.0/self.mu + c2_tip)
            self.clim = delta_x / (self.mu_int * self.l_g)  # the concentration needed in growth zone to add one step

            delta_N = int(
                ((self.cg - csat) * self.l_g * self.mu_int * self.k5 * delta_t)
                / delta_x
            )
            if delta_N <= 0:
                delta_N = 0
            else:
                oldN = self.N
                self.N += delta_N
                # c2's new cells get c2_tip
                c2[oldN:self.N] = c2_tip
                # c4's new cells get 1/mu
                c4[oldN:self.N] = 1.0 / self.mu

            # Increase domain length
            delta_l = (delta_N) * delta_x
            self.l += delta_l
            # (growth velocity just for reference)
            v_g = delta_l / delta_t 

            # Remove mass from growth zone
            self.cg -= (delta_l / self.mu_int) / self.l_g

            # 7. Time increment
            self.time += delta_t

            # 8. Recompute width
            local_mass = c4 * delta_x
            local_volume = local_mass / rho
            self.w = np.sqrt(4.0 * local_volume / delta_x / np.pi)

            # 9. Update total mass
            self.total_mass = (
                np.sum(c2) + np.sum(c4)
            ) * delta_x + self.cg * self.l_g
            self.total_mass_injected += self.m_dot_0 * delta_t

            # 10. Update calc_start_index
            self.calc_start_index = max(self.N - self.calc_tip_range, 0)

        # Optional final .pkl save after all steps
        if self.save_pkl and self.folder_dir is not None:
            pkl_path = os.path.join(self.folder_dir, f"{int(round(self.time))}.pkl")
            with open(pkl_path, 'wb') as f:
                pickle.dump(self, f)

    def plot_variable_dim(self, fig, ax, ax2):
        """
        Plot the concentration profile (c2) over the domain,
        along with the branch width and the growth zone, using
        only the active domain size (self.N).
        """
        ax.clear()
        ax2.clear()

        # --- Colors ---
        color2 = plt.cm.tab20c(5)
        color5 = plt.cm.tab20c(16)
        cmap = plt.get_cmap('binary')
        norm = plt.Normalize(vmin=0, vmax=self.clim)

        # --- Slicing active domain ---
        c2_active = self.c2[:self.N]
        w_active = self.w[:self.N]
        
        x_vals = np.linspace(
            self.delta_x / 2,
            self.l - self.delta_x / 2,
            self.N
        )

        # --- Plot c2 ---
        ax.plot(x_vals, c2_active, label=r'c_2', color=color2)
        ax.fill_between(x_vals, c2_active, color=color2, alpha=0.5)
        max_conc_check = np.max(c2_active)
        ax.set_ylim(0, max_conc_check * 1.1 if max_conc_check > 0 else 1.0)

        # --- Plot the growth zone as a rectangle ---
        ax2.fill_between(
            [self.l, self.l + self.l_g], 
            -self.w0 / 2, 
            self.w0 / 2,
            color=cmap(norm(self.cg)),
            alpha=0.5
        )
        rectangle = plt.Rectangle(
            (self.l, -self.w0 / 2),
            self.l_g,
            self.w0,
            edgecolor='black',
            facecolor='none',
            linestyle='--'
        )
        ax2.add_patch(rectangle)

        # --- Plot growth zone concentration (cg) ---
        ax.plot(
            self.l + self.l_g / 2,
            self.cg,
            color=color5,
            label=r'c_g',
            marker='o',
            markersize=10
        )
        ax.plot(
            [self.l + self.l_g / 2, self.l + self.l_g / 2],
            [0, self.cg],
            color=color5,
            linestyle='--'
        )

        # --- Plot the branch boundary (using w_active) ---
        ax2.plot(x_vals, w_active / 2.0, 'k-')
        ax2.plot(x_vals, -w_active / 2.0, 'k-')
        ax2.fill_between(
            x_vals,
            w_active / 2.0,
            -w_active / 2.0,
            color=color5,
            alpha=0.5
        )

        # --- Axes limits and aspect ratio ---
        max_len = max(self.l + 2.0, 25.0)
        max_wid = max(self.w0 * 2.0, max_len / 2.0)
        ax2.set_xlim(0, max_len)
        ax2.set_ylim(-max_wid, max_wid)
        ax2.set_aspect('equal')

        # --- Time stamp ---
        ax.text(
            0.5, 0.98,
            f'Time: {int(round(self.time))} s',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=12
        )

        # --- Axes labels and style ---
        ax.set_xlabel(r'Length ($\mu m$)', fontsize=12)
        ax.set_ylabel(r'Concentration ($pg/\mu m$)', fontsize=12)
        ax2.set_ylabel(r'Width ($\mu m$)', fontsize=12)
        ax.set_title('Concentration vs Length', fontsize=15)
        ax.set_facecolor('gainsboro')
        ax.legend(facecolor='gainsboro', frameon=False, loc='upper right')
        fig.subplots_adjust(left=0.08)

        # --- Parameter text ---
        c2_sum = np.sum(c2_active)
        c4_sum = np.sum(self.c4[:self.N])
        parameters_text = (
            f'D2 = {self.D2}\n'
            f'k4 = {self.k4}\n'
            f'k5 = {self.k5}\n'
            f'epsilon = {self.epsilon}\n'
            f'csat = {self.csat}\n'
            f'vm = {self.vm}\n'
            f'mu = {self.mu}\n'
            f'mu_int = {self.mu_int}\n'
            f'l_g = {self.l_g}\n'
            f'm_dot_0 = {self.m_dot_0}\n'
            f'delta_t = {self.delta_t}\n'
            f'delta_x = {self.delta_x}\n'
            f'w0 = {self.w0}\n'
            f'rho = {self.rho}\n'
            f'N = {self.N}\n'
            f'l = {self.l}\n'
            f'integrated_mass = {self.total_mass}\n'
            f'mass_injected = {self.total_mass_injected}\n'
            f'mass_difference = {self.total_mass - self.total_mass_injected}\n'
            f'clim = {self.clim}\n'
            f'c2_sum = {c2_sum}\n'
            f'c4_sum = {c4_sum}\n'
            f'cg = {self.cg}\n'
            f'time = {int(round(self.time))}\n'
        )

        fig.subplots_adjust(right=0.75)
        ax2.text(
            1.1, -0.02,
            parameters_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom'
        )


def write_params_to_file(param_dict, file_path):
    """
    Utility function: Given a dictionary {key: value}, 
    writes each parameter as 'key = value' to file_path.
    """
    with open(file_path, 'w') as file:
        for k, v in param_dict.items():
            file.write(f"{k} = {v}\n")


def run_phase_space_study(
    # Provide ranges (lists) for the parameters you want to vary:
    D2_list, 
    k4_list, 
    vm_list,
    # Provide defaults or single values for other parameters that might be held constant:
    k5=100.0,
    csat=0.0,
    m_dot_0=0.01,
    w0=1.0,
    rho=2.71,
    delta_t=0.01,
    delta_x=0.05,
    N=5,
    l_g=0.5,
    # Reaction rate / precipitation param, could vary if desired
    epsilon_factor=10.0,  # the factor multiplied by vm
    # Control saving flags and iteration details
    save_plot=False,
    save_pkl=False,
    base_output_folder=str(OUT_ROOT / "05_branch_transport_model_1p5d"),
    num_simulation_steps=10,   # number of times we do run_simulation_dim(...)
    steps_per_run=1000,       # how many steps each call to run_simulation_dim() uses
    show_plots=False          # set to True if you want to see plots in a live window
    ):
    """
    Runs all combinations of parameters from the provided lists. 
    Each combination gets its own folder, with param file, saved plots, pickles, etc.
    If a combination fails, it is logged to 'error_log.txt' and skipped.
    """
    # 1. Create a timestamped directory for all results
    date_time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    global_folder = os.path.join(base_output_folder, 'phase_space_study_' + date_time_stamp)
    os.makedirs(global_folder, exist_ok=True)

    # 2. Create or open error log file
    error_log_path = os.path.join(global_folder, 'error_log.txt')
    with open(error_log_path, 'w') as f:
        f.write("Error Log\n")
        f.write("=========\n\n")

    # 3. Generate all combinations of D2, k4, vm
    all_combinations = list(product(D2_list, k4_list, vm_list))
    
    # 4. Loop over each combination
    for idx, (D2_val, k4_val, vm_val) in enumerate(all_combinations):
        # Prepare output folder name for this combination
        combo_folder_name = f"D2_{D2_val}_k4_{k4_val}_vm_{vm_val}"
        combo_folder_path = os.path.join(global_folder, combo_folder_name)
        os.makedirs(combo_folder_path, exist_ok=True)

        print(f"\n[{idx+1}/{len(all_combinations)}] Running combination: {combo_folder_name}")

        # Prepare the parameter dictionary for this combination
        epsilon_val = epsilon_factor * vm_val  # e.g. epsilon = 10 * vm
        param_dict = {
            'D2': D2_val,
            'k4': k4_val,
            'k5': k5,
            'epsilon': epsilon_val,
            'csat': csat,
            'vm': vm_val,
            'm_dot_0': m_dot_0,
            'delta_t': delta_t,
            'delta_x': delta_x,
            'N': N,
            'w0': w0,
            'rho': rho,
            'l_g': l_g
        }

        # Write the parameters file
        param_file_name = 'exp_parameters.txt'
        param_file_path = os.path.join(combo_folder_path, param_file_name)
        write_params_to_file(param_dict, param_file_path)

        # Now run the simulation in a try/except block
        try:
            # Create the simulator instance
            simulator = ReactionDiffusionSimulator(
                parameters_file=param_file_path,
                folder_dir=combo_folder_path,
                save_pkl=save_pkl,
                save_plot=save_plot
            )

            # Prepare figure for plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            ax2 = ax.twinx()

            # Possibly show the initial plot
            simulator.plot_variable_dim(fig, ax, ax2)
            plot_name = f"plot_initial.png"
            if save_plot:
                plt.savefig(os.path.join(combo_folder_path, plot_name))
            if show_plots:
                plt.pause(0.1)

            # Perform simulation in increments
            for step_i in range(num_simulation_steps):
                simulator.run_simulation_dim(steps_per_run)

                print(
                    f"Combination {combo_folder_name}, time={simulator.time}, "
                    f"c_end={simulator.c2[simulator.N-1]}, clim={simulator.clim}, cg={simulator.cg}"
                )

                # Plot
                simulator.plot_variable_dim(fig, ax, ax2)
                
                # Save or show plot at intermediate step
                plot_number = str((step_i+1) * steps_per_run).zfill(6)
                if save_plot:
                    plt.savefig(os.path.join(combo_folder_path, f'plot_{plot_number}.png'))
                if show_plots:
                    plt.pause(0.1)

            # Optionally compile images into a video if desired (like in original code)
            # or simply leave the images as is.

            plt.close(fig)  # close the figure to free resources

        except Exception as e:
            # Log the error, skip this combination
            print(f"Error encountered for {combo_folder_name}. See error_log.txt for details.")
            with open(error_log_path, 'a') as f:
                f.write(f"Error for {combo_folder_name}:\n")
                f.write(f"{str(e)}\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
                f.write("\n\n")

            # Move on to the next combination
            continue


def load_final_simulator_state(folder_path):
    """
    Searches for .pkl files in 'folder_path', picks the one with 
    the largest integer time in its filename (e.g. "1000.pkl"),
    and returns the loaded ReactionDiffusionSimulator object.
    Returns None if no PKL files found or loading fails.
    """
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    if not pkl_files:
        return None
    
    # Attempt to parse out the numeric portion from each filename
    # We assume a pattern "NNN.pkl" or "...NNN.pkl" at the end
    time_values = []
    for fn in pkl_files:
        match = re.search(r"(\d+)\.pkl$", fn)
        if match:
            time_values.append((int(match.group(1)), fn))
        else:
            time_values.append((-1, fn))

    if not time_values:
        return None

    # Pick the file with the largest integer time
    time_values.sort(key=lambda x: x[0], reverse=True)
    final_file = time_values[0][1]  # the filename with the largest time
    final_path = os.path.join(folder_path, final_file)

    try:
        with open(final_path, 'rb') as f:
            simulator = pickle.load(f)
        return simulator
    except Exception as e:
        print(f"Could not load PKL file: {final_path}. Error: {e}")
        return None


def revolve_width_profile(x_vals, w_vals):
    """
    Given arrays x_vals (the axial coordinate) and w_vals (the full width),
    create a 2D polyline (x along the horizontal axis, radius = w/2 on Y),
    then revolve around the X-axis to produce a 3D mesh.
    
    Returns a pyvista PolyData object.
    """
    # Create an array of points for the polyline:
    # We'll revolve around the X-axis, so place the radius in Y.
    polyline_points = []
    for x, width in zip(x_vals, w_vals):
        r = 0.5 * width  # radius
        polyline_points.append([x, r, 0])  # (x, y, z=0)

    points = np.array(polyline_points)

    # Create a line from the points by spline fitting
    line = pv.Spline(points, 100)

    # Revolve around the X-axis by 360 degrees
    # (requires pyvista >= 0.38.4 for axis='x')
    mesh_3d = line.extrude_rotate(resolution=100, rotation_axis=[1, 0, 0])
    return mesh_3d


def generate_2d_geometry_profiles_in_parameter_space_vm(
    root_output_folder,
    fixed_vm_index=0,
    scale_factor=1.0,
    scale_bar_length=10.0,  # physical length in microns for the scale bar
    primary_x_scale=1.0,
    primary_y_scale=1.0,
    save_plot=False,
    save_path=None
):
    """
    Traverse 'root_output_folder' to find subfolders corresponding to parameter combinations
    with folder names like "D2_<val>_k4_<val>_vm_<val>".
    
    For a fixed vm value (selected by fixed_vm_index, an integer 0–4 corresponding to the
    sorted unique vm values found), this function loads the final ReactionDiffusionSimulator
    state from each matching folder and extracts its full width profile.
    
    Each branch profile is drawn in the 2D phase space at its simulation's (D₂, k₄) origin,
    where the simulation origin is first scaled by:
      - primary_x_scale (applied to D₂), and
      - primary_y_scale (applied to k₄).
    
    However, the primary x- and y-axis tick labels are remapped (using custom formatters)
    so that the ticks show the actual (unscaled) D₂ and k₄ values.
    
    The branch profiles (which contain geometry width and length data in microns) are drawn 
    using physical coordinates computed via scale_factor. In addition, a secondary x-axis 
    (top) and a secondary y-axis (right) are added. Their transformation functions use the 
    drawn scale bar as reference, so that a branch profile drawn at
       (primary_x_scale * D₂ + x_offset, primary_y_scale * k₄ + offset)
    is mapped to physical branch coordinates
       (x_offset, offset)
    via:
       x_phys = (x - canonical_scaled_D2) / scale_factor
       y_phys = (y - canonical_scaled_k4) / scale_factor,
    where canonical_scaled_D2 and canonical_scaled_k4 are the canonical simulation's origins 
    after primary-axis scaling. This ensures that the secondary axes correctly display the 
    physical dimensions, with the drawn scale bar (of length scale_bar_length×scale_factor) 
    corresponding to scale_bar_length microns.
    
    Parameters:
      root_output_folder : str
          Path to the folder containing the subfolders for each parameter combination.
      fixed_vm_index : int (default 0)
          An index (0 to 4) selecting which vm value (from the sorted unique vm values) to use.
      scale_factor : float (default 1.0)
          Factor to scale the simulation's domain length and width profile (physical units).
      scale_bar_length : float (default 10.0)
          The physical length (in microns) of the scale bar to draw.
      primary_x_scale : float (default 1.0)
          Scaling factor for the primary x-axis (D₂ values). The simulation origin is plotted at
          primary_x_scale * D₂, but the axis tick is remapped to show the original D₂.
      primary_y_scale : float (default 1.0)
          Scaling factor for the primary y-axis (k₄ values). The simulation origin is plotted at
          primary_y_scale * k₄, but the axis tick is remapped to show the original k₄.
      save_plot : bool (default False)
          If True, the final 2D plot will be saved to a file.
      save_path : str (default None)
          The full file path where the plot will be saved if save_plot is True.
    """
    import os, re
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # Assume load_final_simulator_state is defined elsewhere.
    folder_pattern = re.compile(r"D2_([-+]?\d*\.?\d+)_k4_([-+]?\d*\.?\d+)_vm_([-+]?\d*\.?\d+)")
    
    simulation_entries = []  # Each entry: (D2, k4, vm, folder_path)
    for combo_name in os.listdir(root_output_folder):
        combo_path = os.path.join(root_output_folder, combo_name)
        if not os.path.isdir(combo_path):
            continue
        match = folder_pattern.match(combo_name)
        if not match:
            continue
        D2_val = float(match.group(1))
        k4_val = float(match.group(2))
        vm_val = float(match.group(3))
        simulation_entries.append((D2_val, k4_val, vm_val, combo_path))
    
    if not simulation_entries:
        print("No simulation folders found matching the pattern.")
        return

    # Get unique vm values (sorted)
    vm_values = sorted(list({entry[2] for entry in simulation_entries}))
    if len(vm_values) < 5:
        print(f"Warning: Expected 5 distinct vm values but found {len(vm_values)}.")
    if fixed_vm_index < 0 or fixed_vm_index >= len(vm_values):
        print(f"fixed_vm_index ({fixed_vm_index}) is out of range. Using index 0 instead.")
        fixed_vm_index = 0
    selected_vm = vm_values[fixed_vm_index]
    print(f"Selected vm value for plotting: {selected_vm}")

    fig, ax = plt.subplots(figsize=(5, 7))
    
    # For computing overall bounding box (in primary axis coordinates)
    global_x_min, global_x_max = np.inf, -np.inf
    global_y_min, global_y_max = np.inf, -np.inf

    # Choose a canonical simulation (first loaded) to define the physical coordinate transform.
    canonical_origin = None  # will store (D2, k4) of the canonical simulation

    # Loop over each simulation entry with the selected vm value.
    for (D2_val, k4_val, vm_val, combo_path) in simulation_entries:
        if not np.isclose(vm_val, selected_vm, atol=1e-6):
            continue

        sim = load_final_simulator_state(combo_path)
        if sim is None:
            print(f"Could not load simulator state for folder: {combo_path}")
            continue

        # Set canonical origin from the first simulation successfully loaded.
        if canonical_origin is None:
            canonical_origin = (D2_val, k4_val)

        # Compute the simulation origin in primary (scaled) coordinates.
        scaled_D2 = primary_x_scale * D2_val
        scaled_k4 = primary_y_scale * k4_val

        # Create the axial coordinate for the active domain (physical branch coordinate).
        x_vals = np.linspace(sim.delta_x / 2, sim.l - sim.delta_x / 2, sim.N)
        w_vals = sim.w[:sim.N]  # branch width profile
        
        # Compute physical offsets (in microns)
        x_offset = (x_vals - x_vals[0]) * scale_factor
        top_offset = (w_vals / 2) * scale_factor
        bottom_offset = - (w_vals / 2) * scale_factor

        # The branch profile is anchored at the scaled simulation origin.
        profile_x = scaled_D2 + x_offset
        profile_top = scaled_k4 + top_offset
        profile_bottom = scaled_k4 + bottom_offset

        # Plot the branch profile: boundaries and fill.
        ax.plot(profile_x, profile_top, 'k-', lw=0.5)
        ax.plot(profile_x, profile_bottom, 'k-', lw=0.5)
        ax.fill_between(profile_x, profile_bottom, profile_top, color='grey', alpha=0.3)
        # Mark the simulation origin with a small circle.
        ax.plot(scaled_D2, scaled_k4, 'ko', markersize=2)

        # Update global bounding box.
        global_x_min = min(global_x_min, np.min(profile_x))
        global_x_max = max(global_x_max, np.max(profile_x))
        global_y_min = min(global_y_min, np.min(profile_bottom))
        global_y_max = max(global_y_max, np.max(profile_top))
    
    # Add margins to ensure all shapes are fully visible.
    margin_x = 0.05 * (global_x_max - global_x_min)
    margin_y = 0.05 * (global_y_max - global_y_min)
    ax.set_xlim(0, global_x_max + margin_x)
    ax.set_ylim(0, global_y_max + margin_y)
    
    ax.set_aspect('equal', adjustable='datalim')
    # Set the axis labels to be the actual (unscaled) parameter values.
    ax.set_xlabel("$D_f$", fontsize=15)
    ax.set_ylabel("$k_d$", fontsize=15)
    ax.set_title(f"2D Geometry Profiles for $v_m$ = {selected_vm}", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)

    #turn on major and minor gridlines
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)

    # --- Remap the primary x- and y-axis tick labels.
    # Although the plotted coordinates are scaled, we want the tick labels to show the original D2 and k4.
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/primary_x_scale:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y/primary_y_scale:g}"))
    
    #set specific ticks on the x-axis and y-axis
    D2_array = (primary_x_scale*np.linspace(0.01, 0.1, 5)).tolist()
    k4_array = (primary_y_scale*np.linspace(0.01, 0.1, 5)).tolist()
    ax.set_xticks(D2_array)
    ax.set_yticks(k4_array)

    # Draw a horizontal scale bar in the main axes.
    bar_end_x = global_x_max 
    bar_y = global_y_max + 5*margin_y/6
    # The drawn scale bar length in primary coordinates:
    bar_length_plot = scale_bar_length * scale_factor  
    bar_start_x = bar_end_x - bar_length_plot
    ax.plot([bar_start_x, bar_end_x], [bar_y, bar_y], 'k-', lw=3)
    
    # --- Add secondary axes to show physical branch coordinates.
    # Use the canonical simulation's origin (after primary scaling) as reference.
    if canonical_origin is not None:
        canonical_D2, canonical_k4 = canonical_origin
        canonical_scaled_D2 = primary_x_scale * canonical_D2
        canonical_scaled_k4 = primary_y_scale * canonical_k4

        # Adjust the transformation to use the scale_bar reference:
        # For the canonical simulation, a point at
        # (canonical_scaled_D2 + x_offset, canonical_scaled_k4 + offset)
        # should map to physical coordinates (x_offset, offset) via:
        def main_to_phys_x(x):
            return (x - canonical_scaled_D2) / scale_factor

        def phys_to_main_x(x_phys):
            return x_phys * scale_factor + canonical_scaled_D2

        def main_to_phys_y(y):
            return (y - canonical_scaled_k4) / scale_factor

        def phys_to_main_y(y_phys):
            return y_phys * scale_factor + canonical_scaled_k4

        secax_x = ax.secondary_xaxis('top', functions=(main_to_phys_x, phys_to_main_x))
        secax_y = ax.secondary_yaxis('right', functions=(main_to_phys_y, phys_to_main_y))
    
    plt.tight_layout()

    if save_plot and save_path is not None:
        #create plot filename with fixed vm value
        filename = r"2d_geometry_profiles_vm_" + str(selected_vm) + ".png"
        save_path = os.path.join(save_path, filename)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    
    plt.show()


def generate_2d_geometry_profiles_in_parameter_space_fixed_D2(
    root_output_folder,
    fixed_D2_index=0,
    scale_factor=1.0,
    scale_bar_length=10.0,  # physical length in microns for the scale bar
    primary_x_scale=1.0,
    primary_y_scale=1.0,
    save_plot=False,
    save_path=None
):
    """
    Traverse 'root_output_folder' for folders named like "D2_<val>_k4_<val>_vm_<val>".
    For a fixed D₂ value (selected by fixed_D2_index from the sorted unique D₂ values),
    this function loads each simulation state and extracts its branch profile.
    
    In this mode, the free parameters are k₄ and vₘ. Each branch profile is drawn in the 
    2D parameter space with:
      - x-axis: k₄ (scaled by primary_x_scale), and
      - y-axis: vₘ (scaled by primary_y_scale).
      
    The branch profile (geometry in microns, computed via scale_factor) is added as an offset:
       profile_x = primary_x_scale * k₄ + x_offset,
       profile_y = primary_y_scale * vₘ + offset,
    where the offsets are computed from the simulation's domain.
    
    Primary-axis tick labels are reformatted so that even though the coordinates are scaled,
    the ticks show the actual (unscaled) k₄ and vₘ values. Secondary axes (top/right) use the
    drawn scale bar as a reference such that the physical branch coordinates are correctly
    shown (i.e. a drawn length of scale_bar_length×scale_factor maps to scale_bar_length microns).
    
    Parameters:
      root_output_folder : str
          Path to the folder containing the simulation subfolders.
      fixed_D2_index : int (default 0)
          Index into the sorted unique D₂ values to fix.
      scale_factor : float (default 1.0)
          Factor to scale the simulation's branch profile (in physical units).
      scale_bar_length : float (default 10.0)
          Physical length (microns) of the scale bar.
      primary_x_scale : float (default 1.0)
          Scaling factor applied to the k₄ values for plotting.
      primary_y_scale : float (default 1.0)
          Scaling factor applied to the vₘ values for plotting.
      save_plot : bool (default False)
          If True, saves the final plot.
      save_path : str (default None)
          Folder path to save the plot.
    """
    import os, re
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # load_final_simulator_state should be defined elsewhere.
    folder_pattern = re.compile(r"D2_([-+]?\d*\.?\d+)_k4_([-+]?\d*\.?\d+)_vm_([-+]?\d*\.?\d+)")
    simulation_entries = []
    for combo_name in os.listdir(root_output_folder):
        combo_path = os.path.join(root_output_folder, combo_name)
        if not os.path.isdir(combo_path): 
            continue
        match = folder_pattern.match(combo_name)
        if not match:
            continue
        D2_val = float(match.group(1))
        k4_val = float(match.group(2))
        vm_val = float(match.group(3))
        simulation_entries.append((D2_val, k4_val, vm_val, combo_path))
    
    if not simulation_entries:
        print("No simulation folders found matching the pattern.")
        return

    # Get unique D₂ values and select one.
    D2_values = sorted(list({entry[0] for entry in simulation_entries}))
    if len(D2_values) < 1:
        print("No D₂ values found.")
        return
    if fixed_D2_index < 0 or fixed_D2_index >= len(D2_values):
        print(f"fixed_D2_index ({fixed_D2_index}) is out of range. Using index 0 instead.")
        fixed_D2_index = 0
    selected_D2 = D2_values[fixed_D2_index]

    selected_D2 = np.round(selected_D2, 4)

    print(f"Selected D₂ value for plotting: {selected_D2}")

    # Create figure and axis.
    fig, ax = plt.subplots(figsize=(10, 3.2))
    
    # For computing overall bounding box in primary coordinates.
    global_x_min, global_x_max = np.inf, -np.inf
    global_y_min, global_y_max = np.inf, -np.inf

    # Use the first simulation (in the filtered set) as canonical for transformation.
    canonical_origin = None  # will store (k4, vm) of the canonical simulation.

    # Loop over simulations with fixed D₂.
    for (D2_val, k4_val, vm_val, combo_path) in simulation_entries:
        if not np.isclose(D2_val, selected_D2, atol=1e-6):
            continue

        sim = load_final_simulator_state(combo_path)
        if sim is None:
            print(f"Could not load simulator state for folder: {combo_path}")
            continue

        if canonical_origin is None:
            canonical_origin = (k4_val, vm_val)

        # Here, the simulation origin (free parameters) are:
        # x: k₄, y: vₘ. They are scaled by primary_x_scale and primary_y_scale.
        scaled_k4 = primary_x_scale * k4_val
        scaled_vm = primary_y_scale * vm_val

        # Compute branch profile offsets (in microns).
        x_vals = np.linspace(sim.delta_x/2, sim.l - sim.delta_x/2, sim.N)
        w_vals = sim.w[:sim.N]
        x_offset = (x_vals - x_vals[0]) * scale_factor
        top_offset = (w_vals/2) * scale_factor
        bottom_offset = - (w_vals/2) * scale_factor

        # In this parameter space, add x_offset to k₄ and offset to vₘ.
        profile_x = scaled_k4 + x_offset
        profile_top = scaled_vm + top_offset
        profile_bottom = scaled_vm + bottom_offset

        # Plot the branch profile.
        ax.plot(profile_x, profile_top, 'k-', lw=0.5)
        ax.plot(profile_x, profile_bottom, 'k-', lw=0.5)
        ax.fill_between(profile_x, profile_bottom, profile_top, color='grey', alpha=0.3)
        # Mark the simulation origin.
        ax.plot(scaled_k4, scaled_vm, 'ko', markersize=2)

        # Update global bounds.
        global_x_min = min(global_x_min, np.min(profile_x))
        global_x_max = max(global_x_max, np.max(profile_x))
        global_y_min = min(global_y_min, np.min(profile_bottom))
        global_y_max = max(global_y_max, np.max(profile_top))
    
    # Set plot limits with margins.
    margin_x = 0.05 * (global_x_max - global_x_min)
    margin_y = 0.05 * (global_y_max - global_y_min)
    ax.set_xlim(0, global_x_max + margin_x)
    ax.set_ylim(0, global_y_max + margin_y)
    ax.set_aspect('equal', adjustable='datalim')

    # Remap primary axis tick labels to show original k₄ and vₘ.
    ax.set_xlabel("$k_4$", fontsize=15)
    ax.set_ylabel("$v_m$", fontsize=15)
    ax.set_title(f"2D Geometry Profiles for $D_f$ = {selected_D2}", fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/primary_x_scale:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y/primary_y_scale:g}"))

    # (Optional) Set specific tick locations.
    k4_array = (primary_x_scale*np.linspace(0.01, 0.1, 5)).tolist()
    vm_array = (primary_y_scale*np.linspace(0.01, 0.5, 5)).tolist()
    ax.set_xticks(k4_array)
    ax.set_yticks(vm_array)

    # Draw a horizontal scale bar.
    bar_end_x = global_x_max
    bar_y = global_y_max + 5*margin_y/6
    bar_length_plot = scale_bar_length * scale_factor
    bar_start_x = bar_end_x - bar_length_plot
    ax.plot([bar_start_x, bar_end_x], [bar_y, bar_y], 'k-', lw=3)
    
    # Add secondary axes for physical branch coordinates.
    if canonical_origin is not None:
        canonical_k4, canonical_vm = canonical_origin
        canonical_scaled_k4 = primary_x_scale * canonical_k4
        canonical_scaled_vm = primary_y_scale * canonical_vm

        def main_to_phys_x(x):
            return (x - canonical_scaled_k4) / scale_factor
        def phys_to_main_x(x_phys):
            return x_phys * scale_factor + canonical_scaled_k4
        def main_to_phys_y(y):
            return (y - canonical_scaled_vm) / scale_factor
        def phys_to_main_y(y_phys):
            return y_phys * scale_factor + canonical_scaled_vm

        secax_x = ax.secondary_xaxis('top', functions=(main_to_phys_x, phys_to_main_x))
        secax_y = ax.secondary_yaxis('right', functions=(main_to_phys_y, phys_to_main_y))
    
    # Adjust layout and save plot if needed.
    plt.tight_layout()

    if save_plot and save_path is not None:
        import os
        filename = f"2d_geometry_profiles_D2_{selected_D2}.png"
        full_save_path = os.path.join(save_path, filename)
        plt.savefig(full_save_path, dpi=300)
        print(f"Plot saved to {full_save_path}")
    
    plt.show()


def generate_2d_geometry_profiles_in_parameter_space_fixed_k4(
    root_output_folder,
    fixed_k4_index=0,
    scale_factor=1.0,
    scale_bar_length=10.0,  # physical length in microns for the scale bar
    primary_x_scale=1.0,
    primary_y_scale=1.0,
    save_plot=False,
    save_path=None
):
    """
    Traverse 'root_output_folder' for folders named like "D2_<val>_k4_<val>_vm_<val>".
    For a fixed k₄ value (selected by fixed_k4_index from the sorted unique k₄ values),
    this function loads each simulation state and extracts its branch profile.
    
    In this mode, the free parameters are D₂ and vₘ. Each branch profile is drawn in the 
    2D parameter space with:
      - x-axis: D₂ (scaled by primary_x_scale), and
      - y-axis: vₘ (scaled by primary_y_scale).
      
    The branch profile is drawn as:
       profile_x = primary_x_scale * D₂ + x_offset,
       profile_y = primary_y_scale * vₘ + offset,
    where the offsets are computed from the simulation's domain (in microns).
    
    Primary-axis tick labels are reformatted to show the unscaled D₂ and vₘ values. Secondary 
    axes (top/right) are added so that the physical branch coordinates are correctly displayed,
    using the drawn scale bar (of length scale_bar_length×scale_factor) as a reference.
    
    Parameters:
      root_output_folder : str
          Path to the folder containing the simulation subfolders.
      fixed_k4_index : int (default 0)
          Index into the sorted unique k₄ values to fix.
      scale_factor : float (default 1.0)
          Factor to scale the simulation's branch profile (physical units).
      scale_bar_length : float (default 10.0)
          Physical length (microns) of the scale bar.
      primary_x_scale : float (default 1.0)
          Scaling factor applied to the D₂ values for plotting.
      primary_y_scale : float (default 1.0)
          Scaling factor applied to the vₘ values for plotting.
      save_plot : bool (default False)
          If True, saves the final plot.
      save_path : str (default None)
          Folder path to save the plot.
    """
    import os, re
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # load_final_simulator_state should be defined elsewhere.
    folder_pattern = re.compile(r"D2_([-+]?\d*\.?\d+)_k4_([-+]?\d*\.?\d+)_vm_([-+]?\d*\.?\d+)")
    simulation_entries = []
    for combo_name in os.listdir(root_output_folder):
        combo_path = os.path.join(root_output_folder, combo_name)
        if not os.path.isdir(combo_path):
            continue
        match = folder_pattern.match(combo_name)
        if not match:
            continue
        D2_val = float(match.group(1))
        k4_val = float(match.group(2))
        vm_val = float(match.group(3))
        simulation_entries.append((D2_val, k4_val, vm_val, combo_path))
    
    if not simulation_entries:
        print("No simulation folders found matching the pattern.")
        return

    # Get unique k₄ values and select one.
    k4_values = sorted(list({entry[1] for entry in simulation_entries}))
    if len(k4_values) < 1:
        print("No k₄ values found.")
        return
    if fixed_k4_index < 0 or fixed_k4_index >= len(k4_values):
        print(f"fixed_k4_index ({fixed_k4_index}) is out of range. Using index 0 instead.")
        fixed_k4_index = 0
    selected_k4 = k4_values[fixed_k4_index]
    selected_k4 = np.round(selected_k4, 4)
    print(f"Selected k₄ value for plotting: {selected_k4}")

    # Create the figure and axis.
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # For computing overall bounding box in primary coordinates.
    global_x_min, global_x_max = np.inf, -np.inf
    global_y_min, global_y_max = np.inf, -np.inf

    canonical_origin = None  # will store (D2, v_m) of the canonical simulation.

    # Loop over simulations with fixed k₄.
    for (D2_val, k4_val, vm_val, combo_path) in simulation_entries:
        if not np.isclose(k4_val, selected_k4, atol=1e-6):
            continue

        sim = load_final_simulator_state(combo_path)
        if sim is None:
            print(f"Could not load simulator state for folder: {combo_path}")
            continue

        if canonical_origin is None:
            canonical_origin = (D2_val, vm_val)

        # Here the simulation origin (free parameters) are:
        # x: D₂, y: vₘ, scaled by primary_x_scale and primary_y_scale.
        scaled_D2 = primary_x_scale * D2_val
        scaled_vm = primary_y_scale * vm_val

        # Compute branch profile offsets (in microns).
        x_vals = np.linspace(sim.delta_x/2, sim.l - sim.delta_x/2, sim.N)
        w_vals = sim.w[:sim.N]
        x_offset = (x_vals - x_vals[0]) * scale_factor
        top_offset = (w_vals/2) * scale_factor
        bottom_offset = - (w_vals/2) * scale_factor

        # Anchor the branch profile at the scaled simulation origin.
        profile_x = scaled_D2 + x_offset
        profile_top = scaled_vm + top_offset
        profile_bottom = scaled_vm + bottom_offset

        # Plot the branch profile.
        ax.plot(profile_x, profile_top, 'k-', lw=0.5)
        ax.plot(profile_x, profile_bottom, 'k-', lw=0.5)
        ax.fill_between(profile_x, profile_bottom, profile_top, color='grey', alpha=0.3)
        ax.plot(scaled_D2, scaled_vm, 'ko', markersize=2)

        # Update global bounds.
        global_x_min = min(global_x_min, np.min(profile_x))
        global_x_max = max(global_x_max, np.max(profile_x))
        global_y_min = min(global_y_min, np.min(profile_bottom))
        global_y_max = max(global_y_max, np.max(profile_top))
    
    margin_x = 0.05 * (global_x_max - global_x_min)
    margin_y = 0.05 * (global_y_max - global_y_min)
    ax.set_xlim(0, global_x_max + margin_x)
    ax.set_ylim(0, global_y_max + margin_y)
    ax.set_aspect('equal', adjustable='datalim')

    # Remap tick labels so that the original D₂ and vₘ values appear.
    ax.set_xlabel("$D_f$", fontsize=15)
    ax.set_ylabel("$v_m$", fontsize=15)
    ax.set_title(f"2D Geometry Profiles for $k_d$ = {selected_k4}", fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/primary_x_scale:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y/primary_y_scale:g}"))

    # Set specific tick locations.
    D2_array = (primary_x_scale*np.linspace(0.01, 0.1, 5)).tolist()
    vm_array = (primary_y_scale*np.linspace(0.01, 0.5, 5)).tolist()
    ax.set_xticks(D2_array)
    ax.set_yticks(vm_array)

    # Draw horizontal scale bar.
    bar_end_x = global_x_max
    bar_y = global_y_max + 5*margin_y/6
    bar_length_plot = scale_bar_length * scale_factor
    bar_start_x = bar_end_x - bar_length_plot
    ax.plot([bar_start_x, bar_end_x], [bar_y, bar_y], 'k-', lw=3)
    
    # Add secondary axes for physical branch coordinates.
    if canonical_origin is not None:
        canonical_D2, canonical_vm = canonical_origin
        canonical_scaled_D2 = primary_x_scale * canonical_D2
        canonical_scaled_vm = primary_y_scale * canonical_vm

        def main_to_phys_x(x):
            return (x - canonical_scaled_D2) / scale_factor
        def phys_to_main_x(x_phys):
            return x_phys * scale_factor + canonical_scaled_D2
        def main_to_phys_y(y):
            return (y - canonical_scaled_vm) / scale_factor
        def phys_to_main_y(y_phys):
            return y_phys * scale_factor + canonical_scaled_vm

        secax_x = ax.secondary_xaxis('top', functions=(main_to_phys_x, phys_to_main_x))
        secax_y = ax.secondary_yaxis('right', functions=(main_to_phys_y, phys_to_main_y))
    
    plt.tight_layout()

    if save_plot and save_path is not None:
        import os
        filename = f"2d_geometry_profiles_k4_{selected_k4}.png"
        full_save_path = os.path.join(save_path, filename)
        plt.savefig(full_save_path, dpi=300)
        print(f"Plot saved to {full_save_path}")
    
    plt.show()


def plot_profiles_in_Da_invPe_space(
    root_output_folder,
    scale_factor=1.0,
    scale_bar_length=10.0,
    save_plot=False,
    save_path=None,
    primary_x_scale=1.0,
    primary_y_scale=1.0,
):
    """
    Plots branch geometry profiles for a full 5×5×5 parameter sweep in
    Da₄–1/Pe₂ space, where:
        Da₄ = k₄ / vₘ
        1/Pe₂ = D₂ / vₘ

    Parameters:
        root_output_folder : str
            Folder containing simulation results organized as D2_<val>_k4_<val>_vm_<val>
        scale_factor : float
            Converts simulation units into microns.
        scale_bar_length : float
            Physical length (in microns) of the scale bar.
        save_plot : bool
            Whether to save the output plot.
        save_path : str
            Where to save the figure (if save_plot is True).
        primary_x_scale : float
            Scale for 1/Pe₂ axis (optional).
        primary_y_scale : float
            Scale for Da₄ axis (optional).
    """
    import os, re
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    folder_pattern = re.compile(r"D2_([-+]?\d*\.?\d+)_k4_([-+]?\d*\.?\d+)_vm_([-+]?\d*\.?\d+)")

    simulation_entries = []
    for folder in os.listdir(root_output_folder):
        folder_path = os.path.join(root_output_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        match = folder_pattern.match(folder)
        if match:
            D2 = float(match.group(1))
            k4 = float(match.group(2))
            vm = float(match.group(3))
            simulation_entries.append((D2, k4, vm, folder_path))

    if not simulation_entries:
        print("No simulation entries found.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    global_x_min, global_x_max = np.inf, -np.inf
    global_y_min, global_y_max = np.inf, -np.inf

    canonical_origin = None

    for D2, k4, vm, folder_path in simulation_entries:
        sim = load_final_simulator_state(folder_path)
        if sim is None:
            print(f"Skipping: {folder_path}")
            continue

        Da4 = k4 / vm
        inv_Pe2 = D2 / vm
        scaled_Da4 = primary_y_scale * Da4
        scaled_invPe2 = primary_x_scale * inv_Pe2

        if canonical_origin is None:
            canonical_origin = (Da4, inv_Pe2)

        x_vals = np.linspace(sim.delta_x / 2, sim.l - sim.delta_x / 2, sim.N)
        w_vals = sim.w[:sim.N]
        x_offset = (x_vals - x_vals[0]) * scale_factor
        top_offset = (w_vals / 2) * scale_factor
        bottom_offset = - (w_vals / 2) * scale_factor

        profile_x = scaled_invPe2 + x_offset
        profile_top = scaled_Da4 + top_offset
        profile_bottom = scaled_Da4 + bottom_offset

        ax.plot(profile_x, profile_top, 'k-', lw=0.5)
        ax.plot(profile_x, profile_bottom, 'k-', lw=0.5)
        ax.fill_between(profile_x, profile_bottom, profile_top, color='grey', alpha=0.3)
        ax.plot(scaled_invPe2, scaled_Da4, 'ko', markersize=2)

        global_x_min = min(global_x_min, np.min(profile_x))
        global_x_max = max(global_x_max, np.max(profile_x))
        global_y_min = min(global_y_min, np.min(profile_bottom))
        global_y_max = max(global_y_max, np.max(profile_top))

    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel(r"$1/Pe_f = \frac{D_f}{v_m}$", fontsize=15)
    ax.set_ylabel(r"$Da_d = \frac{k_d}{v_m}$", fontsize=15)
    ax.set_title("Geometry Profiles in $Da_d$–$1/Pe_f$ Space", fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/primary_x_scale:.2g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y/primary_y_scale:.2g}"))

    margin_x = 0.05 * (global_x_max - global_x_min)
    margin_y = 0.05 * (global_y_max - global_y_min)
    ax.set_xlim(0, global_x_max + margin_x)
    ax.set_ylim(0, global_y_max + margin_y)

    # Scale bar
    bar_end_x = global_x_max
    bar_y = global_y_max + 5 * margin_y / 6
    bar_length_plot = scale_bar_length * scale_factor
    bar_start_x = bar_end_x - bar_length_plot
    ax.plot([bar_start_x, bar_end_x], [bar_y, bar_y], 'k-', lw=3)

    # Secondary physical axes
    if canonical_origin is not None:
        Da4_canon, invPe2_canon = canonical_origin
        canon_x = primary_x_scale * invPe2_canon
        canon_y = primary_y_scale * Da4_canon

        def main_to_phys_x(x):
            return (x - canon_x) / scale_factor
        def phys_to_main_x(x_phys):
            return x_phys * scale_factor + canon_x
        def main_to_phys_y(y):
            return (y - canon_y) / scale_factor
        def phys_to_main_y(y_phys):
            return y_phys * scale_factor + canon_y

        secax_x = ax.secondary_xaxis('top', functions=(main_to_phys_x, phys_to_main_x))
        secax_y = ax.secondary_yaxis('right', functions=(main_to_phys_y, phys_to_main_y))

    plt.tight_layout()
    if save_plot and save_path:
        filename = "geometry_profiles_Da4_invPe2.png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300)
        print(f"Saved plot to {full_path}")

    plt.show()


def plot_profiles_in_Da_invPe_space_from_ND(
    root_output_folder,
    scale_factor=1.0,
    scale_bar_length=10.0,
    save_plot=False,
    save_path=None,
    primary_x_scale=1.0,
    primary_y_scale=1.0,
):
    """
    Plots branch geometry profiles for parameter sweeps in Da_d–1/Pe space, where:
        Da_d = k₄·w₀ / vₘ  (deposition Damköhler number)
        1/Pe = D₂ / (vₘ·w₀)  (inverse Péclet number)
    
    Works with both cartesian and pairwise sweep outputs from run_phase_space_study_nd().
    
    Parameters:
        root_output_folder : str
            Folder containing simulation results organized as:
            - cartesian mode: D2_<val>_k4_<val>_vm_<val>
            - pairwise mode: idx_XXX_D2_<val>_k4_<val>_vm_<val>[_label_...]
        scale_factor : float
            Converts simulation units into microns for profile display.
        scale_bar_length : float
            Physical length (in microns) of the scale bar.
        save_plot : bool
            Whether to save the output plot.
        save_path : str
            Where to save the figure (if save_plot is True).
        primary_x_scale : float
            Scale for 1/Pe axis (optional, for spacing adjustment).
        primary_y_scale : float
            Scale for Da_d axis (optional, for spacing adjustment).
    """
    import os, re
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # Updated pattern to handle both cartesian and pairwise naming conventions
    folder_pattern = re.compile(
        r"^(?:idx_\d+_)?D2_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)_k4_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)_vm_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    simulation_entries = []
    for folder in os.listdir(root_output_folder):
        folder_path = os.path.join(root_output_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        match = folder_pattern.match(folder)
        if match:
            D2 = float(match.group(1))
            k4 = float(match.group(2))
            vm = float(match.group(3))
            simulation_entries.append((D2, k4, vm, folder_path))

    if not simulation_entries:
        print("No simulation entries found.")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    global_x_min, global_x_max = np.inf, -np.inf
    global_y_min, global_y_max = np.inf, -np.inf

    canonical_origin = None

    for D2, k4, vm, folder_path in simulation_entries:
        sim = load_final_simulator_state(folder_path)
        if sim is None:
            print(f"Skipping: {folder_path}")
            continue

        # Extract w0 from the loaded simulator
        w0 = sim.w0
        
        # Compute dimensionless groups
        Da_d = k4 * w0 / vm      # deposition Damköhler
        inv_Pe = D2 / (vm * w0)  # inverse Péclet
        
        scaled_Da_d = primary_y_scale * Da_d
        scaled_invPe = primary_x_scale * inv_Pe

        if canonical_origin is None:
            canonical_origin = (Da_d, inv_Pe)

        # Extract geometry from the ND simulator
        # The simulator stores barred variables; reconstruct dimensional length and width
        if hasattr(sim, 'l_bar'):
            # ND version: reconstruct dimensional coordinates
            l_dim = sim.l_bar * sim.l0
            x_vals = np.linspace(sim.delta_x / 2, l_dim - sim.delta_x / 2, sim.N)
            
            # Reconstruct dimensional cd for width calculation
            cd_dim_active = sim.cd_bar[:sim.N] * sim.c0
            local_mass = cd_dim_active * sim.delta_x
            local_volume = local_mass / sim.rho
            w_vals = np.sqrt(4.0 * local_volume / sim.delta_x / np.pi)
        else:
            # Dimensional version (fallback)
            x_vals = np.linspace(sim.delta_x / 2, sim.l - sim.delta_x / 2, sim.N)
            w_vals = sim.w[:sim.N]

        # Compute profile offsets
        x_offset = (x_vals - x_vals[0]) * scale_factor
        top_offset = (w_vals / 2) * scale_factor
        bottom_offset = -(w_vals / 2) * scale_factor

        profile_x = scaled_invPe + x_offset
        profile_top = scaled_Da_d + top_offset
        profile_bottom = scaled_Da_d + bottom_offset

        # Plot the branch profile
        ax.plot(profile_x, profile_top, 'k-', lw=0.5)
        ax.plot(profile_x, profile_bottom, 'k-', lw=0.5)
        ax.fill_between(profile_x, profile_bottom, profile_top, color='grey', alpha=0.3)
        ax.plot(scaled_invPe, scaled_Da_d, 'ko', markersize=2)

        global_x_min = min(global_x_min, np.min(profile_x))
        global_x_max = max(global_x_max, np.max(profile_x))
        global_y_min = min(global_y_min, np.min(profile_bottom))
        global_y_max = max(global_y_max, np.max(profile_top))

    # Set axes properties
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel(r"$Pe_{f}^{-1} = \frac{D_f}{v_m w_0}$", fontsize=15)
    ax.set_ylabel(r"$Da_d = \frac{k_d w_0}{v_m}$", fontsize=15)
    ax.set_title("Geometry Profiles in $Da_d$–$Pe^{-1}$ Space", fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/primary_x_scale:.2g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y/primary_y_scale:.2g}"))

    margin_x = 0.05 * (global_x_max - global_x_min)
    margin_y = 0.05 * (global_y_max - global_y_min)
    ax.set_xlim(0, global_x_max + margin_x)
    ax.set_ylim(0, global_y_max + margin_y)

    # Scale bar
    bar_end_x = global_x_max
    bar_y = global_y_max + 5 * margin_y / 6
    bar_length_plot = scale_bar_length * scale_factor
    bar_start_x = bar_end_x - bar_length_plot
    ax.plot([bar_start_x, bar_end_x], [bar_y, bar_y], 'k-', lw=3)
    ax.text((bar_start_x + bar_end_x) / 2, bar_y + margin_y/3, f"{scale_bar_length} μm",
            ha='center', va='bottom', fontsize=12)

    # Secondary physical axes
    if canonical_origin is not None:
        Da_d_canon, invPe_canon = canonical_origin
        canon_x = primary_x_scale * invPe_canon
        canon_y = primary_y_scale * Da_d_canon

        def main_to_phys_x(x):
            return (x - canon_x) / scale_factor
        def phys_to_main_x(x_phys):
            return x_phys * scale_factor + canon_x
        def main_to_phys_y(y):
            return (y - canon_y) / scale_factor
        def phys_to_main_y(y_phys):
            return y_phys * scale_factor + canon_y

        secax_x = ax.secondary_xaxis('top', functions=(main_to_phys_x, phys_to_main_x))
        secax_x.set_xlabel("Branch axial coordinate (μm)", fontsize=12)
        secax_y = ax.secondary_yaxis('right', functions=(main_to_phys_y, phys_to_main_y))
        secax_y.set_ylabel("Branch half-width (μm)", fontsize=12)

    plt.tight_layout()
    if save_plot and save_path:
        filename = "geometry_profiles_Da_d_invPe.png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300)
        print(f"Saved plot to {full_path}")

    plt.show()


def plot_profiles_in_Da_invPe_space_from_ND_loglog(
    root_output_folder,
    scale_factor=1.0,
    scale_bar_length=10.0,
    save_plot=False,
    save_path=None,
    pixels_per_unit=100.0,  # Controls how branch profiles map to figure pixels
):
    """
    Plots branch geometry profiles in Da_d–Pe^{-1} space with:
    - Log-log axes for the parameter space (showing dimensionless groups)
    - Linear-scale branch profiles overlaid at their corresponding parameter locations
    
    The branch profiles are drawn in a separate axes overlay that uses pixel coordinates,
    so the profiles maintain their true physical aspect ratios while the parameter
    positions are displayed on log-log axes.
    
    Parameters:
        root_output_folder : str
            Folder containing simulation results (cartesian or pairwise mode)
        scale_factor : float
            Converts simulation dimensional units to physical microns for profiles
        scale_bar_length : float
            Physical length (in microns) of the scale bar
        save_plot : bool
            Whether to save the output plot
        save_path : str
            Where to save the figure
        pixels_per_unit : float
            Scaling for branch profiles in the pixel overlay (controls profile sizes)
    """
    import os, re
    import numpy as np
    import matplotlib.pyplot as plt

    # Pattern handles both cartesian and pairwise naming
    folder_pattern = re.compile(
        r"^(?:idx_\d+_)?D2_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)_k4_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)_vm_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    simulation_entries = []
    for folder in sorted(os.listdir(root_output_folder), key=str.upper):  # Windows folder order
        folder_path = os.path.join(root_output_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        match = folder_pattern.match(folder)
        if match:
            D2 = float(match.group(1))
            k4 = float(match.group(2))
            vm = float(match.group(3))
            simulation_entries.append((D2, k4, vm, folder_path))

    if not simulation_entries:
        print("No simulation entries found.")
        return

    #remove repeated sets of simulations for clarity. look for identical (D2, k4, vm) tuples within some tolerance
    unique_entries = []
    tolerance = 1e-6
    for entry in simulation_entries:
        D2, k4, vm, folder_path = entry
        is_duplicate = False
        for u_entry in unique_entries:
            u_D2, u_k4, u_vm, _ = u_entry
            if (abs(D2 - u_D2) < tolerance and
                abs(k4 - u_k4) < tolerance and
                abs(vm - u_vm) < tolerance):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_entries.append(entry)

    simulation_entries = unique_entries

    #make groups of simulations with same Da_d values
    da_groups = {}
    for entry in simulation_entries:
        D2, k4, vm, folder_path = entry
        w0 = 1
        Da_d = k4 * w0 / vm  # Compute Da_d for this simulation
        if Da_d not in da_groups:
            da_groups[Da_d] = []
        da_groups[Da_d].append(entry)

    # Now filter each Da_d group to reduce clutter for low Da_d values
    filtered_entries = []
    for Da_d, group in da_groups.items():
        if Da_d < 0.1 and Da_d > 0.01:
            # For low Da_d, keep every other simulation in the group
            filtered_entries.extend(group[::2])
        elif Da_d <= 0.01:
            # For medium Da_d, keep every simulation
            filtered_entries.extend(group[::4])
        else:
            # For high Da_d, keep all simulations
            filtered_entries.extend(group)

    simulation_entries = filtered_entries




    # #choose only alternate simulations for clarity in rows
    # simulation_entries1 = simulation_entries[::2]
    # simulation_entries2 = simulation_entries[1::2]

    # #now from this choose every alternate consecutive 5 entries to get alternative columns
    # simulation_entries1 = [simulation_entries1[i] for i in range(len(simulation_entries1)) if (i//5)%2==0]
    # simulation_entries2 = [simulation_entries2[i] for i in range(len(simulation_entries2)) if (i//5)%2==1]
    # simulation_entries = simulation_entries1 + simulation_entries2




    # #choose only alternate simulations for clarity in rows
    # simulation_entries1 = simulation_entries[::2]
    # # simulation_entries2 = simulation_entries[1::2]

    # #now from this choose every alternate consecutive 5 entries to get alternative columns
    # simulation_entries1 = [simulation_entries1[i] for i in range(len(simulation_entries1)) if (i//5)%2==0]
    # # simulation_entries2 = [simulation_entries2[i] for i in range(len(simulation_entries2)) if (i//5)%2==1]
    # # simulation_entries = simulation_entries1 + simulation_entries2
    # simulation_entries = simulation_entries1

    # #group by Da_d values and thin out low Da_d groups
    # da_groups = {}
    # for entry in simulation_entries:
    #     D2, k4, vm, folder_path = entry
    #     w0 = 1
    #     Da_d = k4 * w0 / vm  # Compute Da_d for this simulation
    #     if Da_d not in da_groups:
    #         da_groups[Da_d] = []
    #     da_groups[Da_d].append(entry)

    # # Thin out low Da_d groups
    # filtered_entries = []
    # for Da_d, group in da_groups.items():
    #     if Da_d <= 0.01:
    #         # For low Da_d, keep every 2nd simulation
    #         filtered_entries.extend(group[::2])
    #     else:
    #         # For high Da_d, keep all simulations
    #         filtered_entries.extend(group)

    # simulation_entries = filtered_entries

    # simulation_entries = simulation_entries[::2]
    # simulation_entries = [simulation_entries[i] for i in range(len(simulation_entries)) if (i//5)%2==0]
    # simulation_entries = simulation_entries[::2]
    # simulation_entries = [simulation_entries[i] for i in range(len(simulation_entries)) if (i//3)%2==0]




    # Create figure with log-log axes
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.12, 0.12, 0.75, 0.75])  # Main log-log axes
    ax.set_xscale('log')
    ax.set_yscale('log')


    # Collect parameter values and geometries FIRST
    invPe_vals = []
    Da_d_vals = []
    geometries = []  # Store (inv_Pe, Da_d, x_vals, w_vals) for each simulation

    for D2, k4, vm, folder_path in simulation_entries:
        sim = load_final_simulator_state(folder_path)
        if sim is None:
            print(f"Skipping: {folder_path}")
            continue

        # Extract w0 from simulator
        w0 = sim.w0
        
        # Compute dimensionless groups
        Da_d = k4 * w0 / vm      # deposition Damköhler
        inv_Pe = D2 / (vm * w0)  # inverse Péclet
        
        invPe_vals.append(inv_Pe)
        Da_d_vals.append(Da_d)

        # Extract geometry (handle both ND and dimensional simulators)
        if hasattr(sim, 'l_bar'):
            # ND version: reconstruct dimensional coordinates
            l_dim = sim.l_bar * sim.l0
            x_vals = np.linspace(sim.delta_x / 2, l_dim - sim.delta_x / 2, sim.N)
            
            # Reconstruct dimensional cd for width calculation
            cd_dim_active = sim.cd_bar[:sim.N] * sim.c0
            local_mass = cd_dim_active * sim.delta_x
            local_volume = local_mass / sim.rho
            w_vals = np.sqrt(4.0 * local_volume / sim.delta_x / np.pi)
        else:
            # Dimensional version (fallback)
            x_vals = np.linspace(sim.delta_x / 2, sim.l - sim.delta_x / 2, sim.N)
            w_vals = sim.w[:sim.N]

        geometries.append((inv_Pe, Da_d, x_vals, w_vals))

    if not invPe_vals or not Da_d_vals:
        print("No valid simulations found.")
        return

    # Set log-log axis limits with padding BEFORE drawing
    xmin, xmax = min(invPe_vals), max(invPe_vals)
    ymin, ymax = min(Da_d_vals), max(Da_d_vals)
    # Pad by ~20% in log space
    x_range = np.log10(xmax) - np.log10(xmin)
    y_range = np.log10(ymax) - np.log10(ymin)
    ax.set_xlim(10**(np.log10(xmin) - 0.02*x_range), 
                10**(np.log10(xmax) + 0.2*x_range))
    ax.set_ylim(10**(np.log10(ymin) - 0.1*y_range), 
                10**(np.log10(ymax) + 0.1*y_range))

    # Plot markers at parameter locations
    ax.plot(invPe_vals, Da_d_vals, 'o', ms=5, color='black', alpha=0.7, zorder=10, markeredgewidth = 0)

    # Axis labels and styling
    ax.set_xlabel(r"$Pe_{f}^{-1} = \frac{D_f}{v_m w_0}$", fontsize=16)
    ax.set_ylabel(r"$Da_d = \frac{k_d w_0}{v_m}$", fontsize=16)
    ax.set_title("Geometry Profiles in $Da_d$–$Pe_{f}^{-1}$ Space (log-log)", fontsize=18)
    ax.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.tick_params(axis='both', labelsize=13)

    # NOW create overlay axes and force draw to get correct pixel coordinates
    ax_bbox = ax.get_position()
    ax_overlay = fig.add_axes([ax_bbox.x0, ax_bbox.y0, ax_bbox.width, ax_bbox.height], 
                               frameon=False)
    ax_overlay.set_axis_off()

    # Force a draw to get accurate pixel coordinates after setting limits
    fig.canvas.draw()
    
    # Get axes dimensions in pixels
    ax_window = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_width_px = ax_window.width * fig.dpi
    ax_height_px = ax_window.height * fig.dpi
    ax_overlay.set_xlim(0, ax_width_px)
    ax_overlay.set_ylim(0, ax_height_px)

    # Helper function: log-log data coordinates -> pixel coordinates in axes
    def data_to_pixels(xdata, ydata):
        """Convert log-log axis data coordinates to pixel offsets within axes"""
        # Transform through log-log data space to display pixels
        xd, yd = ax.transData.transform((xdata, ydata))
        # Get axes window in display pixels
        ax_disp = ax.get_window_extent()
        # Return pixel coordinates relative to axes origin
        return xd - ax_disp.x0, yd - ax_disp.y0

    # Now draw all the branch profiles
    for inv_Pe, Da_d, x_vals, w_vals in geometries:
        # Get anchor point in pixel coordinates (this should now be accurate)
        anchor_x_px, anchor_y_px = data_to_pixels(inv_Pe, Da_d)

        # Compute branch profile in physical units (microns)
        x_offset_um = (x_vals - x_vals[0])  # axial coordinate
        y_top_um = 0.5 * w_vals             # top boundary
        y_bot_um = -0.5 * w_vals            # bottom boundary

        # Convert to pixels (linear scale)
        dx_px = x_offset_um * scale_factor * pixels_per_unit
        dy_top_px = y_top_um * scale_factor * pixels_per_unit
        dy_bot_px = y_bot_um * scale_factor * pixels_per_unit

        # Draw branch profile on overlay axes (in pixel coordinates)
        ax_overlay.plot(anchor_x_px + dx_px, anchor_y_px + dy_top_px, 'k-', lw=0.6, zorder=5)
        ax_overlay.plot(anchor_x_px + dx_px, anchor_y_px + dy_bot_px, 'k-', lw=0.6, zorder=5)
        ax_overlay.fill_between(anchor_x_px + dx_px,
                                anchor_y_px + dy_bot_px,
                                anchor_y_px + dy_top_px,
                                color='0.5', alpha=0.35, linewidth=0, zorder=5)

    # Add scale bar in pixel overlay (top-right corner)
    bar_length_px = scale_bar_length * scale_factor * pixels_per_unit
    pad_px = 20
    x0_bar = ax_width_px - 1.5*pad_px - bar_length_px
    y0_bar = ax_height_px - pad_px
    ax_overlay.plot([x0_bar, x0_bar + bar_length_px], [y0_bar, y0_bar], 'k-', lw=3, zorder=15)
    ax_overlay.text(x0_bar + bar_length_px/2, y0_bar - 10, f"{scale_bar_length} μm",
                    ha='center', va='top', fontsize=12, zorder=15)

    plt.tight_layout()

    if save_plot and save_path:
        filename = "geometry_profiles_Da_d_invPe_loglog_small.eps"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=600, bbox_inches='tight')
        print(f"Saved plot to {full_path}")

    plt.show()


#     )


if __name__ == "__main__":
    my_root_folder = str(OUT_ROOT / "05_branch_transport_model_1p5d" / "sweep_study_ND" / "phase_space_study_ND_pairwise_2025_10_29_12_40_46")  # published sweep; set to your run
    
    plot_folder = os.path.join(my_root_folder, 'plots')
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    plot_profiles_in_Da_invPe_space_from_ND_loglog(
        root_output_folder=my_root_folder,
        scale_factor=0.05,      # Adjust to control profile sizes
        scale_bar_length=10.0,
        pixels_per_unit=50.0,    # Adjust to scale profiles up/down
        save_plot=True,
        save_path=plot_folder
    )
