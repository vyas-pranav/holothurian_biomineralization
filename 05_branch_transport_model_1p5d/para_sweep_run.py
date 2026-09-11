# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2025 Pranav Vyas
"""1.5D advection-reaction-diffusion model: dimensional parameter sweep (Feb 2025).

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 6A (simulated tip; rendered by 3d_time_growth.py);
                Supplementary Video SV8 (advection-reaction-diffusion clips)
What it does  : The 1.5D ARD model solved in dimensional variables (explicit
                finite differences) for every combination of
                D_f in linspace(0.01, 0.1, 5) um^2/s, k_d in linspace(0.01, 0.1, 5) 1/s
                and v_m in linspace(0.01, 0.5, 5) um/s, with eps = 10 v_m,
                mdot_0 = 0.01 pg/s, dt = 0.01 s and dx = 0.05 um. Each run lasts
                500 chunks x 10,000 steps = 50,000 s and saves a PNG frame
                (concentration and branch shape) plus a pickled snapshot every
                100 s. The SV8 clips (D2 = 0.055 with k4/vm pairs) were assembled
                from these frames with
                06_branching_lattice_model_2d/plotting/video_from_frames.py, and
                Fig. 6A shows the run D2 = 0.055, k4 = 0.055, vm = 0.255.
Inputs        : none
Outputs       : OUT_ROOT/05_branch_transport_model_1p5d/sweep_study_dimensional/
                phase_space_study_<timestamp>/D2_*_k4_*_vm_*/
                  exp_parameters.txt, plot_*.png, <t>.pkl
Environment   : environment-models.yml (Python 3.13)
Run           : python para_sweep_run.py   (long: 125 runs of 5e6 steps)

Symbols (code -> SI): D2 -> D_f, k4 -> k_d, k5 -> k_t, epsilon -> eps,
vm -> v_m, m_dot_0 -> mdot_0, c2/c4/cg -> c_f/c_d/c_g (dimensional classes),
cf_bar/cd_bar/cg_bar -> non-dimensional c_f/c_d/c_g, Pe2_inv/Pe_inv -> Pe_f^-1,
Da4/Da_d -> Da_d, Da5/Da_t -> Da_t, Da_epsilon/Da_eps -> Da_eps (SI Eqs. 3-8).
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import shutil
import cv2
import pickle
from itertools import product
import traceback

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

        # Draw the figure in its current state (optional for some backends)
        # fig.canvas.draw()


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


if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # EXAMPLE USE:
    # Define your parameter lists for D2, k4, vm. 
    # Everything else can remain single values or be turned into lists similarly.
    # ----------------------------------------------------------------------
    # D2_list_example = [0.01, 0.05, 0.1]    
    # k4_list_example = [0.01, 0.05, 0.1]  
    # vm_list_example = [0.01, 0.1, 1.0]    

    D2_array = np.linspace(0.01, 0.1, 5)
    k4_array = np.linspace(0.01, 0.1, 5)    
    vm_array = np.linspace(0.01, 0.5, 5)

    D2_list_example = D2_array.tolist()
    k4_list_example = k4_array.tolist()
    vm_list_example = vm_array.tolist()

    # This would result in 2 x 2 x 2 = 8 total combinations.

    # Run the study
    run_phase_space_study(
        D2_list=D2_list_example,
        k4_list=k4_list_example,
        vm_list=vm_list_example,
        # Possibly override other defaults as desired:
        k5=100.0,
        csat=0.0,
        m_dot_0=0.01,
        w0=1.0,
        rho=2.71,
        delta_t=0.01,
        delta_x=0.05,
        N=5,
        l_g=0.5,
        epsilon_factor=10.0,  # means epsilon = 10.0 * vm each time
        save_plot=True,       # set True if you want PNG files
        save_pkl=True,        # set True if you want pickles
        base_output_folder=str(OUT_ROOT / "05_branch_transport_model_1p5d" / "sweep_study_dimensional"),
        num_simulation_steps=500,  # how many times we call run_simulation_dim(...)
        steps_per_run=10000,       # how many steps each call uses
        show_plots=False         # if True, will open windows with live plots
    )
