# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2025 Pranav Vyas
"""1.5D advection-reaction-diffusion branch-growth model: non-dimensional sweep.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 6B; Fig. S19 (simulation data)
What it does  : Solves the 1.5D ARD model (SI Eqs. 3-8, non-dimensional form)
                with an explicit finite-difference scheme (upwind advection,
                central diffusion, forward Euler time stepping) for a 10 x 10
                grid of target (Pe_f^-1, Da_d) values, each log-spaced over
                [0.01, 10]. choose_safe_parameters_for_phase_plane() picks v_m
                inside CFL-type stability limits (v_m <= 0.25 um/s here) and
                sets D_f = Pe_f^-1 v_m w0 and k_d = Da_d v_m / w0. Every run
                lasts 50 chunks x 100,000 steps x dt (0.01 s) = 50,000 s, with
                mdot_0 = 0.01 pg/s and eps = 100 um/s (SI Table 3).
Inputs        : none
Outputs       : OUT_ROOT/05_branch_transport_model_1p5d/sweep_study_ND/
                phase_space_study_ND_pairwise_<timestamp>/idx_###_D2_*_k4_*_vm_*/
                  exp_parameters.txt and <t>.pkl snapshots of the simulator
                  (read by plot_2D_ND.py); PNG frames if save_plot=True.
Environment   : environment-models.yml (Python 3.13)
Run           : python para_sweep_non_dim.py   (long: 100 runs of 5e6 steps)

Notes: the driver sets epsilon = 100 directly, so `epsilon_factor` is unused.
The committed script had save_pkl=False (switched off after the published sweep
had been run); it is True here so the sweep writes the snapshots that
plot_2D_ND.py reads. Pickles store the class defined in this file; load them
with plot_2D_ND.py, which defines the same class.

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
import traceback  # used in the error log below; was missing
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
    1D ARD simulator solved in NON-DIMENSIONAL variables.
    Inputs remain DIMENSIONAL (same file format). Plots are shown in DIMENSIONAL units.
    """

    def __init__(self, parameters_file, folder_dir=None, save_pkl=False, save_plot=False):
        self.folder_dir = folder_dir
        self.save_pkl = save_pkl
        self.save_plot = save_plot

        self.read_parameters(parameters_file)
        self.initialize_variables_nd()  # now initializes BARRED variables

    def read_parameters(self, parameters_file):
        """
        Read dimensional parameters key=value and set attributes.
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

        # Precompute dimensional helpers
        # mu = 4 / (rho * pi * w0^2) (length per unit mass of deposit)
        self.mu = 4.0 / (self.rho * np.pi * self.w0 * self.w0)

        # === NON-DIMENSIONALIZATION SCALES ===
        # l0 = w0 ; t0 = w0/vm ; c0 = 1/mu
        self.l0 = self.w0
        self.t0 = self.w0 / self.vm
        self.c0 = 1.0 / self.mu

        # === DIMENSIONLESS GROUPS ===
        # Pe_f = vm*w0/D2 ; we store Pe_inv = 1/Pe_f = D2/(vm*w0) to use directly
        self.Pe_inv = self.D2 / (self.vm * self.w0)
        # Damkohler numbers
        self.Da_d = self.k4 * self.w0 / self.vm      # deposition
        self.Da_t = self.k5 * self.w0 / self.vm      # tip growth
        self.Da_eps = self.epsilon / self.vm         # tip influx
        # Dimensionless saturation concentration
        self.csat_bar = self.csat / self.c0          # = mu * csat

        # BARRED timestep and spacing
        self.delta_t_bar = self.delta_t * self.vm / self.w0
        self.delta_x_bar = self.delta_x / self.w0
        self.l_g_bar = self.l_g / self.w0

        # Derived/internals
        self.max_N = 2000  # buffer for growth
        self.time_bar = 0.0  # dimensionless time

        # Debug prints
        print(
            '--- DIMENSIONAL (input) ---\n'
            f'D2={self.D2}, k4={self.k4}, k5={self.k5}, epsilon={self.epsilon}, csat={self.csat}, vm={self.vm}\n'
            f'mu={self.mu}, w0={self.w0}, rho={self.rho}, l_g={self.l_g}, m_dot_0={self.m_dot_0}\n'
            f'delta_t={self.delta_t}, delta_x={self.delta_x}, N={self.N}'
        )
        print(
            '--- NON-DIMENSIONAL params ---\n'
            f'Pe_inv={self.Pe_inv} (=> Pe={1.0/self.Pe_inv if self.Pe_inv>0 else np.inf})\n'
            f'Da_d={self.Da_d}, Da_t={self.Da_t}, Da_eps={self.Da_eps}\n'
            f'csat_bar={self.csat_bar}, delta_t_bar={self.delta_t_bar}, delta_x_bar={self.delta_x_bar}, l_g_bar={self.l_g_bar}'
        )

        # Left BC source in dimensionless form: m0_bar = dot{m}_0 * mu / vm
        self.m0_bar = self.m_dot_0 * self.mu / self.vm

    def initialize_variables_nd(self):
        """
        Initialize BARRED variables: cf_bar, cd_bar, cg_bar, l_bar, etc.
        """
        # Allocate big arrays
        self.cf_bar = np.zeros(self.max_N)  # transport field (dimensionless)
        self.cd_bar = np.zeros(self.max_N)  # deposited field (dimensionless)

        # Active domain
        self.cf_bar[:self.N] = 0.0
        self.cd_bar[:self.N] = 1.0  # since c0 = 1/mu, dimensional c_d = 1/mu -> barred = 1

        # Growth zone concentration (dimensionless)
        self.cg_bar = 0.0

        # Dimensionless length from N cells
        self.l_bar = self.delta_x_bar * self.N

        # For local updates near the tip (same idea as before, but in bars)
        # we'll interpret self.calc_tip_range as number of cells; reuse the same scale as before
        self.calc_tip_range = int(100 * self.l_g / self.delta_x)  # number of cells; dimensionless-free
        self.calc_start_index = max(self.N - self.calc_tip_range, 0)

        # mu_int_bar and clim_bar
        self.mu_int_bar = 1.0  # initially 1/(1+cf_tip)=1
        # clim was concentration threshold in growth zone for one spatial step; in dimensionless:
        # clim_bar = delta_x_bar / (mu_int_bar * l_g_bar)
        self.clim_bar = self.delta_x_bar / (self.mu_int_bar * self.l_g_bar)

        # Mass tallies (for plotting/comparison). Track in DIMENSIONAL units to match your previous text.
        # We'll reconstruct dimensional arrays when needed.
        self.total_mass_injected_dim = 0.0
        self.update_total_mass_dim()  # sets total_mass_dim using current state

        # Optional snapshot
        if self.save_pkl and self.folder_dir is not None:
            pkl_path = os.path.join(self.folder_dir, f"{int(round(self.time_bar*self.t0))}.pkl")
            with open(pkl_path, 'wb') as f:
                pickle.dump(self, f)

    def update_total_mass_dim(self):
        """
        Compute total mass in DIMENSIONAL units for reporting:
          total_mass = (int c_f + int c_d)*dx + c_g*l_g
        """
        # Reconstruct dimensional fields for active domain
        cf_dim = self.cf_bar[:self.N] * self.c0
        cd_dim = self.cd_bar[:self.N] * self.c0
        dx_dim = self.delta_x
        cg_dim = self.cg_bar * self.c0

        c2_mass = np.sum(cf_dim) * dx_dim
        c4_mass = np.sum(cd_dim) * dx_dim
        self.total_mass_dim = c2_mass + c4_mass + cg_dim * self.l_g

    def run_simulation_nd(self, num_steps):
        """
        Advance the simulation by 'num_steps' steps in NON-DIMENSIONAL time.
        """
        cf = self.cf_bar
        cd = self.cd_bar

        dx_bar = self.delta_x_bar
        dt_bar = self.delta_t_bar

        Pe_inv = self.Pe_inv
        Da_d = self.Da_d
        Da_t = self.Da_t
        Da_eps = self.Da_eps
        csat_bar = self.csat_bar

        for _ in range(num_steps):
            # 1) Precipitation (dimensionless)
            cf_active = cf[:self.N]
            cd_active = cd[:self.N]

            precip_bar = Da_d * (cf_active - csat_bar)
            precip_bar[precip_bar < 0.0] = 0.0

            # 2) Diffusion + advection in interior (dimensionless)
            start = max(self.calc_start_index + 1, 1)
            end = self.N - 1
            if end > start:
                c_slice = cf_active[start:end]
                c_left = cf_active[start-1:end-1]
                c_right = cf_active[start+1:end+1]
                p_slice = precip_bar[start:end]

                # ∂t cf = Pe_inv*(c_left - 2c + c_right)/dx^2  - (c - c_left)/dx  - precip
                cf_active[start:end] += dt_bar * (
                    Pe_inv * (c_left - 2.0 * c_slice + c_right) / (dx_bar**2)
                    - (c_slice - c_left) / dx_bar
                    - p_slice
                )

            # 3) c_d update (dimensionless)
            cd_active += dt_bar * precip_bar

            # 4) Growth-zone accumulation (dimensionless)
            tip_val = cf_active[self.N - 1]
            # d c_g_bar / dt_bar = (Da_eps / l_g_bar) * c_f_tip - (v_g_bar / l_g_bar)*(1/mu_int_bar)
            # we'll integrate cg later after computing v_g_bar; for now accumulate the influx term
            influx_term = (dt_bar * Da_eps * tip_val) / self.l_g_bar
            # Temporarily add influx; outflux will be subtracted after v_g_bar is computed
            cg_new_bar = self.cg_bar + influx_term

            # 5) LEFT boundary (injection): -Pe_inv * (c1 - c0)/dx + c0 = m0_bar
            # => c0 * (1 + Pe_inv/dx) - (Pe_inv/dx) * c1 = m0_bar
            # => c0 = ( m0_bar + (Pe_inv/dx)*c1 ) / (1 + Pe_inv/dx)
            denom_left = 1.0 + Pe_inv / dx_bar
            cf_active[0] = (self.m0_bar + (Pe_inv / dx_bar) * cf_active[1]) / denom_left

            # 6) RIGHT boundary (tip): -Pe_inv * (cN - cNm1)/dx + cNm1 = Da_eps*cN
            denom_right = (Da_eps) + (Pe_inv / dx_bar)
            cf_active[self.N - 1] = (Pe_inv / dx_bar + 1) * cf_active[self.N - 2] / denom_right

            # 7) Domain growth: update mu_int_bar, v_g_bar, l_bar, extend grid if needed
            cf_tip = cf_active[self.N - 1]
            self.mu_int_bar = 1.0 / (1.0 + cf_tip)
            # v_g_bar = mu_int_bar * l_g_bar * Da_t * (cg_bar - csat_bar)
            v_g_bar = self.mu_int_bar * self.l_g_bar * Da_t * (cg_new_bar - csat_bar)

            # Apply growth to length
            l_bar_old = self.l_bar
            self.l_bar += v_g_bar * dt_bar

            # Outflux term for cg_bar: -(v_g_bar / l_g_bar)*(1/mu_int_bar) * dt_bar
            outflux_term = (v_g_bar / self.l_g_bar) * (1.0 / self.mu_int_bar) * dt_bar
            self.cg_bar = cg_new_bar - outflux_term

            # If length crossed a cell boundary, add cells
            target_N = int(np.floor(self.l_bar / dx_bar))
            if target_N > self.N:
                oldN = self.N
                self.N = min(target_N, self.max_N)
                # New transport cells get cf_tip; new deposit cells get 1
                cf[oldN:self.N] = cf_tip
                cd[oldN:self.N] = 1.0  # since c0=1/mu
                # (Note: no change to cg_bar here; we already accounted the mass removal via outflux_term)

            # 8) Advance time
            self.time_bar += dt_bar

            # 9) Update mass tallies in DIMENSIONAL units & injected mass
            self.update_total_mass_dim()
            # Injected mass increment (dimensional): m_dot_0 * dt
            self.total_mass_injected_dim += self.m_dot_0 * (dt_bar * self.t0)

            # 10) Update calc_start_index
            self.calc_start_index = max(self.N - self.calc_tip_range, 0)

        # Optional final .pkl save
        if self.save_pkl and self.folder_dir is not None:
            pkl_path = os.path.join(self.folder_dir, f"{int(round(self.time_bar*self.t0))}.pkl")
            with open(pkl_path, 'wb') as f:
                pickle.dump(self, f)

    def plot_variable_dim(self, fig, ax, ax2):
        """
        Plot DIMENSIONAL profiles reconstructed from BARRED variables.
        """
        ax.clear()
        ax2.clear()

        # Colors
        color2 = plt.cm.tab20c(5)
        color5 = plt.cm.tab20c(16)
        cmap = plt.get_cmap('binary')

        # Active slices (dimensionless)
        cf_bar_active = self.cf_bar[:self.N]
        cd_bar_active = self.cd_bar[:self.N]

        # Reconstruct DIMENSIONAL fields for plotting
        cf_dim_active = cf_bar_active * self.c0
        cd_dim_active = cd_bar_active * self.c0
        cg_dim = self.cg_bar * self.c0

        # Dimensional coordinates
        x_vals = np.linspace(
            self.delta_x / 2.0,
            self.l_bar * self.l0 - self.delta_x / 2.0,
            self.N
        )

        # Plot c_f (dimensional)
        ax.plot(x_vals, cf_dim_active, label=r'$c_f$', color=color2)
        ax.fill_between(x_vals, cf_dim_active, color=color2, alpha=0.5)
        max_conc_check = np.max(cf_dim_active)
        ax.set_ylim(0, max_conc_check * 1.1 if max_conc_check > 0 else 1.0)

        # Growth zone rectangle (dimensional)
        clim_dim = (self.delta_x_bar / (self.mu_int_bar * self.l_g_bar)) * self.c0  # for color scale
        norm_val = 0.0 if clim_dim == 0 else np.clip(cg_dim / clim_dim, 0, 1)
        ax2.fill_between(
            [self.l_bar * self.l0, self.l_bar * self.l0 + self.l_g],
            -self.w0 / 2,
            self.w0 / 2,
            color=cmap(norm_val),
            alpha=0.5
        )
        rectangle = plt.Rectangle(
            (self.l_bar * self.l0, -self.w0 / 2),
            self.l_g,
            self.w0,
            edgecolor='black',
            facecolor='none',
            linestyle='--'
        )
        ax2.add_patch(rectangle)

        # Plot c_g marker (dimensional)
        ax.plot(
            self.l_bar * self.l0 + self.l_g / 2,
            cg_dim,
            color=color5,
            label=r'$c_g$',
            marker='o',
            markersize=10
        )
        ax.plot(
            [self.l_bar * self.l0 + self.l_g / 2, self.l_bar * self.l0 + self.l_g / 2],
            [0, cg_dim],
            color=color5,
            linestyle='--'
        )

        # Compute width from DIMENSIONAL c_d as before
        local_mass = cd_dim_active * self.delta_x
        local_volume = local_mass / self.rho
        w_active = np.sqrt(4.0 * local_volume / self.delta_x / np.pi)

        # Plot branch boundary (dimensional)
        ax2.plot(x_vals, w_active / 2.0, 'k-')
        ax2.plot(x_vals, -w_active / 2.0, 'k-')
        ax2.fill_between(x_vals, w_active / 2.0, -w_active / 2.0, color=color5, alpha=0.5)

        # Axes limits & aspect
        max_len = max(self.l_bar * self.l0 + 2.0, 25.0)
        max_wid = max(self.w0 * 2.0, max_len / 2.0)
        ax2.set_xlim(0, max_len)
        ax2.set_ylim(-max_wid, max_wid)
        ax2.set_aspect('equal')

        # Time stamp (dimensional time)
        ax.text(
            0.5, 0.98,
            f'Time: {int(round(self.time_bar * self.t0))} s',
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=12
        )

        # Labels
        ax.set_xlabel(r'Length ($\mu m$)', fontsize=12)
        ax.set_ylabel(r'Concentration ($pg/\mu m$)', fontsize=12)
        ax2.set_ylabel(r'Width ($\mu m$)', fontsize=12)
        ax.set_title('Concentration vs Length and Shape Profile', fontsize=15)
        ax.set_facecolor('gainsboro')
        ax.legend(facecolor='gainsboro', frameon=False, loc='upper right')
        fig.subplots_adjust(left=0.08)

        # Parameter text
        parameters_text = (
            f'Pe_inv = {self.Pe_inv}\n'
            f'Da_d = {self.Da_d}\n'
            f'Da_t = {self.Da_t}\n'
            f'Da_eps = {self.Da_eps}\n'
            f'csat_bar = {self.csat_bar}\n'
            f'delta_t_bar = {self.delta_t_bar}\n'
            f'delta_x_bar = {self.delta_x_bar}\n'
            f'l_g_bar = {self.l_g_bar}\n'
            f'm0_bar = {self.m0_bar}\n'
            f'mu_int_bar = {self.mu_int_bar}\n'
            f'l_bar = {self.l_bar}\n'
            f'N = {self.N}\n'
            f'l (dim) = {self.l_bar * self.l0}\n'
            f'integrated_mass = {self.total_mass_dim}\n'
            f'mass_injected = {self.total_mass_injected_dim}\n'
            f'mass_difference = {self.total_mass_dim - self.total_mass_injected_dim}\n'
            f'cg (dim) = {self.cg_bar * self.c0}\n'
            f'time (dim s) = {int(round(self.time_bar * self.t0))}\n'
        )

        fig.subplots_adjust(right=0.75)
        ax2.text(1.1, -0.02, parameters_text, transform=ax.transAxes, fontsize=9, va='bottom')

    # ---------------------- end class ----------------------





def write_params_to_file(param_dict, file_path):
    """
    Utility: write each parameter as 'key = value' to file_path.
    """
    with open(file_path, 'w') as f:
        for k, v in param_dict.items():
            f.write(f"{k} = {v}\n")



# ---- ND phase-space sweep --------------------------------------------------
def run_phase_space_study_nd(
    # Swept parameters (DIMENSIONAL inputs; ND happens inside the simulator)
    D2_list,
    k4_list,
    vm_list,
    # NEW: choose how to combine the lists
    combo_mode="cartesian",   # "cartesian" (default) or "pairwise"
    # Optional labels for pairwise mode (must match length of lists if provided)
    combo_labels=None,

    # Constants / singletons (still DIMENSIONAL)
    k5=100.0,
    csat=0.0,
    m_dot_0=0.01,
    w0=1.0,
    rho=2.71,
    delta_t=0.01,
    delta_x=0.05,
    N=5,
    l_g=0.5,
    # epsilon = epsilon_factor * vm (DIMENSIONAL); becomes Da_eps = epsilon/vm inside
    epsilon_factor=10.0,
    # I/O and run control
    save_plot=False,
    save_pkl=False,
    base_output_folder=str(OUT_ROOT / "05_branch_transport_model_1p5d"),
    num_simulation_steps=10,     # how many chunks
    steps_per_run=1000,          # ND steps per chunk
    show_plots=False
):
    """
    Non-dimensional sweep driver with two modes:

    combo_mode:
      - "cartesian": run all combinations (D2 x k4 x vm) like before.
      - "pairwise":  run only the n-th entries together:
                     (D2_list[n], k4_list[n], vm_list[n]) for n in range(L),
                     where all three lists must have the same length L.

    If combo_labels is provided in "pairwise" mode, it must be a list of length L;
    each label is appended to the combo folder name for easier identification.

    Writes DIMENSIONAL params to exp_parameters.txt; the simulator handles ND internally.
    """
    # Root folder
    stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    global_folder = os.path.join(base_output_folder, f'phase_space_study_ND_{combo_mode}_{stamp}')
    os.makedirs(global_folder, exist_ok=True)

    # Error log
    error_log_path = os.path.join(global_folder, 'error_log.txt')
    with open(error_log_path, 'w') as f:
        f.write("Error Log\n=========\n\n")

    # Build combinations
    if combo_mode.lower() == "pairwise":
        # Validate lengths
        L = len(D2_list)
        if not (len(k4_list) == L and len(vm_list) == L):
            raise ValueError(
                f"In 'pairwise' mode, D2_list (len={len(D2_list)}), "
                f"k4_list (len={len(k4_list)}), and vm_list (len={len(vm_list)}) must have equal length."
            )
        if combo_labels is not None and len(combo_labels) != L:
            raise ValueError(
                f"combo_labels length ({len(combo_labels)}) must match the lists length ({L}) in 'pairwise' mode."
            )

        all_combos = list(zip(D2_list, k4_list, vm_list))
        # Build names with optional labels and an index to keep ordering clear
        def folder_name(i, D2_val, k4_val, vm_val):
            base = f"idx_{i:03d}_D2_{D2_val}_k4_{k4_val}_vm_{vm_val}"
            if combo_labels is not None:
                return f"{base}_label_{str(combo_labels[i])}"
            return base

    elif combo_mode.lower() == "cartesian":
        all_combos = list(product(D2_list, k4_list, vm_list))
        def folder_name(i, D2_val, k4_val, vm_val):
            return f"D2_{D2_val}_k4_{k4_val}_vm_{vm_val}"
    else:
        raise ValueError("combo_mode must be 'cartesian' or 'pairwise'.")

    total = len(all_combos)

    for idx, (D2_val, k4_val, vm_val) in enumerate(all_combos, start=1):
        combo_name = folder_name(idx-1, D2_val, k4_val, vm_val)
        combo_dir = os.path.join(global_folder, combo_name)
        os.makedirs(combo_dir, exist_ok=True)

        print(f"\n[{idx}/{total}] Running combination: {combo_name}")

        # DIMENSIONAL epsilon -> ND Da_eps computed inside simulator as epsilon/vm
        epsilon_val = 100   #epsilon_factor * vm_val

        # Parameter file (DIMENSIONAL)
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
        param_path = os.path.join(combo_dir, 'exp_parameters.txt')
        write_params_to_file(param_dict, param_path)

        try:
            # Create ND simulator (assumes your class non-dimensionalizes internally)
            simulator = ReactionDiffusionSimulator(
                parameters_file=param_path,
                folder_dir=combo_dir,
                save_pkl=save_pkl,
                save_plot=save_plot
            )

            # Figure
            fig, ax = plt.subplots(figsize=(14, 9))
            ax2 = ax.twinx()

            # Initial plot (DIMENSIONAL reconstruction)
            simulator.plot_variable_dim(fig, ax, ax2)
            if save_plot:
                plt.savefig(os.path.join(combo_dir, 'plot_initial.png'), dpi=150, bbox_inches='tight')
            if show_plots:
                plt.pause(0.1)

            # Run in ND steps
            for chunk_i in range(num_simulation_steps):
                simulator.run_simulation_nd(steps_per_run)

                # DIMENSIONAL diagnostics reconstructed from barred variables
                time_dim = simulator.time_bar * simulator.t0
                cf_tip_dim = simulator.cf_bar[simulator.N - 1] * simulator.c0
                clim_dim   = simulator.clim_bar * simulator.c0
                cg_dim     = simulator.cg_bar * simulator.c0
                l_dim      = simulator.l_bar * simulator.l0

                print(
                    f"{combo_name} | t={time_dim:.3f} s, "
                    f"cf_tip={cf_tip_dim:.5g}, clim={clim_dim:.5g}, cg={cg_dim:.5g}, "
                    f"l={l_dim:.5g} µm, N={simulator.N}"
                )

                simulator.plot_variable_dim(fig, ax, ax2)
                if save_plot:
                    step_tag = str((chunk_i + 1) * steps_per_run).zfill(8)
                    plt.savefig(os.path.join(combo_dir, f'plot_{step_tag}.png'), dpi=150, bbox_inches='tight')
                if show_plots:
                    plt.pause(0.05)

            plt.close(fig)

        except Exception as e:
            print(f"Error in {combo_name}. See error_log.txt.")
            with open(error_log_path, 'a') as f:
                f.write(f"Error for {combo_name}:\n{str(e)}\nTraceback:\n")
                f.write(traceback.format_exc())
                f.write("\n\n")
            continue




def choose_safe_parameters_for_phase_plane(
    Pe_inv_vals,            # iterable of target Pe^{-1}
    Da_d_vals,              # iterable of target Da_d
    w0=1.0,                 # μm
    delta_t=0.01,           # s
    delta_x=0.05,           # μm
    # CFL-like stability guardrails for explicit scheme:
    alpha_diff=0.45,        # dt_bar <= alpha_diff * dx_bar^2 / Pe_inv
    beta_adv =0.8,          # dt_bar <= beta_adv * dx_bar
    safety_factor=0.5,      # shrink chosen vm by this factor inside the allowable ceiling
    # User/physics bounds:
    vm_min=1e-3, vm_max=1.0,            # μm/s (search range for vm)
    D2_min=1e-6, D2_max=1e+2,           # μm^2/s
    k4_min=1e-6, k4_max=1e+2,           # 1/s
    dedupe=True,                         # ensure each (Pe_inv,Da_d) only once
    verbose=True
):
    """
    For each (Pe_inv*, Da_d*) target, pick vm as the smallest feasible value that
    satisfies:
        1) CFL-like stability ceilings
        2) Optional physical bounds on D2, k4, vm
    Then set:
        D2 = Pe_inv* * vm * w0
        k4 = Da_d* * vm / w0
    Returns lists D2_list, k4_list, vm_list aligned with (Pe_inv, Da_d) targets.
    """
    Pe_inv_vals = list(Pe_inv_vals)
    Da_d_vals   = list(Da_d_vals)

    dx_bar = delta_x / w0
    # For dt_bar = delta_t * vm / w0:
    #   diffusion ceiling: dt_bar <= alpha*dx_bar^2 / Pe_inv  -> vm <= alpha*dx_bar^2/(Pe_inv) * (w0/delta_t)
    #   advection ceiling: dt_bar <= beta*dx_bar               -> vm <= beta*dx_bar * (w0/delta_t)

    pairs = list(product(Pe_inv_vals, Da_d_vals))
    if dedupe:
        pairs = list(dict.fromkeys(pairs))  # order-preserving unique

    D2_list, k4_list, vm_list = [], [], []
    info = []

    vm_adv_ceiling = beta_adv * dx_bar * (w0 / delta_t)  # Pe_inv-independent
    for (Pe_inv, Da_d) in pairs:
        # CFL ceilings
        vm_diff_ceiling = alpha_diff * (dx_bar**2) * (w0 / delta_t) / max(Pe_inv, 1e-30)
        vm_cfl_max = min(vm_diff_ceiling, vm_adv_ceiling)

        # Bounds induced by D2 and k4 ranges:
        # D2 = Pe_inv * vm * w0  -> vm in [D2_min/(Pe_inv*w0), D2_max/(Pe_inv*w0)]
        # k4 = Da_d   * vm / w0  -> vm in [k4_min*w0/Da_d,     k4_max*w0/Da_d    ]
        vm_from_D2_min = D2_min / max(Pe_inv * w0, 1e-30)
        vm_from_D2_max = D2_max / max(Pe_inv * w0, 1e-30)
        vm_from_k4_min = (k4_min * w0) / max(Da_d, 1e-30)
        vm_from_k4_max = (k4_max * w0) / max(Da_d, 1e-30)

        # Aggregate feasible vm interval:
        lower_bounds = [vm_min, vm_from_D2_min, vm_from_k4_min]
        upper_bounds = [vm_max, vm_from_D2_max, vm_from_k4_max, vm_cfl_max]

        vm_lo = max(lower_bounds)
        vm_hi = min(upper_bounds)

        # If infeasible, relax to closest feasible inside user vm bounds (warn)
        feasible = vm_lo <= vm_hi
        if not feasible:
            # Try to clip to user vm range while respecting CFL (pick smallest possible for stability)
            vm_candidate = min(max(vm_min, 1e-6), vm_cfl_max)
            if verbose:
                info.append({
                    'Pe_inv': Pe_inv, 'Da_d': Da_d,
                    'status': 'RELAXED',
                    'vm_lo_req': vm_lo, 'vm_hi_req': vm_hi,
                    'vm_chosen': vm_candidate,
                    'reason': 'No intersection of constraints; chose smallest within user/CFL.'
                })
            vm_star = max(vm_min, min(vm_candidate, vm_max))
        else:
            # Choose the smallest vm in the feasible interval (most stable), with a safety margin
            vm_star = max(vm_lo, min(vm_hi, safety_factor * vm_hi))
            if verbose:
                info.append({
                    'Pe_inv': Pe_inv, 'Da_d': Da_d,
                    'status': 'OK',
                    'vm_interval': (vm_lo, vm_hi),
                    'vm_cfl_max': vm_cfl_max,
                    'vm_chosen': vm_star
                })

        # Compute D2, k4 that hit the exact (Pe_inv, Da_d)
        D2 = Pe_inv * vm_star * w0
        k4 = Da_d   * vm_star / w0

        # Final clipping to hard bounds (rare; keeps numbers sane)
        D2 = float(np.clip(D2, D2_min, D2_max))
        k4 = float(np.clip(k4, k4_min, k4_max))

        D2_list.append(D2)
        k4_list.append(k4)
        vm_list.append(float(vm_star))

    if verbose:
        # Lightweight summary (optional to inspect)
        print(f"Selected {len(pairs)} unique (Pe_inv, Da_d) pairs.")
        n_relaxed = sum(1 for x in info if x['status'] == 'RELAXED')
        if n_relaxed:
            print(f"Note: {n_relaxed} pairs required relaxation (no feasible intersection).")
    return D2_list, k4_list, vm_list, info


# ---- example invocation ----------------------------------------------------

if __name__ == "__main__":
    # # Example sweeps (DIMENSIONAL inputs)
    # D2_array = np.linspace(0.01, 0.1, 5)     # µm^2/s
    # k4_array = np.linspace(0.01, 0.1, 5)     # 1/s
    # vm_array = np.linspace(0.01, 0.5, 5)     # µm/s

    Pe_inv_targets = np.logspace(-2, 1, 10)    # 1/Pe
    Da_d_targets   = np.logspace(-2, 1, 10)    # Da_d

    D2_list, k4_list, vm_list, log_info = choose_safe_parameters_for_phase_plane(
        Pe_inv_vals=Pe_inv_targets,
        Da_d_vals=Da_d_targets,
        w0=1.0,
        delta_t=0.01,
        delta_x=0.05,
        alpha_diff=0.45,
        beta_adv=0.8,
        safety_factor=0.5,
        vm_min=1e-3, vm_max=0.5,
        D2_min=1e-3, D2_max=1e+0,
        k4_min=1e-3, k4_max=1e+0,
        dedupe=True,
        verbose=True
    )

    print("\nRunning phase space study with the following parameters:")
    for i in range(len(D2_list)):
        D2 = D2_list[i]
        k4 = k4_list[i]
        vm = vm_list[i]
        Pe_inv = D2 / (vm * 1.0)    # w0=1.0
        Da_d   = k4 * (1.0) / vm    # w0
        print(f"Run {i+1}: D2={D2_list[i]:.5g}, k4={k4_list[i]:.5g}, vm={vm_list[i]:.5g}, Pe_inv={Pe_inv:.5g}, Da_d={Da_d:.5g}")

    combo_mode = "pairwise"  # or "cartesian"

    run_phase_space_study_nd(
        # D2_list=D2_array.tolist(),
        # k4_list=k4_array.tolist(),
        # vm_list=vm_array.tolist(),
        D2_list=D2_list,
        k4_list=k4_list,
        vm_list=vm_list,
        k5=100.0,
        csat=0.0,
        m_dot_0=0.01,
        w0=1.0,
        rho=2.71,
        delta_t=0.01,
        delta_x=0.05,
        N=5,
        l_g=0.5,
        epsilon_factor=10.0,  # epsilon = 10 * vm
        save_plot=False,
        save_pkl=True,
        base_output_folder=str(OUT_ROOT / "05_branch_transport_model_1p5d" / "sweep_study_ND"),
        num_simulation_steps=50,     # number of chunks  500
        steps_per_run=100000,         # ND steps per chunk 10000
        show_plots=False,
        combo_mode=combo_mode
    )
