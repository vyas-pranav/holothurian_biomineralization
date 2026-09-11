# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2025 Pranav Vyas
"""Render a simulated growing branch in 3D at several time points.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 6A (left: simulated tip growing in length and thickness)
What it does  : Loads the pickled snapshots of one run of para_sweep_run.py,
                picks n geometrically spaced time points, revolves each width
                profile w(x) about the branch axis into a 3D surface (pyvista)
                and overlays the surfaces with transparency, coloured by time
                (plasma colormap).
Inputs        : one run folder of para_sweep_run.py (set folder_path; the
                published panel used D2 = 0.055, k4 = 0.055, vm = 0.255)
Outputs       : interactive pyvista window (screenshot for the figure)
Environment   : environment-models.yml (Python 3.13)
Run           : python 3d_time_growth.py

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



def load_simulator_from_pkl(pkl_path):
    """
    Loads and returns the ReactionDiffusionSimulator object from a pickle file.
    Returns None if loading fails.
    """
    try:
        with open(pkl_path, 'rb') as f:
            sim = pickle.load(f)
        return sim
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return None

def revolve_width_profile(x_vals, w_vals):
    """
    Given arrays of x-values (axial coordinate) and w-values (full width profile),
    creates a 3D mesh by revolving the 2D profile (polyline) around the X-axis.
    Returns a pyvista PolyData object.
    """
    polyline_points = []
    for x, width in zip(x_vals, w_vals):
        r = 0.5 * width  # radius
        polyline_points.append([x, r, 0])
    points = np.array(polyline_points)
    
    # Create a smooth spline through the points
    line = pv.Spline(points, 100)
    
    # Revolve (extrude rotate) around the X-axis
    mesh_3d = line.extrude_rotate(resolution=100, rotation_axis=[1, 0, 0])
    return mesh_3d

def get_sorted_pkl_files(folder):
    """
    Searches the given folder for .pkl files and returns a list of full paths,
    sorted by the numeric time value extracted from filenames (e.g. "1000.pkl").
    """
    pkl_files = [f for f in os.listdir(folder) if f.endswith('.pkl')]
    file_time_pairs = []
    for fname in pkl_files:
        match = re.search(r"(\d+)\.pkl$", fname)
        if match:
            time_val = int(match.group(1))
            file_time_pairs.append((time_val, fname))
    file_time_pairs.sort(key=lambda x: x[0])
    sorted_files = [os.path.join(folder, fname) for _, fname in file_time_pairs]
    return sorted_files

def plot_3d_growth_over_time(folder, n_time_points=5, scale_factor=1.0, opacity=0.5):
    """
    Loads simulator snapshots (pickle files) from the provided folder,
    selects n evenly spaced time points, generates 3D shapes from their width profiles,
    and plots all shapes in a single PyVista window with transparent overlapping.
    Each shape is colored based on its simulation time using the plasma colormap.
    A manual scale bar (black line with label) is added.
    """
    sorted_pkl_files = get_sorted_pkl_files(folder)
    if not sorted_pkl_files:
        print("No pickle files found in the folder.")
        return

    total_files = len(sorted_pkl_files)
    indices = np.geomspace(1, total_files - 1, n_time_points, dtype=int)
    print(f"Found {total_files} pickle files. Using indices: {indices}")

    # Initialize the PyVista plotter and set a white background
    plotter = pv.Plotter()
    plotter.set_background("white")

    # Collect time values and meshes for normalization and labeling
    times = []
    meshes = []

    for idx in indices:
        pkl_path = sorted_pkl_files[idx]
        sim = load_simulator_from_pkl(pkl_path)
        if sim is None:
            continue

        # Build the x-coordinate array and extract the width profile from simulation data
        x_vals = np.linspace(sim.delta_x / 2, sim.l - sim.delta_x / 2, sim.N)
        w_vals = sim.w[:sim.N]
        mesh = revolve_width_profile(x_vals, w_vals)
        
        # Scale the mesh if desired (to normalize different domain sizes)
        mesh.scale([scale_factor, scale_factor, scale_factor], inplace=True)
        
        times.append(sim.time)
        meshes.append((mesh, sim.time))
    
    if not times:
        print("No valid simulation states loaded.")
        return

    # Normalize time values for colormap mapping
    min_time = min(times)
    max_time = max(times)
    time_range = max_time - min_time if max_time != min_time else 1.0

    for mesh, t in meshes:
        norm_time = (t - min_time) / time_range
        # Use plasma colormap to get an RGB tuple (ignore alpha)
        color = plt.cm.plasma(norm_time)[:3]
        label = f"Time = {t:.2f} s"
        plotter.add_mesh(mesh, color=color, opacity=opacity, label=label)
    
    # Instead of a built-in scale bar, create one manually:
    scale_length = 10.0  # Adjust this value as needed for your data
    # Create a simple line along the X-axis representing the scale bar
    scale_bar_points = np.array([[0, 0, 0], [scale_length, 0, 0]])
    scale_bar = pv.Line(scale_bar_points[0], scale_bar_points[1])
    # plotter.add_mesh(scale_bar, color="black", line_width=4)
    plotter.add_text(f"Scale: {scale_length} units", position="lower_left", font_size=10, color="black")

    # Add a colorbar to represent the time points
    plotter.add_scalar_bar(title="Time (normalized)", n_labels=5, vertical=True, title_font_size=12, label_font_size=10)

    # Add a legend with the time labels
    plotter.add_legend()

    # Finally, show the plot (no grid is added)
    plotter.show()

if __name__ == "__main__":
    # Change this path to the folder containing your pickle files
    folder_path = str(OUT_ROOT / "05_branch_transport_model_1p5d" / "sweep_study_dimensional" / "phase_space_study_2025_02_19_14_02_25" / "D2_0.05500000000000001_k4_0.05500000000000001_vm_0.255")
    
    # Number of time points to plot (evenly spaced over the simulation data)
    n_points = 7
    
    # Scaling factor for the 3D meshes
    scale_factor = 1.0
    
    # Opacity for overlapping shapes (0 = fully transparent, 1 = opaque)
    opacity = 0.5

    plot_3d_growth_over_time(folder_path, n_time_points=n_points, 
                             scale_factor=scale_factor, opacity=opacity)
