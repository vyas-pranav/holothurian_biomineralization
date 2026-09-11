# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Sweep of the cell force alpha and the pair force beta (2D branching model).

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Figs. S21-S23 (alpha-beta grids); Fig. 6I (alpha = 0.034966,
                beta = 0.073487: a point of the Fig. S22 grid)
What it does  : Runs the Ossicle model (ossicle_class3.py) on an m x m geometric grid of
                (alpha, beta), other parameters at the SI Table 4 values, logarithmic
                timing rule (SI Eq. 48). The grid is chosen with PANEL from SETTINGS:
                "as_saved" is the grid active in the original file (alpha 0.0025-0.025,
                beta 0.0005-0.05), "Fig. S22" is alpha 0.01-0.05 x beta 0.05-0.1, and
                "earlier_0.005_0.1" a further grid recorded in the file. The Fig. S21
                (0.001-1.0) and Fig. S23 (0.05-0.5) ranges are not recorded in this file;
                add them to SETTINGS to rerun those grids.
Inputs        : none
Outputs       : OUT_ROOT/06_branching_lattice_model_2d/Network simulations/
                alpha_beta_phase_space_study/<YYYYmmdd_HHMMSS>/alpha_<a>_beta_<b>/:
                parameters.txt, plots/ossicle_time_<ii>.png (labelled "Step: ii+1"),
                final_ossicle_pruned.png, final_oss_graph.gpickle and
                final_oss_graph_pruned.gpickle (standard pickles, read with pickle.load;
                both hold the same, unpruned graph: prune_graph() is never called)
                Frames are saved every 20 steps. Grid figure: ../plotting/plot_grid_alpha_beta_param.py
Environment   : environment-models.yml (Python 3.13); the paper runs used
                environment-analysis.yml (Python 3.7)
Run           : python cell_pair_force_sweep3.py   (on Windows, when output is redirected
                to a file, set PYTHONIOENCODING=utf-8: messages contain Greek letters)
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
import pickle  # replaces nx.write_gpickle (removed in networkx 3.0)
from scipy.special import k0  # For modified Bessel function of the second kind

# Import the Ossicle class from your separate file
from ossicle_class3 import Ossicle

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

# --- SETTINGS ---------------------------------------------------------------
# (alpha, beta) grids recorded in the original file, as (start, stop) of
# np.geomspace(start, stop, num=m). PANEL = "as_saved" reproduces the active grid.
SETTINGS = {
    "as_saved": {"alpha": (0.0025, 0.025), "beta": (0.0005, 0.05)},
    "Fig. S22": {"alpha": (0.01, 0.05), "beta": (0.05, 0.1)},
    "earlier_0.005_0.1": {"alpha": (0.005, 0.1), "beta": (0.005, 0.1)},
}
PANEL = "as_saved"
# ----------------------------------------------------------------------------

#----------------------------------------------------------
# Wrapper function for a single simulation run with given α and β.
def run_simulation(alpha_value, beta_value, sim_params, base_run_folder, save_plots_every_n=2, tsteps=None):
    """
    Run one simulation with given α and β values.
    
    Parameters:
        alpha_value (float): Cell repulsion strength.
        beta_value (float): Pair node attraction strength.
        sim_params (dict): Dictionary of simulation parameters.
        base_run_folder (str): Base folder for saving run-specific data.
        save_plots_every_n (int): Interval for saving plots.
        tsteps (int): Number of time steps to run; if None, uses sim_params['tsteps'].
    """
    # Define initial conditions for cells and nodes.
    init_pos_len = 0.1
    start_cells = np.array([[init_pos_len, 0],
                            [-init_pos_len, 0],
                            [0, init_pos_len],
                            [0, -init_pos_len]])
    start_nodes = np.array([[0.1, 0.01],
                            [-0.1, 0.01],
                            [-0.1, -0.01],
                            [0.1, -0.01]])
    n_cells = start_cells.shape[0]
    
    # Create a folder for this simulation run.
    run_folder = os.path.join(base_run_folder, f"alpha_{alpha_value:.4e}_beta_{beta_value:.4e}")
    os.makedirs(run_folder, exist_ok=True)
    plots_folder = os.path.join(run_folder, "plots")
    os.makedirs(plots_folder, exist_ok=True)
    
    # Update simulation parameters for this run.
    sim_params['alpha'] = alpha_value
    sim_params['beta'] = beta_value

    # Define a helper function to compute the scaling factor for converting morphogen concentration to replication interval decrement.
    def rep_time_scaling_func(s_params, n_cells=n_cells):
        D = s_params.get('morphogen_diffusion_coefficient', 10.0)
        lambda0 = s_params.get('morphogen_decay_rate', 0.02)
        Q = s_params.get('morphogen_release_rate', 100.0)
        r_vals = np.linspace(0.0001, 100, 10000)
        c_total = n_cells * Q / (2 * np.pi * D) * k0(np.sqrt(lambda0 / D) * r_vals)
        z_val = np.log(c_total)
        return np.max(z_val)
    
    
    # Update the scaling factor based on the current parameters.
    # sim_params['morphogen_rep_scale'] = rep_time_scaling_func(sim_params)
    sim_params['morphogen_rep_scale'] = 4.0  # Set a default value for now.


    # For this new simulation code we still use the morphogen replication scaling.
    # Define a replication interval function that uses the current simulation parameters.
    def rep_interval_func_wrapper(cell_list, pos_node):
        """
        Compute the replication interval at a given node position based on the net local morphogen concentration.
        """
        D = sim_params.get('morphogen_diffusion_coefficient', 10.0)
        lambda0 = sim_params.get('morphogen_decay_rate', 0.02)
        Q = sim_params.get('morphogen_release_rate', 100.0)
        tau_max = sim_params.get('morphogen_rep_max', 40)
        tau_min = sim_params.get('morphogen_rep_min', 5)
        scale = sim_params.get('morphogen_rep_scale', 1.0)
        
        c_total = 0.0
        for cell in cell_list:
            cell_pos = np.array(cell.pos)
            r = np.linalg.norm(pos_node - cell_pos)
            r_eff = max(r, 0.01)  # Avoid singularity.
            c_total += Q / (2 * np.pi * D) * k0(np.sqrt(lambda0 / D) * r_eff)
            
        rep_int = max(np.log(c_total) / scale * tau_max, tau_min)
        return rep_int
    
    sim_params['rep_interval_func'] = rep_interval_func_wrapper

    # Save simulation parameters to a text file.
    params_filename = os.path.join(run_folder, "parameters.txt")
    with open(params_filename, 'w') as f:
        f.write("Simulation Parameters:\n")
        for key, value in sim_params.items():
            f.write(f"{key}: {value}\n")
    
    # Instantiate the simulation.
    ossicle = Ossicle(start_nodes, start_cells, sim_params)
    
    # Create a figure for plotting.
    fig, ax = plt.subplots(figsize=(15, 15))
    
    if tsteps is None:
        tsteps = sim_params.get('tsteps', 100)
    
    # Run the simulation loop.
    for ii in range(tsteps):
        print(f"[α = {alpha_value:.4e}, β = {beta_value:.4e}] Time step: {ii}")
        ossicle.sim_growth()         # Grow the structure.
        is_active = ossicle.modify_topology()  # Update topology.
        ossicle.time += 1
        
        # Save a plot every n time steps.
        if ii % save_plots_every_n == 0:
            ossicle.plot_graph(fig, ax)
            plot_filename = os.path.join(plots_folder, f"ossicle_time_{ii}.png")
            fig.savefig(plot_filename, dpi=300)
        
        if not is_active:
            print("No active nodes remaining. Exiting simulation...")
            ossicle.plot_graph(fig, ax)
            plot_filename = os.path.join(plots_folder, f"ossicle_time_{ii}.png")
            fig.savefig(plot_filename, dpi=300)
            final_graph_filename = os.path.join(run_folder, "final_oss_graph.gpickle")
            with open(final_graph_filename, 'wb') as f:
                pickle.dump(ossicle.oss_graph, f, pickle.HIGHEST_PROTOCOL)
            ossicle.plot_graph(fig, ax)
            final_plot_filename = os.path.join(run_folder, "final_ossicle_pruned.png")
            fig.savefig(final_plot_filename, dpi=300)
            final_graph_filename = os.path.join(run_folder, "final_oss_graph_pruned.gpickle")
            with open(final_graph_filename, 'wb') as f:
                pickle.dump(ossicle.oss_graph, f, pickle.HIGHEST_PROTOCOL)
            break

    if is_active:
        ossicle.plot_graph(fig, ax)
        plot_filename = os.path.join(plots_folder, f"ossicle_time_{ii}.png")
        fig.savefig(plot_filename, dpi=300)
        final_graph_filename = os.path.join(run_folder, "final_oss_graph.gpickle")
        with open(final_graph_filename, 'wb') as f:
            pickle.dump(ossicle.oss_graph, f, pickle.HIGHEST_PROTOCOL)
        ossicle.plot_graph(fig, ax)
        final_plot_filename = os.path.join(run_folder, "final_ossicle_pruned.png")
        fig.savefig(final_plot_filename, dpi=300)
        final_graph_filename = os.path.join(run_folder, "final_oss_graph_pruned.gpickle")
        with open(final_graph_filename, 'wb') as f:
            pickle.dump(ossicle.oss_graph, f, pickle.HIGHEST_PROTOCOL)
    
    plt.close(fig)
    print(f"Simulation with α = {alpha_value:.4e} and β = {beta_value:.4e} complete. Data saved in: {run_folder}")

#----------------------------------------------------------
# Main function to run the 2D phase space study for α and β.
def main():
    # Random seed: None = unseeded, as in the original runs; set an int for a repeatable sweep.
    SEED = None
    if SEED is not None:
        np.random.seed(SEED)

    # Define the base folder for the phase space study.
    base_folder = str(OUT_ROOT / "06_branching_lattice_model_2d" / "Network simulations" / "alpha_beta_phase_space_study")
    os.makedirs(base_folder, exist_ok=True)
    
    # Create a timestamped study folder.
    now = datetime.datetime.now()
    study_folder = os.path.join(base_folder, now.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(study_folder, exist_ok=True)
    
    # Define constant simulation parameters.
    constant_sim_params = {
        'alpha': 0.0075,                # Cell director force strength
        'beta': 0.001,                  # Pair node attraction strength
        'eta': 1,                       # Persistence force strength
        'step_length': 0.5,             # Growth step length
        'pair_thresh_length': 10,       # Pairing threshold distance (um)
        'angle': 108 * np.pi / 180,     # Branching angle (radians)
        'seed_size': 0.1,               # Seed size for initial structure
        # The replication interval function will be set for each run.
        'rep_interval_func': None,
        'cell_force_modifier': None,
        'pers_force_modifier': None,
        'growth_step_scaling_func': None,
        'angle_noise': 0.01,
        'time_noise': 0.01,
        'seed_angle_noise': 0.1,
        'initial_branch_count': 4,      # Default initial branch count; will be overwritten.
        'tsteps': 1000,                 # Number of time steps per simulation
        'node_size': 45,
        'edge_width': 5,
        'color_by_topology': False,
        'node_cmap': plt.cm.plasma,
        'cell_translation_step': 0.3,
        # Morphogen parameters (constant for all runs)
        'morphogen_diffusion_coefficient': 10.0,
        'morphogen_decay_rate': 0.010,
        'morphogen_release_rate': 100.0,
        'morphogen_rep_max': 40,
        'morphogen_rep_min': 5,
        'plot_limit': 60,
    }


    save_plots_every_n = 20  # Save a plot every 5 time steps.
    
    # Define the number of values in each parameter dimension (geometric sweep).
    m = 10  # Adjust as needed for a finer or coarser sweep.


    # Grid ranges: selected with PANEL (SETTINGS at the top of the file).
    alpha_values = np.geomspace(*SETTINGS[PANEL]["alpha"], num=m)
    beta_values  = np.geomspace(*SETTINGS[PANEL]["beta"], num=m)

    
    # Loop over all combinations of α and β.
    for alpha_value in alpha_values:
        for beta_value in beta_values:
            # Use a fresh copy of the constant parameters for each run.
            sim_params = constant_sim_params.copy()
            run_simulation(alpha_value, beta_value, sim_params, study_folder, save_plots_every_n)
            
if __name__ == '__main__':
    main()
