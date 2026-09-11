# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Sweep of the morphogen decay rate k and diffusion coefficient D (2D branching model).

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. S24 (k-D grid); Figs. 6L-M (k = 0.0027826 and 0.0046416 with
                D = 1.0, 7.7426 and 59.948 um^2/h: points of the same grid)
What it does  : Runs the Ossicle model (ossicle_class3.py) on a 10 x 10 geometric grid,
                k ('morphogen_decay_rate'; code: lambda) = 0.001-0.1 /h and
                D = 1-100 um^2/h, other parameters at the SI Table 4 values, logarithmic
                timing rule (SI Eq. 48) with kappa = 4.
                Note: as found, 'tsteps' is 1, so every run stops after one step; set it
                (e.g. to 1000 as in the other sweeps; runs end when no tip is active) to
                grow full ossicles.
Inputs        : none
Outputs       : OUT_ROOT/06_branching_lattice_model_2d/Network simulations/
                lambda_D_phase_space_study/<YYYYmmdd_HHMMSS>/lambda_<k>_D_<D>/:
                parameters.txt, plots/ossicle_time_<ii>.png (labelled "Step: ii+1"),
                final_ossicle_pruned.png, final_oss_graph.gpickle and
                final_oss_graph_pruned.gpickle (standard pickles, read with pickle.load;
                both hold the same, unpruned graph: prune_graph() is never called)
                Frames are saved every 5 steps. Grid figure: ../plotting/replot_final_pruned.py,
                then ../plotting/plot_grid_lambda_D_param.py
Environment   : environment-models.yml (Python 3.13); the paper runs used
                environment-analysis.yml (Python 3.7)
Run           : python lambda_D_sweep3.py   (on Windows, when output is redirected to
                a file, set PYTHONIOENCODING=utf-8: messages contain Greek letters)
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

#----------------------------------------------------------
# Wrapper function for a single simulation run with given λ and D.
def run_simulation(lambda_value, D_value, sim_params, base_run_folder, save_plots_every_n=2, tsteps=None):
    """
    Run one simulation with specified morphogen decay rate (λ) and diffusion coefficient (D).
    
    Parameters:
        lambda_value (float): Morphogen decay rate (λ) to use.
        D_value (float): Morphogen diffusion coefficient (D) to use.
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
    run_folder = os.path.join(base_run_folder, f"lambda_{lambda_value:.4e}_D_{D_value:.4e}")
    os.makedirs(run_folder, exist_ok=True)
    plots_folder = os.path.join(run_folder, "plots")
    os.makedirs(plots_folder, exist_ok=True)
    
    # Update simulation parameters for this run.
    sim_params['morphogen_decay_rate'] = lambda_value
    sim_params['morphogen_diffusion_coefficient'] = D_value
    
    # Define a function to compute the scaling factor for converting morphogen concentration 
    # to a replication time decrement. This ensures that the scale is computed based on the
    # current λ and D.
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
    
    # Define a replication interval function that uses the current sim_params.
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
            r_eff = max(r, 0.01)  # avoid singularity
            c_total += Q / (2 * np.pi * D) * k0(np.sqrt(lambda0 / D) * r_eff)
            
        # Example conversion from concentration to replication interval.
        rep_int = max(np.log(c_total) / scale * tau_max, tau_min)
        return rep_int
    
    # Set the replication function in the simulation parameters.
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
        print(f"[λ = {lambda_value:.4e}, D = {D_value:.4e}] Time step: {ii}")
        ossicle.sim_growth()       # Grow the structure.
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
    print(f"Simulation with λ = {lambda_value:.4e} and D = {D_value:.4e} complete. Data saved in: {run_folder}")

#----------------------------------------------------------
# Main function to perform the parameter sweep over λ and D.
def main():
    # Random seed: None = unseeded, as in the original runs; set an int for a repeatable sweep.
    SEED = None
    if SEED is not None:
        np.random.seed(SEED)

    # Define the base folder for the parameter sweep study.
    base_folder = str(OUT_ROOT / "06_branching_lattice_model_2d" / "Network simulations" / "lambda_D_phase_space_study")
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
        'initial_branch_count': 4,
        'tsteps': 1,                  # Number of time steps per simulation
        'node_size': 45,
        'edge_width': 5,
        'color_by_topology': False,
        'node_cmap': plt.cm.plasma,
        'cell_translation_step': 0.3,
        # Default morphogen parameters (will be overwritten per run)
        'morphogen_diffusion_coefficient': 10.0,  # Default D value
        'morphogen_decay_rate': 0.010,            # Default λ value
        'morphogen_release_rate': 100.0,          # Q (arbitrary units)
        'morphogen_rep_max': 40,                  # Maximum replication interval (h)
        'morphogen_rep_min': 5,                   # Minimum replication interval (h)
        'plot_limit': 100,
    }
    
    save_plots_every_n = 5  # Save a plot every 5 time steps.
    
    # Define the number of values in each parameter dimension.
    m = 10 # Adjust for finer or coarser sweeps.
    
    # Define the sweep ranges for λ and D.
    lambda_values = np.geomspace(0.001, 0.1, num=m)  # Example range for morphogen decay rate.
    D_values = np.geomspace(1, 100, num=m)               # Example range for diffusion coefficient.
    
    # Loop over all combinations of λ and D.
    for lambda_value in lambda_values:
        for D_value in D_values:
            # Use a fresh copy of the constant parameters for each run.
            sim_params = constant_sim_params.copy()
            run_simulation(lambda_value, D_value, sim_params, study_folder, save_plots_every_n)
            
if __name__ == '__main__':
    main()
