# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Re-render the final graphs of a sweep study with the current plot style.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. S24 (tile images; also Figs. 6L-M)
What it does  : Walks a sweep study folder; for every final_oss_graph_pruned.gpickle it
                reads parameters.txt from the same folder, builds an Ossicle
                (../sweeps/ossicle_class3.py) with a dummy replication function and four
                dummy cells at the origin (used for the morphogen background), replaces
                its graph by the loaded one, sets the step label to the last node
                creation time + 1 and saves final_ossicle_pruned_new.png next to the
                graph.
Inputs        : outputs of ../sweeps/lambda_D_sweep3.py: OUT_ROOT/
                06_branching_lattice_model_2d/Network simulations/
                lambda_D_phase_space_study/20250325_220918/ (the study recorded in the
                original)
Outputs       : final_ossicle_pruned_new.png in every run folder (read by
                plot_grid_lambda_D_param.py)
Environment   : environment-models.yml (Python 3.13) or environment-analysis.yml
                (Python 3.7)
Run           : python replot_final_pruned.py
"""

import os
import pickle
import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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

# Import the Ossicle class from ../sweeps/ossicle_class3.py (the original imported
# ossicle_class3_plot.py, a byte-identical copy of that module kept next to this script).
sys.path.append(str(_HERE.parent.parent / "sweeps"))
from ossicle_class3 import Ossicle

def convert_value(val_str):
    """
    Attempt to convert a string value to int or float.
    If conversion fails, return the original string.
    """
    try:
        if '.' in val_str or 'e' in val_str.lower():
            return float(val_str)
        else:
            return int(val_str)
    except ValueError:
        return val_str

def read_parameters(param_filepath):
    """
    Read simulation parameters from a parameters.txt file.
    Expects lines of the form:
        key: value
    Ignores the header line.
    Returns a dictionary with converted numerical values where possible.
    """
    sim_params = {}
    with open(param_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Simulation Parameters"):
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                sim_params[key] = convert_value(value)
    return sim_params

def set_final_time(ossicle):
    """
    Set the ossicle.time attribute to the maximum create_time found among all nodes.
    If no node has create_time, defaults to zero.
    """
    try:
        create_times = [data.get("create_time", 0) for _, data in ossicle.oss_graph.nodes(data=True)]
        if create_times:
            ossicle.time = max(create_times)+1
        else:
            ossicle.time = 0
    except Exception as e:
        print("Error determining final time:", e)
        ossicle.time = 0

def replot_final_ossicle_pruned(base_folder):
    """
    Recursively search for final ossicle pruned graph files,
    read the parameters.txt file from the same folder,
    update the parameters with a dummy replication function to circumvent
    the 'NoneType' error, then replot the pruned graph using the Ossicle.plot_graph method.
    The ossicle.time is updated to the final time step (maximum create_time in the graph).
    Saves the new plot as final_ossicle_pruned_new.png in the same folder.
    """
    # Walk through the project base folder.
    for root, dirs, files in os.walk(base_folder):
        if "final_oss_graph_pruned.gpickle" in files:
            gpickle_path = os.path.join(root, "final_oss_graph_pruned.gpickle")
            print(f"Processing file: {gpickle_path}")
            # Read the graph.
            with open(gpickle_path, 'rb') as f:
                G = pickle.load(f)
            
            # Attempt to read the parameters.txt file from the same folder.
            param_filepath = os.path.join(root, "parameters.txt")
            if os.path.exists(param_filepath):
                sim_params = read_parameters(param_filepath)
                print(f"Read parameters from {param_filepath}")
            else:
                print(f"No parameters.txt found in {root}. Using default parameters.")
                sim_params = {}
            
            # Ensure the replication function is defined to avoid errors in __init__.
            sim_params['rep_interval_func'] = lambda cell_list, pos_node: 1e6
            
            # Some necessary parameters for plotting.
            if 'plot_limit' not in sim_params:
                sim_params['plot_limit'] = 100
            
            # If the simulation used a key with diffusion coefficient,
            # ensure consistency. For example, if the Ossicle class expects 'morphogen_diffusion_coeff'
            # copy the value from 'morphogen_diffusion_coefficient' if needed.
            if 'morphogen_diffusion_coefficient' in sim_params and 'morphogen_diffusion_coeff' not in sim_params:
                sim_params['morphogen_diffusion_coeff'] = sim_params['morphogen_diffusion_coefficient']
            
            # Create dummy initial conditions for instantiation.
            dummy_start_nodes = np.array([[0, 0]])
            dummy_start_cells = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
            
            # Instantiate the Ossicle.
            try:
                ossicle = Ossicle(dummy_start_nodes, dummy_start_cells, sim_params)
            except Exception as e:
                print(f"Error instantiating Ossicle: {e}")
                continue
            
            # Override the oss_graph with the loaded pruned graph.
            ossicle.oss_graph = G
            
            # Make sure necessary attributes exist.
            if not hasattr(ossicle, 'growth_center'):
                ossicle.growth_center = np.array([0, 0])
            if not hasattr(ossicle, 'cell_list'):
                ossicle.cell_list = []
            if not hasattr(ossicle, 'plot_lim_size'):
                ossicle.plot_lim_size = sim_params.get('plot_limit', 100)
            
            # Set ossicle.time as the maximum create_time in the graph.
            set_final_time(ossicle)
            print(f"Final simulation time set to: {ossicle.time}")
            
            # Create the figure and replot using the existing plotting function.
            fig, ax = plt.subplots(figsize=(15, 15))
            ossicle.plot_graph(fig, ax)
            
            # Save the new plot.
            new_plot_path = os.path.join(root, "final_ossicle_pruned_new.png")
            fig.savefig(new_plot_path, dpi=300)
            plt.close(fig)
            print(f"Saved new plot to: {new_plot_path}")

if __name__ == "__main__":
    base_folder = str(OUT_ROOT / "06_branching_lattice_model_2d" / "Network simulations" / "lambda_D_phase_space_study" / "20250325_220918")
    replot_final_ossicle_pruned(base_folder)
    print("Replotting completed.")