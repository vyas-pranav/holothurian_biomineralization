# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Growth series of a simulated ossicle drawn from its final graph.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. S20A (growth series; see the note on step labels)
What it does  : Loads a final graph pickle of a model run, picks rows x cols time
                thresholds between the smallest and largest node creation time
                (geometric spacing, or linear when the smallest time is 0 as for seed
                nodes), and draws for each threshold the sub-graph of nodes created up
                to then, with common axis limits, the four cell positions and a scale bar
                (1/10 of the largest node-node distance, in um).
                Note: panels are labelled "Step: <threshold>" with float thresholds; the
                integer step labels 11, 31, ..., 111 of Fig. S20A match the frames
                written by ../sweeps/seed_edge_sweep3.py.
Inputs        : OUT_ROOT/06_branching_lattice_model_2d/Network simulations/old tests/
                2025-03-22_19-34-36/final_oss_graph.gpickle (the run recorded in the
                original; point graph_pickle_path at any final_oss_graph.gpickle)
Outputs       : output_grid.png next to the input graph
Environment   : environment-models.yml (Python 3.13) or environment-analysis.yml
                (Python 3.7)
Run           : python plot_growth_series.py
"""

import os
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

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

#set font to Helvetica
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

def load_graph(pickle_path):
    """Load the NetworkX graph from a pickle file."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def get_global_bounds(graph):
    """
    Compute global x and y bounds from all node positions in the graph.
    This ensures identical physical scaling for all subplots.
    """
    pos_dict = nx.get_node_attributes(graph, 'pos')
    xs = [pos[0] for pos in pos_dict.values()]
    ys = [pos[1] for pos in pos_dict.values()]
    return min(xs), max(xs), min(ys), max(ys)

def get_growth_center(graph):
    """
    Determine the growth center by choosing the node with the earliest create_time.
    Returns its position.
    """
    # Convert create_time to float in case they are stored as strings.
    central_node, _ = min(graph.nodes(data=True), key=lambda x: float(x[1].get("create_time", 0)))
    pos_dict = nx.get_node_attributes(graph, 'pos')
    return pos_dict.get(central_node, (0, 0))

def select_logarithmic_time_points(graph, n_points):
    """
    Choose n_points time thresholds logarithmically between the minimum and maximum
    create_time values in the graph. If the range is too narrow or the minimum is <=0,
    fallback to linear spacing.
    """
    times = [float(attr.get("create_time", 0)) for _, attr in graph.nodes(data=True)]
    min_time, max_time = min(times), max(times)
    # If the values are too close or min_time is non-positive, use linear spacing.
    if min_time <= 0 or max_time/min_time < 1.001:
        return np.linspace(min_time, max_time, n_points)
    else:
        return np.geomspace(min_time, max_time, n_points)

def filter_graph_by_time(graph, time_threshold):
    """
    Return a subgraph containing only nodes with create_time <= time_threshold.
    Edges are kept only if both endpoints are present.
    """
    nodes_to_include = [
        n for n, attr in graph.nodes(data=True)
        if float(attr.get("create_time", 0)) <= time_threshold
    ]
    return graph.subgraph(nodes_to_include).copy()

def compute_max_distance(graph):
    """
    Compute the maximum Euclidean distance between any two nodes in the graph.
    (This is an O(n^2) algorithm and may be slow for very large graphs.)
    """
    pos_dict = nx.get_node_attributes(graph, 'pos')
    nodes = list(pos_dict.keys())
    max_dist = 0
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            p1 = pos_dict[nodes[i]]
            p2 = pos_dict[nodes[j]]
            dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
            if dist > max_dist:
                max_dist = dist
    return max_dist

def plot_graph_state(ax, graph_state, growth_center, current_time, global_bounds, edge_width=2, active_nodes=[], cell_list=[]):
    """
    Plot the current state of the graph on the provided axes using the style exactly
    identical to your provided snippet.
    - Draws grey edges with a rounded cap style and shadow.
    - Marks the central node with a large black marker.
    - Marks active nodes with red markers.
    - Plots cells with an orange circle.
    - Displays a time label in the top-right corner.
    - Uses identical x/y limits for all plots.
    - No ticks, axes, or labels are shown.
    """
    ax.cla()
    pos_dict = nx.get_node_attributes(graph_state, 'pos')
    
    # Draw edges.
    edge_collection = nx.draw_networkx_edges(
        graph_state, 
        pos_dict, 
        edge_color='grey', 
        width=edge_width, 
        ax=ax, 
        alpha=1
    )
    
    # Set capstyle and shadow effects.
    if hasattr(edge_collection, 'set_capstyle'):
        edge_collection.set_capstyle('round')
    edge_collection.set_path_effects([
        pe.SimpleLineShadow(offset=(4, -4), shadow_color="lightgrey", alpha=1),
        pe.Normal()
    ])
    
    # Mark the central node with a black marker.
    ax.scatter([growth_center[0]], [growth_center[1]], color='black', s=800, zorder=4)
    
    # Mark active nodes with red markers.
    for active_node in active_nodes:
        if active_node in pos_dict:
            ax.scatter([pos_dict[active_node][0]], [pos_dict[active_node][1]], 
                       color='red', s=100, zorder=4, alpha=0.5)
    
    # Plot cells with a distinct orange 'o' marker.
    if cell_list:
        cell_positions = cell_list #[cell.pos for cell in cell_list]
        cell_x = [p[0] for p in cell_positions]
        cell_y = [p[1] for p in cell_positions]
        ax.scatter(cell_x, cell_y, color='orange', s=250, marker='o', label='Cells',
                   zorder=5, edgecolors='black', linewidth=2)
    
    ax.set_aspect('equal')
    # Remove ticks, axes and labels.
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.axis('off')
    ax.text(0.98, 0.98, f"Step: {current_time}", fontsize=25,
            ha='right', va='top', transform=ax.transAxes)
    
    # Set identical physical scaling.
    min_x, max_x, min_y, max_y = global_bounds
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

def create_grid_of_plots(graph, grid_rows, grid_cols, output_path, edge_width=2, active_nodes=[], cell_list=[]):
    """
    Create a grid of plots showing the graph state at different times.
    For each time threshold (selected logarithmically from the range of create_time values),
    a subgraph is created containing only nodes (and corresponding edges) present up to that time.
    """
    n_plots = grid_rows * grid_cols
    time_points = select_logarithmic_time_points(graph, n_plots)
    global_bounds = get_global_bounds(graph)
    growth_center = get_growth_center(graph)
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols*6, grid_rows*6))
    # Ensure axes is a 2D array.
    if grid_rows == 1 and grid_cols == 1:
        axes = np.array([[axes]])
    elif grid_rows == 1:
        axes = np.array([axes])
    elif grid_cols == 1:
        axes = np.array([[ax] for ax in axes])
    axes_flat = axes.flatten()
    
    for i, t in enumerate(time_points):
        ax = axes_flat[i]
        subgraph = filter_graph_by_time(graph, t)
        plot_graph_state(ax, subgraph, growth_center, t, global_bounds,
                         edge_width=edge_width, active_nodes=active_nodes, cell_list=cell_list)
    
    # Turn off any extra axes.
    for j in range(i+1, len(axes_flat)):
        axes_flat[j].axis('off')


    # ---- Add a scale bar below the first subplot ----
    # Compute the maximum distance in the final (full) graph.
    max_dist = compute_max_distance(graph)
    # Choose 1/10th of that distance as the scale bar length (in data units).
    scale_bar_data = max_dist / 10.0
    # Create an AnchoredSizeBar on the first subplot.
    ax0 = axes_flat[0]
    fontprops = fm.FontProperties(size=12)
    scalebar = AnchoredSizeBar(ax0.transData,
                               scale_bar_data,
                               f'{scale_bar_data:.2f} units',
                               loc='upper center',
                               pad=0.5,
                               borderpad=0.5,
                               sep=5,
                               frameon=False,
                               size_vertical=scale_bar_data/20,
                               fontproperties=fontprops,
                               bbox_to_anchor=(0.5, -0.1),
                               bbox_transform=ax0.transAxes)
    ax0.add_artist(scalebar)

    
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Grid image saved to: {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    # === Configuration ===
    # Path to the pickled graph file.
    graph_pickle_path = str(OUT_ROOT / "06_branching_lattice_model_2d" / "Network simulations" / "old tests" / "2025-03-22_19-34-36" / "final_oss_graph.gpickle")
    # Output path for the grid image.
    output_grid_path = str(OUT_ROOT / "06_branching_lattice_model_2d" / "Network simulations" / "old tests" / "2025-03-22_19-34-36" / "output_grid.png")
    
    # User-specified grid dimensions.
    grid_rows = 1   # e.g., 2 rows
    grid_cols = 8   # e.g., 4 columns (2x4 = 8 plots)
    # Edge width for drawing edges.
    edge_width = 7
    # Optional: lists for active nodes and cell objects (each with a .pos attribute)
    active_nodes = []  
    cell_list = [[0.1, 0], [-0.1, 0], [0, 0.1], [0, -0.1]]
    
    # --- Load the graph ---
    if not os.path.exists(graph_pickle_path):
        raise FileNotFoundError(f"Graph file not found: {graph_pickle_path}")
    graph = load_graph(graph_pickle_path)
    
    # --- Create the grid of plots ---
    create_grid_of_plots(graph, grid_rows, grid_cols, output_grid_path,
                         edge_width=edge_width, active_nodes=active_nodes, cell_list=cell_list)
