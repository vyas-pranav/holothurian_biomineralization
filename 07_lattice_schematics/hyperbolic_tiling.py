# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Hyperbolic {7,3} tiling drawn as an ossicle-like lattice.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 2J ("Translational asymmetry")
What it does  : Builds a {7,3} hyperbolic tiling with the `hypertiling`
                package (4 layers, centred on a vertex and shifted by 0.14),
                converts it to a NetworkX graph, crops it to a disc of radius
                0.95 and draws it with struts and nodes that shrink with
                distance from the centre, so that cells get smaller towards
                the rim.
Inputs        : none
Outputs       : hyp_graph1.png in the current working directory (600 dpi)
Environment   : environment-analysis.yml (Python 3.7; needs hypertiling)
Run           : python hyperbolic_tiling.py
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cmap
from hypertiling import HyperbolicTiling
from hypertiling.graphics.plot import convert_polygons_to_patches
from hypertiling.graphics.plot import plot_tiling
import networkx as nx
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import is_color_like


def convert_polygons_to_graph(tiling):
    """
    Convert polygons in a hyperbolic tiling to a graph representation.
    Overlapping nodes and edges will be combined into single entities.

    Parameters
    ----------
    tiling: HyperbolicTiling
        A hyperbolic tiling object.

    Returns
    -------
    graph: networkx.Graph
        The graph representation of the tiling.
    """
    graph = nx.Graph()
    position_to_node = {}  # Dictionary to map positions to node ids

    # Iterate over all polygons in the tiling
    for poly in tiling:
        u = poly[1:]  # Extract vertex coordinates
        nodes = []

        # Iterate over the vertices of the polygon
        for vertex in u:
            pos = (vertex.real, vertex.imag)
            # If the position is already in the graph, reuse the existing node
            if pos in position_to_node:
                node_id = position_to_node[pos]
            else:
                # If not, create a new node and remember its id
                node_id = len(position_to_node)
                graph.add_node(node_id, pos=pos)
                position_to_node[pos] = node_id
            nodes.append(node_id)

        # Add edges between consecutive nodes, considering the last one connected to the first if the edge does not exist already
        for i in range(len(nodes)):
            if not graph.has_edge(nodes[i], nodes[(i + 1) % len(nodes)]):
                graph.add_edge(nodes[i], nodes[(i + 1) % len(nodes)])

        #remove all edges and nodes outside a circle of given radius
        cir_rad = 0.95
        center = (0,0)
        nodes_to_remove = []
        for node in graph.nodes():
            node_loc = graph.nodes[node]['pos']
            dist = np.sqrt((node_loc[0] - center[0])**2 + (node_loc[1] - center[1])**2)
            if dist > cir_rad:
                nodes_to_remove.append(node)

    for node in nodes_to_remove:
        graph.remove_node(node)

    return graph

def custom_plot_graph(G):
    fig,ax  = plt.subplots(figsize=(20, 20), dpi = 600)

    #decrease the linewidth of the edges with increasing distance from the origin
    for edge in G.edges():
        x1, y1 = G.nodes[edge[0]]['pos']
        x2, y2 = G.nodes[edge[1]]['pos']
        mid_point = ((x1 + x2)/2, (y1 + y2)/2)
        center = (0,0)
        dist = np.sqrt((mid_point[0] - center[0])**2 + (mid_point[1] - center[1])**2)
        ax.plot([x1, x2], [y1, y2], linewidth = 13*(1.75 - dist/1.5), color='purple')
    # nx.draw_networkx_edges(graph, nx.get_node_attributes(graph, 'pos'), width = 1.5, edge_color = 'purple', ax=ax)


    #create a list of x and y positions of the nodes
    #create a list of sizes for the nodes
    x_list = []
    y_list = []
    size_list = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        x_list.append(x)
        y_list.append(y)
        size_list.append(500 - np.sqrt(x**2 + y**2)*400)

    #plot the nodes
    ax.scatter(x_list, y_list, marker = 'o', color='orange', s = size_list, zorder=2, linewidths=0)

    # nx.draw_networkx_nodes(graph, nx.get_node_attributes(graph, 'pos'), node_size=1.5, node_color='orange', ax=ax)

    ax.scatter([0], [0], color='black', s = 700, zorder=3)

    #draw the circle used to remove nodes
    # circle = plt.Circle((0,0), 0.95, linewidth=1, fill=False)
    # ax.add_artist(circle)

    # Set aspect of the plot to be equal
    ax.set_aspect('equal')
    # Remove the axes for a cleaner look
    plt.axis('off')
    #save the figure
    plt.savefig("hyp_graph1.png", dpi = 600)
    #show the figure
    plt.show()

    return

# Define the parameters for the tiling
p = 7  # The number of edges on each polygonal tile
q = 3  # The number of polygons meeting at each vertex                # need to be 3 to match ossicle
n = 4 # The number of layers to draw

# Create a hyperbolic tiling object
T = HyperbolicTiling(p, q, n, center = 'vertex')
T.translate(0.14)

#convert polygons to graph
graph = convert_polygons_to_graph(T)

#plot the graph
custom_plot_graph(graph)
