# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Hexagonal lattice cropped to a disc, drawn and saved as a NetworkX graph.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 2J ("Symmetric ordered lattice", "Radial asymmetry")
What it does  : Tiles hexagons of the given radius, clips every hexagon edge to
                a disc of radius `circle_radius` (shapely), turns the clipped
                edges into a graph, keeps the largest connected component and
                draws it in the ossicle-graph style (purple struts, orange
                nodes, black origin marker at (0, 3.45)).
Inputs        : none
Outputs       : OUT_ROOT/07_lattice_schematics/graphs_extracted/<timestamp>/
                hexagonal_lattice_<size>_<radius>_<circle_radius>.gpickle.
                The figure panels were saved by enabling the commented
                `plt.savefig("hex1.png", dpi=600)` line below.
Environment   : environment-analysis.yml (Python 3.7)
Run           : python hexagonal_lattice2graph.py
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon
import networkx as nx
import itertools
import datetime
import os

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

#%%
graph_save_loc = str(OUT_ROOT / "07_lattice_schematics" / "graphs_extracted")

#%%
def create_hexagonal_lattice(size, radius):
    lattice_points = []
    center = (radius*3/2, 0)
    lattice_points.append(center)
    for m in range(-size, size + 1):
        for n in range(-size + 1, size):
            x = center[0] + radius * (3/2 * n)
            y = center[1] + radius * (np.sqrt(3) * m + np.sqrt(3)/2 * n)
            if np.sqrt((x-center[0])**2 + (y-center[1])**2) <= radius * size:
                lattice_points.append((x, y))
    return lattice_points

def hexagon_corners(center, radius):
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    return [(center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)) for angle in angles]

def merge_overlapping_nodes(graph, tolerance=1e-5):
    G = graph.copy()
    H = graph.copy()
    nodes_to_remove = set()
    for node1, node2 in itertools.combinations(G.nodes, 2):
        pos1 = graph.nodes[node1]['pos']
        pos2 = graph.nodes[node2]['pos']
        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        if distance < tolerance:
            nodes_to_remove.add(node2)
            # Merge edges of the overlapping nodes
            for neighbor in G.neighbors(node2):
                if (neighbor != node1) and (neighbor != node2) :
                    H.add_edge(node1, neighbor)
    # Remove overlapping nodes
    for node in nodes_to_remove:
        H.remove_node(node)

    return H

# function to find distance between two points as tuples
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# function to find node at a given position
def find_node_at_position(graph, position, tolerance=1e-5):
    for node, data in graph.nodes(data=True):
        if distance(data['pos'], position)<tolerance:
            return node
    return None

def plot_hexagonal_lattice(lattice_points, radius, circle_radius, fig, ax):
    ax.cla()
    ax.set_aspect('equal', 'box')
    ax.axis('off')

    circle = Point(0, 0).buffer(circle_radius)
    graph = nx.Graph()

    node_ctr = 1
    for center in lattice_points:
        corners = hexagon_corners(center, radius)
        for i in range(len(corners)):
            line = LineString([corners[i], corners[(i + 1) % len(corners)]])
            intersection = line.intersection(circle)
            intersection_pt = line.intersection(circle.boundary)
            if intersection.is_empty:
                continue

            if isinstance(intersection, LineString):
                x, y = intersection.xy

                pos1 = (x[0], y[0])
                pos2 = (x[1], y[1])

                found_node_id1 = find_node_at_position(graph, pos1)
                if found_node_id1 is None:
                    graph.add_node(node_ctr, pos=pos1)
                    found_node_id1 = node_ctr
                    node_ctr += 1

                found_node_id2 = find_node_at_position(graph, pos2)
                if found_node_id2 is None:
                    graph.add_node(node_ctr, pos=pos2)
                    found_node_id2 = node_ctr
                    node_ctr += 1

                #now create an edge between the above defined two nodes
                graph.add_edge(found_node_id1, found_node_id2)

            elif isinstance(intersection, (Point, Polygon)):
                x, y = intersection.buffer(0.001).boundary.xy

                pos1 = (x[0], y[0])
                pos2 = (x[1], y[1])

                found_node_id1 = find_node_at_position(graph, pos1)
                if found_node_id1 is None:
                    graph.add_node(node_ctr, pos=pos1)
                    found_node_id1 = node_ctr
                    node_ctr += 1

                found_node_id2 = find_node_at_position(graph, pos2)
                if found_node_id2 is None:
                    graph.add_node(node_ctr, pos=pos2)
                    found_node_id2 = node_ctr
                    node_ctr += 1

                #now create an edge between the above defined two nodes
                graph.add_edge(found_node_id1, found_node_id2)

            else:
                continue


    # merge overlapping nodes - don't need it if we are cleanly adding new nodes
    # graph = merge_overlapping_nodes(graph, tolerance = 1e-3)

    #remove disconnected components in the graph and only keep the largest one
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc).copy()

    node_wise_neighbors = {node: list(graph.neighbors(node)) for node in graph.nodes}

    nx.draw_networkx_edges(graph, nx.get_node_attributes(graph, 'pos'), width = 10, edge_color = 'purple', ax=ax)

    nx.draw_networkx_nodes(graph, nx.get_node_attributes(graph, 'pos'), node_size=150, node_color='orange', ax=ax)

    ax.scatter([0], [3.45], color='black', s = 300)

    max_coord = 11
    ax.set_xlim(-max_coord, max_coord)
    ax.set_ylim(-max_coord, max_coord)

    #draw the cropping circle
    circle = plt.Circle((0, 0), circle_radius, color='black', fill=False, linewidth=10)
    # ax.add_artist(circle)

    return graph

size = 20 #number of layers of lattice
radius = 1.51 #size of each hexagon
circle_radius = 10  #size of cropping circle
n_plts = 10


# Get the current date and time
now = datetime.datetime.now()
# Format the timestamp as a string
folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
# Create a folder with the timestamp
dir_loc = os.path.join(graph_save_loc, folder_name)
os.makedirs(dir_loc)

radii = [2.001] #[1.6, 1.8, 2.001, 2.3, 2.7, 3.3, 3.9, 5.1, 6.1, 8.1]
# radii = [1.6, 1.8, 2.1, 2.5, 3.00001, 3.6, 4.3, 5.1, 6.1, 8.1]

fig, ax = plt.subplots(figsize=(11, 11))

ctr = 0
for radius in radii:
    lattice_points = create_hexagonal_lattice(size, radius)
    lattice_graph = plot_hexagonal_lattice(lattice_points, radius, circle_radius, fig, ax)


    file_loc = os.path.join(dir_loc, "hexagonal_lattice_"+str(size)+"_"+str(radius)+"_"+str(circle_radius)+".gpickle")
    nx.write_gpickle(lattice_graph, file_loc)

    plt.pause(0.1)

    # plt.savefig(file_loc[:-7]+".png", dpi=600)
    # plt.savefig("hex1.png", dpi=600)
    ctr += 1

plt.show()
