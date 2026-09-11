# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Disordered polygonal lattice schematic.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 2J ("Polygonal disorder")
What it does  : Jitters a 10 x 10 hexagonal point lattice, builds its Voronoi
                diagram and draws the finite Voronoi edges in the ossicle-graph
                style (purple struts, orange nodes). The jitter is random, so
                every run gives a different lattice.
Inputs        : none
Outputs       : disordered_lattice.png in the current working directory (600 dpi)
Environment   : environment-analysis.yml (Python 3.7)
Run           : python disorder_graph.py
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial

def create_hexagonal_lattice(rows, cols, jitter=1):
    points = []
    for row in range(rows):
        for col in range(cols):
            x = col * 3/2
            y = np.sqrt(3) * (row + 0.5 * (col % 2))
            points.append((x, y))
    points = np.array(points)

    # Introduce disorder by adding jitter
    jittered_points = points + 2*np.random.uniform(-jitter, jitter, points.shape)

    return jittered_points/10



def calculate_voronoi_regions(points):
    # Create the Voronoi diagram for the given points
    vor = scipy.spatial.Voronoi(points)

    # Store Voronoi vertices and regions
    voronoi_nodes = vor.vertices
    voronoi_edges = []

    for point_idx, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if -1 not in vertices:  # Ignore regions that go to infinity
            # Append the edges that form the Voronoi cell
            voronoi_edges.extend([(vertices[i], vertices[(i+1) % len(vertices)]) for i in range(len(vertices))])

    # Remove duplicates from voronoi_edges
    voronoi_edges = list(set(voronoi_edges))

    return voronoi_nodes, voronoi_edges

# Parameters for lattice generation
rows, cols = 10, 10
jitter_amount = 0.3

# Generate hexagonal lattice
hex_points = create_hexagonal_lattice(rows, cols, jitter=jitter_amount)

# Calculate Voronoi regions
voronoi_nodes, voronoi_edges = calculate_voronoi_regions(hex_points)

# Plot lattice
fig, ax = plt.subplots(figsize=(20, 20))
for edge in voronoi_edges:
    p1, p2 = voronoi_nodes[edge[0]], voronoi_nodes[edge[1]]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth = 13*1.75, color='purple')

ax.scatter(voronoi_nodes[:, 0], voronoi_nodes[:, 1], marker = 'o', color='orange', s = 500, zorder=2, linewidths=0)
# ax.scatter([0.5], [0.5], color='black', s = 700, zorder=3)

# Set aspect of the plot to be equal
ax.set_aspect('equal')

# Remove the axes for a cleaner look
plt.axis('off')

#set the limits of the plot
plt.xlim(0, 1)
plt.ylim(0, 1)

# Save the figure
plt.savefig('disordered_lattice.png', dpi=600)

plt.show()
