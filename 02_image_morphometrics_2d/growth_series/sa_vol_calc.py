# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Volume and surface-area growth of an ossicle from its lattice graphs.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 3H
What it does  : Treats every edge of each full pixel graph as a cylinder (diameter =
                medial thickness), sums volumes and lateral surface areas, and plots both
                against time (0, 6.04, 27.82, 44.68, 96.58 h, paired with the graphs in
                sorted file-name order) with a least-squares line (slope and its standard
                error are printed).
Inputs        : DATA_ROOT/growth_series/graphs/*graph_full.gpickle (outputs of bin2skel.py)
Outputs       : OUT_ROOT/02_image_morphometrics_2d/growth_series/volume_increase.{png,svg},
                surface_area_increase.{png,svg}; graphs/volume_list, graphs/surface_area_list (pickles)
Environment   : environment-analysis.yml (Python 3.7)
Run           : python sa_vol_calc.py
"""

import networkx as nx
import pyvista as pv
import numpy as np
import pyacvd
import os
import pygmsh  # pip install pygmsh
import pickle
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
# output folder of this script (it already existed in the original working folder)
(OUT_ROOT / "02_image_morphometrics_2d" / "growth_series" / "graphs").mkdir(parents=True, exist_ok=True)
# ----------------------------------------------------------------------------


#%%

def integrate_volume(graph):
    #iterate through each edge and add the volume of the cylinder to the total volume
    total_volume = 0
    for edge in graph.edges():
        #get the radius of the edge
        radius = graph.edges[edge]["med_thick"]/2
        #get the length of the edge
        start_point = graph.nodes[edge[0]]["pos"]
        end_point = graph.nodes[edge[1]]["pos"]
        direction_cyl = np.array(end_point) - np.array(start_point)
        length = np.linalg.norm(direction_cyl)
        #calculate the volume of the cylinder
        volume = np.pi * radius**2 * length
        #add the volume of the cylinder to the total volume
        total_volume += volume

    return total_volume

def integrate_surface_area(graph):
    #iterate through each edge and add the surface area of the cylinder to the total surface area
    total_surface_area = 0
    for edge in graph.edges():
        #get the radius of the edge
        radius = graph.edges[edge]["med_thick"]/2
        #get the length of the edge
        start_point = graph.nodes[edge[0]]["pos"]
        end_point = graph.nodes[edge[1]]["pos"]
        direction_cyl = np.array(end_point) - np.array(start_point)
        length = np.linalg.norm(direction_cyl)
        #calculate the surface area of the cylinder
        surface_area = 2 * np.pi * radius * length
        #add the surface area of the cylinder to the total surface area
        total_surface_area += surface_area

    return total_surface_area


#%%


graph_dir = str(DATA_ROOT / "growth_series" / "graphs")

#get file names of all graphs with 'simple_oirgin' some where in the name
graph_file_names = [f for f in sorted(os.listdir(graph_dir), key=str.upper) if 'full' in f]
print(graph_file_names)

#list of all graphs loaded as networkx graphs
all_graphs = []
for i in range(len(graph_file_names)):
    file_name = os.path.join(graph_dir, graph_file_names[i])
    with open(file_name, 'rb') as f:
        graph = pickle.load(f)

    all_graphs.append(graph)

#iterate through all graphs in graph_list and calculate the volume for each
volume_list = []
for graph in all_graphs:
    volume = integrate_volume(graph)
    print(volume)
    volume_list.append(volume)

#iterate through all graphs in graph_list and calculate the surface area for each
surface_area_list = []
for graph in all_graphs:
    surface_area = integrate_surface_area(graph)
    print(surface_area)
    surface_area_list.append(surface_area)

#save the volume list
save_dir = str(OUT_ROOT / "02_image_morphometrics_2d" / "growth_series" / "graphs")
save_name = 'volume_list'
with open(os.path.join(save_dir, save_name), 'wb') as f:
    pickle.dump(volume_list, f)

#save the surface area list
save_dir = str(OUT_ROOT / "02_image_morphometrics_2d" / "growth_series" / "graphs")
save_name = 'surface_area_list'
with open(os.path.join(save_dir, save_name), 'wb') as f:
    pickle.dump(surface_area_list, f)


time_points = [0, 6.04, 27.82, 44.68, 96.58]

#plot the volume increase
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 6))

ax.plot(time_points, volume_list, color='black', marker='o', linestyle='dashed', linewidth=4, markersize=20)
ax.scatter(time_points, volume_list, color='black', marker='o', s=100)
ax.set_xlabel('Time(hrs)')
ax.set_ylabel(r'Segmented volume ($mm^3$)')

#set tick label font size
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(20)

#set axes label font size
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)


#fit a line to the data
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, volume_list)
print('slope: ', slope)
print('intercept: ', intercept)
print('r_value: ', r_value)
print('p_value: ', p_value)
print('std_err: ', std_err)

#plot the line
x = np.linspace(0, 100, 100)
y = slope * x + intercept
ax.plot(x, y, color='red', linestyle='dashed', linewidth=3, alpha=0.5)

#add text to the plot
# text = 'slope = ' + str(np.round(slope,10)) + r'$mm^3/hr$'
# ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=20,
#         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))


#save
save_dir = str(OUT_ROOT / "02_image_morphometrics_2d" / "growth_series")
save_name = 'volume_increase'
plt.savefig(os.path.join(save_dir, save_name)+'.png', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(save_dir, save_name)+'.svg', dpi=300, bbox_inches='tight')

plt.show()


#plot the surface area increase
fig, ax = plt.subplots(figsize=(4, 6))

ax.plot(time_points, surface_area_list, color='black', marker='o', linestyle='dashed', linewidth=4, markersize=20)
ax.scatter(time_points, surface_area_list, color='black', marker='o', s=100)
ax.set_xlabel('Time(hrs)')
ax.set_ylabel(r'Segmented surface area ($mm^2$)')
#set tick label font size
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(20)

#set axes label font size
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)

#fit a line to the data
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, surface_area_list)
print('slope: ', slope)
print('intercept: ', intercept)
print('r_value: ', r_value)
print('p_value: ', p_value)
print('std_err: ', std_err)

#plot the line
x = np.linspace(0, 100, 100)
y = slope * x + intercept
ax.plot(x, y, color='red', linestyle='dashed', linewidth=3, alpha=0.5)

#add text to the plot
# text = 'slope = ' + str(np.round(slope,10)) + r'$mm^2/hr$'
# ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=20,
#         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

#save
save_dir = str(OUT_ROOT / "02_image_morphometrics_2d" / "growth_series")
save_name = 'surface_area_increase'
plt.savefig(os.path.join(save_dir, save_name)+'.png', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(save_dir, save_name)+'.svg', dpi=300, bbox_inches='tight')

plt.show()


