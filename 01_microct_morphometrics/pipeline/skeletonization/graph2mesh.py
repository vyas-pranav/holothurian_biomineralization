# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Show an ossicle skeleton graph as a mesh (inspection utility).

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : None directly (inspection utility for the skeleton graphs)
What it does  : Defines helpers that turn a NetworkX graph with node positions into a
                vedo or PyVista line mesh, then loads the first unaligned ossicle with
                its skeleton, voxel volume and clean_simple_G graph and shows them in a
                vedo window.
Inputs        : microct/animal_1/extracted_ossicles/{vtk,
                skeleton/<ossicle>/(post_skel_line.vtk, vox.mhd, full_G, clean_full_G,
                simple_G, clean_simple_G .gpickle)}
Outputs       : interactive vedo window only
Environment   : environment-analysis.yml (Python 3.7)
Run           : python graph2mesh.py
"""

from vedo import *
import vedo
import pyvista as pv
import networkx as nx
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


def graph_to_vedo_mesh(H):
    #reorder node ids so that the nodes now are numbered as consecutive integers
    G = nx.convert_node_labels_to_integers(H)

    # Get the node positions from the graph
    pos = nx.get_node_attributes(G, 'pos')

    pos_list= []
    for node in G.nodes():
        pos_value = G.nodes[node]['pos']
        pos_list.append((pos_value[0],pos_value[1],pos_value[2]))


    # Extract the edges from the graph
    edge_list = list(G.edges())

    gmesh = vedo.Mesh([pos_list, edge_list])

    return gmesh


def graph_to_pv_mesh(H):

    G = nx.convert_node_labels_to_integers(H)

    pos_list= []
    for node in G.nodes():
        pos_value = G.nodes[node]['pos']
        pos_list.append((pos_value[0],pos_value[1],pos_value[2]))

    nodes = np.array(pos_list)

    # Extract the edges from the graph
    edge_list = list(G.edges())
    edges = np.array(edge_list)

    padding = np.empty(edges.shape[0], int) * 2
    padding[:] = 2
    edges_w_padding = np.vstack((padding, edges.T)).T

    gmesh = pv.PolyData(nodes, edges_w_padding)

    return gmesh


#alternative method is to plot cylinders and spheres at points and segments
# # create a vedo mesh for each node
# for node_id, pos in positions.items():                
#     if node_id == 1363 :
#         mesh = vedo.pointcloud.Point(r=100).pos(pos).color('green')#.alpha(0.5)
#     else:
#         mesh = vedo.pointcloud.Point(r=20).pos(pos).color('red')#.alpha(0.5)
#     meshes.append(mesh)

# # create a vedo mesh for each edge
# for u, v in G.edges():
#     mesh = vedo.Line([positions[u], positions[v]], res = 2, lw = 10).color('k').alpha(0.5)
#     meshes.append(mesh)

# # use points and lines to get scale invariant shapes, but use spheres and cylinders with you need fixed sizes of markers


indir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "vtk")
skeldir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "skeleton")

plt3d = Plotter(bg2='gray2')#, interactive=True) # screen size
all_file_names = os.listdir(indir)
file_names = [f for f in all_file_names if f.endswith('.vtk')]
file_names.sort()
files = [os.path.join(indir,f) for f in file_names]
skel_paths = [os.path.join(skeldir,f) for f in file_names]

n_files = len(file_names)

for ii in range(0,1): # n_files):
    print(files[ii])
    if files[ii][-6:] != '32.vtk' : 
        ossicle = load(files[ii])
        # ossicle.lw(0.05).c('pink6').alpha(0.5)
        skel_path = os.path.join(skel_paths[ii],"post_skel_line.vtk")
        skeleton = load(skel_path)
        vox_path = os.path.join(skel_paths[ii],"vox.mhd")
        voxel_vol = load(vox_path)

        #read skeleton only if there are non zero number of lines in the skeleton
        if len(skeleton.lines())!=0:
            full_G = nx.read_gpickle(os.path.join(skel_paths[ii],"full_G.gpickle"))
            clean_full_G = nx.read_gpickle(os.path.join(skel_paths[ii],"clean_full_G.gpickle"))
            simple_G = nx.read_gpickle(os.path.join(skel_paths[ii],"simple_G.gpickle")) 
            clean_simple_G = nx.read_gpickle(os.path.join(skel_paths[ii],"clean_simple_G.gpickle")) 

        # Create a random 3D graph using networkx
        G = clean_simple_G

        # Get the mesh from the graph
        gmesh = graph_to_vedo_mesh(G)
        gmesh.lw(5)
        gmesh2 = gmesh.clone()
        gmesh2.ps(20).c('r')

        # Show the plot
        plt3d = vedo.Plotter(bg2='gray')#, interactive=True) # screen size
        plt3d.add(gmesh)
        plt3d.add(gmesh2)
        plt3d.add(skeleton)
        plt3d.add(ossicle.alpha(0.5))
        # plt3d.add(voxel_vol.alpha(10))
        plt3d.show()
