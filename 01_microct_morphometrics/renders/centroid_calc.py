# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Ossicle centroids and origins.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Input of the whole-animal renders and the field scripts
What it does  : For every ossicle (unaligned data) stores the centre of the voxel volume
                and the origin (midpoint of the thickest central edge, [0, 0] if the
                ossicle has no skeleton lines).
Inputs        : microct/animal_1/extracted_ossicles/loc_file_id.txt,
                microct/animal_1/extracted_ossicles/{vtk, skeleton/<ossicle>/(vox.mhd,
                *skel_line.vtk, *_G.gpickle)}
Outputs       : centroids.npy, start_points.npy in the working directory (copied to Data
                Calculated/)
Environment   : environment-analysis.yml (Python 3.7)
Run           : python centroid_calc.py
"""

from vedo import *
import vedo
import vtk
import networkx as nx

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


def load_data(ii):
    ossicle = load(files[ii])
    # ossicle.lw(0.05).c('pink6').alpha(0.5)
    vox_path = os.path.join(skel_paths[ii],"vox.mhd")
    voxel_vol = load(vox_path)

    skel_path = os.path.join(skel_paths[ii],"post15_skel_line.vtk")
    cl_skel_path = os.path.join(skel_paths[ii],"clean_skel_line.vtk")
    skeleton = load(skel_path)
    cl_skeleton = load(cl_skel_path)

    return ossicle, voxel_vol, skeleton, cl_skeleton

def load_graphs(ii):
    full_G = nx.read_gpickle(os.path.join(skel_paths[ii],"full_G.gpickle"))
    clean_full_G = nx.read_gpickle(os.path.join(skel_paths[ii],"clean_full_G.gpickle"))
    simple_G = nx.read_gpickle(os.path.join(skel_paths[ii],"simple_G.gpickle")) 
    clean_simple_G = nx.read_gpickle(os.path.join(skel_paths[ii],"clean_simple_G.gpickle")) 

    return full_G, clean_full_G, simple_G, clean_simple_G

type_dir =str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles")
indir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "vtk")
skeldir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "skeleton")

plt3d = Plotter(bg2='gray2')#, interactive=True) # screen size
all_file_names = os.listdir(indir)
file_names = [f for f in all_file_names if f.endswith('.vtk')]
file_names.sort()
files = [os.path.join(indir,f) for f in file_names]
skel_paths = [os.path.join(skeldir,f) for f in file_names]

plt3d = vedo.Plotter(bg2='gray')#, interactive=True) # screen size
# plt3d.show()

n_files = len(file_names)

with open(os.path.join(type_dir,'loc_file_id.txt')) as txtfile:
    loc_file_id = list(map(int, txtfile.read().split('\n')))

meshes = []
centroid_list = []
start_list = []
for ii in loc_file_id: #n_files): #loc_file_id[0:10]: #range(n_files): #[5,15,25,35,45,55,65,75,85,95,105,115,125,135,145,155]:  # range(n_files):  #
    print(ii)
    ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)

    vol_centroid = voxel_vol.center()
    centroid_list.append(vol_centroid)

    #write skeleton only if there are non-zero number of lines in the skeleton
    if len(skeleton.lines())!=0 and len(cl_skeleton.lines())!=0:
        full_G, clean_full_G, simple_G, clean_simple_G = load_graphs(ii)

        G = clean_simple_G

        for u, v, data in G.edges(data=True):
            if (data['branch_degree']==0):
                start_edge = [u,v]


        start_pt = (G.nodes[start_edge[0]]['pos'] + G.nodes[start_edge[1]]['pos'])/2 
        start_list.append(start_pt)

    else:
        start_list.append(np.array([0, 0]))


with open('centroids.npy', 'wb') as ff:
    np.save(ff, np.array(centroid_list))

with open('start_points.npy', 'wb') as gg:
    np.save(gg, np.array(start_list))

#     start_pt_mesh = vedo.Sphere(start_pt, r = 0.001, c = 'k')
#     centroid_mesh = vedo.Sphere(vol_centroid,  r = 0.001, c = 'b')

#     # Show the plot

#     meshes.append(skeleton)
#     meshes.append(ossicle.alpha(0.5))

#     label = Text3D(file_names[ii], 0.01+vol_centroid, s=0.005, depth=0.5, c="g")
#     label.follow_camera() 
#     meshes.append(label)
#     # plt3d.add(voxel_vol.alpha(10))
#     meshes.append(start_pt_mesh)
#     meshes.append(centroid_mesh)

#     # Get the mesh from the graph
#     gmesh = graph_to_vedo_mesh(G)
#     gmesh.lw(5)
#     gmesh2 = gmesh.clone()
#     gmesh2.ps(5).c('r')
#     meshes.append(gmesh)
#     meshes.append(gmesh2)

# vedo.show(meshes, axes = 1)

# plt3d.interactive()
