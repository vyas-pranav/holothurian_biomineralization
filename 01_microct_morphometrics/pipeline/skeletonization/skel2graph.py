# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Convert ossicle skeletons into NetworkX graphs with origin and depth levels.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Pipeline step (graphs used by all micro-CT field panels)
What it does  : Builds the full graph of all skeleton points (node attribute: medial
                thickness) and the simple graph of junctions and tips of the closed-loop
                skeleton (edge thickness = mean medial thickness along the branch);
                removes degree-2 nodes, marks the thickest edge as the central edge,
                inserts the origin node at its midpoint and assigns to every node its
                topological depth (hops) and Dijkstra path-length depth from the origin.
                Active: writes origin_clean_simple_G for loc_file_id[171]; the writes of
                the other graphs are commented. Imports skelCodes.post_skel
                (Coral3D-based, not distributed), so it is not runnable as shipped.
Inputs        : microct/animal_1/extracted_ossicles/aligned/{vtk,
                skeleton/<ossicle>/(post15_skel_line.vtk, clean_skel_line.vtk)},
                microct/animal_1/extracted_ossicles/Data
                Calculated/{ossicle_type_list.txt, loc_file_id.txt}
Outputs       : microct/animal_1/extracted_ossicles/aligned/skeleton/<ossicle>/origin_clean_simple_G.gpickle
                (commented: full_G, clean_full_G, simple_G, clean_simple_G,
                clean_simple_mid_G, area_mid_G)
Environment   : environment-analysis.yml (Python 3.7)
Run           : python skel2graph.py
"""

from vedo import *
import vedo
import vtk
import networkx as nx
from skelCodes.post_skel import *
import pyvista as pv

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

pv.global_theme.background = 'white'
pv.global_theme.font.color = 'black'
pv.global_theme.font.family = 'arial'
pv.global_theme.camera = {'position': [0,0,-1],'viewup': [0, 0, -1]}  #[-1, -0.3, 2.2]


#to get a graph version of skeleton vtk polydata object created by sourcing the vedo mesh object
def graph_from_mesh(skeleton):
    full_G = nx.Graph()
    # add the mesh vertices as nodes to the graph
    for i in range(skeleton.npoints):
        point = skeleton.points()[i]
        thick = skeleton.pointdata["MedialThickness"][i]
        full_G.add_node(i, pos=point, med_thick = thick)

    # add the mesh edges as edges to the graph
    for i in range(skeleton.ncells):
        cell = skeleton.lines()[i]
        for jj in range(len(cell)-1):
            full_G.add_edge(cell[jj], cell[jj+1])

    #remove nodes that do not form an edge
    full_G.remove_nodes_from(list(nx.isolates(full_G)))

    return full_G

# to get a graph with only topologically relevant nodes, i.e. more than 2 degree nodes and tips
def simple_graph_from_mesh(skeleton):
    G = nx.Graph()
    # add the mesh vertices as nodes to the graph
    for i in range(skeleton.npoints):
        point = skeleton.points()[i]
        thick = skeleton.pointdata["MedialThickness"][i]
        G.add_node(i, pos=point, med_thick = thick)
        G.add_node(i, pos=point, med_thick = thick, with_struct = 0)

    # add the mesh edges as edges to the graph
    for i in range(skeleton.ncells):
        cell = skeleton.lines()[i]
        G.add_edge(cell[0], cell[-1])

        thick_sum = 0
        for jj in range(len(cell)):
            thick_sum = thick_sum + G.nodes(data='med_thick')[cell[jj]]
        thick = thick_sum/len(cell)
        G[cell[0]][cell[-1]]['med_thick'] = thick   #add edge thickness as the average value of medial thickness along that line

    #remove nodes that do not form an edge
    G.remove_nodes_from(list(nx.isolates(G)))

    return G

# to get a simple graph with all degree 3 or more nodes and also degree 2 nodes that are mid points of segments
def simple_mid_graph_from_mesh(skeleton):
    G = nx.Graph()
    # add the mesh vertices as nodes to the graph
    for i in range(skeleton.npoints):
        point = skeleton.points()[i]
        thick = skeleton.pointdata["MedialThickness"][i]
        G.add_node(i, pos=point, med_thick = thick, with_struct = 0)



    # add the mesh edges as edges to the graph
    for i in range(skeleton.ncells):
        cell = skeleton.lines()[i]

        if len(cell)%2==0:
            mid = len(cell)//2
        else:
            mid = (len(cell)-1)//2

        G.add_edge(cell[0], cell[mid])
        G.add_edge(cell[mid], cell[-1])

        thick_sum = 0
        for jj in range(mid+1):
            thick_sum = thick_sum + G.nodes(data='med_thick')[cell[jj]]
        thick = thick_sum/(mid+1)
        G[cell[0]][cell[mid]]['med_thick'] = thick   #add edge thickness as the average value of medial thickness along that line

        thick_sum = 0
        for jj in range(mid, len(cell)):
            thick_sum = thick_sum + G.nodes(data='med_thick')[cell[jj]]
        thick = thick_sum/(len(cell)-mid)
        G[cell[mid]][cell[-1]]['med_thick'] = thick   #add edge thickness as the average value of medial thickness along that line

        if G.degree[cell[0]]==2 or G.degree[cell[-1]]==2 :
            G.nodes[cell[mid]]['with_struct'] = 0
        else:
            G.nodes[cell[mid]]['with_struct'] = 1

    #remove nodes that do not form an edge

    G.remove_nodes_from(list(nx.isolates(G)))

    return G

#to remove extra points with degree less than 3 in clean graphs
def clean_seg(test_G):
    G = test_G.copy()

    #remove all degree 2 nodes in the middle of the cycles to only keep degree 3 and above nodes
    node_set = set(G.nodes())
    for node in node_set:
        deg = G.degree[node]
        with_struct = G.nodes[node]['with_struct']
        if (deg == 2) and (with_struct == 0):
            neighbors = list(G.neighbors(node))

            G.add_edge(neighbors[0], neighbors[1])
            G[neighbors[0]][neighbors[1]]['branch_degree'] = 1

            len_a = np.linalg.norm(G.nodes[node]['pos'] - G.nodes[neighbors[0]]['pos'])
            len_b = np.linalg.norm(G.nodes[node]['pos'] - G.nodes[neighbors[0]]['pos'])
            thick_a = G[node][neighbors[0]]['med_thick']
            thick_b = G[node][neighbors[1]]['med_thick']
            thick = (len_a*thick_a+len_b*thick_b)/(len_a + len_b)
            G[neighbors[0]][neighbors[1]]['med_thick'] = thick
            # Remove the node from the graph
            G.remove_node(node)

    #label edge with highest average thickness as the starting edge
    # Iterate through the edges
    max_thick = 0
    for u, v, data in G.edges(data=True):
        thick = data['med_thick']
        if thick > max_thick:
            max_thick = thick 
            max_edge = [u,v]

    for u, v, data in G.edges(data=True):
        data['branch_degree'] = 1
    G[max_edge[0]][max_edge[1]]['branch_degree'] = 0



    return G


def add_origin_node(test_G, id):
    #add the mid point as a new node representing the starting point
    G = test_G.copy()
    for u, v, data in G.edges(data=True):
        if (data['branch_degree']==0):
            start_edge = [u,v]
            break

    start_pt = (G.nodes[start_edge[0]]['pos'] + G.nodes[start_edge[1]]['pos'])/2 
    node_id = id #max(list(G.nodes()))+1 need to get max from the full_G and not the clean_G
    thick =  G[start_edge[0]][start_edge[1]]['med_thick']
    G.add_node(node_id, pos=start_pt, med_thick = thick, with_struct = 0)
    G.add_edge(node_id, start_edge[0])
    G.add_edge(node_id, start_edge[1])
    G[node_id][start_edge[0]]['med_thick'] = thick
    G[node_id][start_edge[1]]['med_thick'] = thick
    G[node_id][start_edge[0]]['branch_degree'] = 1
    G[node_id][start_edge[1]]['branch_degree'] = 1

    G.remove_edge(start_edge[0], start_edge[1])

    node_set = set(G.nodes())
    for node in node_set:
        G.nodes[node]['level'] = 1

    G.nodes[node_id]['level'] = 0

    return G

#works only with graphs on which origin has been added
def add_node_edge_levels(test_G):
    G = test_G.copy()

    #add lengths to edges as attribute to be used for djikstra path length calculation 
    for u, v in G.edges:
        p1 = G.nodes[u]['pos']
        p2 = G.nodes[v]['pos']
        length = np.linalg.norm(p1-p2)
        G[u][v]['length'] = length

    for node, data in G.nodes(data=True):
        if (data['level']==0):
            source_node= node
            break

    distances = nx.single_source_shortest_path_length(G, source_node)
    distances2 = nx.single_source_dijkstra_path_length(G, source_node, weight = 'length')
    nx.set_node_attributes(G, distances, 'level_topo')
    nx.set_node_attributes(G, distances2, 'level_dist')

    #placeholders to be used in 
    nx.set_node_attributes(G, distances2, 'level_bound_topo')      #topological level with boundaries to be marked while base extraction
    nx.set_node_attributes(G, distances2, 'level_bound_dist')      #distance based level with boundaries to be marked while base extraction

    #set level number for boundary nodes to be a very high number when using base extraction from the graph
    #can't perform that boundary allocation easily here as the vertical spikes (vertical ossicle growth) prevent clean identification

    # name edge level as the lower value of both the levels of participating nodes.
    for u, v in G.edges:
        l1 = G.nodes[u]['level_bound_topo']
        l2 = G.nodes[v]['level_bound_topo']
        level = min([l1,l2])
        G[u][v]['level_bound_topo'] = level



    return G

###########################def add_node_degree def add_edge_degree based on the distance from the origin



def graph_to_vedo_mesh(H):
    #reorder node ids so that the nodes now are numbered as consecutive integers
    G = nx.convert_node_labels_to_integers(H)

    # Get the node positions from the graph
    # pos = nx.get_node_attributes(G, 'pos')

    pos_list= []
    for node in G.nodes():
        pos_value = G.nodes[node]['pos']
        pos_list.append((pos_value[0],pos_value[1],pos_value[2]))


    # Extract the edges from the graph
    edge_list = list(G.edges())

    gmesh = vedo.Mesh([pos_list, edge_list])

    return gmesh

base_dir =str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles")
# type_dir = base_dir
# indir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "vtk")
# skeldir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "skeleton")

type_dir =str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned")
indir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned" / "vtk")
skeldir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned" / "skeleton")
datadir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "Data Calculated")


plt3d = Plotter(bg2='gray2')#, interactive=True) # screen size
pl = pv.Plotter(off_screen=False)

all_file_names = os.listdir(indir)
file_names = [f for f in all_file_names if f.endswith('.vtk')]
file_names.sort()
files = [os.path.join(indir,f) for f in file_names]
skel_paths = [os.path.join(skeldir,f) for f in file_names]

with open(os.path.join(datadir,'ossicle_type_list.txt')) as txtfile:
    ossicle_type_list = list(map(int, txtfile.read().split('\n')))

with open(os.path.join(datadir,'loc_file_id.txt')) as txtfile:
    loc_file_id = list(map(int, txtfile.read().split('\n')))

n_files = len(file_names)
for jj in [171]:  #list(range(31))+list(range(32,213)): #range(n_files): #[5,15,25,35,45,55,65,75,85,95,105,115,125,135,145,155]:  # range(n_files):  #
    ii = loc_file_id[jj]

    print(jj, ii, files[ii])
    # if files[ii][-6:] != '32.vtk' : 
    ossicle = load(files[ii])
    # ossicle.lw(0.05).c('pink6').alpha(0.5)
    skel_path = os.path.join(skel_paths[ii],"post15_skel_line.vtk")
    cl_skel_path = os.path.join(skel_paths[ii],"clean_skel_line.vtk")
    skeleton = load(skel_path)
    cl_skeleton = load(cl_skel_path)

    #write skeleton only if there are non-zero number of lines in the skeleton
    if len(skeleton.lines())!=0 and len(cl_skeleton.lines())!=0:
        print('yes')
        full_G = graph_from_mesh(skeleton)
        max_node_id = max(list(full_G.nodes()))
        # clean_full_G = graph_from_mesh(cl_skeleton)
        # simple_G = simple_graph_from_mesh(skeleton)
        clean_simple_G = clean_seg(simple_graph_from_mesh(cl_skeleton))
        # clean_simple_mid_G = clean_seg(simple_mid_graph_from_mesh(cl_skeleton))
        # area_mid_G = simple_mid_graph_from_mesh(cl_skeleton)
        origin_clean_simple_G = add_origin_node(clean_simple_G, max_node_id+1)
        origin_clean_simple_G = add_node_edge_levels(origin_clean_simple_G)

        # nx.write_gpickle(full_G, os.path.join(skel_paths[ii],"full_G.gpickle"))
        # nx.write_gpickle(clean_full_G, os.path.join(    skel_paths[ii],"clean_full_G.gpickle"))
        # nx.write_gpickle(simple_G, os.path.join(skel_paths[ii],"simple_G.gpickle")) 
        # nx.write_gpickle(clean_simple_G, os.path.join(skel_paths[ii],"clean_simple_G.gpickle"))
        # nx.write_gpickle(clean_simple_mid_G, os.path.join(skel_paths[ii],"clean_simple_mid_G.gpickle"))
        # nx.write_gpickle(area_mid_G, os.path.join(skel_paths[ii],"area_mid_G.gpickle"))
        nx.write_gpickle(origin_clean_simple_G, os.path.join(skel_paths[ii],"origin_clean_simple_G.gpickle"))

        print('graphs saved')
