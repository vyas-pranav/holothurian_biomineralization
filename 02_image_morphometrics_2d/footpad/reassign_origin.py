# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Re-assign the origin node of a foot-pad lattice graph.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 2I; Fig. S3B–D; foot-pad series in Fig. 2D–H
What it does  : Loads <name>graph_simple.gpickle (ctr = 0 -> test1, 1 -> test2) and places
                the origin node at the midpoint of a hand-picked edge (changed_edges[ctr])
                instead of the thickest edge, then recomputes node/edge levels and the
                boundary nodes. select_node() is an optional interactive helper (not called).
Inputs        : DATA_ROOT/footpad/graphs/test1graph_simple.gpickle, test2graph_simple.gpickle
                (outputs of bin2skel2.py)
Outputs       : DATA_ROOT/footpad/graphs/<input file name>graph_simple_origin.gpickle
Environment   : environment-analysis.yml (Python 3.7)
Run           : python reassign_origin.py   (edit ctr: 0 or 1)
"""

import numpy as np
import networkx as nx
import sknw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import sys
import math
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


def read_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder), key=str.upper):
        img = cv2.imread(os.path.join(folder,filename),0)
        if img is not None:
            images.append(img)
    return images


#works only with graphs on which origin has been added
def add_node_edge_levels(test_G):
    G = test_G.copy()
    
    #add lengths to edges as attribute to be used for djikstra path length calculation 
    for u, v in G.edges:
        p1 = G.nodes[u]['pos']
        p2 = G.nodes[v]['pos']
        length = np.linalg.norm(p1-p2)
        G[u][v]['length'] = length
    
    #make source node or origin as the node with maximum node id value
    source_node = max(G.nodes())
    
    distances = nx.single_source_shortest_path_length(G, source_node)
    distances2 = nx.single_source_dijkstra_path_length(G, source_node, weight = 'length')
    nx.set_node_attributes(G, distances, 'level_topo')
    nx.set_node_attributes(G, distances2, 'level_dist')

    #placeholders to be used in 
    nx.set_node_attributes(G, distances, 'level_bound_topo')      #topological level with boundaries to be marked while base extraction
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


#get minimum cycle basis to get the polygons composing the graph
def get_min_cycle_basis(G):
    min_cycle_basis = nx.minimum_cycle_basis(G)
    
    min_cycle_list = []
    for cycle in min_cycle_basis:
        if len(cycle) >=3:
            min_cycle_list.append(cycle)
        
    return min_cycle_list

#get all edges in the cycle that are present in the graph
def order_cycle_edges(G, cycle):
    cycle_edges = []

    #cycle through all pairs of nodes in the cycle and add the edge to the list if it is present in the graph
    node_pair_list = []
    #get all pairs of nodes in the cycle
    for ii in range(len(cycle)):
        for jj in range(ii+1, len(cycle)):
            node_pair_list.append([cycle[ii], cycle[jj]])

    #check if the edge is present in the graph and add it to the list
    for node_pair in node_pair_list:
        if G.has_edge(node_pair[0], node_pair[1]) or G.has_edge(node_pair[1], node_pair[0]):
            cycle_edges.append([node_pair[0], node_pair[1]])
        
    return cycle_edges


#get the list of edges of all cycles in min cycle basis
def get_min_cycle_edges(G, min_cycle_list):
    min_cycle_edges = []

    for cycle in min_cycle_list:
        cycle_edges = order_cycle_edges(G, cycle)
        min_cycle_edges.append(cycle_edges)

    return min_cycle_edges


#function to get the area of a polygon projected on the xy plane
def get_cycle_proj_area(H, cycle):
    pos_list = []
    for node in cycle:
        pos = H.nodes[node]['pos']
        pos_list.append([pos[0],pos[1],pos[2]])

    if len(pos_list) == 0:
        area = 0
    else:
        #find the area of the polygon
        pos_array = np.array(pos_list)
        # print(pos_array)
        pos_array = pos_array[:,0:2]

        #find area of the polygon
        area = 0.5*np.abs(np.dot(pos_array[:,0],np.roll(pos_array[:,1],1))-np.dot(pos_array[:,1],np.roll(pos_array[:,0],1)))

    return area

def order_points_by_angle(positions):
        origin = np.mean(np.array(positions), axis = 0)
        angles = []
        
        ctr = 0
        for point in positions:
            angle = math.atan2(point[1] - origin[1], point[0] - origin[0])
            angles.append(angle)
            ctr+=1
        # ordered_points.sort()
        
        sort_ind = np.argsort(angles)
        sort_pts = np.array(positions)[sort_ind].tolist()
        # sorted_nodes = nodes[sort_ind]
        return sort_ind, sort_pts

#compare areas of all the cycles in the min cycle basis and the boundary cycle and return the one with the largest area
def get_real_boundary_nodes_edges(H, min_cycle_list, boundary_nodes):
    cycle_pos_list = []
    for cycle in min_cycle_list:
        cycle_pos = []
        for node in cycle:
            cycle_pos.append(H.nodes[node]['pos'])
        cycle_pos_list.append(cycle_pos)

    #order points in all cycles incluing the boundary cycle based on the angle they subtend
    ordered_min_cycle_list = []
    ctr = 0
    for cycle in min_cycle_list:
        sort_ind, sort_pts = order_points_by_angle(cycle_pos_list[ctr])
        sorted_cycle = np.array(cycle)[sort_ind].tolist()
        ordered_min_cycle_list.append(sorted_cycle)
        ctr+=1

    boundary_pos = []
    for node in boundary_nodes:
        boundary_pos.append(H.nodes[node]['pos'])
    sort_ind, sort_pts = order_points_by_angle(boundary_pos)
    sorted_boundary_nodes = np.array(boundary_nodes)[sort_ind].tolist()
    ordered_min_cycle_list.append(sorted_boundary_nodes)

    #find the area of all the cycles
    area_list = []
    for cycle in ordered_min_cycle_list:
        area = get_cycle_proj_area(H, cycle)
        area_list.append(area)

    #find the cycle with the largest area
    max_area = max(area_list)
    max_area_ind = area_list.index(max_area)

    #find the nodes and edges of the cycle with the largest area
    real_boundary_nodes = ordered_min_cycle_list[max_area_ind]      

    return real_boundary_nodes


#assign boundary nodes and edges
def assign_boundary_nodes_edges2(H):
    G = H.copy()
    node_list = list(G.nodes())
    edge_list = list(G.edges())
    print("node edges done")

    #find boundary nodes in a planar lattice represented as a graph
    #get minimum cycle basis
    min_cycle_list = get_min_cycle_basis(G)
    print("min cycle done")
    #get all edges in the cycle that are present in the graph
    min_cycle_edges = get_min_cycle_edges(G, min_cycle_list)
    print("min edges done")

    #find edges that are part of only one cycle, those are the ones that make up the boundary
    boundary_edges = []
    for edge in edge_list:
        count = 0
        for cycle_edges in min_cycle_edges:              
            if ([edge[0],edge[1]] in cycle_edges) or ([edge[1],edge[0]] in cycle_edges):
                count = count+1
                
        if count == 1:
            boundary_edges.append(edge)
    
    #find all the nodes that are part of the boundary edges
    boundary_nodes = []
    for edge in boundary_edges:
        boundary_nodes.append(edge[0])
        boundary_nodes.append(edge[1])
    
    boundary_nodes = list(set(boundary_nodes))

    real_boundary_nodes = get_real_boundary_nodes_edges(G, min_cycle_list, boundary_nodes)

    #add nodes that have degree 1 (tips of free ends) to the boundary nodes
    for node in node_list:
        if G.degree(node) == 1:
            real_boundary_nodes.append(node)

            #now remove the neighbor of degree 1 node from the boundary nodes
            neighbor = list(G.neighbors(node))[0]
            if neighbor in real_boundary_nodes:
                real_boundary_nodes.remove(neighbor)
            
    real_boundary_nodes = list(set(real_boundary_nodes))

    
    #assign boundary nodes and edges
    for node in real_boundary_nodes:
        G.nodes[node]['level_bound_topo'] = 1000
        G.nodes[node]['level_bound_dist'] = 1000
        
    # name edge level as the lower value of both the levels of participating nodes.
    # level of boundary edges then becomes 1000 as well.
    for u, v in G.edges:
        l1 = G.nodes[u]['level_bound_topo']
        l2 = G.nodes[v]['level_bound_topo']
        level = min([l1,l2])
        G[u][v]['level_bound_topo'] = level

    return G 




# img_folder = str(DATA_ROOT / "footpad" / "images")
    
# # # scale_factor = 50/181.583 *0.001
# # scale_factor = 20/162.333 *0.001 

# #read in the images
# images = read_images_from_folder(img_folder)


#write a function to allow usere to select a node in the graph by manually clicking on it
def select_node(graph):
    #draw the graph
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')

    #draw the image
    #read the image
    img = images[1]
    #show the image
    ax.imshow(img, cmap='gray', alpha=0.5)

    #draw the graph with node positions determined by pos attribute of nodes
    #get positions of nodes from pos attribute of nodes
    pos = nx.get_node_attributes(graph, 'pos')
    #remove z coordinate from pos and take only x and y coordinates
    pos = np.array(list(pos.values()))/scale_factor
    #take only x and y coordinates
    pos = pos[:,0:2]
    #flip x and y coordinates
    pos = np.flip(pos, axis=1)

    #draw the nodes
    nx.draw_networkx_nodes(graph, ax=ax, pos=pos, node_size=10, node_color='r')

    #draw the edges
    nx.draw_networkx_edges(graph, ax=ax, pos=pos, width=0.5, edge_color='b')


    #add labels to the nodes as node ids
    labels = {}
    for node in graph.nodes():
        labels[node] = node

    nx.draw_networkx_labels(graph, ax=ax, pos=pos, labels=labels, font_size=8)

    plt.show()
    #get the node that the user clicked on
    node = input("Enter the node number: ")
    #return the node
    return node

#execute the function
graph_folder = str(DATA_ROOT / "footpad" / "graphs")


ctr = 0

#get file names of all graphs with 'simple_oirgin' some where in the name
graph_file_names = [f for f in sorted(os.listdir(graph_folder), key=str.upper) if 'simple.gpickle' in f]
G = nx.read_gpickle(os.path.join(graph_folder, graph_file_names[ctr]))

# node = select_node(G)
# print(node)


#change edges list
changed_edges = [(433,407),(71,76)]

#change the origin of the graph
origin_edge = changed_edges[ctr]


max_edge = G[origin_edge[0]][origin_edge[1]]['med_thick']
max_edge_nodes = origin_edge

#create a simple graph with origin
#create a networkx graph
sim_G_origin = G.copy()

#add a new node at the origin as the mid point of the edge with maximum thickness
#get mid pt of the edge with maximum thickness
mid_pt = (sim_G_origin.nodes[max_edge_nodes[0]]['pos'] + sim_G_origin.nodes[max_edge_nodes[1]]['pos'])/2
#add the node to the graph
node_id = len(sim_G_origin.nodes())        

sim_G_origin.add_node(node_id, pos = mid_pt)
#add edges between the new node and the two nodes in the edge with maximum thickness and remove the edge
sim_G_origin.add_edge(node_id, max_edge_nodes[0])
sim_G_origin.add_edge(node_id, max_edge_nodes[1])
sim_G_origin.remove_edge(max_edge_nodes[0], max_edge_nodes[1])

#add thickness attribute to the new edges as maximum thickness
sim_G_origin[node_id][max_edge_nodes[0]]['med_thick'] = max_edge
sim_G_origin[node_id][max_edge_nodes[1]]['med_thick'] = max_edge

#add node level attributes
sim_G_origin = add_node_edge_levels(sim_G_origin)
sim_G_origin = assign_boundary_nodes_edges2(sim_G_origin)

nx.write_gpickle(sim_G_origin, graph_folder + '\\' + graph_file_names[ctr] + 'graph_simple_origin.gpickle')
