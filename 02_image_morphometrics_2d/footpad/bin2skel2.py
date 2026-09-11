# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Adult foot-pad ossicle: binary mask to skeleton to lattice graphs.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 2I; Fig. S3B–D (the graphs also feed the foot-pad series in Fig. 2D–H)
What it does  : For one foot-pad ossicle (ctr = 0 -> test1, 1 -> test2; scale_factor_list
                holds the matching mm/pixel) it cleans the mask, takes the medial-axis
                skeleton, builds networkx graphs (sknw, full pixel graph and simple graph
                with medial thickness) and a copy with an origin node at the midpoint of
                the thickest edge, with node levels and boundary nodes marked.
Inputs        : DATA_ROOT/footpad/images/test1.png, test2.png (DIC images, only these two);
                DATA_ROOT/footpad/masks/test1_mask.bmp, test2_mask.bmp
Outputs       : DATA_ROOT/footpad/graphs/<name>graph_{sknw,full,simple,simple_origin}.gpickle
                and <name>_graph.png (read by reassign_origin.py and the *_foot scripts);
                OUT_ROOT/02_image_morphometrics_2d/footpad/labels/ and skeletons/ (QC images)
Environment   : environment-analysis.yml (Python 3.7)
Run           : python bin2skel2.py   (edit ctr near the end of the file: 0 or 1)
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys
import glob
import pandas as pd
import time
import math
from skimage import measure
from skimage import morphology
from skimage import segmentation
from skimage import filters
from skimage import util
from skimage import color
from skimage import io
from skimage import img_as_ubyte
from skimage import img_as_uint
from skimage import img_as_float
from skimage import exposure
from skimage import feature
import sknw
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



def read_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder), key=str.upper):
        img = cv2.imread(os.path.join(folder,filename),0)
        if img is not None:
            images.append(img)
    return images


#function to segment the image and label the blobs
def get_labeled_image(image, bin_image, name):
    #convert the image to binary
    thresh = filters.threshold_otsu(bin_image)
    binary = bin_image > thresh

    #remove smaller objects
    binary = morphology.remove_small_objects(binary, 200)
    
    #smoothen boundaries of binary regions, fill holes and identify identify objects under a certain size
    #remove objects larger than 1000 pixels squared
    binary = morphology.binary_closing(binary)

    #remove small holes
    binary = morphology.remove_small_holes(binary, 750)

    #create a labelled image
    label_image = measure.label(binary)

    #remove n largest regions
    props = measure.regionprops(label_image)
    areas = []
    for prop in props:
        areas.append(prop.area)
    areas.sort(reverse=True)
    areas = areas[0:1]
    for prop in props:
        if prop.area in areas:
            label_image[label_image == prop.label] = 0

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(label_image, cmap=plt.cm.gray)
    ax.axis('off')
    plt.savefig(label_folder + '\\' + name + '_labelled_image_3.png')

    #overlay labelled image on original image and save overlaid image
    fig1, ax1 = plt.subplots(figsize=(20, 20))
    ax1.imshow(image, cmap='gray')
    ax1.imshow(label_image, alpha=0.5)
    plt.axis('off')
    # plt.savefig('overlaid_image_3.png')

    #save the binary image
    binary_clean = img_as_ubyte(binary)


    return label_image, binary_clean

#find distances from center of the structure to the center of all the holes
def find_area_distances(label_image):
    #scale conversion factor
    #1px = 50/181.583 um
    # scale = 50/181.583
    scale = 20/162.333  

    #input the center of the image
    # center0 = [849.500, 812.333]
    center0 = [786,724]

    #find the center of all the holes
    props = measure.regionprops(label_image)
    centers = []
    for prop in props:
        centers.append(prop.centroid)

    #find the distance between the center of the structure and the center of all the holes
    distances = []
    for center in centers:
        distance = math.sqrt((center0[0] - center[0])**2 + (center0[1] - center[1])**2)
        distances.append(distance)

    #convert distances to um
    for i in range(len(distances)):
        distances[i] = distances[i] * scale

    #find the area of all the holes
    areas = []
    for prop in props:
        areas.append(prop.area)

    #convert areas to um^2
    for i in range(len(areas)):
        areas[i] = areas[i] * scale**2


    return areas, distances

#function to get the skeleton of the binary image
def get_skeleton(binary_clean, name):


    #invert the binary image
    binary_inv = util.invert(binary_clean)

    #get the skeleton of the binary image
    skeleton, distance = morphology.medial_axis(binary_inv, return_distance = True)

    #only keep the biggest connected component in the skeleton
    skeleton = measure.label(skeleton)
    props = measure.regionprops(skeleton)
    areas = []
    for prop in props:
        areas.append(prop.area)
    areas.sort(reverse=True)
    areas = areas[0:1]
    for prop in props:
        if prop.area not in areas:
            skeleton[skeleton == prop.label] = 0


    #thin the skeleton so that it is 1 pixel connected
    skeleton = morphology.thin(skeleton)    

    dist_on_skel = 2*distance * skeleton

    #overlay skeleton on original image and save overlaid image
    fig3, ax3 = plt.subplots(figsize=(20, 20))
    ax3.imshow(binary_clean, cmap='gray')   
    ax3.imshow(dist_on_skel, alpha=0.5, cmap = 'plasma')
    plt.axis('off')
    plt.savefig(skeleton_folder + '\\' + name + '_skeleton_3.png')

    #save skeleton as a binary image with 0 or 1 as pixel values
    skeleton = img_as_ubyte(skeleton)

    #save the skeleton as a bmp file
    cv2.imwrite(skeleton_folder + '\\' + name + '_skeleton_3.bmp', skeleton)

    return skeleton, dist_on_skel

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





#convert skeleton to graph with nodes as pixels and edges between connected pixels
def get_graph(skeleton, dist_on_skel, name):
    #build graph from skeleton
    graph = sknw.build_sknw(skeleton)

    # #draw image
    plt.imshow(skeleton, cmap='gray')

    #draw edges by pts
    for (s,e) in graph.edges():
        ps = graph[s][e]['pts']
        plt.plot(ps[:,1], ps[:,0], 'green')
        
    #draw node by o
    nodes = graph.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])
    plt.plot(ps[:,1], ps[:,0], 'r.')

    #title and show
    plt.title('Build Graph')
    #save the plot
    plt.savefig(graph_folder + '\\' + name + '_graph.png')

    # plt.show()


    # print(graph)
    # print(ps, len(ps))

    #convert sknw graph to a networkx graph and save it
    #get all edges with attributes
    edges = []
    for (s,e) in graph.edges():
        edges.append([s,e,graph[s][e]]) 

    #get all nodes with attributes
    nodes = []
    for n in graph.nodes():
        nodes.append([n,graph.nodes[n]])


    #remove edges with the same node at both ends
    edges_to_remove = []
    for edge in edges:
        if edge[0] == edge[1]:
            edges_to_remove.append(edge)

    for edge in edges_to_remove:
        edges.remove(edge)
        graph.remove_edge(edge[0], edge[1])

    #from graph get rid of nodes with degree 0
    nodes_to_remove = []
    for node in nodes:
        if graph.degree(node[0]) == 0:
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        nodes.remove(node)
        graph.remove_node(node[0])

    # print(nodes)
    # print(edges)



    #create a networkx graph
    full_G = nx.Graph()

    #add edges and nodes to the graph, with all points within the edge as separate edges
    for node in nodes:
        position = np.array([node[1]['o'][0], node[1]['o'][1], 0])
        full_G.add_node(node[0], pos = node[1]['o'])

    #total nodes in the graph
    total_nodes = len(nodes)

    #add nodes from the points in the edges
    for edge in edges:
        node_list = []
        #add the first node of the edge in the node list
        node_list.append(edge[0])
        for point in edge[2]['pts']:
            position = np.array([point[0], point[1], 0])
            full_G.add_node(total_nodes, pos = position)
            node_list.append(total_nodes)
            total_nodes += 1
        #add the last node of the edge in the node list
        node_list.append(edge[1])
        
        #add edges between all the nodes in the node list
        for i in range(len(node_list)-1):
            full_G.add_edge(node_list[i], node_list[i+1])

    #for all the nodes add the dist_on_skel as an attribute
    for node in full_G.nodes():
        position = full_G.nodes[node]['pos']
        #find the pixel in which the position lies
        position = np.array([int(position[0]), int(position[1])])
        full_G.nodes[node]['med_thick'] = dist_on_skel[position[0], position[1]]*scale_factor

    #set edge thickness as the average of the thickness of the two nodes
    for edge in full_G.edges():
        full_G[edge[0]][edge[1]]['med_thick'] = (full_G.nodes[edge[0]]['med_thick'] + full_G.nodes[edge[1]]['med_thick'])/2
    
    #scale all positions with the scale factor
    for node in full_G.nodes():
        position = full_G.nodes[node]['pos']
        position = np.array([position[0]*scale_factor, position[1]*scale_factor, 0])
        full_G.nodes[node]['pos'] = position




    #create a simple graph
    #create a networkx graph
    sim_G = nx.Graph()

    #add all the nodes to the graph
    for node in graph.nodes():
        position = np.array([graph.nodes[node]['o'][0], graph.nodes[node]['o'][1], 0])*scale_factor
        sim_G.add_node(node, pos = position)

    #add a single edge between nodes if a path exists between them
    for edge in graph.edges():
        sim_G.add_edge(edge[0], edge[1])

    #add thickness attribute to all the edges
    for edge in sim_G.edges():
        #find path between the two nodes in full_G
        path = nx.shortest_path(full_G, edge[0], edge[1])

        if(len(path)) <=2:
            print('path length less than 2', name)    
            print(edge[0], edge[1])
            #remove the edge from the graph
            sim_G.remove_edge(edge[0], edge[1])
            continue
            
        #find sum over all the thicknesses of the edges in the path
        thickness = 0
        for i in range(len(path)-1):
            thickness += full_G[path[i]][path[i+1]]['med_thick']
        #take average of the thickness
        print("path length", len(path))
        thickness = thickness/(len(path)-1)
        #add thickness as an attribute to the edge
        sim_G[edge[0]][edge[1]]['med_thick'] = thickness





    #create a simple graph with origin
    #find the edge in graph with the maximum edge thickness
    max_edge = 0    
    for edge in sim_G.edges():
        if sim_G[edge[0]][edge[1]]['med_thick'] > max_edge:
            max_edge = sim_G[edge[0]][edge[1]]['med_thick']
            max_edge_nodes = edge

    #create a simple graph with origin
    #create a networkx graph
    sim_G_origin = sim_G.copy()

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
    
    
    nx.write_gpickle(graph, graph_folder + '\\' + name + 'graph_sknw.gpickle')
    nx.write_gpickle(full_G, graph_folder + '\\' + name + 'graph_full.gpickle')
    nx.write_gpickle(sim_G, graph_folder + '\\' + name + 'graph_simple.gpickle')
    nx.write_gpickle(sim_G_origin, graph_folder + '\\' + name + 'graph_simple_origin.gpickle')

    return graph



#get data from network

img_folder = str(DATA_ROOT / "footpad" / "images")
mask_folder = str(DATA_ROOT / "footpad" / "masks")
label_folder = str(OUT_ROOT / "02_image_morphometrics_2d" / "footpad" / "labels")
skeleton_folder = str(OUT_ROOT / "02_image_morphometrics_2d" / "footpad" / "skeletons")
graph_folder = str(DATA_ROOT / "footpad" / "graphs")


#make folders if they don't exist
if not os.path.exists(label_folder):
    os.makedirs(label_folder)
if not os.path.exists(skeleton_folder):
    os.makedirs(skeleton_folder)
if not os.path.exists(graph_folder):
    os.makedirs(graph_folder)


# #main function
# def main():
    
# scale_factor = 
scale_factor_list = [50/181.583 *0.001, 20/162.333 *0.001] #in mm 

    
#read in the images
images = read_images_from_folder(img_folder)
masks = read_images_from_folder(mask_folder)


#list of filenames without extension
file_names = []
for filename in sorted(os.listdir(img_folder), key=str.upper):
    file_names.append(filename.split('.')[0])

for ii in range(len(images)):
    #read the original image
    image = images[ii]
    mask = masks[ii]
    name = file_names[ii]


ctr = 1
scale_factor = scale_factor_list[ctr]
for image in [images[ctr]]:
    #read in the binary image
    binary_image = masks[ctr]

    name = file_names[ctr]

    #segment the image
    label_image, binary_clean = get_labeled_image(image, binary_image, name)

    #get the skeleton of the binary image
    skeleton, dist_on_skel = get_skeleton(binary_clean, name)

    #convert skeleton to graph with nodes as pixels and edges between connected pixels
    graph = get_graph(skeleton, dist_on_skel, name)

    # #find the areas and distances of all the holes
    # areas, distances = find_area_distances(label_image)

    # #create a csv file to store the data
    # with open('data1.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Area (um2)", "Distance(um)"])
    #     for i in range(len(areas)):
    #         writer.writerow([areas[i], distances[i]])

    #plot scatter plot of areas with distances
    # fig2, ax2 = plt.subplots(figsize=(20, 20))
    # ax2.scatter(distances, areas)
    # plt.xlabel('Distance (um)')
    # plt.ylabel('Area (um2)')
    # plt.savefig('scatter_plot_3.png')
