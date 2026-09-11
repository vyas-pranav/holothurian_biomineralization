# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Scaled medial thickness vs scaled path length (Fig. 2H inset).

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 2H (inset)
What it does  : Pools the scaled, binned thickness curves of the pillared tables, other
                plates and foot pads and draws their mean curves (seaborn lineplot) on a
                2 x 2 inch axes with limits and ticks 0/0.5/1. Bands, points, labels,
                legend and the savefig of thickness_vs_distance_scaled_small.svg/.png
                are commented out, as in the original.
Inputs        : microct/animal_1/extracted_ossicles/Data
                Calculated/{thickness_data_lumped_scaled/*.pickle, loc_file_id*.txt,
                selected_*.npy},
                footpad/graphs/thick_data/thick_with_dist_lump_scaled_2*.pkl,
                footpad/graphs/test{1,2}graph_{simple_origin,full}.gpickle (foot-pad
                graphs from 02_image_morphometrics_2d/footpad)
Outputs       : matplotlib window (savefig commented out)
Environment   : environment-analysis.yml (Python 3.7)
Run           : python plot_thickness_scaled_small.py
"""

# %matplotlib qt

# from vedo import *
import vedo
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
import re
import pyvista as pv
from scipy import interpolate
import pyacvd
import igl
import matplotlib as mpl
import scipy.spatial as spatial
import os
import math
from matplotlib import pyplot as plt
import pickle
import random
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit



"""pygeodesic library to compute geodesic distances"""
from pygeodesic import geodesic
import potpourri3d as pp3d

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

#%%



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


def unit_vec(vector):
  """Converts a vector to a unit vector.

  Args:
    vector: A 1-D numpy array.

  Returns:
    The unit vector of the input vector.
  """
  magnitude = np.linalg.norm(vector)
  if magnitude == 0:
    return np.zeros_like(vector)
  else:
    return vector / magnitude


def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vec(v1)
    v2_u = unit_vec(v2)
    return 180/np.pi*np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rotation_matrix(a, b):
    """
    Find the rotational transformation matrix between two sets of orthogonal axes represented as unit vectors.

    Parameters
    ----------
    a : list of numpy arrays
        The first set of unit vectors.
    b : list of numpy arrays
        The second set of unit vectors.

    Returns
    -------
    R : numpy array
        The rotational transformation matrix.
    """
    R = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            R[i, j] = np.dot(a[i], b[j])
    return R


def surface_fitting_pts(fit_pos):

    x = fit_pos[:,0]
    y = fit_pos[:,1]
    z = fit_pos[:,2]

    fit_pos_xy = np.zeros((len(x),2))
    for ii in range(len(x)):
        fit_pos_xy[ii,0] = fit_pos[ii,0]  
        fit_pos_xy[ii,1] = fit_pos[ii,1]  

    N = 100
    fac = 2
    grid_x, grid_y = np.mgrid[-0.05*fac:0.05*fac:N*1j*fac, -0.05*fac:0.05*fac:N*1j*fac]
    f_dg = interpolate.griddata(fit_pos_xy, z, (grid_x, grid_y), method='linear')

    good_ind = np.argwhere(np.isnan(f_dg)==False)

    ctr = 0
    new_pos_list = []
    for ind in good_ind.tolist():
         new_pos_list.append([grid_x[ind[0],ind[1]], grid_y[ind[0],ind[1]], f_dg[ind[0],ind[1]]])

    new_fit_pos = np.array(new_pos_list)

    return new_fit_pos


def remesh_func(mesh, n_subdivide, n_cluster):
    clus = pyacvd.Clustering(mesh)
    # mesh is not dense enough for uniform remeshing
    clus.subdivide(n_subdivide)
    clus.cluster(n_cluster)

    # plot clustered cow mesh
    # clus.plot()

    # remesh
    remesh = clus.create_mesh()

    return remesh


def load_data(ii):
    ossicle = vedo.load(files[ii])
    # ossicle.lw(0.05).c('pink6').alpha(0.5)
    vox_path = os.path.join(skel_paths[ii],"vox.mhd")
    voxel_vol = vedo.load(vox_path)

    skel_path = os.path.join(skel_paths[ii],"post15_skel_line.vtk")
    cl_skel_path = os.path.join(skel_paths[ii],"clean_skel_line.vtk")
    skeleton = vedo.load(skel_path)
    cl_skeleton = vedo.load(cl_skel_path)

    return ossicle, voxel_vol, skeleton, cl_skeleton

def load_graphs(ii):
    clean_full_G = nx.read_gpickle(os.path.join(skel_paths[ii],"clean_full_G.gpickle"))
    clean_simple_G = nx.read_gpickle(os.path.join(skel_paths[ii],"clean_simple_G.gpickle")) 
    origin_clean_simple_G = nx.read_gpickle(os.path.join(skel_paths[ii],"origin_clean_simple_G.gpickle")) 
    clean_full_base_G = nx.read_gpickle(os.path.join(skel_paths[ii],"clean_full_base_G.gpickle"))
    clean_simple_base_G = nx.read_gpickle(os.path.join(skel_paths[ii],"clean_simple_base_G.gpickle"))
    origin_clean_simple_base_G = nx.read_gpickle(os.path.join(skel_paths[ii],"origin_clean_simple_base_G.gpickle"))
    return clean_full_G, clean_simple_G, origin_clean_simple_G, clean_full_base_G, clean_simple_base_G,  origin_clean_simple_base_G


#align ossicle 60 (170 id) to the z axis
def get_base_mesh():
    ii = 170

    ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)

    full_G, clean_full_G, simple_G, clean_simple_G = load_graphs(ii)

    G = clean_simple_G

    for u, v, data in G.edges(data=True):
        if (data['branch_degree']==0):
            start_edge = [u,v]

    base_nodes = set()
    for node in start_edge:
        base_nodes = base_nodes.union(set(G.neighbors(node)))

    base_pos_list = []
    for node in base_nodes:
        base_pos_list.append(G.nodes[node]['pos'])
    base_pos = np.array(base_pos_list)

    # find the start point and volumetric centroid
    start_pt = (G.nodes[start_edge[0]]['pos'] + G.nodes[start_edge[1]]['pos'])/2 

    # perform PCA on the base position coordinates 
    pca = PCA(n_components = 3)
    pca.fit(base_pos)

    ossicle_align = ossicle.clone().shift(dx= -start_pt[0], dy = -start_pt[1],dz = -start_pt[2])
    rot_axis = np.cross(unit_vec(- pca.components_[2]), np.array([0,0,1]))
    rot_angle = angle_between(unit_vec(- pca.components_[2]), np.array([0,0,1]))
    ossicle_align.rotate(rot_angle, rot_axis)

    ossicle_align.write(os.path.join(type_dir, 'aligned', 'stl_files', file_names[ii][0:-4]+'.stl'))

    return ossicle_align

def closest_point(points, towhat):
    closest_distance = float('inf')
    closest_point = None
    for point in points:
        distance = np.linalg.norm(point-towhat) # Euclidean distance
        if distance < closest_distance:
            closest_distance = distance
            closest_point = point
    return closest_point

def furthest_point(points, towhat):
    furthest_distance = -float('inf')
    furthest_point = None
    for point in points:
        distance = np.linalg.norm(point-towhat) # Euclidean distance
        if distance > furthest_distance:
            furthest_distance = distance
            furthest_point = point
    return furthest_point

def furthest_point_by_x(points):
    furthest_x = -float('inf')
    furthest_point = None
    for point in points:
        if point[0] > furthest_x:
            furthest_x = point[0]
            furthest_point = point
    return furthest_point


def least_angle_point_below_point(points, reference_point):
    # Find the distance of each point to the reference_point
    distances_to_reference_point = np.linalg.norm(points - reference_point, axis=1)

    # Find the index of the points below the reference_point
    below_reference_point = np.where(points[:, 2] < reference_point[2])[0]

    # take only the point below the reference_point
    below_reference_points = points[below_reference_point]

    # Find the dot product of each point's vector with the z-axis vector (0,0,1)
    dot_products = np.dot((below_reference_points-reference_point), [0, 0, 1])

    # Normalize the dot products to get the cosine of the angle between the point vector and the z-axis vector
    cos_angles = dot_products / np.linalg.norm(below_reference_points-reference_point, axis=1)

    # Find the index of the point with the lowest cosine value (i.e. the point that subtends the least angle on the z-axis)
    index_of_point_with_least_angle = np.argmin(cos_angles)

    # Return the point with the lowest cosine value
    return below_reference_points[index_of_point_with_least_angle]

def xy_plane_distance(point, points):
    # subtract the point from all points in the array
    differences = points - point
    # extract x and y coordinates
    x_differences, y_differences = differences[:, 0], differences[:, 1]
    # calculate the xy plane projected distance using the Pythagorean theorem
    distances = np.sqrt(x_differences**2 + y_differences**2)
    return distances


def map_to_skel_pt(mesh, skel):

    skel_pts = skel.points
    skl_med_thick = skel.point_data['MedialThickness']
    mesh_med_thick = np.zeros([mesh.n_points,1]) 
    ctr = 0
    for point in mesh.points:        
        distances = np.linalg.norm(skel_pts - point, axis=1)
        index = np.argmin(distances)
        sort_ind = np.argsort(distances)
        mesh_med_thick[ctr] = np.average(skl_med_thick[sort_ind[0:100]])
        # if (ctr%1000 == 0):
        ctr = ctr+1

    mesh.point_data['MedialThickness'] = mesh_med_thick
    return


def get_min_cycle_basis(G):
    min_cycle_basis = nx.minimum_cycle_basis(G)

    min_cycle_list = []
    for cycle in min_cycle_basis:
        if len(cycle) >=3:
            min_cycle_list.append(cycle)

    return min_cycle_list

def get_hole_surf_list(min_cycle_basis, G):
    surf_list = []
    for cycle in min_cycle_basis:
        pts = []
        for node in cycle:
            pos_value = G.nodes[node]['pos']
            pts.append(pos_value)

        points = pv.wrap(np.array(pts))
        surf = points.reconstruct_surface(nbr_sz = 50, sample_spacing = 0.005)
        surf_list.append(surf)

    return surf_list


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
        return sort_pts

def cut_mesh_with_loop(pv_mesh, loop, kdtree, mesh_pts):

    # Get the indices of the closest vertices in the mesh for each polygon point
    _, closest_indices = kdtree.query(loop)
    loop_mesh_mapped = mesh_pts[closest_indices]


    # Convert the PyVista mesh into a vedo mesh
    vedo_mesh = vedo.Mesh(pv_mesh)
    pts = vedo.Points(loop_mesh_mapped)

    # Cut the vedo mesh with the closed loop of points
    cut_mesh = vedo_mesh.cut_with_point_loop(loop_mesh_mapped, on = "points", include_boundary=True)

    # # Convert the cut vedo mesh back into a PyVista mesh
    # pv_cut_mesh = pv.PolyData(cut_mesh.points(), cut_mesh.faces())

    return cut_mesh


def get_hole_mesh_list(surface, min_cycle_basis, G):
    # Create a KDTree for searching the mesh vertices
    mesh_pts = surface.points
    kdtree = spatial.KDTree(mesh_pts)

    hole_mesh_list = []
    ctr = 0
    for cycle in min_cycle_basis:
        pts = []
        for node in cycle:
            pos_value = G.nodes[node]['pos']
            pts.append(pos_value)

        loop = order_points_by_angle(pts)
        hole_mesh = cut_mesh_with_loop(surface, loop, kdtree, mesh_pts)
        hole_mesh_list.append(hole_mesh)
        ctr+=1

    return hole_mesh_list

def get_hole_area_list(hole_mesh_list):
    area_list = []
    for mesh in hole_mesh_list:
        area_list.append(mesh.area())
    return area_list

def get_hole_center_list(hole_mesh_list):
    center_list = []
    for mesh in hole_mesh_list:
        center_list.append(mesh.center_of_mass())
    return center_list    


def nx_to_pv_cylinders(graph, radius=0.1, resolution=5):
    """Convert a NetworkX graph into a list of PyVista cylindrical meshes.

    Parameters
    ----------
    graph: networkx.Graph
        The input graph
    radius: float
        The radius of the cylindrical meshes
    resolution: int
        The number of points to use in constructing the cylindrical meshes

    Returns
    -------
    list of pyvista.PolyData
        A list of PyVista cylindrical meshes representing the edges of the graph
    """
    meshes = []
    lengths = []
    for (u, v) in graph.edges():
        p1 = np.array(graph.nodes[u]["pos"])
        p2 = np.array(graph.nodes[v]["pos"])
        direction = (p2 - p1)/np.linalg.norm(p2-p1)
        height = np.linalg.norm(p2-p1)
        center = (p1+p2)/2
        cylinder = pv.Cylinder(center = center, direction = direction, radius = radius, height = height, resolution = resolution)
        cylinder.point_data['length'] = height*np.ones(cylinder.n_points) #fill length value as point data in all the points making up the cylinder

        meshes.append(cylinder)
        lengths.append(height)

    return meshes, lengths

#assign origin in the G as the node closest to the origin from bound_G
def assign_origin(G, bound_G):
    #find the origin node which is the max node id in bound_G
    origin = max(bound_G.nodes())

    #find node in G closest to origin
    origin_pos = bound_G.nodes[origin]['pos']
    dist = float('inf')
    for node in G.nodes():
        node_pos = G.nodes[node]['pos']
        if np.linalg.norm(node_pos - origin_pos) < dist:
            dist = np.linalg.norm(node_pos - origin_pos)
            origin = node

    return origin

#find shortest path between orign and boundary node while avoiding any node in the list of nodes to avoid
def get_shortest_path(G, bound_G, origin, boundary_node, all_boundary_nodes):
    print("level0")
    #create a graph with all the nodes to avoid removed except the boundary_node
    G_copy = G.copy()
    bound_rem_G = G.copy()


    #create a list of nodes to remove and remove boundary_node if it exists in the list
    nodes_to_remove = []
    for node in G_copy.nodes():
        if node in all_boundary_nodes:
            nodes_to_remove.append(node)

    if origin in nodes_to_remove:
        nodes_to_remove.remove(origin)

    if boundary_node in nodes_to_remove:
        nodes_to_remove.remove(boundary_node)

    #remove nodes to remove from the graph
    bound_rem_G.remove_nodes_from(nodes_to_remove)

    #check if G has path from origin to boundary node
    if nx.has_path(bound_rem_G, origin, boundary_node):
        #first find the shortest path between origin and boundary node 
        path = nx.shortest_path(bound_rem_G, origin, boundary_node)
    else:
        print("level1")
        #if no path exists, remove neighbors of boundary node from nodes to remove and check again
        neighbor_set = set([boundary_node])
        for node in bound_G.neighbors(boundary_node):
            neighbor_set.add(node)

        #remove neighbors of boundary node from nodes to remove and check again
        for node in neighbor_set:
            if node in nodes_to_remove:
                nodes_to_remove.remove(node)

        #remove nodes to remove from the graph
        bound_rem_G = G.copy()
        bound_rem_G.remove_nodes_from(nodes_to_remove)

        #check if G has path from origin to boundary node
        if nx.has_path(bound_rem_G, origin, boundary_node):
            #first find the shortest path between origin and boundary node 
            path = nx.shortest_path(bound_rem_G, origin, boundary_node)
        else:
            print("level2")
            #get all neighbor set of neighbors of boundary node
            neighbor_set2 = neighbor_set.copy()
            for node in neighbor_set:
                for neighbor in bound_G.neighbors(node):
                    neighbor_set2.add(neighbor)
            neighbor_set = neighbor_set2.copy()

            #remove neighbors of boundary node from nodes to remove and check again
            for node in neighbor_set:
                if node in nodes_to_remove:
                    nodes_to_remove.remove(node)

            #remove nodes to remove from the graph
            bound_rem_G = G.copy()
            bound_rem_G.remove_nodes_from(nodes_to_remove)

            #check if G has path from origin to boundary node
            if nx.has_path(bound_rem_G, origin, boundary_node):
                #find the shortest path between origin and boundary node
                path = nx.shortest_path(bound_rem_G, origin, boundary_node)
            else:
                print("level3")
                #get all neighbor set of neighbors of neighbors of boundary node
                neighbor_set2 = neighbor_set.copy()
                for node in neighbor_set:
                    for neighbor in bound_G.neighbors(node):
                        neighbor_set2.add(neighbor)
                neighbor_set = neighbor_set2.copy()

                #remove neighbors of boundary node from nodes to remove and check again
                for node in neighbor_set:
                    if node in nodes_to_remove:
                        nodes_to_remove.remove(node)

                #remove nodes to remove from the graph
                bound_rem_G = G.copy()
                bound_rem_G.remove_nodes_from(nodes_to_remove)

                #check if G has path from origin to boundary node
                if nx.has_path(bound_rem_G, origin, boundary_node):
                    #find the shortest path between origin and boundary node
                    path = nx.shortest_path(bound_rem_G, origin, boundary_node)
                else:
                    print("level4")
                    #get all neighbor set of neighbors of neighbors of boundary node
                    neighbor_set2 = neighbor_set.copy()
                    for node in neighbor_set:
                        for neighbor in bound_G.neighbors(node):
                            neighbor_set2.add(neighbor)
                    neighbor_set = neighbor_set2.copy()

                    #remove neighbors of boundary node from nodes to remove and check again
                    for node in neighbor_set:
                        if node in nodes_to_remove:
                            nodes_to_remove.remove(node)

                    #remove nodes to remove from the graph
                    bound_rem_G = G.copy()
                    bound_rem_G.remove_nodes_from(nodes_to_remove)

                    #check if G has path from origin to boundary node
                    if nx.has_path(bound_rem_G, origin, boundary_node):
                        #find the shortest path between origin and boundary node
                        path = nx.shortest_path(bound_rem_G, origin, boundary_node)
                    else:
                        print("level5")
                        return


    return path

#get thickness attributes from all the nodes lying on the shortest path connecting origin and a boundary node
def get_thickness_along_path(G, bound_G, origin, boundary_node, boundary_nodes):
    #first find the shortest path between origin and boundary node 
    path = get_shortest_path(G, bound_G, origin, boundary_node, boundary_nodes)

    #get the thickness attribute for all the nodes in the shortest path and store it with the length along the path
    thickness_list = []
    length_list = []
    for node in path:
        thickness_list.append(G.nodes[node]['med_thick'])
        if node == origin:
            length_list.append(0)
        else:
            length_list.append(length_list[-1] + np.linalg.norm(G.nodes[node]['pos'] - G.nodes[path[path.index(node)-1]]['pos']))

    #create a list with (node, length, thickness) tuples
    data_list = list(zip(path, length_list, thickness_list))

    return data_list

#find the boundary nodes of the graph and get the thickness along the shortest path connecting origin and each boundary node
def get_all_path_thickness_origin_to_boundary(G, bound_G):
    #get the boundary nodes of the graph
    boundary_nodes = [node for node in bound_G.nodes() if bound_G.nodes[node]['level_bound_dist'] == 1000]

    #assign origin in G
    origin = assign_origin(G, bound_G)
    print(origin)

    #get the thickness along the shortest path connecting origin and each boundary node
    data_whole_list = []
    for node in boundary_nodes:
        data_list = get_thickness_along_path(G, bound_G, origin, node, boundary_nodes)
        data_whole_list.append(data_list)

    return data_whole_list

#function to assign shortest distances as sum of all edge lengths along the shortest path from origin to all nodes in the graph
def assign_distances(G, origin):
    #assign all edges attribute length as distance between node positions
    for u, v, data in G.edges(data=True):
        data['length'] = np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos'])

    distances = nx.single_source_dijkstra_path_length(G, origin, weight = 'length')

    nx.set_node_attributes(G, distances, 'level_dist')

#function to take in full base graph, define origin as the point closest to the origin in simple graph, and get a list of tuple (node, dist from origin, thickness) for all nodes in the full base graph
def get_dist_thickness_origin_to_all(G, bound_G):
    #assign origin in G
    origin = assign_origin(G, bound_G)

    #assign distance from origin to all the nodes in the graph
    assign_distances(G, origin)

    #get the thickness along the shortest path connecting origin and each boundary node
    data_list = []
    for node in G.nodes():
        data = (node, G.nodes[node]['level_dist'], G.nodes[node]['med_thick'])
        data_list.append(data)

    return data_list


#%%
base_dir =str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles")

datadir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "Data Calculated")

type_dir =str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned")
indir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned" / "vtk")
skeldir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned" / "skeleton")

basemeshdir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned" / "base" / "vtk")
curv_dir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned" / "curvature")

all_file_names = os.listdir(indir)
file_names = [f for f in all_file_names if f.endswith('.vtk')]
file_names.sort()
files = [os.path.join(indir,f) for f in file_names]
skel_paths = [os.path.join(skeldir,f) for f in file_names]
n_files = len(file_names)
file_id = []
for ii in range(n_files):
    txt = file_names[ii]
    num = [int(s) for s in re.findall(r'\d+',txt)]
    file_id.append(num[0])

with open(os.path.join(datadir,'loc_file_id.txt')) as txtfile:
    loc_file_id = list(map(int, txtfile.read().split('\n')))

with open(os.path.join(datadir,'loc_file_id_table.txt')) as txtfile:
    loc_file_id_table = list(map(int, txtfile.read().split('\n')))

with open(os.path.join(datadir,'loc_file_id_all_flat.txt')) as txtfile:
    loc_file_id_all_flat = list(map(int, txtfile.read().split('\n')))

with open(os.path.join(datadir,'loc_file_id_other_flat.txt')) as txtfile:
    loc_file_id_other_flat = list(map(int, txtfile.read().split('\n')))

#read selected numpy files
with open(os.path.join(datadir, 'selected_table.npy'), 'rb') as f:
    selected_table = np.load(f).tolist()

with open(os.path.join(datadir, 'selected_other_flat.npy'), 'rb') as f:
    selected_other_flat = np.load(f).tolist()


plt3d = vedo.Plotter(bg2='gray2')#, interactive=True) # screen size
pl = pv.Plotter(off_screen=False)
# pl = pv.Plotter(off_screen=True)
meshes = []

#%%
with open(os.path.join(datadir, 'centroids.npy'), 'rb') as f:
    centroids = np.load(f)

with open(os.path.join(datadir, 'avg_mean_curvature.npy'), 'rb') as f:
    data_list = np.load(f)


center = np.mean(centroids, axis = 0)
sphere = pv.Sphere(radius = 0.005, center = center)
sphere.point_data['data'] = 0.0000001
merged = sphere
ctr = 0
max_val = 0
min_val = 0

#create empty lists for hole vs area data
whole_dist_list1 = []
whole_thick_list1 = []

active_loc_list = selected_table[:]

for jj in list(range(len(active_loc_list))): #list(range(0,10))+ list(range(11,len(active_loc_list))): #range(n_files):      print(ii, files[ii])

    kk = active_loc_list[jj] - 1
    ii = loc_file_id[kk]

    print(jj, ii, files[ii])

    #check if file exists
    if os.path.isfile(os.path.join(datadir, 'thickness_data_lumped_scaled', file_names[ii][0:-4]+'.pickle')):
        print('file exists')
        #load pickle file
        with open(os.path.join(datadir, 'thickness_data_lumped_scaled', file_names[ii][0:-4]+'.pickle'), 'rb') as f:
            data_whole_list = pickle.load(f)

        #append data from data_whole_list to list area and dist lists
        for data in data_whole_list:
            if (np.isnan(data[0]) or np.isnan(data[1])):
                continue
            else:
                whole_dist_list1.append(data[0])
                whole_thick_list1.append(data[1])

#create empty lists for hole vs area data
whole_dist_list2 = []
whole_thick_list2 = []

active_loc_list = selected_other_flat[:]

for jj in list(range(len(active_loc_list))): #list(range(0,10))+ list(range(11,len(active_loc_list))): #range(n_files):      print(ii, files[ii])

    kk = active_loc_list[jj] - 1
    ii = loc_file_id[kk]

    print(jj, ii, files[ii])

    #check if file exists
    if os.path.isfile(os.path.join(datadir, 'thickness_data_lumped_scaled', file_names[ii][0:-4]+'.pickle')):
        print('file exists')
        #load pickle file
        with open(os.path.join(datadir, 'thickness_data_lumped_scaled', file_names[ii][0:-4]+'.pickle'), 'rb') as f:
            data_whole_list = pickle.load(f)

        #append data from data_whole_list to list area and dist lists
        for data in data_whole_list:
            if (np.isnan(data[0]) or np.isnan(data[1])):
                continue
            else:
                whole_dist_list2.append(data[0])
                whole_thick_list2.append(data[1])



#get file list in folder
graph_folder = str(DATA_ROOT / "footpad" / "graphs")

#get file names of all graphs with 'simple_oirgin' some where in the name
simple_graph_file_names = [f for f in sorted(os.listdir(graph_folder), key=str.upper) if 'simple_origin' in f]
full_graph_file_names = [f for f in sorted(os.listdir(graph_folder), key=str.upper) if 'full' in f] 

whole_dist_list3 = []
whole_thick_list3 = []

for jj in [0,1]: #list(range(0,10))+ list(range(11,len(active_loc_list))): #range(n_files):      print(ii, files[ii])

    # kk = active_loc_list[jj] - 1
    # ii = loc_file_id[kk]


    # ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)
    x = []
    y = []

    if 1==2:
        continue
    else:
        print('yes')
        # clean_full_G, clean_simple_G, origin_clean_simple_G, clean_full_base_G, clean_simple_base_G,  origin_clean_simple_base_G = load_graphs(ii)
        # sim_G_origin = nx.read_gpickle(os.path.join(graph_folder, simple_graph_file_names[jj]))
        # full_G = nx.read_gpickle(os.path.join(graph_folder, full_graph_file_names[jj]))

        # G = sim_G_origin

        # pv_ossicle = pv.wrap(ossicle.polydata())
        # pv_skeleton = pv.wrap(skeleton.polydata())
        # # pv_rem_recon = pv.wrap(rem_recon_mesh.polydata()).subdivide(2)
        # pv_graph =  graph_to_pv_mesh(G)    #.subgraph(min_cycle_basis_G[20])

        data_file_name = graph_folder + '\\thick_data\\thick_with_dist_lump_scaled_2' + simple_graph_file_names[jj] + '.pkl'

        #load pickle file
        with open(data_file_name, 'rb') as f:
            data_whole_list = pickle.load(f)        

        #append data from data_whole_list to list area and dist lists
        for data in data_whole_list:
            if (np.isnan(data[0]) or np.isnan(data[1])):
                continue
            else:
                whole_dist_list3.append(data[0])
                whole_thick_list3.append(data[1])



# #get file list in folder
# graph_folder = str(DATA_ROOT / "footpad" / "graphs")

# #get file names of all graphs with 'simple_oirgin' some where in the name
# simple_graph_file_names = [f for f in sorted(os.listdir(graph_folder), key=str.upper) if 'simple_origin' in f]
# full_graph_file_names = [f for f in sorted(os.listdir(graph_folder), key=str.upper) if 'full' in f] 

# whole_dist_list4 = []
# whole_thick_list4 = []

# for jj in [1]: #list(range(0,10))+ list(range(11,len(active_loc_list))): #range(n_files):      print(ii, files[ii])

#     # kk = active_loc_list[jj] - 1
#     # ii = loc_file_id[kk]


#     # ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)
#     x = []
#     y = []

#     if 1==2:
#         continue
#     else:
#         # clean_full_G, clean_simple_G, origin_clean_simple_G, clean_full_base_G, clean_simple_base_G,  origin_clean_simple_base_G = load_graphs(ii)
#         sim_G_origin = nx.read_gpickle(os.path.join(graph_folder, simple_graph_file_names[jj]))
#         full_G = nx.read_gpickle(os.path.join(graph_folder, full_graph_file_names[jj]))

#         G = sim_G_origin

#         # pv_ossicle = pv.wrap(ossicle.polydata())
#         # pv_skeleton = pv.wrap(skeleton.polydata())
#         # # pv_rem_recon = pv.wrap(rem_recon_mesh.polydata()).subdivide(2)
#         pv_graph =  graph_to_pv_mesh(G)    #.subgraph(min_cycle_basis_G[20])

#         data_file_name = graph_folder + '\\thick_data\\thick_with_dist_lump_scaled_2' + simple_graph_file_names[jj] + '.pkl'

#         #load pickle file
#         with open(data_file_name, 'rb') as f:
#             data_whole_list = pickle.load(f)        

#         #append data from data_whole_list to list area and dist lists
#         for data in data_whole_list:
#             if (np.isnan(data[0]) or np.isnan(data[1])):
#                 continue
#             else:
#                 whole_dist_list4.append(data[0])
#                 whole_thick_list4.append(data[1])

# show the meshes

# # pl.add_scalar_bar(title='Length (in mm)', mapper = None, n_labels=5, italic=False, bold=False, title_font_size=None, label_font_size=None, color=None, font_family=None, shadow=False, width=None, height=None, position_x=None, position_y=None, vertical=None, interactive=None, fmt=None, use_opacity=True, outline=False, nan_annotation=False, below_label=None, above_label=None, background_color=None, n_colors=None, fill=False, render=False, theme=None)

# pl.camera.roll = 90.0
# # # pl.screenshot('206_front_length.png', window_size=(10000,10000), transparent_background= True)
# pl.show()

# # # plot all the thickness vs length curves
# # fig, ax = plt.subplots()
# # for data_list in data_whole_list:
# #     ax.plot([data[1] for data in data_list], [data[2] for data in data_list])
# # ax.set_xlabel('Length (in mm)')
# # ax.set_ylabel('Thickness (in mm)')
# # plt.show()


#create a mean and shaded error plot of thickness vs distance values
fig, ax = plt.subplots(figsize = (2,2))


#get lists of lengths and thicknesses from data_whole_list
x1 = whole_dist_list1
y1 = whole_thick_list1

#find a list of unique values of distances from origin present in the second entry of all the tuples in data_whole_list
bin_centers1 = list(set(whole_dist_list1))
#sort the list of bin centers
bin_centers1.sort()

#find mean and standard deviation of thickness values for each unique distance value
bin_means1 = []
bin_stdevs1 = []
# for bin_center in bin_centers1:
#     bin_list = []
#     ctr = 0
#     for ii in range(len(x1)):
#         if x1[ii] == bin_center:
#             bin_list.append(y1[ii])
#             ctr = ctr+1
#     if ctr>2:
#         bin_means1.append(np.mean(np.array(bin_list)))
#         bin_stdevs1.append(np.std(np.array(bin_list)))
#     else:
#         bin_means1.append(np.nan)
#         bin_stdevs1.append(np.nan)

for bin_center in bin_centers1:
    bin_list = []
    ctr = 0
    for ii in range(len(x1)):
        if x1[ii] == bin_center:
            bin_list.append(y1[ii])
            ctr = ctr+1

    bin_means1.append(np.mean(np.array(bin_list)))
    bin_stdevs1.append(np.std(np.array(bin_list)))

#convert bin_centers, bin_means and bin_stdevs to numpy arrays
bin_centers1 = np.array(bin_centers1)
bin_means1 = np.array(bin_means1)
bin_stdevs1 = np.array(bin_stdevs1)


# #find min and max values of distances from origin present in the second entry of all the tuples in data_whole_list
# min_level = min(whole_dist_list1)
# max_level = max(whole_dist_list1)

# #create bins from range of distances from origin
# bin_edges = np.arange(-0.5, max_level+1,1) 
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# # Bin data
# df = pd.DataFrame({'x': x, 'y': y})
# df['bin'] = pd.cut(df['x'], bins=bin_edges)

# # Compute mean and standard deviation for each bin
# bin_means = df.groupby('bin')['y'].mean().values
# bin_stdevs = df.groupby('bin')['y'].std().values

color1 = 'orange'
color1_1 = 'chocolate'
# Create the lineplot with shaded error bars (confidence interval) using seaborn
sns.lineplot(x=bin_centers1, y=bin_means1, color=color1_1, alpha = 0.75, linewidth = 3, ax=ax)
# plt.fill_between(bin_centers1, bin_means1 - bin_stdevs1, bin_means1 + bin_stdevs1, color=color1, alpha=0.3, zorder = 1)
# #also mark points on the mean curve
# plt.scatter(x1, y1, marker = 'o', color = color1, alpha = 0.1, s = 25, zorder = 2, linewidth = 0)
# plt.scatter(bin_centers1, bin_means1, marker = 'o', color = color1_1, s = 100, zorder = 3, alpha = 0.75, label = 'table', linewidth = 0)



#get lists of lengths and thicknesses from data_whole_list
x = whole_dist_list2
y = whole_thick_list2

#find a list of unique values of distances from origin present in the second entry of all the tuples in data_whole_list
bin_centers = list(set(whole_dist_list2))
#sort the list of bin centers
bin_centers.sort()

#find mean and standard deviation of thickness values for each unique distance value
bin_means = []
bin_stdevs = []
# for bin_center in bin_centers:
#     bin_list = []
#     ctr = 0
#     for ii in range(len(x)):
#         if x[ii] == bin_center:
#             bin_list.append(y[ii])
#             ctr = ctr+1
#     if ctr>2:
#         bin_means.append(np.mean(np.array(bin_list)))
#         bin_stdevs.append(np.std(np.array(bin_list)))
#     else:
#         bin_means.append(np.nan)
#         bin_stdevs.append(np.nan)

for bin_center in bin_centers:
    bin_list = []
    ctr = 0
    for ii in range(len(x)):
        if x[ii] == bin_center:
            bin_list.append(y[ii])
            ctr = ctr+1

    bin_means.append(np.mean(np.array(bin_list)))
    bin_stdevs.append(np.std(np.array(bin_list)))


#convert bin_centers, bin_means and bin_stdevs to numpy arrays
bin_centers = np.array(bin_centers)
bin_means = np.array(bin_means)
bin_stdevs = np.array(bin_stdevs)

# #change the number scale on x and y axis into scientific notation
# ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
# ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

#set max x value as 7
# ax.set_xlim([-0.2, 12.2])

color2 = 'steelblue'
# Create the lineplot with shaded error bars (confidence interval) using seaborn
sns.lineplot(x=bin_centers, y=bin_means, color=color2, alpha = 0.5, linewidth = 3, ax=ax)
# plt.fill_between(bin_centers, bin_means - bin_stdevs, bin_means + bin_stdevs, color=color2, alpha=0.2, zorder = 1)
# #also mark points on the mean curve
# plt.scatter(x, y, marker = 'o', color = color2, alpha = 0.1, s = 25, zorder = 2, linewidth = 0)
# plt.scatter(bin_centers, bin_means, marker = 'o', color = color2, s = 100, zorder = 3, alpha = 0.5, label = 'other flat', linewidth = 0)

bin_centers2 = bin_centers
bin_means2 = bin_means
bin_stdevs2 = bin_stdevs



#get lists of lengths and thicknesses from data_whole_list
x = whole_dist_list3
y = whole_thick_list3

#find a list of unique values of distances from origin present in the second entry of all the tuples in data_whole_list
bin_centers = list(set(whole_dist_list3))
#sort the list of bin centers
bin_centers.sort()

#find mean and standard deviation of thickness values for each unique distance value
bin_means = []
bin_stdevs = []
# for bin_center in bin_centers:
#     bin_list = []
#     ctr = 0
#     for ii in range(len(x)):
#         if x[ii] == bin_center:
#             bin_list.append(y[ii])
#             ctr = ctr+1
#     if ctr>2:
#         bin_means.append(np.mean(np.array(bin_list)))
#         bin_stdevs.append(np.std(np.array(bin_list)))
#     else:
#         bin_means.append(np.nan)
#         bin_stdevs.append(np.nan)

for bin_center in bin_centers:
    bin_list = []
    ctr = 0
    for ii in range(len(x)):
        if x[ii] == bin_center:
            bin_list.append(y[ii])
            ctr = ctr+1

    bin_means.append(np.mean(np.array(bin_list)))
    bin_stdevs.append(np.std(np.array(bin_list)))


#convert bin_centers, bin_means and bin_stdevs to numpy arrays
bin_centers = np.array(bin_centers)
bin_means = np.array(bin_means)
bin_stdevs = np.array(bin_stdevs)

# #change the number scale on x and y axis into scientific notation
# ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
# ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

#set max x value as 7
# ax.set_xlim([-0.2, 12.2])

# color3 = 'yellowgreen'
color3 = 'green'
# Create the lineplot with shaded error bars (confidence interval) using seaborn
sns.lineplot(x=bin_centers, y=bin_means, color=color3, alpha = 0.5, linewidth = 3, ax=ax)
# plt.fill_between(bin_centers, bin_means - bin_stdevs, bin_means + bin_stdevs, color=color3, alpha=0.2, zorder = 1)
# #also mark points on the mean curve
# plt.scatter(x, y, marker = 'o', color = color3, alpha = 0.1, s = 25, zorder = 2, linewidth = 0)
# plt.scatter(bin_centers, bin_means, marker = 'o', color = color3, s = 100, zorder = 3, alpha = 0.5, label = 'foot pad', linewidth = 0)

bin_centers3 = bin_centers
bin_means3 = bin_means
bin_stdevs3 = bin_stdevs



# #get lists of lengths and thicknesses from data_whole_list
# x = whole_dist_list3
# y = whole_thick_list3

# #find a list of unique values of distances from origin present in the second entry of all the tuples in data_whole_list
# bin_centers = list(set(whole_dist_list3))
# #sort the list of bin centers
# bin_centers.sort()

# #find mean and standard deviation of thickness values for each unique distance value
# bin_means = []
# bin_stdevs = []
# # for bin_center in bin_centers:
# #     bin_list = []
# #     ctr = 0
# #     for ii in range(len(x)):
# #         if x[ii] == bin_center:
# #             bin_list.append(y[ii])
# #             ctr = ctr+1
# #     if ctr>2:
# #         bin_means.append(np.mean(np.array(bin_list)))
# #         bin_stdevs.append(np.std(np.array(bin_list)))
# #     else:
# #         bin_means.append(np.nan)
# #         bin_stdevs.append(np.nan)

# for bin_center in bin_centers:
#     bin_list = []
#     ctr = 0
#     for ii in range(len(x)):
#         if x[ii] == bin_center:
#             bin_list.append(y[ii])
#             ctr = ctr+1

#     bin_means.append(np.mean(np.array(bin_list)))
#     bin_stdevs.append(np.std(np.array(bin_list)))


# #convert bin_centers, bin_means and bin_stdevs to numpy arrays
# bin_centers = np.array(bin_centers)
# bin_means = np.array(bin_means)
# bin_stdevs = np.array(bin_stdevs)

# # #change the number scale on x and y axis into scientific notation
# # ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
# # ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

# #set max x value as 7
# # ax.set_xlim([-0.2, 12.2])

# color3 = 'yellowgreen'
# # Create the lineplot with shaded error bars (confidence interval) using seaborn
# sns.lineplot(x=bin_centers, y=bin_means, color=color3, alpha = 0.5, linewidth = 3)
# plt.fill_between(bin_centers, bin_means - bin_stdevs, bin_means + bin_stdevs, color=color3, alpha=0.2, zorder = 1)
# #also mark points on the mean curve
# plt.scatter(x, y, marker = 'o', color = color3, alpha = 0.1, s = 25, zorder = 2, linewidth = 0)
# plt.scatter(bin_centers, bin_means, marker = 'o', color = color3, s = 100, zorder = 3, alpha = 0.5, label = 'foot pad 1', linewidth = 0)



# #get lists of lengths and thicknesses from data_whole_list
# x = whole_dist_list4
# y = whole_thick_list4

# #find a list of unique values of distances from origin present in the second entry of all the tuples in data_whole_list
# bin_centers = list(set(whole_dist_list4))
# #sort the list of bin centers
# bin_centers.sort()

# #find mean and standard deviation of thickness values for each unique distance value
# bin_means = []
# bin_stdevs = []
# # for bin_center in bin_centers:
# #     bin_list = []
# #     ctr = 0
# #     for ii in range(len(x)):
# #         if x[ii] == bin_center:
# #             bin_list.append(y[ii])
# #             ctr = ctr+1
# #     if ctr>2:
# #         bin_means.append(np.mean(np.array(bin_list)))
# #         bin_stdevs.append(np.std(np.array(bin_list)))
# #     else:
# #         bin_means.append(np.nan)
# #         bin_stdevs.append(np.nan)

# for bin_center in bin_centers:
#     bin_list = []
#     ctr = 0
#     for ii in range(len(x)):
#         if x[ii] == bin_center:
#             bin_list.append(y[ii])
#             ctr = ctr+1

#     bin_means.append(np.mean(np.array(bin_list)))
#     bin_stdevs.append(np.std(np.array(bin_list)))


# #convert bin_centers, bin_means and bin_stdevs to numpy arrays
# bin_centers = np.array(bin_centers)
# bin_means = np.array(bin_means)
# bin_stdevs = np.array(bin_stdevs)

# # #change the number scale on x and y axis into scientific notation
# # ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
# # ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

# #set max x value as 7
# # ax.set_xlim([-0.2, 12.2])

# color3 = 'olivedrab'
# # Create the lineplot with shaded error bars (confidence interval) using seaborn
# sns.lineplot(x=bin_centers, y=bin_means, color=color3, alpha = 0.5, linewidth = 3)
# plt.fill_between(bin_centers, bin_means - bin_stdevs, bin_means + bin_stdevs, color=color3, alpha=0.2, zorder = 1)
# #also mark points on the mean curve
# plt.scatter(x, y, marker = 'o', color = color3, alpha = 0.1, s = 25, zorder = 2, linewidth = 0)
# plt.scatter(bin_centers, bin_means, marker = 'o', color = color3, s = 100, zorder = 3, alpha = 0.5, label = 'foot pad 2', linewidth = 0)



# # #change axes to log log scale
# # ax.set_xscale('log')
# # ax.set_yscale('log')

# # fit exponential curve to the data


# ctr = 15
# bin_centers_fit = (bin_centers1[ctr:] + bin_centers2[ctr:] + bin_centers3[ctr:]) / 3
# bin_means_fit = (bin_means1[ctr:] + bin_means2[ctr:] + bin_means3[ctr:]) / 3

# popt, pcov = curve_fit(func1, bin_centers_fit, bin_means_fit)

# x_fit = np.linspace(bin_centers1[ctr], 1, 100)
# y_fit = func1(x_fit, *popt)

# plt.plot(x_fit, y_fit, 'k-', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))

# popt, pcov = curve_fit(func, bin_centers, bin_means)

# x_fit = np.linspace(0, 1, 100)
# y_fit = func(x_fit, *popt)

# plt.plot(x_fit, y_fit, 'k-', label='fit: a=%5.3f' % tuple(popt))


# plt.xlabel('Scaled path length from center') 
# plt.ylabel('Scaled medial thickness')

# plt.xlabel('Path length (in mm)')
# plt.ylabel('Medial thickness (in mm)')

#set tick label size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

#set label size
ax.xaxis.label.set_size(18)
ax.yaxis.label.set_size(18)


# #set aspect ratio
ax.set_aspect('equal')
#set axis limits
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

#set x and y tick label
ax.set_xticks([0, 0.5, 1])
ax.set_xticklabels(['0', '0.5', '1'])
ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels(['0', '0.5', '1'])


# #set plot font
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Helvetica'

# plt.legend()

#set legent size
# plt.legend(prop={'size': 14})

# # #save figure
# plt.savefig(os.path.join('thickness_vs_distance_scaled_3.svg'), dpi = 600, bbox_inches = 'tight')
# plt.savefig(os.path.join('thickness_vs_distance_scaled_3.png'), dpi = 600, bbox_inches = 'tight')

# plt.savefig(os.path.join('thickness_vs_distance_scaled_small.svg'), dpi = 600, bbox_inches = 'tight')
# plt.savefig(os.path.join('thickness_vs_distance_scaled_small.png'), dpi = 600, bbox_inches = 'tight')


print(min(x), max(x), min(y), max(y))

plt.show()
