# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Cut pillars from ossicle graphs and mark the lattice boundary.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Pipeline step (base graphs used by Fig. 2C-H, S2C-D, S3)
What it does  : For the selected aligned ossicle (active: loc_file_id[209]) removes
                every graph node above z = 0.003 mm (pillar cut), finds the outer
                boundary from the minimum cycle basis (edges belonging to exactly one
                cycle, largest projected polygon, plus degree-1 tips) and gives boundary
                nodes/edges the level 1000. Saves the four *_base_G graphs and renders
                the ossicle, skeleton, graph, boundary nodes (blue) and origin (red).
Inputs        : microct/animal_1/extracted_ossicles/aligned/{vtk,
                skeleton/<ossicle>/(vox.mhd, *skel_line.vtk, full_G, clean_full_G,
                simple_G, clean_simple_G, area_mid_G, origin_clean_simple_G .gpickle)},
                microct/animal_1/extracted_ossicles/Data Calculated/{loc_file_id.txt,
                loc_file_id_table.txt, centroids.npy, avg_mean_curvature.npy}
Outputs       : microct/animal_1/extracted_ossicles/aligned/skeleton/<ossicle>/{clean_simple_base_G,
                area_mid_base_G, clean_full_base_G, origin_clean_simple_base_G}.gpickle
                (data tree); PyVista window
Environment   : environment-analysis.yml (Python 3.7)
Run           : python extract_base_graphs.py
"""

from vedo import *
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
import matplotlib.pyplot as plt
import math
import os

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
pv.global_theme.camera = {'position': [0,0,-1],'viewup': [0, 0, -1]}

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
    area_mid_G = nx.read_gpickle(os.path.join(skel_paths[ii],"area_mid_G.gpickle")) 
    origin_clean_simple_G = nx.read_gpickle(os.path.join(skel_paths[ii],"origin_clean_simple_G.gpickle")) 

    return full_G, clean_full_G, simple_G, clean_simple_G, area_mid_G, origin_clean_simple_G


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


def remove_vertical_nodes(H):
    G = H.copy()
    node_list = list(G.nodes)
    rem_list = []
    for node in node_list:
        if (G.nodes[node]['pos'][2] > 0.003) :      # 0.003 by deafault
        # if (G.nodes[node]['pos'][2] > 0.005) :      # 0.003 by deafault
        # if (G.nodes[node]['pos'][2] > 0.01) :      # 0.003 by deafault
            G.remove_node(node)

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

# #assign boundary nodes and edges

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

plt3d = Plotter(bg2='gray2')#, interactive=True) # screen size
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

for jj in [209]: #53,54,56,62,63,65,73,87,94,102,103,106,110,111,117,118,119,120,143,144,145,150,153,154,155,157,158,161,166,171,177,183,185,189,195,197,198,199,201,209   # list(range(0,13))+list(range(14,18))+list(range(19,23))+list(range(24,31))+list(range(32,67))+list(range(68,95))+list(range(96,182))+list(range(183,213)): # list(range(31))+list(range(32,213)):      

    ii = loc_file_id[jj]
    print(jj, ii, files[ii])
    ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)

    if len(skeleton.lines())!=0 and len(cl_skeleton.lines())!=0:
        full_G, clean_full_G, simple_G, clean_simple_G, area_mid_G, origin_clean_simple_G = load_graphs(ii)

        # rem_recon_mesh = load(os.path.join(basemeshdir, file_names[ii]))

        clean_simple_base_G = remove_vertical_nodes(clean_simple_G)
        area_mid_base_G = remove_vertical_nodes(area_mid_G)
        clean_full_base_G = remove_vertical_nodes(clean_full_G)
        origin_clean_simple_base_G = remove_vertical_nodes(origin_clean_simple_G)
        print("vert removed")
        origin_clean_simple_base_G = assign_boundary_nodes_edges2(origin_clean_simple_base_G)
        print("boundary established")
        nx.write_gpickle(clean_simple_base_G, os.path.join(skel_paths[ii],"clean_simple_base_G.gpickle"))
        nx.write_gpickle(area_mid_base_G, os.path.join(skel_paths[ii],"area_mid_base_G.gpickle"))
        nx.write_gpickle(clean_full_base_G, os.path.join(skel_paths[ii],"clean_full_base_G.gpickle"))
        nx.write_gpickle(origin_clean_simple_base_G, os.path.join(skel_paths[ii],"origin_clean_simple_base_G.gpickle"))


        G = origin_clean_simple_base_G

        # node_list = list(G.nodes())
        # edge_list = list(G.edges())
        # pos_list= []
        # for node in G.nodes():
        #     pos_value = G.nodes[node]['pos']
        #     pos_list.append([pos_value[0],pos_value[1],pos_value[2]])

        # edges = []
        # for edge in edge_list:
        #     edges.append([2, edge[0], edge[1]])

        # # vedo_mesh = vedo.Mesh([pos_list, edge_list])
        # # pv_mesh = pv.PolyData(np.array(pos_list), lines = np.hstack(edges))

        # cloud = pv.PolyData(pos_list)
        # pv_mesh = cloud.delaunay_2d()        
        # vedo_mesh = vedo.Mesh(pv_mesh)
        # pids = vedo_mesh.boundaries(return_point_ids=True)
        # bpts = vedo_mesh.points()

        # for u, v, data in G.edges(data=True):
        #     if (data['branch_degree']==0):
        #         start_edge = [u,v]

        # # find the start point and volumetric centroid
        # start_pt = (G.nodes[start_edge[0]]['pos'] + G.nodes[start_edge[1]]['pos'])/2 
        # centroid_pt = sort_cent[jj]

        # start_sph_mesh = vedo.Sphere(start_pt, r=0.006, res=16)
        # centroid_sph_mesh = vedo.Sphere(centroid_pt, r=0.006    , res=16)


    pv_ossicle = pv.wrap(ossicle.polydata())
    pv_skeleton = pv.wrap(skeleton.polydata())
    # # pv_rem_recon = pv.wrap(rem_recon_mesh.polydata())
    pv_graph =  graph_to_pv_mesh(G)


    # # pv_start = pv.wrap(start_sph_mesh.polydata())
    # # pv_centroid = pv.wrap(centroid_sph_mesh.polydata())
    # # pv_ossicle.point_data['data'] = data_list[jj]  #*(10**4)*2
    cmap = mpl.cm.get_cmap('plasma')
    # # rgba = cmap(data_list[jj]*(10**4)*2)
    # # osc_color = pv.Color([rgba[0], rgba[1], rgba[2] ])
    # # merged = pv_ossicle.merge(merged)

    # # pl.add_mesh(pv_start, color = 'steelblue')
    # # pl.add_mesh(pv_centroid, color = 'steelblue')
    #     # pl.add_text(str(jj), position =0,0,0 , color='blue', shadow=True, font_size=26)

    pl.add_mesh(pv_ossicle, cmap = cmap, opacity = 0.5, diffuse = 0.9, smooth_shading = True, show_scalar_bar= False)#, clim = [0,0.13])#osc_color)#, specular=1.0, specular_power=10)
    pl.add_mesh(pv_skeleton, cmap= cmap)
    # # pl.add_mesh(pv_rem_recon, color = 'yellow', opacity =0.5)    
    pl.add_mesh(pv_graph, color = 'black', render_lines_as_tubes=True, render_points_as_spheres = True, style='wireframe', line_width=10, point_size = 20, show_scalar_bar=False)
    # # pl.add_mesh(pv_graph, color = 'red', render_lines_as_tubes=True, render_points_as_spheres = True, style='points', line_width=10, point_size = 20, show_scalar_bar=False)

    # # # pl.add_mesh()
    boundary_nodes = [node for node in origin_clean_simple_base_G.nodes() if origin_clean_simple_base_G.nodes[node]['level_bound_topo'] == 1000]
    for node in boundary_nodes:
        node_pos = origin_clean_simple_base_G.nodes[node]['pos']
        node_sph_mesh = pv.Sphere(center = node_pos, radius=0.001, theta_resolution=16, phi_resolution=10)
        pl.add_mesh(node_sph_mesh, color = 'blue')

    #plot origin node marked as highest node id
    # find max node id
    origin_node = max(origin_clean_simple_base_G.nodes(), key=int)    
    origin_node_pos = origin_clean_simple_base_G.nodes[origin_node]['pos']
    origin_node_sph_mesh = pv.Sphere(center = origin_node_pos, radius=0.002, theta_resolution=16, phi_resolution=10)
    pl.add_mesh(origin_node_sph_mesh, color = 'red')


#     # Get the mesh from the graph
#     gmesh = graph_to_vedo_mesh(G)
#     gmesh.lw(5)
#     gmesh2 = gmesh.clone()
#     gmesh2.ps(15).c('r')
#     # meshes.extend([gmesh,gmesh2])
#     meshes.extend([ossicle.alpha(0.2)])
#     meshes.append(cl_skeleton)


# #     pts = Points(bpts[pids], r=10, c='red5')
# #     meshes.append(vedo_mesh.boundaries())
# #     meshes.append(pts)

# vedo.show(meshes, axes=1)


# show the meshes

pl.add_axes()
# pl.add_scalar_bar(title='Medial Thickness(in mm)', mapper = None, n_labels=5, italic=False, bold=False, title_font_size=None, label_font_size=None, color=None, font_family=None, shadow=False, width=None, height=None, position_x=None, position_y=None, vertical=None, interactive=None, fmt=None, use_opacity=True, outline=False, nan_annotation=False, below_label=None, above_label=None, background_color=None, n_colors=None, fill=False, render=False, theme=None)

pl.camera.roll = 90.0
# # pl.screenshot('ani_volume_map.png', window_size=(10000,10000), transparent_background= True)
pl.show()
# 
