# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Bifurcation angle vs node depth; mean and SD per ossicle group.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Bifurcation-angle panels (Fig. 2G; Fig. S2C)
What it does  : Loads the bifurcation angles and node depths of pillared tables, other
                plates and foot pads, plots angle vs node depth (per-depth mean +/- SD,
                raw points, dashed group means) and prints the mean and SD of each group
                (Fig. 2G values). The commented blocks record how the arrays were
                computed from origin_clean_simple_base_G (split triplets) and saved with
                np.save.
Inputs        : tables/angle/{level,angle}_{table,other_flat,foot_pad}.npy;
                microct/animal_1/extracted_ossicles/Data Calculated lists
Outputs       : printed mean/SD; matplotlib figure (savefig commented out)
Environment   : environment-analysis.yml (Python 3.7)
Run           : python all_angle_plots2.py
"""

#code to make combined plot for all the angles

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
from matplotlib import pyplot as plt
import scipy.spatial as spatial
import os
import math
import itertools
import seaborn as sns
import pandas as pd

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
    full_G = nx.read_gpickle(os.path.join(skel_paths[ii],"full_G.gpickle"))
    clean_full_G = nx.read_gpickle(os.path.join(skel_paths[ii],"clean_full_G.gpickle"))
    simple_G = nx.read_gpickle(os.path.join(skel_paths[ii],"simple_G.gpickle")) 
    clean_simple_G = nx.read_gpickle(os.path.join(skel_paths[ii],"clean_simple_G.gpickle")) 
    area_mid_G = nx.read_gpickle(os.path.join(skel_paths[ii],"area_mid_G.gpickle")) 
    full_base_G = nx.read_gpickle(os.path.join(skel_paths[ii],"full_base_G.gpickle"))
    clean_simple_base_G = nx.read_gpickle(os.path.join(skel_paths[ii],"clean_simple_base_G.gpickle"))
    area_mid_base_G = nx.read_gpickle(os.path.join(skel_paths[ii],"area_mid_base_G.gpickle"))
    origin_clean_simple_base_G = nx.read_gpickle(os.path.join(skel_paths[ii],"origin_clean_simple_base_G.gpickle"))
    return full_G, clean_full_G, simple_G, clean_simple_G, area_mid_G, full_base_G, clean_simple_base_G, area_mid_base_G, origin_clean_simple_base_G


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
        cylinder.point_data['length'] = height*np.ones(cylinder.n_points)

        meshes.append(cylinder)
        lengths.append(height)

    return meshes, lengths


def get_all_angle_triplets(G):
    triplets = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        for triplet in itertools.combinations(neighbors, 2):
            triplets.append((node,) + triplet)

    return triplets

def get_split_angle_triplets(G):
    triplets = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if not neighbors:  # skip node if it has no neighbors
            continue

        level_str = 'level_bound_dist'
        node_level = G.nodes[node][level_str]
        for n1, n2 in itertools.combinations(neighbors, 2):
            if node_level < G.nodes[n1][level_str] and node_level < G.nodes[n2][level_str]:
                triplets.append((node, n1, n2))
    return triplets

def get_triplet_angles(G, triplets):
    angles = {}
    for node, n1, n2 in triplets:
        x, y, z = G.nodes[node]['pos']
        x1, y1, z1 = G.nodes[n1]['pos']
        x2, y2, z2 = G.nodes[n2]['pos']

        # compute the vectors of the edges
        u = (x1 - x, y1 - y, z1 - z)
        v = (x2 - x, y2 - y, z2 - z)

        # compute the dot product and the norm of the vectors
        dot = u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
        norm_u = math.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
        norm_v = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

        # compute the splitting angle
        cos_theta = dot / (norm_u * norm_v)
        if cos_theta > 1:
            cos_theta = 1
        elif cos_theta < -1:
            cos_theta = -1
        angle = math.acos(cos_theta)
        angles[(node, n1, n2)] = angle*180/np.pi
    return angles

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

with open(os.path.join(datadir,'loc_file_id_front_rear.txt')) as txtfile:
    loc_file_id_front_rear = list(map(int, txtfile.read().split('\n')))

#read selected numpy files
with open(os.path.join(datadir, 'selected_table.npy'), 'rb') as f:
    selected_table = np.load(f).tolist()

with open(os.path.join(datadir, 'selected_other_flat.npy'), 'rb') as f:
    selected_other_flat = np.load(f).tolist()

plt3d = vedo.Plotter(bg2='gray2')#, interactive=True) # screen size
# pl = pv.Plotter(off_screen=False)
pl = pv.Plotter(off_screen=True)
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

#%%

# level_all1 = []
# angle_all1 = []
# data_all1 = []

# active_loc_list = selected_table

# for jj in list(range(len(active_loc_list))): #list(range(0,10))+ list(range(11,len(active_loc_list))): #range(n_files):      print(ii, files[ii])

#     kk = active_loc_list[jj] - 1
#     ii = loc_file_id[kk]


#     ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)

#     if len(skeleton.lines())!=0 and len(cl_skeleton.lines())!=0: #  and ossicle_ind not in [13, 23,67,182, 192, 201,209]:  #and jj not in [13,142, 154]
#         full_G, clean_full_G, simple_G, clean_simple_G, area_mid_G, full_base_G, clean_simple_base_G, area_mid_base_G, origin_clean_simple_base_G = load_graphs(ii)
#         G = origin_clean_simple_base_G

#         #load mesh represnting the base surface
#         # rem_recon_mesh = vedo.load(os.path.join(basemeshdir, file_names[ii]))

#         all_triplets = get_all_angle_triplets(G)
#         split_triplets = get_split_angle_triplets(G)
#         all_angles = get_triplet_angles(G, all_triplets)
#         split_angles = get_triplet_angles(G, split_triplets)

#         all_angle_vals = list(all_angles.values())
#         split_angle_vals = list(split_angles.values())


#         pos_list= []
#         angle_list = []
#         level_list = []
#         data_list = []
#         for triplet in split_triplets:
#             angle = split_angles[triplet]
#             node = triplet[0]
#             pos = G.nodes[node]['pos']
#             level =  G.nodes[node]['level_topo']
#             # G.nodes[triplet[0]]['angle'] = 
#             pos_list.append(pos)
#             angle_list.append(angle)
#             level_list.append(level)
#             data = [kk,  pos, angle, level]
#             data_list.append(data)



#         level_all1.extend(level_list)
#         angle_all1.extend(angle_list)
#         data_all1.extend(data_list)

# # vedo.show(meshes, axes=1)

# with open('level_table.npy', 'wb') as ff:
#     np.save(ff, np.array(level_all1))

# with open('angle_table.npy', 'wb') as gg:
#     np.save(gg, np.array(angle_all1))

# with open('data_table.npy', 'wb') as ff:
#     np.save(ff, np.array(data_all1))



# level_all2 = []
# angle_all2 = []
# data_all2 = []

# active_loc_list = selected_other_flat

# for jj in list(range(len(active_loc_list))): #list(range(0,10))+ list(range(11,len(active_loc_list))): #range(n_files):      print(ii, files[ii])

#     kk = active_loc_list[jj] - 1
#     ii = loc_file_id[kk]
#     # ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)

#     ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)

#     if len(skeleton.lines())!=0 and len(cl_skeleton.lines())!=0: #  and ossicle_ind not in [13, 23,67,182, 192, 201,209]:  #and jj not in [13,142, 154]
#         full_G, clean_full_G, simple_G, clean_simple_G, area_mid_G, full_base_G, clean_simple_base_G, area_mid_base_G, origin_clean_simple_base_G = load_graphs(ii)
#         G = origin_clean_simple_base_G

#         #load mesh represnting the base surface
#         # rem_recon_mesh = vedo.load(os.path.join(basemeshdir, file_names[ii]))

#         all_triplets = get_all_angle_triplets(G)
#         split_triplets = get_split_angle_triplets(G)
#         all_angles = get_triplet_angles(G, all_triplets)
#         split_angles = get_triplet_angles(G, split_triplets)

#         all_angle_vals = list(all_angles.values())
#         split_angle_vals = list(split_angles.values())


#         pos_list= []
#         angle_list = []
#         level_list = []
#         data_list = []
#         for triplet in split_triplets:
#             angle = split_angles[triplet]
#             node = triplet[0]
#             pos = G.nodes[node]['pos']
#             level =  G.nodes[node]['level_topo']
#             # G.nodes[triplet[0]]['angle'] = 
#             pos_list.append(pos)
#             angle_list.append(angle)
#             level_list.append(level)
#             data = [kk,  pos, angle, level]
#             data_list.append(data)



#         level_all2.extend(level_list)
#         angle_all2.extend(angle_list)
#         data_all2.extend(data_list)

# # vedo.show(meshes, axes=1)

# with open('level_other_flat.npy', 'wb') as ff:
#     np.save(ff, np.array(level_all2))

# with open('angle_other_flat.npy', 'wb') as gg:
#     np.save(gg, np.array(angle_all2))

# with open('data_other_flat.npy', 'wb') as ff:
#     np.save(ff, np.array(data_all2))



# level_all3 = []
# angle_all3 = []
# data_all3 = []

# #get file list in folder
# graph_folder = str(DATA_ROOT / "footpad" / "graphs")

# #get file names of all graphs with 'simple_oirgin' some where in the name
# simple_graph_file_names = [f for f in sorted(os.listdir(graph_folder), key=str.upper) if 'simple_origin' in f]
# full_graph_file_names = [f for f in sorted(os.listdir(graph_folder), key=str.upper) if 'full' in f] 


# for jj in [0,1]: #list(range(0,10))+ list(range(11,len(active_loc_list))): #range(n_files):      print(ii, files[ii])

#     # kk = active_loc_list[jj] - 1
#     kk = jj
#     # ii = loc_file_id[kk]


#     # ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)

#     if 1==2:
#         continue
#     else:
#         # clean_full_G, clean_simple_G, origin_clean_simple_G, clean_full_base_G, clean_simple_base_G,  origin_clean_simple_base_G = load_graphs(ii)
#         sim_G_origin = nx.read_gpickle(os.path.join(graph_folder, simple_graph_file_names[jj]))
#         full_G = nx.read_gpickle(os.path.join(graph_folder, full_graph_file_names[jj]))

#         G = sim_G_origin

#         #load mesh represnting the base surface
#         # rem_recon_mesh = vedo.load(os.path.join(basemeshdir, file_names[ii]))

#         all_triplets = get_all_angle_triplets(G)
#         split_triplets = get_split_angle_triplets(G)
#         all_angles = get_triplet_angles(G, all_triplets)
#         split_angles = get_triplet_angles(G, split_triplets)

#         all_angle_vals = list(all_angles.values())
#         split_angle_vals = list(split_angles.values())


#         pos_list= []
#         angle_list = []
#         level_list = []
#         data_list = []
#         for triplet in split_triplets:
#             angle = split_angles[triplet]
#             node = triplet[0]
#             pos = G.nodes[node]['pos']
#             level =  G.nodes[node]['level_topo']
#             # G.nodes[triplet[0]]['angle'] = 
#             pos_list.append(pos)
#             angle_list.append(angle)
#             level_list.append(level)
#             data = [kk,  pos, angle, level]
#             data_list.append(data)



#         level_all3.extend(level_list)
#         angle_all3.extend(angle_list)
#         data_all3.extend(data_list)

# # vedo.show(meshes, axes=1)

# with open('level_foot_pad.npy', 'wb') as ff:
#     np.save(ff, np.array(level_all3))

# with open('angle_foot_pad.npy', 'wb') as gg:
#     np.save(gg, np.array(angle_all3))

# with open('data_foot_pad.npy', 'wb') as ff:
#     np.save(ff, np.array(data_all3))



#%% Load files

angle_dir = str(DATA_ROOT / "tables" / "angle")

level_1 = np.load(os.path.join(angle_dir, 'level_table.npy'))
angle_1 = np.load(os.path.join(angle_dir, 'angle_table.npy'))

level_2 = np.load(os.path.join(angle_dir, 'level_other_flat.npy'))
angle_2 = np.load(os.path.join(angle_dir, 'angle_other_flat.npy'))

level_3 = np.load(os.path.join(angle_dir, 'level_foot_pad.npy'))
angle_3 = np.load(os.path.join(angle_dir, 'angle_foot_pad.npy'))


#%%

#create a mean and shaded error plot of thickness vs distance values
fig, ax = plt.subplots(figsize = (10,5))
#get lists of lengths and thicknesses from data_whole_list
x = level_1
y = angle_1

#find min and max values of distances from origin present in the second entry of all the tuples in data_whole_list
min_level = min(level_1)
max_level = max(level_1)

#create bins from range of distances from origin
bin_edges = np.arange(-0.5, max_level+1,1) 
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Bin data
df = pd.DataFrame({'x': x, 'y': y})
df['bin'] = pd.cut(df['x'], bins=bin_edges)

# Compute mean and standard deviation for each bin
bin_means = df.groupby('bin')['y'].mean().values
bin_stdevs = df.groupby('bin')['y'].std().values

color1 = 'orange'
color1_1 = 'chocolate'
# Create the lineplot with shaded error bars (confidence interval) using seaborn
sns.lineplot(x=bin_centers, y=bin_means, color=color1_1, alpha = 0.8, linewidth = 3)
plt.fill_between(bin_centers, bin_means - bin_stdevs, bin_means + bin_stdevs, color=color1, alpha=0.3, zorder = 1)
#also mark points on the mean curve
plt.scatter(x, y, marker = 'o', color = color1, alpha = 0.05, s = 25, zorder = 2, linewidth = 0)
plt.scatter(bin_centers, bin_means, marker = 'o', color = color1_1, s = 100, zorder = 3, alpha = 0.5, label = 'table', linewidth = 0)


#add a horiontal line with mean angle value
plt.axhline(y=np.mean(y), color=color1_1, linestyle='--', alpha = 0.5, linewidth = 3)


#get lists of lengths and thicknesses from data_whole_list
x = level_2
y = angle_2

#find min and max values of distances from origin present in the second entry of all the tuples in data_whole_list
min_level = min(level_2)
max_level = max(level_2)

#create bins from range of distances from origin
bin_edges = np.arange(-0.5, max_level+1,1) 
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Bin data
df = pd.DataFrame({'x': x, 'y': y})
df['bin'] = pd.cut(df['x'], bins=bin_edges)

# Compute mean and standard deviation for each bin
bin_means = df.groupby('bin')['y'].mean().values
bin_stdevs = df.groupby('bin')['y'].std().values

# #change the number scale on x and y axis into scientific notation
# ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
# ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

color2 = 'steelblue'
# Create the lineplot with shaded error bars (confidence interval) using seaborn
sns.lineplot(x=bin_centers, y=bin_means, color=color2, alpha = 0.5, linewidth = 3)
plt.fill_between(bin_centers, bin_means - bin_stdevs, bin_means + bin_stdevs, color=color2, alpha=0.2, zorder = 1)
#also mark points on the mean curve
plt.scatter(x, y, marker = 'o', color = color2, alpha = 0.1, s = 25, zorder = 2, linewidth = 0)
plt.scatter(bin_centers, bin_means, marker = 'o', color = color2, s = 100, zorder = 3, alpha = 0.5, label = 'other flat', linewidth = 0)

#add a horiontal line with mean angle value
plt.axhline(y=np.mean(y), color=color2, linestyle='--', alpha = 0.5, linewidth = 3)



#get lists of lengths and thicknesses from data_whole_list
x = level_3
y = angle_3

#find min and max values of distances from origin present in the second entry of all the tuples in data_whole_list
min_level = min(level_3)
max_level = max(level_3)

#create bins from range of distances from origin
bin_edges = np.arange(-0.5, max_level+1,1) 
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Bin data
df = pd.DataFrame({'x': x, 'y': y})
df['bin'] = pd.cut(df['x'], bins=bin_edges)

# Compute mean and standard deviation for each bin
bin_means = df.groupby('bin')['y'].mean().values
bin_stdevs = df.groupby('bin')['y'].std().values

# #change the number scale on x and y axis into scientific notation
# ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
# ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

color3 = 'green'
# Create the lineplot with shaded error bars (confidence interval) using seaborn
sns.lineplot(x=bin_centers, y=bin_means, color=color3, alpha = 0.5, linewidth = 3)
plt.fill_between(bin_centers, bin_means - bin_stdevs, bin_means + bin_stdevs, color=color3, alpha=0.2, zorder = 1)
#also mark points on the mean curve
plt.scatter(x, y, marker = 'o', color = color3, alpha = 0.1, s = 25, zorder = 2, linewidth = 0)
plt.scatter(bin_centers, bin_means, marker = 'o', color = color3, s = 100, zorder = 3, alpha = 0.5, label = 'foot pad', linewidth = 0)

#add a horiontal line with mean angle value
plt.axhline(y=np.mean(y), color=color3, linestyle='--', alpha = 0.5, linewidth = 3)



#set max x value as 7
# ax.set_xlim([-0.2, 12.2])

plt.xlabel('Node depth from origin') 
plt.ylabel('Angle (in degrees)')

#set tick label size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

#set label size
ax.xaxis.label.set_size(18)
ax.yaxis.label.set_size(18)

# #set plot font
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Helvetica'

plt.legend()

#set legent size
plt.legend(prop={'size': 14})

#save figure
# plt.savefig(os.path.join('edge_length_vs_level3.svg'), dpi = 300, bbox_inches = 'tight')
# plt.savefig(os.path.join('edge_length_vs_level3.png'), dpi = 300, bbox_inches = 'tight')

# plt.show()

#print mean values of angles for both datasets
print('mean angle for table: ', np.mean(angle_1))
print('mean angle for other flat: ', np.mean(angle_2))
print('mean angle for foot pad: ', np.mean(angle_3))

#print standard deviation of angles for both datasets
print('std angle for table: ', np.std(angle_1))
print('std angle for other flat: ', np.std(angle_2))
print('std angle for foot pad: ', np.std(angle_3))



# sorted_list = []
# for ii in range(1,13):
#     sorted_list.append(angle_other[np.where(level_other == ii)[0]])

# # plt.scatter(level_table, angle_table)
# fig, ax = plt.subplots(figsize=(24, 12))
# ax.set_ylim(0,180)

# # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
# sns.set(style="darkgrid")
# sns.set(font_scale=3)
# # Grouped violinplot
# g = sns.violinplot(data=sorted_list, palette="vlag", ax=ax, cut = 0)
# sns.stripplot(data=sorted_list, size=3, color=".3", linewidth=0, alpha = 0.1, ax=ax)
# g.set_xticklabels(['1','2','3','4','5','6', '7', '8', '9', '10', '11', '12'])
# fig.show()

# # plt.savefig('other_flat_angle.png', dpi=600)


# with open('level_table.npy', 'rb') as f:
#     level_table = np.load(f)

# with open('angle_table.npy', 'rb') as f:
#     angle_table = np.load(f)

# sorted_list = []
# for ii in range(1,13):
#     sorted_list.append(angle_table[np.where(level_table == ii)[0]])

# # plt.scatter(level_table, angle_table)
# fig1, ax1 = plt.subplots(figsize=(24, 12))
# ax1.set_ylim(0,180)

# # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
# sns.set(style="darkgrid")
# sns.set(font_scale=3)
# # Grouped violinplot
# g = sns.violinplot(data=sorted_list, palette="vlag", ax=ax1, cut = 0)
# sns.stripplot(data=sorted_list, size=3, color=".3", linewidth=0, alpha = 0.1, ax=ax1)
# g.set_xticklabels(['1','2','3','4','5','6', '7', '8', '9', '10', '11', '12'])
# fig1.show()

# plt.savefig('table_angle.png', dpi=600)


# with open('level_all_flat.npy', 'rb') as f:
#     level_all = np.load(f)

# with open('angle_all_flat.npy', 'rb') as f:
#     angle_all = np.load(f)

# sorted_list = []
# for ii in range(1,13):
#     sorted_list.append(angle_all[np.where(level_all == ii)[0]])

# # plt.scatter(level_table, angle_table)
# fig2, ax2 = plt.subplots(figsize=(24, 12))
# ax2.set_ylim(0,180)

# # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
# sns.set(style="darkgrid")
# sns.set(font_scale=3)
# # Grouped violinplot
# g = sns.violinplot(data=sorted_list, palette="vlag", ax=ax2, cut = 0)
# sns.stripplot(data=sorted_list, size=3, color=".3", linewidth=0, alpha = 0.1, ax=ax2)
# g.set_xticklabels(['1','2','3','4','5','6', '7', '8', '9', '10', '11', '12'])
# fig2.show()

# plt.savefig('all_flat_angle.png', dpi=600)

# with open('level_front_rear.npy', 'rb') as f:
#     level_fr = np.load(f)

# with open('angle_front_rear.npy', 'rb') as f:
#     angle_fr = np.load(f)

# sorted_list = []
# for ii in range(1,13):
#     sorted_list.append(angle_fr[np.where(level_fr == ii)[0]])

# # plt.scatter(level_table, angle_table)
# fig3, ax3 = plt.subplots(figsize=(24, 12))
# ax3.set_ylim(0,180)

# # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
# sns.set(style="darkgrid")
# sns.set(font_scale=3)
# # Grouped violinplot
# g = sns.violinplot(data=sorted_list, palette="vlag", ax=ax3, cut = 0)
# sns.stripplot(data=sorted_list, size=3, color=".3", linewidth=0, alpha = 0.1, ax=ax3)
# g.set_xticklabels(['1','2','3','4','5','6', '7', '8', '9', '10', '11', '12'])
# fig3.show()

# plt.savefig('front_rear_angle.png', dpi=600)

# show the meshes

# pl.add_scalar_bar(title='Angle (in degrees)', mapper = None, n_labels=5, italic=False, bold=False, title_font_size=None, label_font_size=None, color=None, font_family=None, shadow=False, width=None, height=None, position_x=None, position_y=None, vertical=None, interactive=None, fmt=None, use_opacity=True, outline=False, nan_annotation=False, below_label=None, above_label=None, background_color=None, n_colors=None, fill=False, render=False, theme=None)

# pl.camera.roll = 90.0
# pl.screenshot('206_front_angle.png', window_size=(10000,10000), transparent_background= True)
# pl.show()
# plt3d.show(skeleton, hole_mesh_list)
# 

# # Plot the frequency distribution of values using seaborn
# sns.distplot(all_angle_vals)

# # Show the plot
# plt.show()
