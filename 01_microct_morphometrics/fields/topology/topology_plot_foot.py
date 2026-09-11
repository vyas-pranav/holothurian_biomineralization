# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Topological-depth tree of a foot-pad graph.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Topology panels (Fig. 2C, 2E; Fig. S2C-D)
What it does  : Same flat topology tree as topology_plot.py for the second foot-pad
                graph (jj = 1).
Inputs        : footpad/graphs/test{1,2}graph_{simple_origin,full}.gpickle (foot-pad
                graphs from 02_image_morphometrics_2d/footpad);
                microct/animal_1/extracted_ossicles/Data Calculated lists (loaded,
                unused)
Outputs       : 2_topo_plot.png (10000 x 10000 px) in the working directory
Environment   : environment-analysis.yml (Python 3.7)
Run           : python topology_plot_foot.py
"""

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

        level_str = 'level_bound'   #'level_dist'
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

#a function to plot nodes at different levels of topology in different rows starting from the center node 
def plot_topology(G):
    graph = G.copy()
    #get level topo values for all nodes and assign positions
    level_topo_dict = nx.get_node_attributes(graph, 'level_topo')

    #create a list of nodes based on increasing level_topo values
    sort_nodes = []
    for node in graph.nodes():
        sort_nodes.append((node, level_topo_dict[node]))
    sort_nodes.sort(key=lambda x: x[1])

    #create a dictionary with level_topo values as keys and a list with all nodes with that level_topo value as value
    topo2node_dict = {}
    for node in sort_nodes:
        if node[1] in topo2node_dict.keys():
            topo2node_dict[node[1]].append(node[0])
        else:
            topo2node_dict[node[1]] = [node[0]]


    #set plot position of nodes as with attribute 'plot_pos' with all nodes at the same level_topo value having the same y value and x value increasing with their angle orientation 
    #of the node with the center node

    #add parent node to each node, traverse each node starting from the center and its neighbors and assign parent node based on the level_topo value
    for node in graph.nodes():
        if level_topo_dict[node] == 0:
            graph.nodes[node]['parent'] = []
        else:
            neighbors = list(graph.neighbors(node))
            parent = []
            for neighbor in neighbors:
                if level_topo_dict[neighbor] < level_topo_dict[node]:
                    parent.append(neighbor)
            graph.nodes[node]['parent'] = parent


    for topo in topo2node_dict.keys():
        #assign plot positions for nodes at each level_topo value
        #get list of nodes at this level_topo value

        if topo == 0:
            node_val = topo2node_dict[topo][0]
            graph.nodes[node_val]['plot_pos'] = np.array([0,0])
        else:
            nodes = topo2node_dict[topo]
            n_nodes = len(nodes)
            angle_list = []
            for node in nodes:
                pos_value = graph.nodes[node]['pos']
                angle = math.atan2(pos_value[1], pos_value[0])
                angle_list.append(angle)

            #sort nodes based on angle
            sort_ind = np.argsort(angle_list)
            nodes = np.array(nodes)[sort_ind].tolist()

            #assign plot positions
            #symmetrically arrange x vaues of all nodes around 0
            # x_vals = np.linspace(-n_nodes/2,n_nodes/2,n_nodes)
            x_vals = np.linspace(-(n_nodes-1)/2,(n_nodes-1)/2,n_nodes)
            for ii in range(n_nodes):
                node = nodes[ii]
                graph.nodes[node]['plot_pos'] = np.array([x_vals[ii], -topo])


    return graph



# #given a graph, plot it as a binary tree starting with the central node and then looking for daugters at different topology levels from center



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


plt3d = vedo.Plotter(bg2='gray2')#, interactive=True) # screen size



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

#get file list in folder
graph_folder = str(DATA_ROOT / "footpad" / "graphs")

#get file names of all graphs with 'simple_oirgin' some where in the name
simple_graph_file_names = [f for f in sorted(os.listdir(graph_folder), key=str.upper) if 'simple_origin' in f]
full_graph_file_names = [f for f in sorted(os.listdir(graph_folder), key=str.upper) if 'full' in f] 


for jj in [1]: #list(range(0,10))+ list(range(11,len(active_loc_list))): #range(n_files):      print(ii, files[ii])

    # kk = active_loc_list[jj] - 1
    kk = jj
    # ii = loc_file_id[kk]


    # ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)
    do_screenshot = True
    if do_screenshot:
        pl = pv.Plotter(off_screen=True)
    else:
        pl = pv.Plotter(off_screen=False)

    meshes = []


    if 1==2:
        continue
    else:
        print('yes')
        # clean_full_G, clean_simple_G, origin_clean_simple_G, clean_full_base_G, clean_simple_base_G,  origin_clean_simple_base_G = load_graphs(ii)
        sim_G_origin = nx.read_gpickle(os.path.join(graph_folder, simple_graph_file_names[jj]))
        full_G = nx.read_gpickle(os.path.join(graph_folder, full_graph_file_names[jj]))

        G = sim_G_origin
        # pv_rem_recon = pv.wrap(rem_recon_mesh.polydata()).subdivide(2)
        pv_graph =  graph_to_pv_mesh(G)    #.subgraph(min_cycle_basis_G[20])


        # test_loop = pv.PolyData(pv.wrap(hole_mesh_list[0].polydata()).points)
        # ctr = 0
        # pv_hole_list = []
        # for surf in hole_mesh_list:
        #     pv_hole = pv.wrap(surf.polydata())
        #     pv_hole.point_data['area_value'] = hole_area_list[ctr] * np.ones([pv_hole.n_points,1])
        #     pv_hole_list.append(pv_hole)
        #     ctr+=1
        # cyl_list, length_list = nx_to_pv_cylinders(G, 0.0001, 10)

        # pv_start = pv.wrap(start_sph_mesh.polydata())
        # pv_centroid = pv.wrap(centroid_sph_mesh.polydata())
        # pv_ossicle.point_data['data'] = data_list[jj]  #*(10**4)*2
        cmap1 = mpl.cm.get_cmap('plasma')
        cmap2 = mpl.cm.get_cmap('plasma')


        # rgba = cmap(hole_area_list[ctr]*(10**3))
        # osc_color = pv.Color([rgba[0], rgba[1], rgba[2] ])
        # merged = pv_ossicle.merge(merged)

        # pl.add_mesh(pv_start, color = 'steelblue')
        # pl.add_mesh(pv_centroid, color = 'steelblue')
            # pl.add_text(str(jj), position =0,0,0 , color='blue', shadow=True, font_size=26)

        # pl.add_mesh(pv_ossicle, color = 'grey', opacity = 0.2, diffuse = 0.9, smooth_shading = True, show_scalar_bar= True)#, clim = [0,0.13])#osc_color)#, specular=1.0, specular_power=10)
        # pl.add_mesh(pv_skeleton, cmap= cmap)
        # pl.add_mesh(pv_rem_recon, color = 'yellow', opacity =0.5,style = 'wireframe')    
        # pl.add_mesh(pv_graph, color = 'black', render_lines_as_tubes=True, render_points_as_spheres = True, style='wireframe', line_width=10, point_size = 40, show_scalar_bar=False) #line_width = 50
        # pl.add_mesh(pv_graph, color = 'red', render_lines_as_tubes=True, render_points_as_spheres = True, style='points', line_width=10, point_size = 20, show_scalar_bar=False)
        # pl.add_mesh(test_loop, color= 'green', render_points_as_spheres = True, point_size = 20)    



        # #add arrows for all the edges of the graph with direction going from node with lower level_topo to higher level_topo
        # for edge in G.edges():
        #     topo1 = G.nodes[edge[0]]['level_bound_topo']
        #     topo2 = G.nodes[edge[1]]['level_bound_topo']

        #     if topo1 >= 1000 and topo2 >= 1000:
        #         continue
        #     else:
        #         if topo1 < topo2:
        #             node1 = edge[0]
        #             node2 = edge[1]
        #         else:
        #             node1 = edge[1]
        #             node2 = edge[0]

        #         pos_value1 = G.nodes[node1]['pos']
        #         pos_value2 = G.nodes[node2]['pos']
        #         dir_vector = (pos_value2 - pos_value1)/np.linalg.norm(pos_value2 - pos_value1)
        #         dir_mag = np.linalg.norm(pos_value2 - pos_value1)
        #         mag_vector = dir_vector*dir_mag
        #         arrow = pv.Arrow(pos_value1, mag_vector, shaft_radius=0.00075/dir_mag, tip_radius=0.0015/dir_mag, tip_length=0.003/dir_mag, scale='auto')

        #         if topo1 >= 1000 or topo2 >= 1000:
        #             arrow.point_data['level_topo'] = G.nodes[node1]['level_topo']
        #         else:
        #             arrow.point_data['level_topo'] = min(G.nodes[edge[0]]['level_topo'], G.nodes[edge[1]]['level_topo'])

        #         pl.add_mesh(arrow, cmap= cmap1, clim = [0, 12], show_scalar_bar=False)


        # #plot nodes as spheres with level_topo values as colors
        # sph_rad = 0.0015
        # for node in G.nodes():
        #     pos_value = G.nodes[node]['pos']
        #     sph_mesh = pv.Sphere(radius = sph_rad, center = pos_value)
        #     sph_mesh.point_data['level_topo'] = G.nodes[node]['level_topo']
        #     pl.add_mesh(sph_mesh, cmap= cmap2, clim = [0, 12], show_scalar_bar=False)

        # #plot topology circle plot
        topo_graph = plot_topology(G)


        #plot edges
        for edge in topo_graph.edges():
            topo1 = topo_graph.nodes[edge[0]]['level_bound_topo']
            topo2 = topo_graph.nodes[edge[1]]['level_bound_topo']

            if topo1 >= 1000 and topo2 >= 1000:
                continue
            else:
                if topo1 < topo2:
                    node1 = edge[0]
                    node2 = edge[1]
                else:
                    node1 = edge[1]
                    node2 = edge[0]

                pos_value1 = topo_graph.nodes[node1]['plot_pos']
                pos_value2 = topo_graph.nodes[node2]['plot_pos']
                #add zero z value to all points
                pos_value1 = np.append(pos_value1, 0)
                pos_value2 = np.append(pos_value2, 0)
                dir_vector = (pos_value2 - pos_value1)/np.linalg.norm(pos_value2 - pos_value1)
                dir_mag = np.linalg.norm(pos_value2 - pos_value1)
                mag_vector = dir_vector*dir_mag
                arrow = pv.Arrow(pos_value1, mag_vector, shaft_radius=0.075/dir_mag, tip_radius=0.15/dir_mag, tip_length=0.35/dir_mag, scale='auto')

                if topo1 >= 1000 or topo2 >= 1000:
                    arrow.point_data['level_topo'] = topo_graph.nodes[node1]['level_topo']
                else:
                    arrow.point_data['level_topo'] = min(topo_graph.nodes[edge[0]]['level_topo'], G.nodes[edge[1]]['level_topo'])

                pl.add_mesh(arrow, cmap= cmap1, clim = [0, 12], show_scalar_bar=False)


        #plot nodes as spheres with level_topo values as colors
        sph_rad = 0.2
        for node in topo_graph.nodes():
            pos_value = topo_graph.nodes[node]['plot_pos']
            pos_value = np.append(pos_value, 0)
            sph_mesh = pv.Sphere(radius = sph_rad, center = pos_value)
            sph_mesh.point_data['level_topo'] = topo_graph.nodes[node]['level_topo']
            pl.add_mesh(sph_mesh, cmap= cmap2, clim = [0, 12], show_scalar_bar=False)


            # show the meshes

    pl.add_scalar_bar(title='Topological Depth', mapper = None, n_labels=5, italic=False, bold=False, title_font_size=None, label_font_size=None, color=None, font_family=None, shadow=False, width=None, height=None, position_x=None, position_y=None, vertical=None, interactive=None, fmt=None, use_opacity=True, outline=False, nan_annotation=False, below_label=None, above_label=None, background_color=None, n_colors=None, fill=False, render=False, theme=None)
    #enforce parallel projection
    pl.enable_parallel_projection()
    pl.camera.roll = -360.0

    if do_screenshot:
        filename_png = str(jj+1) + '_topo_plot.png'
        pl.screenshot(filename_png, window_size=(10000,10000), transparent_background= True)
    else:    
        pl.show()
