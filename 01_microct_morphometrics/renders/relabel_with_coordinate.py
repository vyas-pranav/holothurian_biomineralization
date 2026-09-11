# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Order ossicles along the body axis (centroid y).

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Input of the whole-animal renders
What it does  : Sorts the ossicle centroids by descending y, saves the sort order and
                draws the index labels at the sorted centroid positions, off-screen.
Inputs        : microct/animal_1/extracted_ossicles/loc_file_id.txt,
                microct/animal_1/extracted_ossicles/Data Calculated/centroids.npy,
                microct/animal_1/extracted_ossicles/{vtk, skeleton}
Outputs       : centroid_sorted_indices.npy in the working directory (copied to Data
                Calculated/ and loaded by whole_animal_coordinate_curvature.py)
Environment   : environment-analysis.yml (Python 3.7)
Run           : python relabel_with_coordinate.py
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
pv.global_theme.camera = {'position': [-1, -0.3, 2.2],'viewup': [0, 0, -1]}

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

    return full_G, clean_full_G, simple_G, clean_simple_G


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

def sort_points_by_y(points):
    """Returns a tuple of the sorted points array and a mapping of indices from the original array to the sorted array, based on the y-coordinate of the points."""
    sort_indices = np.argsort(points[:, 1])
    sorted_points = points[sort_indices]
    sort_indices = np.flip(sort_indices, 0)
    sorted_points = np.flip(sorted_points, 0)

    return sorted_points, sort_indices



#%%
base_dir =str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles")
# type_dir =str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned")
# indir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned" / "vtk")
# skeldir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned" / "skeleton")

type_dir =str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles")
indir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "vtk")
skeldir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "skeleton")
datadir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "Data Calculated")

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


with open(os.path.join(base_dir,'loc_file_id.txt')) as txtfile:
    loc_file_id = list(map(int, txtfile.read().split('\n')))

with open(os.path.join(datadir, 'centroids.npy'), 'rb') as f:
    centroids = np.load(f)

#sort based on centroid coordinates
sort_cent, sort_indices = sort_points_by_y(centroids)

plt3d = Plotter(bg2='gray2')#, interactive=True) # screen size
# pl = pv.Plotter(off_screen=False)
pl = pv.Plotter(off_screen=True)
light = pv.Light(position=(0, 0, 2), color='white')
light.positional = True
# pl.add_light(light)
meshes = []

#%%
# ctr = 0
# for jj in range(0,213): #[loc_file_id[4]]: #range(n_files):      print(ii, files[ii])

#     ii = loc_file_id[sort_indices[jj]]
#     ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)

#     if len(skeleton.lines())!=0 and len(cl_skeleton.lines())!=0:
#         full_G, clean_full_G, simple_G, clean_simple_G = load_graphs(ii)

#         G = clean_simple_G

#         for u, v, data in G.edges(data=True):
#             if (data['branch_degree']==0):
#                 start_edge = [u,v]

#         # find the start point and volumetric centroid
#         start_pt = (G.nodes[start_edge[0]]['pos'] + G.nodes[start_edge[1]]['pos'])/2 
#         centroid_pt = sort_cent[jj]

#         start_sph_mesh = vedo.Sphere(start_pt, r=0.006, res=16)
#         centroid_sph_mesh = vedo.Sphere(centroid_pt, r=0.006    , res=16)


#         pv_ossicle = pv.wrap(ossicle.polydata())
#         pv_start = pv.wrap(start_sph_mesh.polydata())
#         pv_centroid = pv.wrap(centroid_sph_mesh.polydata())

#         cmap = mpl.cm.get_cmap('Paired')
#         pl.add_mesh(pv_ossicle, color = 'whitesmoke', opacity = 1, diffuse = 0.9, smooth_shading = True)#, clim = [0,0.13])#osc_color)#, specular=1.0, specular_power=10)

#         pl.add_mesh(pv_start, color = 'steelblue')
        # pl.add_mesh(pv_centroid, color = 'steelblue')
        # pl.add_text(str(jj), position =0,0,0 , color='blue', shadow=True, font_size=26)

with open('centroid_sorted_indices.npy', 'wb') as gg:
    np.save(gg, sort_indices)

# show the meshes

pset = pv.PolyData(sort_cent[0:213])
pset["label"] = [str(i) for i in range(pset.n_points)]
pl.add_point_labels(pset, "label", italic=False, bold=True, font_size=200, text_color=None, font_family=None, shadow=False, show_points=False, point_color='red', point_size=10, name=None, shape_color='yellow', shape=None, fill_shape=True, margin=2, shape_opacity=0.8, pickable=False, render_points_as_spheres=False, tolerance=0.001, reset_camera=None, always_visible=True, render=True)

pl.add_axes()
pl.camera.roll = 180.0
# pl.screenshot('whole_ani_coord.png', window_size=(10000,10000), transparent_background= True)
# pl.show()
# 

#%%

# for ii in [loc_file_id[205]]: #[loc_file_id[4]]: #range(n_files):      print(ii, files[ii])

#     ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)
#     full_G, clean_full_G, simple_G, clean_simple_G = load_graphs(ii)

#     G = clean_simple_G

#     for u, v, data in G.edges(data=True):
#         if (data['branch_degree']==0):
#             start_edge = [u,v]

#     # find the start point and volumetric centroid
#     start_pt = (G.nodes[start_edge[0]]['pos'] + G.nodes[start_edge[1]]['pos'])/2 
#     start_mesh = vedo.Sphere(r=0.0005).pos(start_pt).color('k').alpha(0.5)

#     geoalg = geodesic.PyGeodesicAlgorithmExact(cl_skeleton.points(), cl_skeleton.lines())

#     # Use source and target point ids
#     pt1 = np.argwhere(np.all(cl_skeleton.points() == closest_point(cl_skeleton.points(), start_pt), axis=1)) 
#     pt2 = np.argwhere(np.all(cl_skeleton.points() == furthest_point_by_x(cl_skeleton.points()), axis=1)) 

#     # distance, path = geoalg.geodesicDistance(pt1[0][0], pt2[0][0])
#     # distances, _   = geoalg.geodesicDistances([pt1[0][0], pt2[0][0]]) # any of the two
#     # distance, path = geoalg.geodesicDistance(pt1[0][0], pt2[0][0])
#     center = np.array([0,0,0])
#     distances, best_source   = geoalg.geodesicDistances(pt1[0], None) # any of the two


#     # line = vedo.Line(path).c("k").lw(4)
#     # cl_skeleton.cmap("jet", distances, name="GeodesicDistance")
#     # vedo.show(cl_skeleton, axes=1)


#     center_sph_mesh = vedo.Sphere(cl_skeleton.points()[pt1[0][0]], r=0.0005, c="r5", alpha=1, res=8)
#     origin_sph_mesh = vedo.Sphere((0,0,0), r=0.005, c="r1", alpha=1, res=8)

#     pv_cl_skeleton = pv.wrap(cl_skeleton.polydata())
#     pv_center = pv.wrap(center_sph_mesh.polydata())
#     pv_origin = pv.wrap(origin_sph_mesh.polydata())
#     pv_cl_skeleton.point_data.set_scalars(distances, 'geodesic distances from center')

#     cmap = mpl.cm.get_cmap('tab20b')
#     # rgba = cmap(1/22 + 1/11)
#     # osc_color = pv.Color([rgba[0], rgba[1], rgba[2] ])
#     pl.add_mesh(pv_cl_skeleton, cmap=cmap, opacity = 1)#osc_color)#, specular=1.0, specular_power=10)
#     pl.add_mesh(pv_center, color='red')#osc_color)#, specular=1.0, specular_power=10)
#     # pl.add_mesh(pv_origin, color='orange')#osc_color)#, specular=1.0, specular_power=10)


# # show the meshes

# pl.camera.roll = 90.0
# # pl.screenshot('curved_top_sc.png', transparent_background= True, window_size=(10000,10000))
# pl.show()
