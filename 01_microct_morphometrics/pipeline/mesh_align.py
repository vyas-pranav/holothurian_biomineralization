# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Align every ossicle mesh to a reference ossicle.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Pipeline step (upstream of the micro-CT panels Fig. 1C-D, 2A, 2C-H,
                S2C-D, S3)
What it does  : The reference ossicle (file index 170) is shifted so that the midpoint
                of its thickest central edge (the origin) is at 0 and rotated so that
                the PCA normal of the origin's neighbouring nodes points along +z. Every
                ossicle not yet aligned is shifted to its own origin (voxel-volume
                centre if it has no skeleton), rotated 90 deg about x and rigidly
                aligned to the reference (vedo align_to, 100 iterations). Shows the
                result in a vedo window.
Inputs        : microct/animal_1/extracted_ossicles/{vtk, skeleton/<ossicle>/(vox.mhd,
                post15_skel_line.vtk, clean_skel_line.vtk,
                full_G/clean_full_G/simple_G/clean_simple_G.gpickle),
                loc_file_id_table.txt, aligned/stl_files/{table,other_types,no_skeleton}
                (already aligned meshes)}
Outputs       : microct/animal_1/extracted_ossicles/aligned/stl_files/*.stl (data tree);
                interactive vedo window
Environment   : environment-analysis.yml (Python 3.7)
Run           : python mesh_align.py
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
    grid_x, grid_y = np.mgrid[-0.05:0.05:N*1j, -0.05:0.05:N*1j]
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


#align ossicle 60 (170 id) to the z axis and use it as a reference to align all other ossicles
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


#%%
type_dir =str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles")
indir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "vtk")
skeldir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "skeleton")

plt3d = Plotter(bg2='gray2')#, interactive=True) # screen size
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


with open(os.path.join(type_dir,'loc_file_id_table.txt')) as txtfile:
    loc_file_id = list(map(int, txtfile.read().split('\n')))

ossicle_base = get_base_mesh()
# ossicle_base = load(os.path.join(type_dir, 'aligned', 'stl_files', file_names[170][0:-4]+'.stl'))

done_file_names = os.listdir(os.path.join(type_dir, 'aligned', 'stl_files', 'table'))
done_file_names2 = os.listdir(os.path.join(type_dir, 'aligned', 'stl_files', 'other_types'))
done_file_names3 = os.listdir(os.path.join(type_dir, 'aligned', 'stl_files', 'no_skeleton'))
done_file_names.extend(done_file_names2)
done_file_names.extend(done_file_names3)
done_file_names = [f[0:-4]+'.vtk' for f in done_file_names if f.endswith('.stl')]

left_file_names = [ ele for ele in file_names]
for a in done_file_names:
  if a in file_names:
    left_file_names.remove(a)

left_id_list = []
for left_file in left_file_names:
    ind = file_names.index(left_file)
    left_id_list.append(ind)

meshes = []
for ii in left_id_list: #[left_id_list[1]]: #range(n_files):      print(ii, files[ii]) left_id_list: #

    ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)

    #write skeleton only if there are non-zero number of lines in the skeleton
    if len(skeleton.lines())!=0 and len(cl_skeleton.lines())!=0:
        full_G, clean_full_G, simple_G, clean_simple_G = load_graphs(ii)

        G = clean_simple_G

        for u, v, data in G.edges(data=True):
            if (data['branch_degree']==0):
                start_edge = [u,v]

        # find the start point and volumetric centroid
        start_pt = (G.nodes[start_edge[0]]['pos'] + G.nodes[start_edge[1]]['pos'])/2 
    else:
        start_pt = voxel_vol.center()

    # plane_mesh = vedo.Plane()
    ossicle_rot = ossicle.clone().shift(dx=-start_pt[0],dy= -start_pt[1],dz= -start_pt[2]).rotate_x(90)
    ossicle_align = ossicle_rot.clone().align_to(ossicle_base, iters = 100, rigid = True)
    # ossicle_align = ossicle.clone().shift(dx=-start_pt[0],dy= -start_pt[1],dz= -start_pt[2]).align_to(ossicle_base, rigid = True)
    meshes.append(ossicle_align)
    meshes.append(ossicle_rot.c('blue'))
    meshes.append(ossicle.shift(dx=-start_pt[0],dy= -start_pt[1],dz= -start_pt[2]).c('green'))
    meshes.append(ossicle_base.c('orange'))
    # meshes.append(plane_mesh)

    ossicle_align.write(os.path.join(type_dir, 'aligned', 'stl_files', file_names[ii][0:-4]+'.stl'))

# show the meshes
vedo.show(meshes, axes=1)



#%%
