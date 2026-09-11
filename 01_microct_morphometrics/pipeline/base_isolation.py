# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Fit a smooth base surface to an aligned ossicle.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 2A (base-surface fit used for the curvature analysis); pipeline
                step
What it does  : Takes the clean_full_G skeleton nodes below z = 0.003 mm (base without
                pillars), interpolates them on a regular xy grid (scipy griddata),
                reconstructs a surface (vedo reconstruct_surface), remeshes it uniformly
                (pyacvd, 5000 points), smooths it and computes principal curvatures with
                libigl (radius 50). The active cell renders ossicle loc_file_id[115]
                with the fitted points; the commented cell loops over the pillared
                tables and records the export of the base meshes (aligned/base/vtk) read
                by curvature_calc.py.
Inputs        : microct/animal_1/extracted_ossicles/aligned/{vtk,
                skeleton/<ossicle>/(vox.mhd, *skel_line.vtk, *_G.gpickle)},
                microct/animal_1/extracted_ossicles/Data Calculated/{loc_file_id.txt,
                loc_file_id_table.txt, centroids.npy}
Outputs       : interactive PyVista window (the base-mesh export is commented out)
Environment   : environment-analysis.yml (Python 3.7)
Run           : python base_isolation.py
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
pv.global_theme.camera = {'position': [-1, -0.3, 0],'viewup': [0, 0, -1]} #[-1, -0.3, 0] [0,0,1]

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


#%%
base_dir =str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles")
type_dir =str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned")
indir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned" / "vtk")
skeldir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned" / "skeleton")
datadir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "Data Calculated")
basemeshdir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned" / "base")


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


with open(os.path.join(datadir,'loc_file_id.txt')) as txtfile:
    loc_file_id = list(map(int, txtfile.read().split('\n')))

with open(os.path.join(datadir,'loc_file_id_table.txt')) as txtfile:
    loc_file_id_table = list(map(int, txtfile.read().split('\n')))

with open(os.path.join(datadir, 'centroids.npy'), 'rb') as f:
    centroids = np.load(f)

pl = pv.Plotter(off_screen=False)


#%% reconstruct surface and save it

# meshes = []
# ctr = 0
# for ii in loc_file_id_table: #[loc_file_id[4]]: #range(n_files):      print(ii, files[ii])
#     ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)
#     # meshes.append(ossicle.lw(1))

#     #write skeleton only if there are non-zero number of lines in the skeleton
#     if len(skeleton.lines())!=0 and len(cl_skeleton.lines())!=0:
#         full_G, clean_full_G, simple_G, clean_simple_G = load_graphs(ii)

#         G = clean_simple_G

#         for u, v, data in G.edges(data=True):
#             if (data['branch_degree']==0):
#                 start_edge = [u,v]

#         all_pos =  np.array([pos for _, pos in clean_full_G.nodes(data='pos')])

#         fit_pos = np.delete(all_pos, np.where(all_pos[:,2] >= 0.003)[0], axis=0)
#         surf_fit_pos = surface_fitting_pts(fit_pos)

#         fit_sph_mesh = vedo.Spheres(fit_pos.tolist(), r=0.0005, c="r5", alpha=1, res=8)
#         surf_fit_sph_mesh = vedo.Spheres(surf_fit_pos.tolist(), r=0.0005, c="gray", alpha=1, res=8)

#         points_mesh = vedo.Points(surf_fit_pos)
#         recon_mesh = points_mesh.reconstruct_surface(dims=(100, 100, 100), padding=0.05).color('yellow').lw(1)

#         rem_recon_mesh = Mesh(remesh_func(pv.wrap(recon_mesh.polydata()), 1, 5000))
#         rem_recon_mesh.smooth(niter=100, pass_band=0.1, edge_angle=15, feature_angle=60)


#         # vedo.write(rem_recon_mesh, os.path.join(basemeshdir, file_names[ii][:-4]+'.vtk'), binary=False)

#         # #use libigl library to calculate curvature as it has a variable radius option
#         v = rem_recon_mesh.points()
#         f = np.array(rem_recon_mesh.cells())
#         pd1, pd2, pv1, pv2 = igl.principal_curvature(v, f, radius=50)
#         mean_curv = (pv1+pv2)/2
#         gauss_curv = pv1*pv2

#         rem_recon_mesh.cmap('viridis', (pv1+pv2)/2).add_scalarbar(title='mean curvature')

#         # meshes.append(fit_sph_mesh)
#         # meshes.append(surf_fit_sph_mesh)        
#         # meshes.append(rem_recon_mesh)

#         # # Get the mesh from the graph
#         # gmesh = graph_to_vedo_mesh(G)
#         # gmesh.lw(5)
#         # gmesh2 = gmesh.clone()
#         # gmesh2.ps(5).c('r')

#         # label = Text3D(file_id[ii], 0.01+center_pt, s=0.005, depth=0.5, c="g")
#         # label.follow_camera()

#         # meshes.extend([gmesh,gmesh2])
#         # meshes.extend([ossicle.alpha(0.5), start_mesh, center_mesh])
#         # meshes.append(skeleton)
#         # # # meshes.append(voxel_vol)
#         # meshes.append(label)

#         # # plane_mesh = vedo.Plane()
#         # meshes.append(ossicle_align)
#         # meshes.append(ossicle_base.c('orange'))

#         pv_ossicle = pv.wrap(ossicle.polydata())
#         pv_fit_sp = pv.wrap(fit_sph_mesh.polydata())
#         pv_surf_fit_sp = pv.wrap(surf_fit_sph_mesh.polydata())
#         pv_rem_recon = pv.wrap(rem_recon_mesh.polydata())

# #         pv_rem_recon.point_data.set_scalars(mean_curv, 'mean curvature')
# #         # pv_rem_recon.point_data.set_scalars(gauss_curv, 'gaussian curvature')

#         cmap = mpl.cm.get_cmap('viridis')
# #         # rgba = cmap(1/22 + 1/11)
# #         # osc_color = pv.Color([rgba[0], rgba[1], rgba[2] ])
#         # pl.add_mesh(pv_ossicle, color='violet', opacity = 0.25)#osc_color)#, specular=1.0, specular_power=10)
#         pl.add_mesh(pv_fit_sp, color='red')#osc_color)#, specular=1.0, specular_power=10)
#         # pl.add_mesh(pv_surf_fit_sp, color='grey')#osc_color)#, specular=1.0, specular_power=10)
#         pl.add_mesh(pv_rem_recon, cmap=cmap, clim = [-50,50])#osc_color)#, specular=1.0, specular_power=10)

#     ctr = ctr+1



# # # show the meshes
# # # vedo.show(meshes, axes=1)


# # # show the meshes

# # pl.parallel_projection = True
# pl.camera.roll = 90.0
# # # pl.screenshot('25_side.png', transparent_background= True, window_size=(10000,10000))
# pl.show()

#%% plot surface with ossicle

ctr = 0
for ii in [loc_file_id[115]]: #[loc_file_id[4]]: #range(n_files):      print(ii, files[ii])
    print(ii)
    ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)
    # meshes.append(ossicle.lw(1))

    #write skeleton only if there are non-zero number of lines in the skeleton
    if len(skeleton.lines())!=0 and len(cl_skeleton.lines())!=0:
        full_G, clean_full_G, simple_G, clean_simple_G = load_graphs(ii)

        G = clean_simple_G

        for u, v, data in G.edges(data=True):
            if (data['branch_degree']==0):
                start_edge = [u,v]

        all_pos =  np.array([pos for _, pos in clean_full_G.nodes(data='pos')])

        fit_pos = np.delete(all_pos, np.where(all_pos[:,2] >= 0.003)[0], axis=0)
        surf_fit_pos = surface_fitting_pts(fit_pos)

        fit_sph_mesh = vedo.Spheres(fit_pos.tolist(), r=0.0005, c="r5", alpha=1, res=8)
        surf_fit_sph_mesh = vedo.Spheres(surf_fit_pos.tolist(), r=0.0005, c="gray", alpha=1, res=8)

        points_mesh = vedo.Points(surf_fit_pos)
        recon_mesh = points_mesh.reconstruct_surface(dims=(100, 100, 100), padding=0.05).color('yellow').lw(1)

        rem_recon_mesh = Mesh(remesh_func(pv.wrap(recon_mesh.polydata()), 1, 5000))
        rem_recon_mesh.smooth(niter=100, pass_band=0.1, edge_angle=15, feature_angle=60)


        # vedo.write(rem_recon_mesh, os.path.join(basemeshdir, file_names[ii][:-4]+'.vtk'), binary=False)

        # #use libigl library to calculate curvature as it has a variable radius option
        v = rem_recon_mesh.points()
        f = np.array(rem_recon_mesh.cells())
        pd1, pd2, pv1, pv2 = igl.principal_curvature(v, f, radius=50)
        mean_curv = (pv1+pv2)/2
        gauss_curv = pv1*pv2

        rem_recon_mesh.cmap('viridis', (pv1+pv2)/2).add_scalarbar(title='mean curvature')

        # meshes.append(fit_sph_mesh)
        # meshes.append(surf_fit_sph_mesh)        
        # meshes.append(rem_recon_mesh)

        # # Get the mesh from the graph
        # gmesh = graph_to_vedo_mesh(G)
        # gmesh.lw(5)
        # gmesh2 = gmesh.clone()
        # gmesh2.ps(5).c('r')

        # label = Text3D(file_id[ii], 0.01+center_pt, s=0.005, depth=0.5, c="g")
        # label.follow_camera()

        # meshes.extend([gmesh,gmesh2])
        # meshes.extend([ossicle.alpha(0.5), start_mesh, center_mesh])
        # meshes.append(skeleton)
        # # # meshes.append(voxel_vol)
        # meshes.append(label)

        # # plane_mesh = vedo.Plane()
        # meshes.append(ossicle_align)
        # meshes.append(ossicle_base.c('orange'))

        pv_ossicle = pv.wrap(ossicle.polydata())
        pv_fit_sp = pv.wrap(fit_sph_mesh.polydata())
        pv_surf_fit_sp = pv.wrap(surf_fit_sph_mesh.polydata())
        pv_rem_recon = pv.wrap(rem_recon_mesh.polydata())

#         pv_rem_recon.point_data.set_scalars(mean_curv, 'mean curvature')
#         # pv_rem_recon.point_data.set_scalars(gauss_curv, 'gaussian curvature')

        cmap = mpl.cm.get_cmap('viridis')
#         # rgba = cmap(1/22 + 1/11)
#         # osc_color = pv.Color([rgba[0], rgba[1], rgba[2] ])
        pl.add_mesh(pv_ossicle, color='violet', opacity = 0.25)#osc_color)#, specular=1.0, specular_power=10)
        pl.add_mesh(pv_fit_sp, color='red')#osc_color)#, specular=1.0, specular_power=10)
        pl.add_mesh(pv_surf_fit_sp, color='grey')#osc_color)#, specular=1.0, specular_power=10)
        # pl.add_mesh(pv_rem_recon, cmap=cmap, clim = [-50,50])#osc_color)#, specular=1.0, specular_power=10)

        pset = pv.PolyData(np.array([0,0,0])+np.array([ctr/1000,ctr/1000,ctr/1000]))
        pset["label"] = [str(i) for i in range(pset.n_points)]
        pl.add_point_labels(pset, "label", italic=False, bold=True, font_size=200, text_color=None, font_family=None, shadow=False, show_points=False, point_color='red', point_size=10, name=None, shape_color='yellow', shape=None, fill_shape=True, margin=2, shape_opacity=0.8, pickable=False, render_points_as_spheres=False, tolerance=0.001, reset_camera=None, always_visible=True, render=True)


    ctr = ctr+1



# # show the meshes
# # vedo.show(meshes, axes=1)


# # show the meshes

pl.add_axes()
# pl.parallel_projection = True
pl.camera.roll = 90.0
# # pl.screenshot('25_side.png', transparent_background= True, window_size=(10000,10000))
pl.show()
