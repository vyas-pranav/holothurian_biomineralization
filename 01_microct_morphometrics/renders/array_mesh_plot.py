# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Gallery of all ossicles of one animal arranged by size.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 1D
What it does  : Loads all aligned ossicle meshes, sorts them by volume (largest first)
                and places them on concentric hexagonal rings, rendering them in orange
                with PyVista off-screen; saves a 10000 x 10000 px screenshot with
                transparent background. Commented cells record alternative layouts
                (grid, diagonal) and colouring by ossicle type.
Inputs        : microct/animal_1/extracted_ossicles/{vtk (file list),
                skeleton/<ossicle>/(vox.mhd, post15/clean skeletons), aligned/stl/*.stl,
                loc_file_id.txt, ossicle_type_list.txt, vol_list.txt}
Outputs       : spiral_all_wo_scale2.png in the working directory
Environment   : environment-analysis.yml (Python 3.7)
Run           : python array_mesh_plot.py
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
pv.global_theme.camera = {'position': [0, 0, 1],'viewup': [0, 0, -1]}

def load_data(ii):
    ossicle = load(al_files[ii])
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

def diagonal_coordinates(n):
  for i in range(n):
    for j in range(i+1):
      yield (j, i-j)


#%%
type_dir =str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles")
indir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "vtk")
skeldir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "skeleton")
aldir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned")

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

al_file_names = [f for f in os.listdir(os.path.join(aldir, 'stl')) if f.endswith('.stl')]
al_file_names.sort()
al_files = [os.path.join(aldir, 'stl',f) for f in al_file_names]

with open(os.path.join(type_dir,'loc_file_id.txt')) as txtfile:
    loc_file_id = list(map(int, txtfile.read().split('\n')))

with open(os.path.join(type_dir,'ossicle_type_list.txt')) as txtfile:
    ossicle_type_list = list(map(int, txtfile.read().split('\n')))

# done_file_names = os.listdir(os.path.join(type_dir, 'aligned', 'stl_files'))

# done_id_list = []
# for file in done_file_names:
#     ind = file_names.index(file)
#     done_id_list.append(ind)

with open(os.path.join(type_dir,'vol_list.txt')) as txtfile:
    vol_list = list(map(float, txtfile.read().split('\n')))

# table_vol_list = []
# for ii in loc_file_id:
#     table_vol_list.append(vol_list[ii])

# sort_ids = np.argsort(table_vol_list).tolist()

sort_ids = np.argsort(vol_list).tolist()

plt3d = vedo.Plotter()
pl = pv.Plotter(off_screen=True)
meshes = []
# ctr = 0
# for id in sort_ids: #range(n_files): #[left_id_list[1]]: #range(n_files):      print(ii, files[ii]) left_id_list: #
#     ii = loc_file_id[id]
#     ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)

#     ossicle.c('orange').lighting(style = 'plastic', ambient = False, diffuse = True).shift(dx = (ctr%12)/13, dy = ctr//12*1/13)
#     meshes.append(ossicle)

#     pv_ossicle = pv.wrap(ossicle.polydata())
#     pl.add_mesh(pv_ossicle)

#     ctr = ctr+1

# ctr = 0
# for x, y in diagonal_coordinates(17):
#     id = sort_ids[ctr]
#     ii = loc_file_id[id]
#     ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)

#     dx = (-x+y)*1/13*1/2
#     dy = (x+y)*1/13*np.sqrt(3)/2

#     ossicle.c('orange').lighting(style = 'plastic', ambient = False, diffuse = True).shift(dx=dx, dy=dy)
#     meshes.append(ossicle)

#     pv_ossicle = pv.wrap(ossicle.polydata())
#     pl.add_mesh(pv_ossicle)


#     ctr = ctr+1
#     if ctr > 20: #len(sort_ids) -1:
#         break


# ctr = 0
# n = 0
# sort_ids.reverse()
# for fid in sort_ids: #range(n_files): #[left_id_list[1]]: #range(n_files):      print(ii, files[ii]) left_id_list: #
#     ii = loc_file_id[fid]
#     osc_id = ossicle_type_list[fid]
#     ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)

#     if ctr > 3*(n+1)*n:
#         n = n+1


#     if n <= 1:        
#         rad = n*1/6.5
#     elif n == 2:        
#         rad = 1/6.5 + 1/8
#     elif n==3:
#         rad = 1/6.5 + 1/8 + 1/13
#     else:
#         rad = 1/6.5 + 1/8 + 1/13 + (n-3)*1/(13+(n-3)/4)

#     # rad = (n)*1/11

#     if n==0:
#         theta = 0
#     elif n%2 == 1:
#         theta = (((ctr-1)%(6*n)))*2*np.pi/(6*n)
#     else:
#         theta = (((ctr-1)%(6*n)))*2*np.pi/(6*n) + np.pi

#     dx = rad*np.cos(theta)
#     dy = rad*np.sin(theta)

#     ossicle.c('orange').lighting(style = 'plastic', ambient = False, diffuse = True).shift(dx=dx, dy=dy)
#     meshes.append(ossicle)

#     pv_ossicle = pv.wrap(ossicle.polydata())
#     cmap = mpl.cm.get_cmap('nipy_spectral_r')
#     rgba = cmap(1/22 + osc_id*1/11)
#     osc_color = pv.Color([rgba[0], rgba[1], rgba[2] ])

#     pl.add_mesh(pv_ossicle, color=osc_color)#, specular=1.0, specular_power=10)


#     ctr = ctr+1

ctr = 0
n = 0
sort_ids.reverse()
for fid in sort_ids: #range(n_files): #[left_id_list[1]]: #range(n_files):      print(ii, files[ii]) left_id_list: #
    print(ii)
    ii = loc_file_id[fid]
    osc_id = ossicle_type_list[fid]
    print(ii, file_names[ii], osc_id)
    ossicle, voxel_vol, skeleton, cl_skeleton = load_data(ii)

    if ctr > 3*(n+1)*n:
        n = n+1


    if n <= 1:        
        rad = n*1/6.5
    elif n == 2:        
        rad = 1/6.5 + 1/8
    elif n==3:
        rad = 1/6.5 + 1/8 + 1/13
    else:
        rad = 1/6.5 + 1/8 + 1/13 + (n-3)*1/(13+(n-3)/4)

    # rad = (n)*1/11

    if n==0:
        theta = 0
    elif n%2 == 1:
        theta = (((ctr-1)%(6*n)))*2*np.pi/(6*n)
    else:
        theta = (((ctr-1)%(6*n)))*2*np.pi/(6*n) + np.pi

    dx = rad*np.cos(theta)
    dy = rad*np.sin(theta)

    ossicle.c('orange').lighting(style = 'plastic', ambient = False, diffuse = True).shift(dx=dx, dy=dy)
    meshes.append(ossicle)

    pv_ossicle = pv.wrap(ossicle.polydata())
    cmap = mpl.cm.get_cmap('nipy_spectral_r')
    rgba = cmap(1/22 + osc_id*1/11)
    osc_color = pv.Color([rgba[0], rgba[1], rgba[2] ])

    pl.add_mesh(pv_ossicle, color='orange')#osc_color)#osc_color)#, specular=1.0, specular_power=10)


    ctr = ctr+1


print('done')
# show the meshes

pl.camera.roll = 0.0
pl.screenshot('spiral_all_wo_scale2.png', transparent_background= True, window_size=(10000,10000))


# pl.show()
